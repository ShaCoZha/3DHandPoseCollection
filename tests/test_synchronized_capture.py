import asyncio
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, patch

from align_capture import aligned_time, calibration_points, align_file
from server import SampleWriter
from synchronized_capture import clock_ms, clock_sample


class SimulatedDevice:
    def __init__(self, writer, key, offset, arm_ok=True):
        self.writer, self.key, self.offset = writer, key, offset
        self.arm_ok = arm_ok
        self.messages = []
        self.tasks = set()

    def write(self, data):
        message = json.loads(data)
        self.messages.append(message)
        kind = message["type"]
        if kind == "clock_ping":
            now = clock_ms() + self.offset
            reply = dict(type="clock_pong", requestId=message["requestId"], receiveMs=now, sendMs=now)
        elif kind in ("pose_arm", "pose_commit"):
            reply = dict(type="pose_armed", requestId=message["requestId"], ok=self.arm_ok, error="No skeleton")
        else:
            return
        task = asyncio.create_task(self.submit(reply))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def drain(self):
        await asyncio.sleep(0)

    async def submit(self, message):
        await self.writer.write_message(json.dumps(message), self.key, self, "tcp")

    async def wait_message(self, kind):
        async def poll():
            while True:
                for message in self.messages:
                    if message["type"] == kind:
                        return message
                await asyncio.sleep(.005)
        return await asyncio.wait_for(poll(), 12)


class ClockTests(unittest.TestCase):
    def test_ntp_offset_excludes_remote_processing(self):
        sample = clock_sample(1000, 6050, 6060, 1110)
        self.assertEqual(sample["offsetMs"], 5000)
        self.assertEqual(sample["rttMs"], 100)

    def test_clock_validation(self):
        for values in ((1, 2, 1, 3), (1, 2, 3, float("nan")), (10, 20, 30, 5)):
            with self.assertRaises(ValueError):
                clock_sample(*values)

    def test_linear_clock_drift(self):
        # Watch runs 100 ppm fast, with an initial +5-second offset.
        rows = [dict(device="watch", phoneMonotonicMs=0, deviceMinusPhoneMs=5000),
                dict(device="watch", phoneMonotonicMs=900000, deviceMinusPhoneMs=5090)]
        points = calibration_points(rows, "watch")
        self.assertAlmostEqual(aligned_time(455045, points, 1700000000000), 1700000450000)
        self.assertEqual(aligned_time(4990, points, 0), -10)

    def test_alignment_preserves_raw_and_reports_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.jsonl"
            dest = Path(directory) / "aligned.jsonl"
            lines = [dict(deviceMonotonicMs=100, unixTimeMs=111, sampleIndex=0),
                     dict(deviceMonotonicMs=130, unixTimeMs=141, sampleIndex=3)]
            original = "".join(json.dumps(x) + "\n" for x in lines)
            source.write_text(original)
            result = align_file(source, dest, [(100, 10)], 1000)
            self.assertEqual(result["sequenceGaps"], 2)
            self.assertEqual(result["firstUnixTimeMs"], 1090)
            self.assertEqual(source.read_text(), original)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.writer = SampleWriter(None, self.root, self.root, self.root, None, False, True, 100, 100, False, 1)
        self.phone = SimulatedDevice(self.writer, "phone", 15000)
        self.quest = SimulatedDevice(self.writer, "quest", -32000)
        self.sid = str(uuid.uuid4())

    async def asyncTearDown(self):
        tasks = list(self.writer.capture.tasks)
        self.writer.close()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.temp.cleanup()

    async def prepare(self):
        await self.quest.submit(dict(type="client_hello", role="quest_visual_cue", poseProtocol=1))
        await self.phone.submit(dict(type="capture_prepare", sessionId=self.sid, phoneEpochOffsetMs=1700000000000,
            watchSync=dict(rttMs=4, phoneMonotonicMs=clock_ms()+15000, deviceMinusPhoneMs=500)))
        return await self.phone.wait_message("capture_prepared")

    async def test_clock_retries_one_timeout_then_uses_valid_replies(self):
        now = clock_ms()
        response = (dict(receiveMs=now + 100, sendMs=now + 100), now)
        peer = dict(key="quest", name="Quest")
        with patch.object(self.writer.capture, "request", new=AsyncMock(
                side_effect=[TimeoutError("cold start")] + [response] * 8)), \
             patch("synchronized_capture.clock_ms", return_value=now):
            result = await self.writer.capture.calibrate(peer)
        self.assertEqual(result["offsetMs"], 100)
        self.assertEqual(len(result["samples"]), 8)
        self.assertEqual(result["failedAttempts"], ["cold start"])

    async def test_high_rtt_is_accepted_and_preserved(self):
        now = clock_ms()
        response = (dict(receiveMs=now + 140, sendMs=now + 140), now + 80)
        with patch.object(self.writer.capture, "request", new=AsyncMock(return_value=response)), \
             patch("synchronized_capture.clock_ms", return_value=now):
            result = await self.writer.capture.calibrate(dict(key="quest", name="Quest"))
        self.assertEqual(result["rttMs"], 80)
        self.assertEqual(len(result["samples"]), 8)

    async def test_high_watch_rtt_allows_prepare_and_periodic_sync(self):
        await self.quest.submit(dict(type="client_hello", role="quest_visual_cue", poseProtocol=1))
        sync = dict(rttMs=450, phoneMonotonicMs=clock_ms()+15000, deviceMinusPhoneMs=500)
        await self.phone.submit(dict(type="capture_prepare", sessionId=self.sid,
                                    phoneEpochOffsetMs=1700000000000, watchSync=sync))
        await self.phone.wait_message("capture_prepared")
        await self.phone.submit(dict(type="watch_clock_sync", sessionId=self.sid,
                                    watchSync=dict(sync, rttMs=600)))
        rows = [json.loads(line) for line in (self.root/self.sid/"clock_sync.jsonl").read_text().splitlines()]
        self.assertEqual([row["rttMs"] for row in rows if row.get("device") == "watch"], [450, 600])

    async def test_invalid_watch_rtt_is_still_rejected(self):
        await self.quest.submit(dict(type="client_hello", role="quest_visual_cue", poseProtocol=1))
        for rtt in (-1, float("nan"), float("inf")):
            with self.subTest(rtt=rtt):
                self.phone.messages.clear()
                await self.phone.submit(dict(type="capture_prepare", sessionId=self.sid,
                                            phoneEpochOffsetMs=1, watchSync=dict(rttMs=rtt)))
                result = await self.phone.wait_message("capture_finished")
                self.assertEqual(result["status"], "failed")
                self.assertIn("Invalid Watch clock calibration", result["error"])
                self.assertIsNone(self.writer.capture.active)

    async def test_all_clock_timeouts_have_nonempty_diagnostic(self):
        with patch.object(self.writer.capture, "request", new=AsyncMock(side_effect=TimeoutError("no pong"))):
            with self.assertRaisesRegex(TimeoutError, r"iPhone.*0/8 valid replies.*no pong"):
                await self.writer.capture.calibrate(dict(key="phone", name="iPhone"))

    async def test_prepare_commit_pose_imu_and_early_stop(self):
        prepared = await self.prepare()
        self.assertAlmostEqual(prepared["endPhoneMs"] - prepared["startPhoneMs"], 900000, delta=0.001)
        state = self.writer.capture.active
        self.assertAlmostEqual(state["questMinusPhone"], -47000, delta=5)
        await self.phone.submit(dict(type="capture_commit", sessionId=self.sid))
        await self.phone.wait_message("capture_committed")
        self.assertTrue(state["committed"])
        # Sample at the same physical instant despite radically different device clocks.
        sample_phone = prepared["startPhoneMs"] + 12
        await self.quest.submit(dict(type="quest_hand_pose", sessionId=self.sid, sampleIndex=0,
                                    deviceMonotonicMs=sample_phone + state["questMinusPhone"], hands=[]))
        await self.phone.submit(dict(source="apple_watch", sessionId=self.sid, sampleIndex=0,
            unixTimeMs=sample_phone+state["epoch"], deviceMonotonicMs=sample_phone+500,
            watchMinusPhoneMs=500, clockSyncRttMs=4, gestureLabel="synchronized_pose"))
        root = self.root / self.sid
        pose = json.loads((root / "hand_pose.jsonl").read_text())
        imu = json.loads((root / "imudata.jsonl").read_text())
        self.assertAlmostEqual(pose["unixTimeMs"], imu["unixTimeMs"], delta=.01)
        self.assertIn("deviceMonotonicMs", imu)
        self.assertIn("watchMinusPhoneMs", imu)
        await self.phone.submit(dict(type="capture_stop", sessionId=self.sid))
        self.assertIsNone(self.writer.capture.active)
        self.assertEqual(json.loads((root / "capture.json").read_text())["status"], "stopped")
        self.assertFalse((self.root / "unknown").exists())

    async def test_missing_quest_rejects_without_recording(self):
        await self.phone.submit(dict(type="capture_prepare", sessionId=self.sid, phoneEpochOffsetMs=1,
                                    watchSync={"rttMs": 1}))
        result = await self.phone.wait_message("capture_finished")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(self.writer.capture.active)
        self.assertEqual(list(self.root.iterdir()), [])

    async def test_no_commit_aborts_armed_session(self):
        await self.prepare()
        result = await self.phone.wait_message("capture_finished")
        self.assertEqual(result["status"], "failed")
        self.assertIn("commit", result["error"])
        self.assertTrue(any(m["type"] == "pose_stop" for m in self.quest.messages))

    async def test_wrong_client_cannot_stop_or_inject_pose(self):
        await self.prepare()
        stranger = SimulatedDevice(self.writer, "stranger", 0)
        await stranger.submit(dict(type="capture_stop", sessionId=self.sid))
        self.assertIsNotNone(self.writer.capture.active)
        await stranger.submit(dict(type="quest_hand_pose", sessionId=self.sid, deviceMonotonicMs=0))
        self.assertFalse((self.root / self.sid / "hand_pose.jsonl").exists())

    async def test_disconnect_marks_partial(self):
        await self.prepare()
        await self.writer.unregister_client("quest")
        manifest = json.loads((self.root / self.sid / "capture.json").read_text())
        self.assertEqual(manifest["status"], "interrupted")
        self.assertIsNone(self.writer.capture.active)

    async def test_full_15_minute_deadline_and_half_open_sample_window(self):
        advanced = [0]
        with patch("synchronized_capture.clock_ms", side_effect=lambda: clock_ms() + advanced[0]):
            prepared = await self.prepare()
            await self.phone.submit(dict(type="capture_commit", sessionId=self.sid))
            await self.phone.wait_message("capture_committed")
            state = self.writer.capture.active
            for index, elapsed in enumerate((-1, 0, 899999, 900000)):
                await self.quest.submit(dict(type="quest_hand_pose", sessionId=self.sid, sampleIndex=index,
                    deviceMonotonicMs=prepared["startPhoneMs"] + state["questMinusPhone"] + elapsed, hands=[]))
            lines = (self.root / self.sid / "hand_pose.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["sampleIndex"] for line in lines], [1, 2])
            advanced[0] = 911000  # Simulated elapsed monotonic time, no 15-minute wall-clock wait.
            result = await self.phone.wait_message("capture_finished")
            self.assertEqual(result["status"], "finished")
            self.assertIsNone(self.writer.capture.active)

    async def test_quest_arm_failure(self):
        self.quest.arm_ok = False
        await self.quest.submit(dict(type="client_hello", role="quest_visual_cue", poseProtocol=1))
        await self.phone.submit(dict(type="capture_prepare", sessionId=self.sid, phoneEpochOffsetMs=1,
                                    watchSync={"rttMs": 1}))
        result = await self.phone.wait_message("capture_finished")
        self.assertEqual(result["status"], "failed")
        self.assertIn("skeleton", result["error"])

    async def test_legacy_cue_and_imu_still_work(self):
        await self.quest.submit(dict(type="client_hello", role="quest_visual_cue"))
        cue = dict(type="quest_visual_cue", sessionId=self.sid, action="show_target", ballIndex=1)
        await self.phone.submit(cue)
        self.assertIn(cue, self.quest.messages)
        await self.phone.submit(dict(source="apple_watch", sessionId=self.sid, sampleIndex=0,
                                     unixTimeMs=1000, gestureLabel="pinch"))
        self.assertTrue((self.root / self.sid / "imudata.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
