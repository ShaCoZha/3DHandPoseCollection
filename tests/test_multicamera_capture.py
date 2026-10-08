import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from align_multicamera import pair_frames, align_session
import test_synchronized_capture as fixtures


class FakeRecorder:
    def __init__(self, config, path, on_error):
        self.opened = self.committed = self.stopped = False
        self.on_error = on_error

    async def open(self):
        self.opened = True

    async def arm(self, start, end, sync, epoch):
        self.window = (start,end)
        return {'ok': True}

    async def commit(self):
        self.committed = True
        return {'ok': True}

    async def stop(self):
        self.stopped = True
        return {'error': None, 'cameras': {}}

    def close(self):
        self.stopped = True

    async def process_handpose(self):
        return 'complete'


class CameraProtocolTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.CaptureTests.asyncSetUp
    asyncTearDown = fixtures.CaptureTests.asyncTearDown
    async def test_camera_mode_without_quest_and_no_start_before_commit(self):
        self.writer.capture.camera_config = object()
        with patch('multicamera_capture.CameraRecorder', FakeRecorder):
            await self.phone.submit(dict(type='capture_prepare', sessionId=self.sid,
                phoneEpochOffsetMs=1700000000000, watchSync={'rttMs': 2}))
            await self.phone.wait_message('capture_prepared')
            state = self.writer.capture.active
            camera = state['camera']
            self.assertTrue(camera.opened)
            self.assertFalse(camera.committed)
            self.assertIsNone(state['quest'])
            records = [json.loads(x) for x in (self.root/self.sid/'clock_sync.jsonl').read_text().splitlines()]
            pc = next(r for r in records if r['device'] == 'pc')
            self.assertAlmostEqual(pc['deviceMinusPhoneMs'], -15000, delta=5)
            stranger = fixtures.SimulatedDevice(self.writer,'stranger',0)
            await stranger.submit(dict(type='capture_commit',sessionId=self.sid))
            self.assertFalse(camera.committed)
            await self.phone.submit(dict(type='capture_commit',sessionId=self.sid))
            await self.phone.wait_message('capture_committed')
            self.assertTrue(camera.committed)
            await self.phone.submit(dict(type='capture_stop',sessionId=self.sid,reason='user_stop'))
            await asyncio.sleep(.01)
            self.assertTrue(camera.stopped)
            manifest = json.loads((self.root/self.sid/'capture.json').read_text())
            self.assertEqual(manifest['poseSource'],'multicamera')
            self.assertEqual(manifest['status'],'stopped')
            self.assertEqual(manifest['handPoseProcessing'],'complete')

    async def test_camera_failure_and_phone_disconnect_release_camera(self):
        self.writer.capture.camera_config = object()
        with patch('multicamera_capture.CameraRecorder', FakeRecorder):
            await self.phone.submit(dict(type='capture_prepare', sessionId=self.sid,
                phoneEpochOffsetMs=1, watchSync={'rttMs': 2}))
            await self.phone.wait_message('capture_prepared')
            state = self.writer.capture.active
            await self.writer.unregister_client('phone')
            self.assertTrue(state['camera'].stopped)
            self.assertEqual(state['manifest']['status'],'interrupted')

    async def test_camera_initialization_failure_never_prepares_or_commits(self):
        self.writer.capture.camera_config = object()
        class BrokenRecorder(FakeRecorder):
            async def open(self):
                raise RuntimeError('Missing camera 25132928')
        with patch('multicamera_capture.CameraRecorder', BrokenRecorder):
            await self.phone.submit(dict(type='capture_prepare',sessionId=self.sid,
                phoneEpochOffsetMs=1,watchSync={'rttMs':2}))
            result = await self.phone.wait_message('capture_finished')
            self.assertEqual(result['status'],'failed')
            self.assertIn('Missing camera',result['error'])
            self.assertFalse(any(m['type']=='capture_prepared' for m in self.phone.messages))
            self.assertIsNone(self.writer.capture.active)


class PairingTests(unittest.TestCase):
    def test_bracketing_group_recovers_free_running_camera_phases(self):
        streams = {name: [dict(unixTimeMs=50*i+phase, sampleIndex=i)
                          for i in range(8)]
                   for name, phase in zip('abcd', (0, 38, 27, 19))}
        result = list(pair_frames(streams, 35))
        self.assertGreaterEqual(len(result), 6)
        self.assertTrue(all(p['cameraSkewMs'] <= 35 for p in result))
        for name in streams:
            indices = [p['frames'][name]['sampleIndex'] for p in result]
            self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(p['unixTimeMs'] == p['frames']['a']['unixTimeMs'] for p in result))

    def test_pairs_use_timestamps_never_reuse_or_force_large_skew(self):
        def rows(times):
            return [dict(unixTimeMs=t, sampleIndex=i) for i,t in enumerate(times)]
        streams = {'a':rows([0,50,100,150]), 'b':rows([3,104,155]),
                   'c':rows([1,51,101,151]), 'd':rows([2,52,102,152])}
        result = list(pair_frames(streams,10))
        self.assertEqual([r['unixTimeMs'] for r in result],[0,100,150])
        self.assertEqual([r['frames']['b']['sampleIndex'] for r in result],[0,1,2])
        self.assertTrue(all(r['cameraSkewMs']<=10 for r in result))

    def test_composed_drift_alignment_preserves_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cam = root/'multicamera'; cam.mkdir()
            (root/'capture.json').write_text(json.dumps(dict(status='finished',phoneEpochOffsetMs=100000)))
            # PC clock +1000 relative to phone, camera clock +5000 relative to PC.
            (root/'clock_sync.jsonl').write_text(json.dumps(dict(device='pc',phoneMonotonicMs=100,
                deviceMinusPhoneMs=1000,uncertaintyMs=1))+'\n')
            (cam/'config.json').write_text(json.dumps(dict(cameras=[{'name':f'cam0{i}'} for i in range(1,5)],maxPairSkewMs=10)))
            raw = json.dumps(dict(cameraTimestampNs=6100*1000000,cameraFrameId=10,sampleIndex=0,unixTimeMs=0))+'\n'
            for i in range(1,5):
                (cam/f'cam0{i}_clock.jsonl').write_text(json.dumps(dict(cameraTimestampNs=6000*1000000,
                    cameraMinusPcMs=5000,uncertaintyMs=2))+'\n')
                (cam/f'cam0{i}_frames.jsonl').write_text(raw)
            report = align_session(root)
            self.assertEqual(report['pairedFrames'],1)
            result = json.loads((cam/'cam01_frames_aligned.jsonl').read_text())
            self.assertEqual(result['unixTimeMs'],100100)
            self.assertEqual(result['clockUncertaintyEstimateMs'],3)
            self.assertEqual((cam/'cam01_frames.jsonl').read_text(),raw)


if __name__ == '__main__':
    unittest.main()
