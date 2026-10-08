"""15-minute pose/IMU coordination. All clock arithmetic uses monotonic milliseconds."""
import asyncio
import json
import math
import time
import uuid
from pathlib import Path


def clock_ms():
    return time.monotonic_ns() / 1_000_000


def clock_sample(t0, t1, t2, t3):
    values = (t0, t1, t2, t3)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
        raise ValueError("Invalid clock timestamps")
    rtt = (t3 - t0) - (t2 - t1)
    if t2 < t1 or t3 < t0 or rtt < -1:
        raise ValueError("Invalid clock exchange")
    return dict(offsetMs=((t1 - t0) + (t2 - t3)) / 2,
                rttMs=max(0, rtt), serverMonotonicMs=(t0 + t3) / 2)


class SynchronizedCapture:
    def __init__(self, writer):
        self.writer = writer
        self.pending = {}
        self.tasks = set()
        self.active = None
        self.sessions = {}
        self.camera_config = None
        self.processing_lock = asyncio.Lock()

    def spawn(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def send(self, peer, message):
        await asyncio.wait_for(self.writer._send_message_to_transport(
            message, peer["transport"], peer["protocol"]), 3)

    async def request(self, peer, message, timeout=3):
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = (peer["key"], future)
        try:
            await self.send(peer, dict(message, requestId=request_id))
            return await asyncio.wait_for(future, timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise TimeoutError(f"{peer.get('name', peer['key'])}: {message['type']} reply/send timeout ({timeout:.1f}s)") from exc
        finally:
            self.pending.pop(request_id, None)

    async def calibrate(self, peer):
        samples = []
        failures = []
        name = peer.get("name", peer["key"])
        deadline = clock_ms() + 20000
        for attempt in range(16):
            remaining = (deadline - clock_ms()) / 1000
            if remaining <= 0:
                break
            t0 = clock_ms()
            try:
                reply, t3 = await self.request(peer, {"type": "clock_ping"}, timeout=min(3, remaining))
                samples.append(clock_sample(t0, reply["receiveMs"], reply["sendMs"], t3))
            except (TimeoutError, asyncio.TimeoutError) as exc:
                failures.append(str(exc))
                print(f"clock_sync {name}: attempt {attempt + 1} timed out; retrying", flush=True)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{name}: invalid clock reply; update the device app ({exc})") from exc
            if len(samples) >= 8:
                break
        if len(samples) < 8:
            detail = failures[-1] if failures else "calibration time budget exhausted"
            raise TimeoutError(f"{name} clock sync failed: {len(samples)}/8 valid replies; {detail}")
        best = min(samples, key=lambda s: s["rttMs"])
        print(f"clock_sync {name}: {len(samples)} valid replies, {len(failures)} timeouts, "
              f"best RTT={best['rttMs']:.2f} ms", flush=True)
        return dict(best, samples=samples, failedAttempts=failures)

    def log(self, state, name, message):
        with (state["path"] / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")

    def manifest(self, state, **updates):
        state["manifest"].update(updates)
        path = state["path"] / "capture.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state["manifest"], indent=2), encoding="utf-8")
        temp.replace(path)

    async def measure_pair(self, state):
        if state.get("camera"):
            phone = await self.calibrate(state["phone"])
            state["phoneSync"] = phone
            sample = dict(device="pc", deviceMinusPhoneMs=-phone["offsetMs"],
                          phoneMonotonicMs=phone["serverMonotonicMs"] + phone["offsetMs"],
                          phoneEpochOffsetMs=state["epoch"], phone=phone,
                          uncertaintyMs=phone["rttMs"] / 2)
            self.log(state, "clock_sync.jsonl", sample)
            return sample
        results = await asyncio.gather(self.calibrate(state["phone"]),
                                       self.calibrate(state["quest"]), return_exceptions=True)
        errors = []
        for name, result in zip(("iPhone", "Quest"), results):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                error = str(result) or type(result).__name__
                self.log(state, "clock_sync.jsonl", dict(type="sync_error", device=name, error=error))
                errors.append(error)
        if errors:
            raise ValueError("; ".join(errors))
        phone, quest = results
        sample = dict(device="quest", deviceMinusPhoneMs=quest["offsetMs"] - phone["offsetMs"],
                      phoneMonotonicMs=phone["serverMonotonicMs"] + phone["offsetMs"],
                      phoneEpochOffsetMs=state["epoch"], phone=phone, quest=quest,
                      uncertaintyMs=(phone["rttMs"] + quest["rttMs"]) / 2)
        self.log(state, "clock_sync.jsonl", sample)
        state["phoneSync"], state["questSync"] = phone, quest
        # Keep the initial online map stable; later measurements support offline drift correction.
        return sample

    async def prepare(self, state, request):
        try:
            if state.get("camera"):
                await state["camera"].open()
            sample = await self.measure_pair(state)
            start_server = clock_ms() + 10000
            start_phone = start_server + state["phoneSync"]["offsetMs"]
            state.update(startServer=start_server, startPhone=start_phone, endPhone=start_phone + 900000,
                         questMinusPhone=sample["deviceMinusPhoneMs"])
            if state.get("camera"):
                reply = await state["camera"].arm(start_server, start_server + 900000,
                                                   state["phoneSync"], state["epoch"])
            else:
                reply, _ = await self.request(state["quest"], dict(
                    type="pose_arm", sessionId=state["id"],
                    startMs=start_phone + state["questMinusPhone"],
                    endMs=start_phone + state["questMinusPhone"] + 900000), timeout=4)
            if not reply.get("ok"):
                raise ValueError(reply.get("error", "Quest pose recorder not ready"))
            self.manifest(state, status="armed", startPhoneMonotonicMs=start_phone,
                          endPhoneMonotonicMs=state["endPhone"],
                          startUnixTimeMs=start_phone + state["epoch"],
                          endUnixTimeMs=state["endPhone"] + state["epoch"],
                          **({"pcMinusPhoneMs": sample["deviceMinusPhoneMs"]} if state.get("camera")
                             else {"questMinusPhoneMs": state["questMinusPhone"]}))
            await self.send(state["phone"], dict(type="capture_prepared", sessionId=state["id"],
                            startPhoneMs=start_phone, endPhoneMs=state["endPhone"],
                            uncertaintyMs=sample["uncertaintyMs"]))
            # A lost Watch acknowledgment must never silently start a Quest-only recording.
            await asyncio.sleep(max(0, (start_server - clock_ms() - 1500) / 1000))
            if not state.get("committed"):
                raise ValueError("Phone/Watch did not commit before the start deadline")
            self.manifest(state, status="recording")
            while clock_ms() < start_server + 900000:
                await asyncio.sleep(min(60, max(0, (start_server + 900000 - clock_ms()) / 1000)))
                try:
                    await self.measure_pair(state)
                except Exception as exc:
                    self.log(state, "clock_sync.jsonl", dict(type="sync_error", error=str(exc)))
            await self.finish(state, "finished")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.finish(state, "failed", str(exc))

    async def finish(self, state, status, error=None):
        if state.get("closed"):
            return
        state["closed"] = True
        print(f"capture {state['id']}: {status}" + (f" - {error}" if error else ""), flush=True)
        task = state.get("task")
        if task and task is not asyncio.current_task():
            task.cancel()
        if state.get("camera"):
            try:
                result = await state["camera"].stop()
                self.manifest(state, cameraStatus=result)
                if result.get("error"):
                    status, error = "failed", result["error"]
            except Exception as exc:
                status, error = "failed", f"Camera finalization: {exc}"
        self.manifest(state, status=status, error=error,
                      poseFrames=state.get("frames", 0),
                      finishedServerUnixTimeMs=int(time.time() * 1000))
        if self.active is state:
            self.active = None
        for peer, message in (
            (state["quest"], dict(type="pose_stop", sessionId=state["id"])),
            (state["phone"], dict(type="capture_finished", sessionId=state["id"],
                                   status=status, error=error or "")),
        ):
            if peer is None:
                continue
            try:
                await self.send(peer, message)
            except Exception:
                pass
        if state.get("camera") and state.get("committed") and status in ("stopped", "finished") and not error:
            self.spawn(self.process_camera(state))

    async def process_camera(self, state):
        self.manifest(state, handPoseProcessing="queued")
        async with self.processing_lock:
            self.manifest(state, handPoseProcessing="processing")
            try:
                result = await state["camera"].process_handpose()
                self.manifest(state, handPoseProcessing=result)
            except Exception as exc:
                self.manifest(state, handPoseProcessing="failed", handPoseError=str(exc))

    async def handle(self, message, client, transport, protocol):
        kind = message.get("type", "")
        key = self.writer._client_key(client)
        if kind in ("clock_pong", "pose_armed"):
            pending = self.pending.get(message.get("requestId"))
            if pending and pending[0] == key and not pending[1].done():
                pending[1].set_result((message, clock_ms()))
            return True
        if kind == "capture_prepare":
            phone = dict(key=key, name="iPhone", transport=transport, protocol=protocol)
            sid = message.get("sessionId", "")
            try:
                if not self.writer.use_session_folders:
                    raise ValueError("Pose capture requires the default per-session dataset layout")
                if not isinstance(sid, str) or not re_session_id(sid):
                    raise ValueError("Invalid session ID")
                if self.active:
                    raise ValueError("Another synchronized capture is active")
                quests = [dict(info, key=k, name="Quest") for k, info in self.writer._quest_clients.items()
                          if info.get("poseProtocol") == 1]
                if self.camera_config is None and len(quests) != 1:
                    raise ValueError("Connect exactly one Quest pose-capable client")
                epoch = float(message["phoneEpochOffsetMs"])
                watch_sync = message["watchSync"]
                if (not math.isfinite(epoch) or
                        not isinstance(watch_sync.get("rttMs"), (float, int)) or
                        not math.isfinite(watch_sync["rttMs"]) or watch_sync["rttMs"] < 0):
                    raise ValueError("Invalid Watch clock calibration")
                path = self.writer.session_dir / sid
                if (path / "capture.json").exists():
                    raise ValueError("Session already exists; start a new recording")
                path.mkdir(parents=True, exist_ok=True)
                state = dict(id=sid, path=path, phone=phone, quest=quests[0] if self.camera_config is None else None, epoch=epoch,
                             manifest=dict(schemaVersion=1, sessionId=sid, durationSeconds=900,
                                           phoneEpochOffsetMs=epoch, poseTimestampMeaning="Unity LateUpdate observation",
                                           coordinateSpace="Unity world, metres, quaternion xyzw"))
                if self.camera_config is not None:
                    from multicamera_capture import CameraRecorder
                    state["camera"] = CameraRecorder(self.camera_config, path,
                        lambda error: self.finish(state, "interrupted", error))
                    state["manifest"].update(schemaVersion=2, poseSource="multicamera",
                        poseTimestampMeaning="Camera hardware timestamps mapped to PC then iPhone; offline triangulation",
                        coordinateSpace="Camera calibration world, metres; not Watch spatial coordinates",
                        handPoseProcessing="pending", hardwareExposureSynchronized=False)
                self.active = state
                self.sessions[sid] = state
                self.log(state, "clock_sync.jsonl", dict(watch_sync, device="watch", phoneEpochOffsetMs=epoch))
                self.manifest(state, status="synchronizing")
                state["task"] = self.spawn(self.prepare(state, message))
            except (ValueError, KeyError, TypeError) as exc:
                await self.send(phone, dict(type="capture_finished", sessionId=sid, status="failed", error=str(exc)))
            return True
        if kind in ("capture_commit", "capture_stop", "watch_clock_sync"):
            state = self.sessions.get(message.get("sessionId"))
            if not state or state["phone"]["key"] != key or state.get("closed"):
                return True
            if kind == "capture_commit":
                # Commit is acknowledged by Quest before phone reports the session armed.
                self.spawn(self.commit(state))
            elif kind == "capture_stop":
                self.manifest(state, stopReason=message.get("reason"))
                await self.finish(state, "stopped")
            else:
                sync = message.get("watchSync", {})
                if (isinstance(sync.get("rttMs"), (float, int)) and
                        math.isfinite(sync["rttMs"]) and sync["rttMs"] >= 0):
                    self.log(state, "clock_sync.jsonl", dict(sync, device="watch", phoneEpochOffsetMs=state["epoch"]))
            return True
        if kind == "quest_hand_pose":
            state = self.sessions.get(message.get("sessionId"))
            if not state or not state.get("quest") or state["quest"]["key"] != key or not state.get("committed"):
                return True
            raw = message.get("deviceMonotonicMs")
            if not isinstance(raw, (int, float)) or not math.isfinite(raw):
                return True
            phone_ms = raw - state["questMinusPhone"]
            # Compare in the device clock to avoid cancellation rounding at end.
            if not state["startPhone"] + state["questMinusPhone"] <= raw < state["endPhone"] + state["questMinusPhone"]:
                return True
            self.log(state, "hand_pose.jsonl", dict(message,
                     unixTimeMs=phone_ms + state["epoch"], phoneMonotonicMs=phone_ms,
                     serverReceivedUnixTimeMs=time.time_ns() / 1e6))
            state["frames"] = state.get("frames", 0) + 1
            return True
        if kind == "pose_status":
            state = self.sessions.get(message.get("sessionId"))
            if state and state.get("quest") and state["quest"]["key"] == key:
                self.log(state, "pose_status.jsonl", message)
                self.manifest(state, questStatus=message, poseFrames=state.get("frames", 0))
                if message.get("reason") not in ("completed", "stopped") and not state.get("closed"):
                    await self.finish(state, "interrupted", "Quest: " + str(message.get("reason")))
            return True
        return False

    async def commit(self, state):
        try:
            if state.get("closed") or state.get("committed"):
                return
            if clock_ms() >= state.get("startServer", 0) - 1500:
                raise ValueError("Commit arrived after the start deadline")
            if state.get("committing"):
                return
            state["committing"] = True
            if state.get("camera"):
                reply = await state["camera"].commit()
            else:
                reply, _ = await self.request(state["quest"], dict(type="pose_commit", sessionId=state["id"]), 3)
            if not reply.get("ok"):
                raise ValueError(reply.get("error", "Quest commit failed"))
            if state.get("closed"):
                return
            state["committed"] = True
            await self.send(state["phone"], dict(type="capture_committed", sessionId=state["id"]))
        except Exception as exc:
            await self.finish(state, "failed", str(exc))

    async def disconnect(self, client):
        key = self.writer._client_key(client)
        for expected_key, future in list(self.pending.values()):
            if key == expected_key and not future.done():
                future.set_exception(ConnectionError("Device disconnected"))
        state = self.active
        if state and key in (state["phone"]["key"], (state.get("quest") or {}).get("key")):
            await self.finish(state, "interrupted", "Device disconnected; inspect partial data")

    def close(self):
        for task in self.tasks:
            task.cancel()
        if self.active:
            if self.active.get("camera"):
                self.active["camera"].close()
            self.manifest(self.active, status="interrupted", error="Receiver shut down")


def re_session_id(value):
    # Phone generates UUIDs; strict validation also prevents path traversal.
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False
