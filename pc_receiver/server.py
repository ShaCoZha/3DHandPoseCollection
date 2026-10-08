import argparse
import asyncio
import json
import re
import socket
import time
from pathlib import Path
from synchronized_capture import SynchronizedCapture


def parse_args():
    parser = argparse.ArgumentParser(
        description="Receive Apple Watch IMU samples and Quest pinch events over TCP JSONL or WebSocket"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pose-source", choices=("quest", "multicamera"), default="quest",
                        help="Synchronized capture source; multicamera replaces Quest, using the same phone protocol")
    parser.add_argument("--camera-config", type=Path, help="Multicamera JSON configuration (required for multicamera)")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Root directory for the default dataset layout.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional aggregate JSONL log for all IMU samples. Disabled by default.",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Optional directory for per-session files. "
        "Defaults to --dataset-dir.",
    )
    parser.add_argument(
        "--imudata-dir",
        default=None,
        help="Optional directory for per-session *_imudata.jsonl files. "
        "When unset together with --timestamp-dir, the default layout is "
        "--dataset-dir/<sessionId>/imudata.jsonl and timestamp.json.",
    )
    parser.add_argument(
        "--timestamp-dir",
        default=None,
        help="Optional directory for per-session *_timestamp.json files. "
        "When unset together with --imudata-dir, the default layout is "
        "--dataset-dir/<sessionId>/imudata.jsonl and timestamp.json.",
    )
    parser.add_argument(
        "--event-save",
        default=None,
        help="Optional JSONL output for discrete events such as Quest pinch detections. "
        "Disabled by default.",
    )
    parser.add_argument(
        "--quest",
        action="store_true",
        help="Enable Quest recording mode. Quest events are saved per session as "
        "dataset/<sessionId>/questlog.jsonl instead of an aggregate event log.",
    )
    parser.add_argument(
        "--protocol",
        choices=("tcp", "websocket"),
        default="tcp",
        help="Use plain TCP with newline-delimited JSON or WebSocket",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print every Nth sample index for quick monitoring",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=100.0,
        help="Target sample rate used for diagnostics and optional downsampling",
    )
    parser.add_argument(
        "--downsample-above-target",
        action="store_true",
        help="When enabled, keep at most the target rate per session based on unixTimeMs",
    )
    parser.add_argument(
        "--rate-log-seconds",
        type=float,
        default=1.0,
        help="Print estimated incoming/written sample rate every N seconds of sensor time",
    )
    return parser.parse_args()


def collect_local_ipv4_addresses():
    addresses = set()

    try:
        host_info = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        for item in host_info:
            ip = item[4][0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if not ip.startswith("127."):
                addresses.add(ip)
    except OSError:
        pass

    return sorted(addresses)


def log_banner(args):
    addresses = collect_local_ipv4_addresses()
    print(f"Listening with {args.protocol} on {args.host}:{args.port}")
    if addresses:
        for ip in addresses:
            if args.protocol == "tcp":
                print(f"Try Apple Watch target host: {ip}, port: {args.port}")
            else:
                print(f"Try Apple Watch URL: ws://{ip}:{args.port}")
    else:
        print("Could not infer LAN IP automatically; use your PC's Wi-Fi IPv4 address.")


def ensure_dataset_readme(dataset_dir: Path):
    readme_path = dataset_dir / "README.md"
    if readme_path.exists():
        return

    readme_text = """# Dataset Layout

This directory stores per-session Apple Watch IMU recordings received by `server.py`.

## Default structure

```
dataset/
  README.md
  <sessionId>/
    imudata.jsonl
    timestamp.json
    questlog.jsonl  # only when server.py runs with --quest
```

## File meanings

- `imudata.jsonl`: one raw IMU sample per line. Each line is a JSON object.
- `timestamp.json`: session-level metadata and prompt timing annotations.
- `questlog.jsonl`: Quest pinch events matched into this watch session when `--quest` is enabled.

## Raw sample fields

Each line in `imudata.jsonl` contains:

- `source`
- `sessionId`
- `sampleIndex`
- `unixTimeMs`
- `gestureLabel`
- `accelerometer`
- `gyroscope`
- `userAcceleration`
- `rotationRate`
- `gravity`
- `attitude`

## Timestamp metadata

`timestamp.json` contains:

- `sessionId`
- `gestureType`
- `recordingMode`
- `repeatCountTarget`
- `durationSecondsTarget`
- `countdownSeconds`
- `promptWindowSeconds`
- `startedAtUnixMs`
- `stoppedAtUnixMs`
- `completedPromptCount`
- `timestamps`: prompt windows with `promptIndex`, `countdownStartUnixMs`, `doUnixMs`, `windowEndUnixMs`

## Quest log

`questlog.jsonl` contains raw Quest event JSON objects, one per line. Each row preserves the Quest-side
`unixTimeMs` and also adds `serverReceivedUnixTimeMs` when the server receives the event. Events are matched
to a session by Apple Watch session id when available, otherwise by `unixTimeMs` within the watch recording window.

## Notes

- Aggregate logs are optional. Use `--save` and `--event-save` if you also want combined logs.
- Use `--quest` for per-session Quest logs. Do not combine `--quest` with `--event-save`.
- If you pass `--imudata-dir` or `--timestamp-dir`, the server switches to the explicit custom output paths instead of the default per-session folder layout.
"""
    readme_path.write_text(readme_text, encoding="utf-8")


class SampleWriter:
    RAW_SAMPLE_FIELDS = (
        "source",
        "sessionId",
        "sampleIndex",
        "unixTimeMs",
        "gestureLabel",
        "accelerometer",
        "gyroscope",
        "userAcceleration",
        "rotationRate",
        "gravity",
        "attitude",
        "deviceMonotonicMs",
        "rawWatchUnixTimeMs",
        "watchMinusPhoneMs",
        "clockSyncRttMs",
    )

    def __init__(
        self,
        save_path,
        session_dir: Path,
        imudata_dir: Path,
        timestamp_dir: Path,
        event_save_path,
        quest_mode: bool,
        use_session_folders: bool,
        print_every: int,
        target_hz: float,
        downsample_above_target: bool,
        rate_log_seconds: float,
    ):
        self.save_path = save_path
        self.session_dir = session_dir
        self.imudata_dir = imudata_dir
        self.timestamp_dir = timestamp_dir
        self.event_save_path = event_save_path
        self.quest_mode = quest_mode
        self.use_session_folders = use_session_folders
        self.print_every = max(1, print_every)
        self._lock = asyncio.Lock()
        self._handle = self.save_path.open("a", encoding="utf-8") if self.save_path else None
        self._event_handle = self.event_save_path.open("a", encoding="utf-8") if self.event_save_path else None
        self.target_hz = max(1e-6, target_hz)
        self.downsample_above_target = downsample_above_target
        self.target_interval_ms = 1000.0 / self.target_hz
        self.rate_log_window_ms = max(100, int(rate_log_seconds * 1000))
        self._session_states = {}
        self._session_outputs = {}
        self._pending_quest_events = []
        self._quest_clients = {}
        self._latest_quest_control = None
        self.capture = SynchronizedCapture(self)

    def _sanitize_filename_part(self, value):
        text = str(value).strip() or "unknown"
        return re.sub(r"[^A-Za-z0-9._-]+", "_", text)

    def _extract_raw_sample(self, message):
        return {
            field: message[field]
            for field in self.RAW_SAMPLE_FIELDS
            if field in message
        }

    def _get_gesture_type(self, message):
        return (
            message.get("recordingTarget")
            or message.get("gestureLabel")
            or "unknown_gesture"
        )

    def _get_output_key(self, message, client):
        return (
            message.get("sessionId") or str(client),
            self._get_gesture_type(message),
        )

    def _get_output_state(self, key):
        state = self._session_outputs.get(key)
        if state is not None:
            return state

        session_id, gesture_type = key
        session_part = self._sanitize_filename_part(session_id)
        gesture_part = self._sanitize_filename_part(gesture_type)

        if self.use_session_folders:
            session_path = self.session_dir / session_part
            session_path.mkdir(parents=True, exist_ok=True)
            imu_path = session_path / "imudata.jsonl"
            timestamp_path = session_path / "timestamp.json"
            quest_path = session_path / "questlog.jsonl" if self.quest_mode else None
        else:
            prefix = f"{session_part}_{gesture_part}"
            imu_path = self.imudata_dir / f"{prefix}_imudata.jsonl"
            timestamp_path = self.timestamp_dir / f"{prefix}_timestamp.json"
            quest_path = None

        state = {
            "session_id": session_id,
            "gesture_type": gesture_type,
            "imu_path": imu_path,
            "imu_handle": imu_path.open("a", encoding="utf-8"),
            "timestamp_path": timestamp_path,
            "timestamp_summary": {
                "sessionId": session_id,
                "gestureType": gesture_type,
            },
            "prompt_events": {},
            "observed_min_unix_ms": None,
            "observed_max_unix_ms": None,
            "quest_path": quest_path,
            "quest_handle": quest_path.open("a", encoding="utf-8") if quest_path is not None else None,
            "quest_event_ids": set(),
        }
        self._session_outputs[key] = state
        if quest_path is None:
            print(f"session files opened: {imu_path}, {timestamp_path}")
        else:
            print(f"session files opened: {imu_path}, {timestamp_path}, {quest_path}")
        return state

    def _update_timestamp_summary(self, output_state, message):
        summary = output_state["timestamp_summary"]
        changed = False

        field_map = {
            "source": "source",
            "recordingMode": "recordingMode",
            "recordingRepeatCountTarget": "repeatCountTarget",
            "recordingDurationSecondsTarget": "durationSecondsTarget",
            "recordingCountdownSeconds": "countdownSeconds",
            "recordingPromptWindowSeconds": "promptWindowSeconds",
            "recordingStartedAtUnixMs": "startedAtUnixMs",
            "recordingStoppedAtUnixMs": "stoppedAtUnixMs",
            "recordingCompletedPromptCount": "completedPromptCount",
        }

        for message_key, summary_key in field_map.items():
            value = message.get(message_key)
            if value is None:
                continue
            if summary.get(summary_key) != value:
                summary[summary_key] = value
                changed = True

        prompt_index = message.get("recordingPromptIndex")
        if isinstance(prompt_index, int):
            prompt_event = {
                "promptIndex": prompt_index,
                "countdownStartUnixMs": message.get("recordingPromptCountdownStartUnixMs"),
                "doUnixMs": message.get("recordingPromptDoUnixMs"),
                "windowEndUnixMs": message.get("recordingPromptWindowEndUnixMs"),
            }
            existing_prompt = output_state["prompt_events"].get(prompt_index)
            if existing_prompt != prompt_event:
                output_state["prompt_events"][prompt_index] = prompt_event
                changed = True

        if changed or "timestamps" not in summary:
            summary["timestamps"] = [
                output_state["prompt_events"][index]
                for index in sorted(output_state["prompt_events"])
            ]
            with output_state["timestamp_path"].open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

    def _update_observed_window(self, output_state, ts_ms):
        if not isinstance(ts_ms, int):
            return

        current_min = output_state["observed_min_unix_ms"]
        current_max = output_state["observed_max_unix_ms"]
        if current_min is None or ts_ms < current_min:
            output_state["observed_min_unix_ms"] = ts_ms
        if current_max is None or ts_ms > current_max:
            output_state["observed_max_unix_ms"] = ts_ms

    def _quest_event_identity(self, message):
        return (
            message.get("sessionId"),
            message.get("source"),
            message.get("unixTimeMs"),
            message.get("hand"),
            message.get("pinchFinger"),
            message.get("pinchEvent"),
            message.get("frameCount"),
            message.get("gameObjectName"),
        )

    def _event_session_candidates(self, message):
        return {
            value
            for value in (
                message.get("sessionId"),
                message.get("watchSessionId"),
                message.get("appleSessionId"),
            )
            if isinstance(value, str) and value
        }

    def _session_time_window(self, output_state):
        summary = output_state["timestamp_summary"]
        started_at = summary.get("startedAtUnixMs")
        stopped_at = summary.get("stoppedAtUnixMs")
        observed_min = output_state["observed_min_unix_ms"]
        observed_max = output_state["observed_max_unix_ms"]

        start_ts = started_at if isinstance(started_at, int) else observed_min
        stop_ts = stopped_at if isinstance(stopped_at, int) else observed_max
        if isinstance(start_ts, int) and isinstance(stop_ts, int):
            return start_ts, stop_ts

        return None

    def _quest_event_matches_output(self, message, output_state):
        if output_state["quest_handle"] is None:
            return False

        if output_state["session_id"] in self._event_session_candidates(message):
            return True

        event_ts = message.get("unixTimeMs")
        if not isinstance(event_ts, int):
            return False

        window = self._session_time_window(output_state)
        if window is None:
            return False

        start_ts, stop_ts = window
        return start_ts <= event_ts <= stop_ts

    def _write_quest_event_to_output(self, output_state, message):
        handle = output_state["quest_handle"]
        if handle is None:
            return False

        event_id = self._quest_event_identity(message)
        if event_id in output_state["quest_event_ids"]:
            return False

        handle.write(json.dumps(message, ensure_ascii=False) + "\n")
        handle.flush()
        output_state["quest_event_ids"].add(event_id)
        return True

    def _write_quest_event_to_matching_outputs(self, message):
        wrote = False
        for output_state in self._session_outputs.values():
            if self._quest_event_matches_output(message, output_state):
                wrote = self._write_quest_event_to_output(output_state, message) or wrote
        return wrote

    def _write_or_buffer_quest_event(self, message):
        if not self.quest_mode:
            return False

        if self._write_quest_event_to_matching_outputs(message):
            return True

        self._pending_quest_events.append(message)
        return False

    def _drain_pending_quest_events(self):
        if not self.quest_mode or not self._pending_quest_events:
            return

        still_pending = []
        for message in self._pending_quest_events:
            if not self._write_quest_event_to_matching_outputs(message):
                still_pending.append(message)
        self._pending_quest_events = still_pending

    def _get_session_key(self, message, client):
        return (
            message.get("sessionId") or str(client),
            message.get("source") or "unknown",
        )

    def _get_session_state(self, key):
        state = self._session_states.get(key)
        if state is None:
            state = {
                "next_keep_ts": None,
                "window_start_ts": None,
                "window_end_ts": None,
                "window_incoming": 0,
                "window_written": 0,
                "last_rate_log_monotonic": time.monotonic(),
                "seen_samples": set(),
            }
            self._session_states[key] = state
        return state

    def _update_rate_state(self, state, ts_ms, wrote_message):
        if not isinstance(ts_ms, int):
            return None

        if state["window_start_ts"] is None:
            state["window_start_ts"] = ts_ms
            state["window_end_ts"] = ts_ms

        state["window_incoming"] += 1
        if wrote_message:
            state["window_written"] += 1
        state["window_end_ts"] = ts_ms

        duration_ms = state["window_end_ts"] - state["window_start_ts"]
        if duration_ms < self.rate_log_window_ms:
            return None

        duration_ms = max(1, duration_ms)
        incoming_hz = state["window_incoming"] * 1000.0 / duration_ms
        written_hz = state["window_written"] * 1000.0 / duration_ms

        state["window_start_ts"] = ts_ms
        state["window_end_ts"] = ts_ms
        state["window_incoming"] = 0
        state["window_written"] = 0

        return incoming_hz, written_hz

    def _expand_messages(self, obj):
        if isinstance(obj, list):
            return [item for item in obj if isinstance(item, dict)]

        if isinstance(obj, dict) and isinstance(obj.get("samples"), list):
            chunk_meta = {
                "chunkType": obj.get("type", "imu_chunk"),
                "chunkIndex": obj.get("chunkIndex"),
                "chunkCreatedAtUnixMs": obj.get("createdAtUnixMs"),
                "chunkSampleCount": len(obj["samples"]),
                "relaySource": obj.get("relaySource", "iphone"),
            }
            messages = []
            for sample in obj["samples"]:
                if not isinstance(sample, dict):
                    continue
                merged = dict(sample)
                for key, value in chunk_meta.items():
                    if value is not None and key not in merged:
                        merged[key] = value
                messages.append(merged)
            return messages

        if isinstance(obj, dict):
            return [obj]

        return []

    def _client_key(self, client):
        return str(client)

    def _is_client_hello_message(self, message):
        return message.get("type") == "client_hello"

    def _is_quest_visual_client(self, message):
        return message.get("role") == "quest_visual_cue"

    def _is_quest_control_message(self, message):
        return message.get("type") == "quest_visual_cue"

    def _is_event_message(self, message):
        return message.get("type") == "quest_pinch_event"

    async def unregister_client(self, client):
        await self.capture.disconnect(client)
        async with self._lock:
            self._quest_clients.pop(self._client_key(client), None)

    async def _send_message_to_transport(self, message, transport, protocol):
        encoded = json.dumps(message, ensure_ascii=False)
        if protocol == "tcp":
            transport.write((encoded + "\n").encode("utf-8"))
            await transport.drain()
            return

        await transport.send(encoded)

    async def _forward_quest_control_message(self, message):
        delivered = 0
        stale_keys = []
        clients = list(self._quest_clients.items())
        for client_key, client_info in clients:
            try:
                await self._send_message_to_transport(
                    message,
                    client_info["transport"],
                    client_info["protocol"],
                )
                delivered += 1
            except Exception as exc:
                print(f"Quest control forward failed for {client_info['client']}: {exc}")
                stale_keys.append(client_key)

        if stale_keys:
            async with self._lock:
                for client_key in stale_keys:
                    self._quest_clients.pop(client_key, None)

        return delivered

    async def write_message(self, line: str, client, transport=None, protocol="tcp"):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            print(f"Bad JSON from {client}: {line[:200]}")
            return

        messages = self._expand_messages(obj)
        remaining = []
        for message in messages:
            if not await self.capture.handle(message, client, transport, protocol):
                remaining.append(message)
        if messages and not remaining:
            return
        messages = remaining
        if not messages:
            print(f"Unsupported JSON payload from {client}: {type(obj).__name__}")
            return

        sample_messages_to_write = []
        event_messages_to_write = []
        quest_control_messages_to_forward = []
        rate_logs = []
        latest_control_to_replay = None
        quest_client_registered = False

        async with self._lock:
            for message in messages:
                if self._is_client_hello_message(message):
                    if transport is not None and self._is_quest_visual_client(message):
                        self._quest_clients[self._client_key(client)] = {
                            "client": client,
                            "transport": transport,
                            "protocol": protocol,
                            "poseProtocol": message.get("poseProtocol"),
                        }
                        latest_control_to_replay = self._latest_quest_control
                        quest_client_registered = True
                    continue

                if self._is_quest_control_message(message):
                    self._latest_quest_control = message
                    quest_control_messages_to_forward.append(message)
                    continue

                if self._is_event_message(message):
                    if "serverReceivedUnixTimeMs" not in message:
                        message["serverReceivedUnixTimeMs"] = int(time.time() * 1000)
                    event_messages_to_write.append(message)
                    if self.quest_mode:
                        self._write_or_buffer_quest_event(message)
                    elif self._event_handle is not None:
                        self._event_handle.write(json.dumps(message, ensure_ascii=False) + "\n")
                    continue

                key = self._get_session_key(message, client)
                state = self._get_session_state(key)
                ts_ms = message.get("unixTimeMs")
                sample_index = message.get("sampleIndex")
                sample_identity = None

                if isinstance(sample_index, int):
                    sample_identity = (
                        message.get("sessionId"),
                        message.get("source"),
                        sample_index,
                    )

                output_key = self._get_output_key(message, client)
                output_state = self._get_output_state(output_key)
                self._update_timestamp_summary(output_state, message)
                self._update_observed_window(output_state, ts_ms)
                self._drain_pending_quest_events()

                should_write = True
                if sample_identity is not None:
                    if sample_identity in state["seen_samples"]:
                        should_write = False
                    else:
                        state["seen_samples"].add(sample_identity)

                if should_write and self.downsample_above_target and isinstance(ts_ms, int):
                    next_keep_ts = state["next_keep_ts"]
                    tolerance_ms = max(1.0, self.target_interval_ms * 0.2)

                    if next_keep_ts is None:
                        state["next_keep_ts"] = ts_ms + self.target_interval_ms
                    elif ts_ms < (next_keep_ts - tolerance_ms):
                        should_write = False
                    else:
                        next_keep_ts = next_keep_ts or float(ts_ms)
                        while next_keep_ts <= ts_ms:
                            next_keep_ts += self.target_interval_ms
                        state["next_keep_ts"] = next_keep_ts

                if should_write:
                    sample_messages_to_write.append(message)
                    if self._handle is not None:
                        self._handle.write(json.dumps(message, ensure_ascii=False) + "\n")
                    raw_sample = self._extract_raw_sample(message)
                    output_state["imu_handle"].write(json.dumps(raw_sample, ensure_ascii=False) + "\n")

                rate_log = self._update_rate_state(state, ts_ms, should_write)
                if rate_log is not None:
                    rate_logs.append((key, rate_log[0], rate_log[1]))

            if sample_messages_to_write:
                if self._handle is not None:
                    self._handle.flush()
                for output_key in {self._get_output_key(message, client) for message in sample_messages_to_write}:
                    self._session_outputs[output_key]["imu_handle"].flush()
            if event_messages_to_write and self._event_handle is not None:
                self._event_handle.flush()

        if quest_client_registered:
            print(f"registered Quest visual-cue client: {client}")
        if latest_control_to_replay is not None and transport is not None:
            try:
                await self._send_message_to_transport(latest_control_to_replay, transport, protocol)
            except Exception as exc:
                print(f"Quest control replay failed for {client}: {exc}")
                await self.unregister_client(client)

        for message in quest_control_messages_to_forward:
            delivered = await self._forward_quest_control_message(message)
            print(
                "quest_control",
                message.get("action"),
                "prompt=",
                message.get("promptIndex"),
                "session=",
                message.get("sessionId"),
                "clients=",
                delivered,
            )

        for key, incoming_hz, written_hz in rate_logs:
            session_id, source = key
            print(
                f"rate session={session_id} source={source} "
                f"in={incoming_hz:.1f}Hz out={written_hz:.1f}Hz target={self.target_hz:.1f}Hz"
            )

        for message in sample_messages_to_write:
            sample_index = message.get("sampleIndex")
            if isinstance(sample_index, int) and sample_index % self.print_every == 0:
                print(
                    "sample",
                    sample_index,
                    "client=",
                    client,
                    "gesture=",
                    message.get("gestureLabel"),
                    "acc=",
                    message.get("accelerometer"),
                    "gyro=",
                    message.get("gyroscope"),
                    "chunk=",
                    message.get("chunkIndex"),
                )

        for message in event_messages_to_write:
            print(
                "event",
                message.get("pinchEvent"),
                "client=",
                client,
                "hand=",
                message.get("hand"),
                "strength=",
                message.get("pinchStrength"),
                "session=",
                message.get("sessionId"),
            )

    def close(self):
        self.capture.close()
        if self._handle is not None:
            self._handle.close()
        if self._event_handle is not None:
            self._event_handle.close()
        for state in self._session_outputs.values():
            state["imu_handle"].close()
            if state["quest_handle"] is not None:
                state["quest_handle"].close()


async def run_tcp_server(args, writer: SampleWriter):
    async def handle_client(reader: asyncio.StreamReader, stream_writer: asyncio.StreamWriter):
        client = stream_writer.get_extra_info("peername")
        print(f"TCP client connected: {client}")
        stream_writer.write(b'{"type":"ready","protocol":"tcp-jsonl"}\n')
        await stream_writer.drain()

        try:
            while True:
                raw = await reader.readline()
                if not raw:
                    break

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                await writer.write_message(line, client, transport=stream_writer, protocol="tcp")
        except asyncio.IncompleteReadError:
            pass
        finally:
            print(f"TCP client disconnected: {client}")
            await writer.unregister_client(client)
            stream_writer.close()
            await stream_writer.wait_closed()

    server = await asyncio.start_server(handle_client, args.host, args.port)
    async with server:
        await server.serve_forever()


async def run_websocket_server(args, writer: SampleWriter):
    try:
        import websockets
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "WebSocket mode requires the 'websockets' package. "
            "Install it with: pip install websockets"
        ) from exc

    async def handler(websocket):
        client = websocket.remote_address
        print(f"WebSocket client connected: {client}")
        await websocket.send("ready")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    line = message.decode("utf-8", errors="replace")
                else:
                    line = message

                await writer.write_message(line.strip(), client, transport=websocket, protocol="websocket")
        except websockets.ConnectionClosed:
            pass
        finally:
            print(f"WebSocket client disconnected: {client}")
            await writer.unregister_client(client)

    async with websockets.serve(handler, args.host, args.port, max_size=None):
        await asyncio.Future()


async def main():
    args = parse_args()
    if args.quest and args.event_save:
        raise SystemExit("--quest writes per-session questlog.jsonl files; do not combine it with --event-save.")
    if args.quest and (args.imudata_dir or args.timestamp_dir):
        raise SystemExit("--quest requires the default per-session dataset layout. Use --dataset-dir or --session-dir instead.")

    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ensure_dataset_readme(dataset_dir)

    save_path = Path(args.save) if args.save else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    event_save_path = Path(args.event_save) if args.event_save else None
    if event_save_path is not None:
        event_save_path.parent.mkdir(parents=True, exist_ok=True)

    session_root_dir = Path(args.session_dir) if args.session_dir else dataset_dir
    use_session_folders = args.imudata_dir is None and args.timestamp_dir is None

    if use_session_folders:
        session_root_dir.mkdir(parents=True, exist_ok=True)
        imudata_dir = session_root_dir
        timestamp_dir = session_root_dir
    else:
        imudata_dir = Path(args.imudata_dir) if args.imudata_dir else session_root_dir
        timestamp_dir = Path(args.timestamp_dir) if args.timestamp_dir else session_root_dir
        imudata_dir.mkdir(parents=True, exist_ok=True)
        timestamp_dir.mkdir(parents=True, exist_ok=True)

    log_banner(args)
    print(f"Dataset root: {dataset_dir}")
    if use_session_folders:
        files = "imudata.jsonl,timestamp.json,questlog.jsonl" if args.quest else "imudata.jsonl,timestamp.json"
        print(f"Session folders: {session_root_dir}" + f" / <sessionId>/{{{files}}}")
    if save_path is not None:
        print(f"Aggregate IMU log: {save_path}")
    if event_save_path is not None:
        print(f"Aggregate event log: {event_save_path}")
    if args.quest:
        print("Quest mode: per-session Quest events are saved as questlog.jsonl")

    writer = SampleWriter(
        save_path,
        session_root_dir,
        imudata_dir,
        timestamp_dir,
        event_save_path,
        args.quest,
        use_session_folders,
        args.print_every,
        args.target_hz,
        args.downsample_above_target,
        args.rate_log_seconds,
    )
    if args.pose_source == "multicamera":
        if not args.camera_config or not use_session_folders:
            raise SystemExit("multicamera requires --camera-config and the default session layout")
        from multicamera_capture import CameraConfig
        writer.capture.camera_config = CameraConfig.load(args.camera_config)
        print("Synchronized pose source: multicamera; Quest is not required. Hand pose is processed offline.")

    try:
        if args.protocol == "tcp":
            await run_tcp_server(args, writer)
        else:
            await run_websocket_server(args, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    asyncio.run(main())
