"""Async control bridge. PySpin runs in its own Python environment/process."""
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraConfig:
    path: Path
    data: dict

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        for key in ("python", "calibration", "wilorPython", "aniposePython", "detectorScript"):
            value = Path(data[key]).expanduser()
            value = value if value.is_absolute() else path.parent / value
            if not value.is_file():
                raise ValueError(f"Missing {key}: {value}")
            data[key] = str(value.resolve()) if key not in ("python", "wilorPython", "aniposePython") else str(value.absolute())
        if data.get('regularizeHandpose',False):
            value=Path(data.get('regularizationPython',data['wilorPython'])).expanduser()
            value=value if value.is_absolute() else path.parent/value
            if not value.is_file():raise ValueError(f'Missing regularizationPython: {value}')
            data['regularizationPython']=str(value.absolute())
        cams = data["cameras"]
        if (len(cams) != 4 or len({c['serial'] for c in cams}) != 4 or
                [c['name'] for c in cams] != ['cam01', 'cam02', 'cam03', 'cam04']):
            raise ValueError("Expected four unique serials ordered cam01..cam04")
        for key, default in (("fps", 20), ("clockIntervalSeconds", 30), ("maxPairSkewMs", 30)):
            data.setdefault(key, default)
            if not isinstance(data[key], (int, float)) or not math.isfinite(data[key]) or data[key] <= 0:
                raise ValueError(f"Invalid {key}")
        if 'exposureTimeUs' in data:
            value=data['exposureTimeUs']
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('exposureTimeUs must be a positive finite number in microseconds')
        data.setdefault('hand', 'right')
        if data['hand'] not in ('right', 'left'):
            raise ValueError('hand must be right or left (single-hand recording)')
        return cls(path, data)


class CameraRecorder:
    def __init__(self, config, session, on_error):
        self.config, self.session, self.on_error = config, Path(session), on_error
        self.process = None
        self.reader = None
        self.pending = {}
        self.counter = 0
        self.stopping = False
        self.final = None

    async def open(self):
        self.root = self.session / 'multicamera'
        self.root.mkdir(exist_ok=False)
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps(self.config.data, indent=2))
        self.log = (self.root / 'worker.log').open('wb')
        try:
            self.process = await asyncio.create_subprocess_exec(
                self.config.data['python'], '-u', str(Path(__file__).with_name('multicamera_worker.py')),
                '--config', str(config_path.resolve()), '--output', str(self.root.resolve()),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=self.log)
        finally:
            self.log.close()
        self.ready = asyncio.get_running_loop().create_future()
        self.reader = asyncio.create_task(self.read_messages())
        await asyncio.wait_for(self.ready, 30)

    async def read_messages(self):
        failure = None
        try:
            while line := await self.process.stdout.readline():
                msg = json.loads(line)
                if msg.get('type') == 'ready' and not self.ready.done():
                    self.ready.set_result(msg)
                if msg.get('type') == 'finished':
                    self.final = msg
                    if msg.get('error'):
                        failure = msg['error']
                fut = self.pending.get(msg.get('requestId'))
                if fut and not fut.done():
                    fut.set_result(msg)
            code = await self.process.wait()
            if code or not self.final:
                failure = failure or f'Camera worker exited ({code}); inspect multicamera/worker.log'
        except Exception as exc:
            failure = str(exc)
        finally:
            error = RuntimeError(failure or 'Camera worker closed')
            if not self.ready.done():
                self.ready.set_exception(error)
            for fut in list(self.pending.values()):
                if not fut.done():
                    fut.set_exception(error)
            if failure and not self.stopping:
                # Do not await finish here: finish drains this reader.
                asyncio.create_task(self.on_error(failure))

    async def request(self, kind, **fields):
        if not self.process or self.process.returncode is not None:
            raise RuntimeError('Camera worker is not running')
        self.counter += 1
        key = str(self.counter)
        fut = asyncio.get_running_loop().create_future()
        self.pending[key] = fut
        try:
            self.process.stdin.write((json.dumps(dict(type=kind, requestId=key, **fields))+'\n').encode())
            await self.process.stdin.drain()
            result = await asyncio.wait_for(fut, 4)
            if not result.get('ok'):
                raise RuntimeError(result.get('error', 'Camera request rejected'))
            return result
        finally:
            self.pending.pop(key, None)

    async def arm(self, start, end, phone_sync, epoch):
        return await self.request('arm', startPcMs=start, endPcMs=end,
                                  phoneMinusPcMs=phone_sync['offsetMs'],
                                  phoneSyncUncertaintyMs=phone_sync['rttMs']/2,
                                  phoneEpochOffsetMs=epoch)

    async def commit(self):
        return await self.request('commit')

    async def stop(self):
        self.stopping = True
        if self.process and self.process.returncode is None:
            if not self.process.stdin.is_closing():
                self.process.stdin.write(b'{"type":"stop"}\n')
                await self.process.stdin.drain()
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 15)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
                raise RuntimeError('Camera worker did not finalize within 15 seconds')
        if self.reader:
            await self.reader
        return self.final or dict(error='Camera recorder did not produce final status')

    def close(self):
        # EOF stops capture even if the PC server crashes; the worker has its own deadline too.
        self.stopping = True
        if self.process and self.process.stdin:
            self.process.stdin.close()

    async def process_handpose(self):
        if not self.config.data.get('autoProcess', True):
            return 'pending'
        with (self.root/'postprocess.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(
                self.config.data['wilorPython'], '-u', str(Path(__file__).with_name('process_multicamera.py')),
                str(self.session.resolve()), stdout=log, stderr=log)
            # Offline processing can complete independently if the receiver shuts down.
            code = await process.wait()
        if code:
            raise RuntimeError(f'Hand pose processing exited {code}; see multicamera/postprocess.log')
        return 'complete'
