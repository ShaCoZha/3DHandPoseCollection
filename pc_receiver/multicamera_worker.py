"""Four independent PySpin capture threads; stdin JSON control, stdout JSON replies.

No arrival-time fallback: hardware timestamp/latch failure aborts capture.
Video playback FPS is nominal; JSONL timestamps are the authoritative sample times.
"""
import argparse
import hashlib
import json
import math
import queue
import shutil
import sys
import threading
import time
from pathlib import Path


def now_ms():
    return time.monotonic_ns() / 1e6


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def append(handle, row):
    handle.write(json.dumps(row, allow_nan=False) + '\n')


class Worker:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.stop = threading.Event()
        self.armed = None
        self.committed = False
        self.errors = queue.Queue()
        self.ready = [threading.Event() for _ in config['cameras']]
        self.summaries = {}
        self.threads = []
        self.reason = 'stopped'

    def run_camera(self, camera, spec, ready, expected_size):
        import PySpin
        import cv2
        name = spec['name']
        writer = None
        started = False
        restore = []
        stats = dict(camera=name, serial=spec['serial'], frames=0, incomplete=0, incompleteDuringCapture=0, frameIdGaps=0,
                     firstPcMonotonicMs=None, lastPcMonotonicMs=None)
        self.summaries[name] = stats
        try:
            camera.Init()
            nodes = camera.GetNodeMap()

            def set_node(key, typ, value):
                node = typ(nodes.GetNode(key))
                if not PySpin.IsWritable(node):
                    raise RuntimeError(f'{name}: {key} not writable')
                if typ is PySpin.CEnumerationPtr:
                    old = node.GetIntValue()
                    node.SetIntValue(node.GetEntryByName(value).GetValue())
                    restore.append(lambda n=node, v=old: n.SetIntValue(v))
                else:
                    old = node.GetValue()
                    node.SetValue(value)
                    restore.append(lambda n=node, v=old: n.SetValue(v))

            set_node('TriggerMode', PySpin.CEnumerationPtr, 'Off')
            set_node('AcquisitionMode', PySpin.CEnumerationPtr, 'Continuous')
            set_node('AcquisitionFrameRateEnable', PySpin.CBooleanPtr, True)
            set_node('AcquisitionFrameRate', PySpin.CFloatPtr, float(self.config['fps']))
            set_node('ChunkModeActive', PySpin.CBooleanPtr, True)
            for key in ('Timestamp', 'FrameID', 'ExposureTime'):
                set_node('ChunkSelector', PySpin.CEnumerationPtr, key)
                set_node('ChunkEnable', PySpin.CBooleanPtr, True)
            stream = camera.GetTLStreamNodeMap()
            handling = PySpin.CEnumerationPtr(stream.GetNode('StreamBufferHandlingMode'))
            old_handling = handling.GetIntValue()
            handling.SetIntValue(handling.GetEntryByName('OldestFirst').GetValue())
            restore.append(lambda: handling.SetIntValue(old_handling))
            size = [int(camera.Width.GetValue()), int(camera.Height.GetValue())]
            if size != expected_size:
                raise RuntimeError(f'{name}: image size {size} != calibration {expected_size}')
            stats.update(size=size, nominalFps=float(camera.AcquisitionFrameRate.GetValue()),
                         pixelFormat=camera.PixelFormat.GetCurrentEntry().GetSymbolic(),
                         ptpEnabled=bool(camera.GevIEEE1588.GetValue()),
                         timestampMeaning='Camera image hardware timestamp (ns); not host arrival time',
                         hardwareExposureSynchronized=False)
            latch = PySpin.CCommandPtr(nodes.GetNode('TimestampLatch'))
            value = PySpin.CIntegerPtr(nodes.GetNode('TimestampLatchValue'))
            if not PySpin.IsWritable(latch) or not PySpin.IsReadable(value):
                raise RuntimeError('Camera timestamp latch unavailable')
            clock_file = (self.output / f'{name}_clock.jsonl').open('x', buffering=1)
            stamps = (self.output / f'{name}_frames.jsonl').open('x', buffering=1)
            with clock_file, stamps:
                def measure():
                    samples = []
                    for _ in range(8):
                        before = now_ms()
                        latch.Execute()
                        raw = int(value.GetValue())
                        after = now_ms()
                        samples.append(dict(cameraTimestampNs=raw, pcMonotonicMs=(before+after)/2,
                                            cameraMinusPcMs=raw/1e6-(before+after)/2,
                                            bracketMs=after-before))
                    best = min(samples, key=lambda x: x['bracketMs'])
                    append(clock_file, dict(best, samples=samples,
                           uncertaintyMs=best['bracketMs']/2, method='device_timestamp_latch_host_bracket'))
                    return best

                initial = measure()
                last_measure = now_ms()
                previous_raw = previous_id = None
                processor = PySpin.ImageProcessor()
                processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
                camera.BeginAcquisition()
                started = True
                segment = None
                local_index = 0
                output_dir = self.output / 'rgb' / name
                output_dir.mkdir(parents=True, exist_ok=False)
                while not self.stop.is_set():
                    if now_ms() - last_measure >= self.config['clockIntervalSeconds']*1000:
                        current = measure()
                        if abs(current['cameraMinusPcMs'] - initial['cameraMinusPcMs']) > 100:
                            raise RuntimeError('Camera clock discontinuity >100 ms; check PTP/reset')
                        last_measure = now_ms()
                    image = camera.GetNextImage(2000)
                    received = now_ms()
                    try:
                        if image.IsIncomplete():
                            stats['incomplete'] += 1
                            if (self.committed and self.armed is not None and
                                    self.armed['startPcMs'] <= received < self.armed['endPcMs']):
                                stats['incompleteDuringCapture'] += 1
                            continue
                        chunk = image.GetChunkData()
                        raw = int(chunk.GetTimestamp())
                        frame_id = int(chunk.GetFrameID())
                        exposure = float(chunk.GetExposureTime())
                        if previous_raw is not None and raw <= previous_raw:
                            raise RuntimeError('Camera hardware timestamp went backwards')
                        previous_raw = raw
                        pc = raw/1e6 - initial['cameraMinusPcMs']
                        if abs(received-pc) > 2000:
                            raise RuntimeError('Camera timestamp does not agree with its latch clock')
                        ready.set()
                        arm = self.armed
                        if not self.committed or arm is None or pc < arm['startPcMs']:
                            continue
                        if pc >= arm['endPcMs']:
                            break
                        current_segment = int((pc-arm['startPcMs'])//60000)
                        if current_segment != segment:
                            if writer:
                                writer.release()
                            segment = current_segment
                            relative = f'rgb/{name}/segment_{segment:04d}.mp4'
                            writer = cv2.VideoWriter(str(self.output/relative), cv2.VideoWriter_fourcc(*'mp4v'),
                                                     self.config['fps'], tuple(size))
                            if not writer.isOpened():
                                raise RuntimeError('Video encoder did not open')
                            local_index = 0
                        converted = processor.Convert(image, PySpin.PixelFormat_BGR8)
                        frame = converted.GetNDArray()
                        writer.write(frame)
                        del frame, converted
                        if previous_id is not None:
                            if frame_id <= previous_id:
                                raise RuntimeError('Camera frame ID reset')
                            stats['frameIdGaps'] += frame_id-previous_id-1
                        previous_id = frame_id
                        phone = pc + arm['phoneMinusPcMs']
                        append(stamps, dict(camera=name, serial=spec['serial'], sampleIndex=stats['frames'],
                            cameraFrameId=frame_id, cameraTimestampNs=raw, exposureTimeUs=exposure,
                            hostReceivedMonotonicMs=received, deviceMonotonicMs=pc,
                            phoneMonotonicMs=phone, unixTimeMs=phone+arm['phoneEpochOffsetMs'],
                            initialClockUncertaintyMs=initial['bracketMs']/2+arm['phoneSyncUncertaintyMs'],
                            video=relative, videoFrameIndex=local_index))
                        stats['frames'] += 1
                        local_index += 1
                        if stats['firstPcMonotonicMs'] is None:
                            stats['firstPcMonotonicMs'] = pc
                        stats['lastPcMonotonicMs'] = pc
                    finally:
                        image.Release()
                measure()
        except Exception as exc:
            self.errors.put(f'{name}: {exc}')
            self.stop.set()
        finally:
            if writer:
                writer.release()
            if started:
                try:
                    camera.EndAcquisition()
                except Exception as exc:
                    self.errors.put(f'{name} EndAcquisition: {exc}')
            for reset in reversed(restore):
                try:
                    reset()
                except Exception:
                    pass
            if camera.IsInitialized():
                camera.DeInit()
            restore.clear()

    def run(self):
        import PySpin
        import toml
        calibration_path = Path(self.config['calibration'])
        calibration = toml.load(calibration_path)
        for i, spec in enumerate(self.config['cameras']):
            if calibration[f'cam_{i}']['name'] != spec['name']:
                raise ValueError('Calibration camera order mismatch')
        shutil.copyfile(calibration_path, self.output/'calibration.toml')
        metadata = dict(schemaVersion=1, cameras=self.config['cameras'], units='metres',
                        calibrationSha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                        calibrationSource=str(calibration_path), hardwareExposureSynchronized=False,
                        clockMethod='camera latch -> PC monotonic -> iPhone frozen epoch',
                        rgbEncoding='BGR converted RGB video, mp4v lossy; 60-second segments')
        (self.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
        if shutil.disk_usage(self.output).free < 1024**3:
            raise RuntimeError('Less than 1 GiB free disk space')
        system = PySpin.System.GetInstance()
        cameras = system.GetCameras()
        owned = []
        camera = None
        commands = queue.Queue()

        def read_commands():
            try:
                for line in sys.stdin:
                    commands.put(json.loads(line))
            finally:
                commands.put(dict(type='eof'))

        threading.Thread(target=read_commands, daemon=True).start()
        try:
            for i, spec in enumerate(self.config['cameras']):
                camera = cameras.GetBySerial(spec['serial'])
                if not camera.IsValid():
                    raise RuntimeError(f"Camera {spec['serial']} not found")
                owned.append(camera)
                thread = threading.Thread(target=self.run_camera,
                    args=(camera, spec, self.ready[i], calibration[f'cam_{i}']['size']))
                self.threads.append(thread)
                thread.start()
            deadline = now_ms()+20000
            while not all(event.is_set() for event in self.ready):
                if self.stop.wait(.02) or now_ms() > deadline:
                    raise RuntimeError('Camera preflight did not receive all four timestamped frames')
            emit(dict(type='ready', cameras=self.config['cameras']))
            while not self.stop.is_set():
                try:
                    message = commands.get(timeout=.05)
                except queue.Empty:
                    message = {}
                kind = message.get('type')
                error = None
                if kind == 'arm':
                    if self.armed is not None:
                        error = 'Already armed'
                    elif not all(isinstance(message.get(k), (int,float)) and math.isfinite(message[k])
                                 for k in ('startPcMs','endPcMs','phoneMinusPcMs','phoneEpochOffsetMs','phoneSyncUncertaintyMs')):
                        error = 'Invalid arm clocks'
                    elif message['startPcMs'] < now_ms()+1500 or not message['startPcMs'] < message['endPcMs'] <= message['startPcMs']+900000:
                        error = 'Invalid capture window'
                    else:
                        self.armed = message
                elif kind == 'commit':
                    if self.armed is None or now_ms() >= self.armed['startPcMs']-1500:
                        error = 'Commit missed start deadline'
                    else:
                        self.committed = True
                elif kind in ('stop','eof'):
                    self.reason = 'stopped' if kind == 'stop' else 'server_disconnected'
                    self.stop.set()
                if message.get('requestId'):
                    emit(dict(type='ack', requestId=message['requestId'], ok=error is None, error=error))
                if self.armed:
                    if not self.committed and now_ms() >= self.armed['startPcMs']-1500:
                        raise RuntimeError('No commit before start deadline')
                    # Allow in-flight images through the half-open hardware-time window.
                    if now_ms() >= self.armed['endPcMs']+2000:
                        self.reason = 'completed'
                        self.stop.set()
                    if self.committed and all(not t.is_alive() for t in self.threads):
                        self.reason = 'completed'
                        self.stop.set()
                if shutil.disk_usage(self.output).free < 1024**3:
                    raise RuntimeError('Free disk space below 1 GiB; stopping capture')
        finally:
            self.stop.set()
            for thread in self.threads:
                thread.join()
            owned.clear()
            camera = None
            import gc
            gc.collect()
            cameras.Clear()
            gc.collect()
            system.ReleaseInstance()

    def finish(self, error=None):
        errors = [error] if error else []
        if self.committed:
            for name, stats in self.summaries.items():
                if not stats['frames']:
                    errors.append(f'{name}: no frames saved after commit')
        while not self.errors.empty():
            errors.append(self.errors.get())
        result = dict(type='finished', reason=self.reason, cameras=self.summaries,
                      error='; '.join(errors) if errors else None)
        (self.output/'status.json').write_text(json.dumps(result, indent=2))
        emit(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    worker = Worker(json.loads(args.config.read_text()), args.output)
    error = None
    try:
        worker.run()
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        error = str(exc)
    worker.finish(error)


if __name__ == '__main__':
    main()
