"""Record a bounded four-camera ChArUco take, using the PC clock only."""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

from multicamera_capture import CameraConfig, CameraRecorder
from align_multicamera import align_session


async def record(args):
    config = CameraConfig.load(args.config)
    config.data.update(autoProcess=False, weightedHandpose=False)
    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d_%H-%M-%S')
    session = args.output_root.resolve()/stamp
    session.mkdir(parents=True, exist_ok=False)
    errors = []
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def on_error(error):
        errors.append(str(error))
        stop.set()

    recorder = CameraRecorder(config, session, on_error)
    manifest = dict(status='preparing', purpose='charuco_calibration', phoneClockSimulated=True,
                    clockReference='PC monotonic and PC Unix; no iPhone synchronization')
    (session/'capture.json').write_text(json.dumps(manifest, indent=2))
    (session/'recording.json').write_text(json.dumps(dict(
        durationSeconds=args.seconds, boardDimensionsConfirmedByUser=args.confirm_board,
        board=dict(squares=[10, 8], squareMetres=.022, markerMetres=.016, dictionary='DICT_4X4_50')), indent=2))
    result = None
    try:
        await recorder.open()
        now = time.monotonic()*1000
        epoch = time.time()*1000-now
        start = now+3000
        end = start+args.seconds*1000
        await recorder.arm(start, end, {'offsetMs': 0., 'rttMs': 0.}, epoch)
        await recorder.commit()
        manifest.update(status='armed', phoneEpochOffsetMs=epoch)
        (session/'capture.json').write_text(json.dumps(manifest, indent=2))
        (session/'clock_sync.jsonl').write_text(json.dumps(dict(device='pc', phoneMonotonicMs=now,
            deviceMinusPhoneMs=0., uncertaintyMs=0., simulated=True))+'\n')
        print(f'All four cameras ready. Starts in 3 seconds; duration {args.seconds}s.\n{session}', flush=True)
        while time.monotonic()*1000 < end and not stop.is_set():
            await asyncio.sleep(.2)
        actual_end = min(end, time.monotonic()*1000)
        if not stop.is_set(): await asyncio.sleep(2.5)
        result = await recorder.stop()
        if result.get('error'): errors.append(result['error'])
        if actual_end <= start: errors.append('Stopped before capture began')
        (session/'timestamp.json').write_text(json.dumps(dict(startedAtUnixMs=start+epoch,
            stoppedAtUnixMs=max(start, actual_end)+epoch), indent=2))
        manifest['status'] = 'failed' if errors else 'stopped' if stop.is_set() else 'finished'
        manifest['errors'] = errors
        (session/'capture.json').write_text(json.dumps(manifest, indent=2))
        if errors: raise RuntimeError('; '.join(errors))
        report = dict(cameraStatus=result, alignment=align_session(session))
        (session/'capture_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        print(f'Recording complete: {session}', flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        (session/'capture.json').write_text(json.dumps(manifest, indent=2))
        raise
    finally:
        if result is None: await recorder.stop()
        for sig in (signal.SIGINT, signal.SIGTERM): loop.remove_signal_handler(sig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--seconds', default=120, type=float)
    ap.add_argument('--confirm-board', action='store_true', help='Confirm the physical board is 10x8, 22 mm squares, 16 mm markers, DICT_4X4_50')
    args = ap.parse_args()
    if not 1 <= args.seconds <= 900: ap.error('--seconds must be between 1 and 900')
    asyncio.run(record(args))


if __name__ == '__main__':
    main()
