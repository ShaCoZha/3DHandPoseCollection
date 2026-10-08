"""Offline ChArUco calibration of timestamped, headless four-camera recordings.

Raw videos are never modified. Detection caches and candidate parameters go in
the session's calibration directory; deploying a candidate is a separate step.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from align_capture import aligned_time


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def board():
    return cv2.aruco.CharucoBoard(
        (10, 8), .022, .016,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def detect(session, stride=1):
    root = session / 'multicamera'
    out = session / 'calibration'
    out.mkdir(exist_ok=True)
    config = json.loads((root / 'config.json').read_text())
    summary = {}
    cv2.setNumThreads(2)
    for spec in config['cameras']:
        name = spec['name']
        cache = out / f'{name}_detections_stride{stride}.jsonl'
        if cache.exists():
            observations = read_rows(cache)
            print(f'{name}: cached {len(observations)} samples', flush=True)
        else:
            detector = cv2.aruco.CharucoDetector(board())
            stamps = read_rows(root / f'{name}_frames.jsonl')
            latch = read_rows(root / f'{name}_clock.jsonl')
            clock = sorted((x['cameraTimestampNs'] / 1e6, x['cameraMinusPcMs']) for x in latch)
            selected = {(x['video'], x['videoFrameIndex']): x
                        for x in stamps if x['sampleIndex'] % stride == 0}
            observations = []
            for relative in dict.fromkeys(x['video'] for x in stamps):
                cap = cv2.VideoCapture(str(root / relative))
                frame = 0
                while cap.grab():
                    stamp = selected.get((relative, frame))
                    frame += 1
                    if stamp is None:
                        continue
                    ok, image = cap.retrieve()
                    if not ok:
                        raise RuntimeError(f'Cannot decode {relative}:{frame - 1}')
                    corners, ids, _, _ = detector.detectBoard(image)
                    observations.append(dict(
                        camera=name, sampleIndex=stamp['sampleIndex'],
                        pcMonotonicMs=aligned_time(stamp['cameraTimestampNs'] / 1e6, clock, 0),
                        clockUncertaintyMs=max(x['uncertaintyMs'] for x in latch),
                        video=relative, videoFrameIndex=stamp['videoFrameIndex'],
                        ids=[] if ids is None else ids.flatten().tolist(),
                        xy=[] if corners is None else corners.reshape(-1, 2).tolist()))
                cap.release()
                print(f'{name} {relative}: {len(observations)} samples', flush=True)
            tmp = cache.with_suffix('.tmp')
            tmp.write_text(''.join(json.dumps(x) + '\n' for x in observations))
            tmp.replace(cache)
        # Neighboring observations measure board motion in ORIGINAL image pixels.
        for i, row in enumerate(observations):
            speeds = []
            a = dict(zip(row['ids'], row['xy']))
            step = max(1, round(4 / stride))
            for j in (i - step, i + step):
                if not 0 <= j < len(observations):
                    continue
                other = observations[j]
                dt = abs(other['pcMonotonicMs'] - row['pcMonotonicMs']) / 1000
                b = dict(zip(other['ids'], other['xy']))
                common = sorted(a.keys() & b.keys())
                if len(common) >= 10 and 0 < dt < .35:
                    speeds.append(float(np.median(np.linalg.norm(
                        np.array([a[k] for k in common]) - np.array([b[k] for k in common]), axis=1))) / dt)
            row['motionPixelsPerSecond'] = max(speeds) if len(speeds) == 2 else None
        cache.write_text(''.join(json.dumps(x) + '\n' for x in observations))
        summary[name] = dict(samples=len(observations),
            atLeast12Corners=sum(len(x['ids']) >= 12 for x in observations),
            stationary12Corners=sum(len(x['ids']) >= 12 and x['motionPixelsPerSecond'] is not None
                                    and x['motionPixelsPerSecond'] <= 15 for x in observations))
    (out / 'detection_report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session', type=Path)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    detect(args.session, args.stride)
