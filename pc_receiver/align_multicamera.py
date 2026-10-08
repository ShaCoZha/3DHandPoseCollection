"""Compose camera->PC and PC->iPhone drift maps; pair frames without index assumptions."""
import argparse
import bisect
import json
from itertools import product
from pathlib import Path
from align_capture import aligned_time, calibration_points


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pair_frames(streams, max_skew_ms):
    names = list(streams)
    times = {name: [r['unixTimeMs'] for r in rows] for name, rows in streams.items()}
    last = {name: -1 for name in names}
    for reference in streams[names[0]]:
        target = reference['unixTimeMs']
        chosen = {}
        candidates = {}
        for name in names:
            index = bisect.bisect_left(times[name], target)
            options = [i for i in (index-1, index) if last[name] < i < len(times[name])]
            if name == names[0]:
                options = [i for i in options if times[name][i] == target]
            if not options:
                break
            candidates[name] = options
            best = min(options, key=lambda i: abs(times[name][i]-target))
            chosen[name] = best
        if len(chosen) != len(names):
            continue
        values = [times[name][i] for name, i in chosen.items()]
        skew = max(values)-min(values)
        if skew > max_skew_ms:
            # Individually nearest frames can straddle the reference and exceed
            # the group limit even when another bracketing combination fits.
            # Preserve existing nearest matches when valid; never relax the
            # skew limit or reuse frames to recover a rejected group.
            combinations = product(*(candidates[name] for name in names))
            def score(indices):
                values = [times[name][i] for name, i in zip(names, indices)]
                return max(values)-min(values), sum(abs(t-target) for t in values)
            best = min(combinations, key=score)
            skew = score(best)[0]
            if skew > max_skew_ms:
                continue
            chosen = dict(zip(names, best))
        frames = {name: streams[name][i] for name, i in chosen.items()}
        last.update(chosen)
        yield dict(unixTimeMs=target, cameraSkewMs=skew, frames=frames,
                   timestampReference=names[0], hardwareExposureSynchronized=False)


def align_session(root):
    root = Path(root)
    manifest = json.loads((root/'capture.json').read_text())
    if manifest['status'] in ('synchronizing','armed','recording'):
        raise ValueError('Finish capture before alignment')
    camera_root = root/'multicamera'
    config = json.loads((camera_root/'config.json').read_text())
    clocks = read_jsonl(root/'clock_sync.jsonl')
    pc_points = calibration_points(clocks, 'pc')
    pc_uncertainty = max(row.get('uncertaintyMs', 0) for row in clocks if row.get('device') == 'pc')
    report = dict(hardwareExposureSynchronized=False, maxPairSkewMs=config['maxPairSkewMs'], cameras={})
    streams = {}
    for spec in config['cameras']:
        name = spec['name']
        samples = read_jsonl(camera_root/f'{name}_clock.jsonl')
        points = sorted({row['cameraTimestampNs']/1e6: row['cameraMinusPcMs'] for row in samples}.items())
        if not points:
            raise ValueError(f'No camera clock samples for {name}')
        rows = read_jsonl(camera_root/f'{name}_frames.jsonl')
        uncertainty = max(s['uncertaintyMs'] for s in samples) + pc_uncertainty
        previous = None
        intervals = []
        with (camera_root/f'{name}_frames_aligned.jsonl').open('w') as handle:
            for row in rows:
                pc_ms = aligned_time(row['cameraTimestampNs']/1e6, points, 0)
                timestamp = aligned_time(pc_ms, pc_points, manifest['phoneEpochOffsetMs'])
                if previous is not None:
                    if timestamp <= previous:
                        raise ValueError(f'{name}: non-monotonic corrected timestamps')
                    intervals.append(timestamp-previous)
                row.update(initialAlignedUnixTimeMs=row['unixTimeMs'], unixTimeMs=timestamp,
                           correctedPcMonotonicMs=pc_ms, clockUncertaintyEstimateMs=uncertainty,
                           clockAlignment='camera latch drift -> PC/phone drift -> frozen phone epoch')
                handle.write(json.dumps(row)+'\n')
                previous = timestamp
        streams[name] = rows
        report['cameras'][name] = dict(frames=len(rows), clockSamples=len(points),
            maximumFrameIntervalMs=max(intervals, default=0),
            frameIdGaps=sum(max(0,b['cameraFrameId']-a['cameraFrameId']-1) for a,b in zip(rows,rows[1:])),
            maximumClockCalibrationGapSeconds=max((b[0]-a[0] for a,b in zip(points,points[1:])),default=0)/1000,
            clockUncertaintyEstimateMs=uncertainty,
            firstUnixTimeMs=rows[0]['unixTimeMs'] if rows else None,
            lastUnixTimeMs=rows[-1]['unixTimeMs'] if rows else None)
    count = 0
    max_skew = 0
    with (camera_root/'frame_sets.jsonl').open('w') as handle:
        for count, row in enumerate(pair_frames(streams, config['maxPairSkewMs']), 1):
            row['sampleIndex'] = count-1
            max_skew = max(max_skew, row['cameraSkewMs'])
            handle.write(json.dumps(row)+'\n')
    report.update(pairedFrames=count, maximumPairedSkewMs=max_skew,
                  pcClockSamples=len(pc_points),
                  maximumPcCalibrationGapSeconds=max((b[0]-a[0] for a,b in zip(pc_points,pc_points[1:])),default=0)/1000)
    for summary in report['cameras'].values():
        summary['unpairedFrames'] = summary['frames']-count
    (camera_root/'alignment_report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    print(json.dumps(align_session(parser.parse_args().session), indent=2))
