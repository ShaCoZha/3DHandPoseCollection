"""Apply measured clock drift to a completed synchronized capture, preserving raw files."""
import argparse
import bisect
import json
from pathlib import Path


def calibration_points(records, device):
    points = {}
    for row in records:
        if row.get("device") != device:
            continue
        offset = row["deviceMinusPhoneMs"]
        points[row["phoneMonotonicMs"] + offset] = offset
    if not points:
        raise ValueError(f"No {device} clock calibration")
    return sorted(points.items())


def aligned_time(raw_ms, points, epoch):
    index = bisect.bisect_right(points, (raw_ms, float("inf")))
    if index == 0:
        offset = points[0][1]
    elif index == len(points):
        offset = points[-1][1]
    else:
        left, right = points[index - 1], points[index]
        weight = (raw_ms - left[0]) / (right[0] - left[0])
        offset = left[1] + weight * (right[1] - left[1])
    return raw_ms - offset + epoch


def align_file(source, destination, points, epoch):
    count = 0
    missing = 0
    invalid_hands = 0
    minimum = maximum = None
    previous_index = None
    previous_time = None
    max_interval = 0
    with source.open(encoding="utf-8") as src, destination.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            timestamp = aligned_time(row["deviceMonotonicMs"], points, epoch)
            if previous_time is not None and timestamp < previous_time:
                raise ValueError(f"Non-monotonic aligned timestamp in {source}")
            if previous_time is not None:
                max_interval = max(max_interval, timestamp - previous_time)
            row["initialAlignedUnixTimeMs"] = row.get("unixTimeMs")
            row["unixTimeMs"] = timestamp
            row["clockAlignment"] = "piecewise_linear_offset; nearest_offset_outside_calibrations"
            index = row["sampleIndex"]
            if previous_index is not None:
                missing += max(0, index - previous_index - 1)
            else:
                missing += max(0, index)
            previous_index, previous_time = index, timestamp
            invalid_hands += sum(not hand["trackingValid"] for hand in row.get("hands", []))
            minimum = timestamp if minimum is None else min(minimum, timestamp)
            maximum = timestamp if maximum is None else max(maximum, timestamp)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return dict(samples=count, sequenceGaps=missing, invalidHandObservations=invalid_hands,
                firstUnixTimeMs=minimum, lastUnixTimeMs=maximum,
                observedDurationSeconds=(maximum-minimum)/1000 if count else 0,
                maximumSampleIntervalMs=max_interval,
                calibrationCount=len(points),
                maximumCalibrationGapSeconds=max((b[0]-a[0] for a, b in zip(points, points[1:])), default=0)/1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--pose-source", type=Path, help="Optional Quest local hand_pose_raw.jsonl for network gap recovery")
    args = parser.parse_args()
    root = args.session_dir
    manifest = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    if manifest["status"] in ("synchronizing", "armed", "recording"):
        raise SystemExit("Wait until capture and Watch file upload have finished")
    calibrations = [json.loads(line) for line in (root / "clock_sync.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {}
    sources = [("watch", root / "imudata.jsonl", "imudata_aligned.jsonl")]
    if manifest.get("poseSource") == "multicamera":
        from align_multicamera import align_session
        report["multicamera"] = align_session(root)
    else:
        sources.append(("quest", args.pose_source or root / "hand_pose.jsonl", "hand_pose_aligned.jsonl"))
    for device, source, output in sources:
        report[device] = align_file(source, root / output, calibration_points(calibrations, device), manifest["phoneEpochOffsetMs"])
        summary = report[device]
        if summary["samples"]:
            summary["startCoverageGapMs"] = max(0, summary["firstUnixTimeMs"] - manifest["startUnixTimeMs"])
            summary["endCoverageGapMs"] = max(0, manifest["endUnixTimeMs"] - summary["lastUnixTimeMs"])
    (root / "alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
