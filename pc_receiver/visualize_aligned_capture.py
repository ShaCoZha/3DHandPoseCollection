"""Build a self-contained HTML viewer from aligned Quest pose and Watch IMU JSONL.

Usage: python3 visualize_aligned_capture.py /path/to/session
Only Python's standard library is required. The HTML works offline in Chrome/Edge.
No inter-stream resampling, timestamp shifts, or writes to the source files occur.
"""

import argparse
from array import array
import base64
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import sys


JOINT_NAMES = [
    "Palm", "Wrist", "Thumb metacarpal", "Thumb proximal", "Thumb distal", "Thumb tip",
    "Index metacarpal", "Index proximal", "Index intermediate", "Index distal", "Index tip",
    "Middle metacarpal", "Middle proximal", "Middle intermediate", "Middle distal", "Middle tip",
    "Ring metacarpal", "Ring proximal", "Ring intermediate", "Ring distal", "Ring tip",
    "Little metacarpal", "Little proximal", "Little intermediate", "Little distal", "Little tip",
]
# Meta XRHand uses the OpenXR 26-joint layout. The capture's enum name strings
# alias other skeletons; use skeletonType and numeric id, never the name strings.
JOINT_SOURCE = "https://developers.meta.com/horizon/documentation/unity/unity-handtracking-interactions/"
IMU_STRIDE = 7
POSE_STRIDE = 159  # time, left/right validity, left/right 26 * xyz


def read_rows(path, session_id):
    previous = -math.inf
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("sessionId") != session_id:
                raise ValueError(f"{path}:{line_no}: sessionId mismatch")
            if not row.get("clockAlignment"):
                raise ValueError(f"{path}:{line_no}: expected an aligned sample")
            timestamp = row["unixTimeMs"]
            if not math.isfinite(timestamp) or timestamp <= previous:
                raise ValueError(f"{path}:{line_no}: timestamps must strictly increase")
            previous = timestamp
            yield row


def finite_xyz(vector):
    values = [float(vector[axis]) for axis in "xyz"]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite coordinate in source data")
    return values


def pack_session(root):
    manifest = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    session_id = manifest["sessionId"]
    origin = manifest["startUnixTimeMs"]
    imu = array("f")
    pose = array("f")
    activity = Counter()
    source_files = ["imudata_aligned.jsonl", "hand_pose_aligned.jsonl"]
    for row in read_rows(root / source_files[0], session_id):
        t = (row["unixTimeMs"] - origin) / 1000
        acc = finite_xyz(row["userAcceleration"])
        gyro = finite_xyz(row["rotationRate"])
        imu.extend([t, *acc, *gyro])
        activity[int(t // 10)] += math.sqrt(sum(v * v for v in gyro))
    print(f"Read {len(imu) // IMU_STRIDE:,} aligned IMU samples", flush=True)
    invalid = Counter()
    low_confidence = Counter()
    index_gaps = Counter()
    previous_index = None
    for row in read_rows(root / source_files[1], session_id):
        t = (row["unixTimeMs"] - origin) / 1000
        if previous_index is not None:
            index_gaps["pose"] += max(0, row["sampleIndex"] - previous_index - 1)
        previous_index = row["sampleIndex"]
        coords = [math.nan] * 156
        flags = [0, 0]
        seen = set()
        for hand in row["hands"]:
            side = hand["hand"]
            if side not in ("left", "right") or side in seen:
                raise ValueError(f"Unexpected or duplicate hand: {side}")
            seen.add(side)
            h = 0 if side == "left" else 1
            if hand["skeletonType"] != ("XRHandLeft" if h == 0 else "XRHandRight"):
                raise ValueError(f"Unsupported skeleton: {hand['skeletonType']}")
            if not hand["trackingValid"]:
                continue
            bones = {bone["id"]: bone for bone in hand["bones"]}
            if len(bones) != 26 or set(bones) != set(range(26)):
                raise ValueError("Tracked XRHand must contain 26 unique joint IDs")
            flags[h] = 2 if hand.get("highConfidence", False) else 1
            low_confidence[side] += flags[h] == 1
            for joint_id, bone in bones.items():
                start = h * 78 + joint_id * 3
                coords[start:start + 3] = finite_xyz(bone["position"])
        for h, side in enumerate(("left", "right")):
            invalid[side] += flags[h] == 0
        pose.extend([t, *flags, *coords])
    print(f"Read {len(pose) // POSE_STRIDE:,} aligned pose frames", flush=True)
    if not imu or not pose:
        raise ValueError("Both aligned streams must contain samples")
    duration = (manifest["endUnixTimeMs"] - origin) / 1000
    # Begin at an active ten-second interval, away from the session boundaries.
    candidates = {k: v for k, v in activity.items() if 1 <= k < duration // 10 - 1}
    initial_time = (max(candidates, key=candidates.get) * 10 + 5) if candidates else min(5, duration / 2)
    report_path = root / "alignment_report.json"
    metadata = {
        "sessionId": session_id,
        "originUnixMs": origin,
        "duration": duration,
        "initialTime": initial_time,
        "imuCount": len(imu) // IMU_STRIDE,
        "poseCount": len(pose) // POSE_STRIDE,
        "imuStride": IMU_STRIDE,
        "poseStride": POSE_STRIDE,
        "imuFields": ["userAcceleration", "rotationRate"],
        "imuUnits": ["m/s²", "rad/s"],
        "invalidHands": dict(invalid),
        "lowConfidenceHands": dict(low_confidence),
        "sequenceGaps": dict(index_gaps),
        "sourceFiles": source_files,
        "jointNames": JOINT_NAMES,
        "jointDefinitionSource": JOINT_SOURCE,
        "report": json.loads(report_path.read_text()) if report_path.exists() else None,
    }
    # Relative seconds retain sub-millisecond precision in float32 over 900 s.
    # All samples/frames and all 26 xyz positions per valid hand are retained.
    if sys.byteorder != "little":
        imu.byteswap()
        pose.byteswap()
    raw = imu.tobytes() + pose.tobytes()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    metadata["packedBytes"] = len(raw)
    metadata["compressedBytes"] = len(compressed)
    return metadata, base64.b64encode(compressed).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output", type=Path, help="Default: <session>/aligned_capture_viewer.html")
    args = parser.parse_args()
    root = args.session_dir.resolve()
    for name in ("capture.json", "imudata_aligned.jsonl", "hand_pose_aligned.jsonl"):
        if not (root / name).is_file():
            parser.error(f"Missing {root / name}; run align_capture.py first")
    output = args.output or root / "aligned_capture_viewer.html"
    if output.suffix.lower() != ".html":
        parser.error("Output must have an .html extension")
    metadata, payload = pack_session(root)
    template = Path(__file__).with_name("aligned_capture_viewer.template.html").read_text(encoding="utf-8")
    html = template.replace("__METADATA_JSON__", json.dumps(metadata, ensure_ascii=False).replace("<", "\\u003c"))
    html = html.replace("__PACKED_DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved {output} ({output.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print(json.dumps({key: metadata[key] for key in ("initialTime", "imuCount", "poseCount", "invalidHands")}, indent=2))


if __name__ == "__main__":
    main()
