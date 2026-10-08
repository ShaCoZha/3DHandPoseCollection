import argparse
import copy
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


AXES = ("x", "y", "z")
VECTOR_FIELDS = ("accelerometer", "gyroscope", "userAcceleration", "rotationRate")
DEFAULT_LABELS = (
    "negative",
    "double_pinch",
    "pinch",
    "swipe_backward",
    "swipe_forward",
    "swipe_left",
    "swipe_right",
    "thumb_tap",
)
NEGATIVE_LABEL = "negative"
PROMPT_SUFFIX = "Prompt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a per-person 1000 ms prompt-window dataset from IMUGestureDC/*Prompt "
            "sessions, handling both timestamp-only and Quest-labeled prompt recordings."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="IMUGestureDC",
        help="Root containing <Person>Prompt folders or one specific prompt folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="prompt_1000ms_windows_by_person",
        help="Output root laid out as output/<person>/<label>/*.jsonl.",
    )
    parser.add_argument(
        "--people",
        default=None,
        help="Optional comma-separated allowlist, e.g. Jiawei,Jesse.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=1000,
        help="Target window size in milliseconds.",
    )
    parser.add_argument(
        "--quest-tail-ms",
        type=int,
        default=2000,
        help="Keep a prompt window only if it contains a Quest event in-window or within this tail.",
    )
    parser.add_argument(
        "--negative-step-ms",
        type=int,
        default=250,
        help="Stride in milliseconds for negative-session sliding windows.",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=100,
        help="Write exactly this many IMU samples per output window.",
    )
    parser.add_argument(
        "--pad-to-points",
        type=int,
        default=None,
        help=(
            "Optional intermediate padded length. When set, each extracted window is "
            "resampled to the number of points implied by window-ms within pad-window-ms, "
            "center-padded to this length, then downsampled to --target-points."
        ),
    )
    parser.add_argument(
        "--pad-window-ms",
        type=int,
        default=2560,
        help="Virtual padded window duration in milliseconds when --pad-to-points is used.",
    )
    parser.add_argument(
        "--pad-mode",
        default="edge",
        choices=("edge", "zero"),
        help="How to fill synthetic samples added by --pad-to-points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used when interpolating sparse windows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir before writing.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.window_ms <= 0:
        raise SystemExit("--window-ms must be > 0")
    if args.quest_tail_ms < 0:
        raise SystemExit("--quest-tail-ms must be >= 0")
    if args.negative_step_ms <= 0:
        raise SystemExit("--negative-step-ms must be > 0")
    if args.target_points <= 0:
        raise SystemExit("--target-points must be > 0")
    if args.pad_to_points is not None:
        if args.pad_to_points <= 0:
            raise SystemExit("--pad-to-points must be > 0")
        if args.pad_to_points < args.target_points:
            raise SystemExit("--pad-to-points must be >= --target-points")
        if args.pad_window_ms <= 0:
            raise SystemExit("--pad-window-ms must be > 0")
        if args.pad_window_ms < args.window_ms:
            raise SystemExit("--pad-window-ms must be >= --window-ms")


def normalize_label(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def load_json(path: Path):
    if not path.exists():
        raise SystemExit(f"JSON file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Bad JSON in {path}: {exc}") from exc


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Bad JSON in {path} line {line_number}: {exc}") from exc
    return rows


def numeric_unix_time(row):
    value = row.get("unixTimeMs")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def clean_vector(value):
    if not isinstance(value, dict):
        return None
    cleaned = {}
    for axis in AXES:
        raw = value.get(axis)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            cleaned[axis] = float(raw)
    if len(cleaned) == len(AXES):
        return cleaned
    return None


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_people_allowlist(raw_people):
    if not raw_people:
        return None
    allowlist = set()
    for item in raw_people.split(","):
        text = item.strip()
        if not text:
            continue
        allowlist.add(text)
        allowlist.add(strip_prompt_suffix(text))
    return allowlist or None


def strip_prompt_suffix(name: str):
    return name[: -len(PROMPT_SUFFIX)] if name.endswith(PROMPT_SUFFIX) else name


def list_prompt_person_dirs(dataset_dir: Path):
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if (dataset_dir / "timestamp.json").exists() and (dataset_dir / "imudata.jsonl").exists():
        return [dataset_dir]

    if dataset_dir.name.endswith(PROMPT_SUFFIX):
        return [dataset_dir]

    prompt_dirs = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir() and path.name.endswith(PROMPT_SUFFIX)
    )
    if not prompt_dirs:
        raise SystemExit(f"No *{PROMPT_SUFFIX} folders found in {dataset_dir}")
    return prompt_dirs


def list_session_dirs(prompt_person_dir: Path):
    return sorted(
        path
        for path in prompt_person_dir.iterdir()
        if path.is_dir()
        and (path / "timestamp.json").exists()
        and (path / "imudata.jsonl").exists()
    )


def ensure_person_label_dirs(person_output_dir: Path, labels):
    person_output_dir.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (person_output_dir / label).mkdir(parents=True, exist_ok=True)


def canonical_session_label(label):
    normalized = normalize_label(label)
    if normalized in ("random_motion", "quest_recording", "negative", "none", "unknown"):
        return NEGATIVE_LABEL
    return normalized


def timestamp_session_label(timestamp_data):
    return canonical_session_label(timestamp_data.get("gestureType", "unknown"))


def parse_prompt_windows(timestamp_data, target_window_ms):
    windows = []
    for item in timestamp_data.get("timestamps", []):
        start_ms = item.get("doUnixMs")
        end_ms = item.get("windowEndUnixMs")
        prompt_index = item.get("promptIndex")
        if not isinstance(start_ms, (int, float)) or isinstance(start_ms, bool):
            continue
        if not isinstance(end_ms, (int, float)) or isinstance(end_ms, bool):
            continue

        original_start_ms = float(start_ms)
        original_end_ms = float(end_ms)
        duration_ms = original_end_ms - original_start_ms
        if duration_ms <= 0:
            continue
        if duration_ms + 1e-6 < target_window_ms:
            continue

        trim_ms = max((duration_ms - target_window_ms) / 2.0, 0.0)
        final_start_ms = original_start_ms + trim_ms
        final_end_ms = original_end_ms - trim_ms
        windows.append(
            {
                "promptIndex": int(prompt_index) if isinstance(prompt_index, int) else len(windows) + 1,
                "originalStartMs": original_start_ms,
                "originalEndMs": original_end_ms,
                "startMs": final_start_ms,
                "endMs": final_end_ms,
                "durationMs": final_end_ms - final_start_ms,
            }
        )
    return windows


def load_sorted_imu_points(session_dir: Path):
    points = []
    for row in load_jsonl(session_dir / "imudata.jsonl"):
        timestamp_ms = numeric_unix_time(row)
        if timestamp_ms is None:
            continue
        fields = {}
        for field in VECTOR_FIELDS:
            vector = clean_vector(row.get(field))
            if vector is not None:
                fields[field] = vector
        if not fields:
            continue
        points.append({"timestampMs": float(timestamp_ms), "fields": fields})
    points.sort(key=lambda item: item["timestampMs"])
    return points


def extract_points(points, start_ms, end_ms):
    return [
        copy.deepcopy(point)
        for point in points
        if start_ms <= point["timestampMs"] < end_ms
    ]


def quest_row_label(row):
    for field in ("gestureEvent", "label", "pinchEvent", "microgesture", "eventType", "type"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            normalized = normalize_label(value)
            if normalized not in ("quest_gesture_event", "quest_pinch_event", "quest_event"):
                return normalized
    return None


def is_pinch_like_label(label):
    return label == "pinch" or label.startswith("pinch_")


def load_quest_events(session_dir: Path):
    quest_path = session_dir / "questlog.jsonl"
    if not quest_path.exists():
        return None

    raw_events = []
    has_double_pinch = False
    for row in load_jsonl(quest_path):
        timestamp_ms = numeric_unix_time(row)
        label = quest_row_label(row)
        if timestamp_ms is None or label is None:
            continue
        if label == "double_pinch":
            has_double_pinch = True
        raw_events.append({"timestampMs": float(timestamp_ms), "rawLabel": label})

    if not raw_events:
        return []

    events = []
    for event in raw_events:
        label = event["rawLabel"]
        if has_double_pinch and is_pinch_like_label(label):
            label = "double_pinch"
        events.append(
            {
                "timestampMs": event["timestampMs"],
                "label": label,
            }
        )
    events.sort(key=lambda item: item["timestampMs"])
    return events


def dominant_label(labels):
    if not labels:
        return None
    counts = Counter(labels)
    max_count = max(counts.values())
    best_labels = sorted(label for label, count in counts.items() if count == max_count)
    return best_labels[0]


def matched_quest_events(events, window, quest_tail_ms):
    search_start_ms = window["originalStartMs"]
    search_end_ms = window["originalEndMs"] + quest_tail_ms
    return [
        event
        for event in events
        if search_start_ms <= event["timestampMs"] <= search_end_ms
    ]


def choose_window_label(events, anchor_ms, preferred_label=None):
    if not events:
        return None

    counts = Counter(event["label"] for event in events)
    if preferred_label is not None and preferred_label in counts and len(counts) > 1:
        return preferred_label
    max_count = max(counts.values())
    candidate_labels = [label for label, count in counts.items() if count == max_count]
    if len(candidate_labels) == 1:
        return candidate_labels[0]

    def tie_break(label):
        distances = [
            abs(event["timestampMs"] - anchor_ms)
            for event in events
            if event["label"] == label
        ]
        return (min(distances), label)

    return min(candidate_labels, key=tie_break)


def session_time_range(timestamp_data, imu_points):
    if not imu_points:
        return None

    first_ts = imu_points[0]["timestampMs"]
    last_ts = imu_points[-1]["timestampMs"]
    started_at = timestamp_data.get("startedAtUnixMs")
    stopped_at = timestamp_data.get("stoppedAtUnixMs")
    if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
        first_ts = max(first_ts, float(started_at))
    if isinstance(stopped_at, (int, float)) and not isinstance(stopped_at, bool):
        last_ts = min(last_ts, float(stopped_at))
    if last_ts <= first_ts:
        return None
    return first_ts, last_ts


def iter_negative_windows(timestamp_data, imu_points, window_ms, step_ms):
    time_range = session_time_range(timestamp_data, imu_points)
    if time_range is None:
        return

    session_start_ms, session_end_ms = time_range
    last_start_ms = session_end_ms - window_ms
    current_start_ms = session_start_ms
    while current_start_ms <= last_start_ms + 1e-6:
        yield current_start_ms, current_start_ms + window_ms
        current_start_ms += step_ms


def interpolate_vector(left, right, alpha):
    return {
        axis: float((1.0 - alpha) * left[axis] + alpha * right[axis])
        for axis in AXES
    }


def interpolate_point(left, right, rng):
    if right["timestampMs"] <= left["timestampMs"]:
        alpha = 0.5
        timestamp_ms = float(left["timestampMs"])
    else:
        alpha = float(rng.uniform(0.0, 1.0))
        timestamp_ms = float(
            left["timestampMs"] + alpha * (right["timestampMs"] - left["timestampMs"])
        )

    fields = {}
    for field in VECTOR_FIELDS:
        left_vector = left["fields"].get(field)
        right_vector = right["fields"].get(field)
        if left_vector is not None and right_vector is not None:
            fields[field] = interpolate_vector(left_vector, right_vector, alpha)
        elif left_vector is not None:
            fields[field] = copy.deepcopy(left_vector)
        elif right_vector is not None:
            fields[field] = copy.deepcopy(right_vector)

    return {"timestampMs": timestamp_ms, "fields": fields}


def ensure_target_points(points, target_points, rng):
    if not points:
        return []

    ordered = sorted(points, key=lambda item: item["timestampMs"])
    if len(ordered) == 1:
        return [copy.deepcopy(ordered[0]) for _ in range(target_points)]

    while len(ordered) < target_points:
        segment_index = int(rng.integers(0, len(ordered) - 1))
        new_point = interpolate_point(ordered[segment_index], ordered[segment_index + 1], rng)
        ordered.insert(segment_index + 1, new_point)

    if len(ordered) > target_points:
        indices = np.linspace(0, len(ordered) - 1, target_points, dtype=int)
        ordered = [ordered[index] for index in indices]

    return ordered


def copy_point_with_timestamp(point, timestamp_ms):
    return {
        "timestampMs": float(timestamp_ms),
        "fields": copy.deepcopy(point["fields"]),
    }


def zero_point_like(point, timestamp_ms):
    return {
        "timestampMs": float(timestamp_ms),
        "fields": {
            field: {axis: 0.0 for axis in AXES}
            for field in point["fields"].keys()
        },
    }


def padded_sample_point(edge_point, timestamp_ms, pad_mode):
    if pad_mode == "zero":
        return zero_point_like(edge_point, timestamp_ms)
    return copy_point_with_timestamp(edge_point, timestamp_ms)


def downsample_points_evenly(points, target_points):
    if len(points) == target_points:
        return points
    indices = np.linspace(0, len(points) - 1, target_points, dtype=int)
    return [points[index] for index in indices]


def prepare_window_points(raw_points, window_start_ms, args, rng):
    if not args.pad_to_points:
        return ensure_target_points(raw_points, args.target_points, rng), window_start_ms

    content_points = int(round(args.pad_to_points * args.window_ms / args.pad_window_ms))
    content_points = max(2, min(args.pad_to_points, content_points))
    content = ensure_target_points(raw_points, content_points, rng)
    if not content:
        return [], window_start_ms

    left_pad = (args.pad_to_points - content_points) // 2
    right_pad = args.pad_to_points - content_points - left_pad
    step_ms = float(args.pad_window_ms) / float(args.pad_to_points)
    padded_start_ms = float(window_start_ms) - left_pad * step_ms

    padded = []
    first_content = content[0]
    last_content = content[-1]
    for index in range(left_pad):
        timestamp_ms = padded_start_ms + index * step_ms
        padded.append(padded_sample_point(first_content, timestamp_ms, args.pad_mode))

    for index, point in enumerate(content, start=left_pad):
        timestamp_ms = padded_start_ms + index * step_ms
        padded.append(copy_point_with_timestamp(point, timestamp_ms))

    for index in range(right_pad):
        padded_index = left_pad + content_points + index
        timestamp_ms = padded_start_ms + padded_index * step_ms
        padded.append(padded_sample_point(last_content, timestamp_ms, args.pad_mode))

    return downsample_points_evenly(padded, args.target_points), padded_start_ms


def write_window_jsonl(path: Path, points, window_start_ms):
    with path.open("w", encoding="utf-8") as handle:
        for sample_index, point in enumerate(points):
            row = {
                "sampleIndex": int(sample_index),
                "tMs": int(round(point["timestampMs"] - window_start_ms)),
                "unixTimeMs": int(round(point["timestampMs"])),
            }
            for field, vector in point["fields"].items():
                row[field] = {axis: float(vector[axis]) for axis in AXES}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_labeled_window(
    person_output_dir: Path,
    label: str,
    label_counts: Counter,
    session_dir: Path,
    sample_tag: str,
    points,
    window_start_ms,
):
    label_dir = person_output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    label_counts[label] += 1
    file_name = f"{label}_{label_counts[label]:06d}_{session_dir.name}_{sample_tag}.jsonl"
    output_path = label_dir / file_name
    write_window_jsonl(output_path, points, window_start_ms)
    return output_path


def build_positive_timestamp_windows(
    session_dir,
    timestamp_data,
    imu_points,
    person_output_dir,
    label,
    args,
    rng,
    label_counts,
    skipped_reasons,
):
    written = 0
    for window in parse_prompt_windows(timestamp_data, args.window_ms):
        raw_points = extract_points(imu_points, window["startMs"], window["endMs"])
        if not raw_points:
            skipped_reasons["empty_positive_window"] += 1
            continue
        sample_points, sample_start_ms = prepare_window_points(
            raw_points,
            window["startMs"],
            args,
            rng,
        )
        if not sample_points:
            skipped_reasons["failed_to_prepare_positive_window"] += 1
            continue
        write_labeled_window(
            person_output_dir=person_output_dir,
            label=label,
            label_counts=label_counts,
            session_dir=session_dir,
            sample_tag=f"prompt_{window['promptIndex']:03d}",
            points=sample_points,
            window_start_ms=sample_start_ms,
        )
        written += 1
    return written


def build_negative_session_windows(
    session_dir,
    timestamp_data,
    imu_points,
    person_output_dir,
    args,
    rng,
    label_counts,
    skipped_reasons,
):
    written = 0
    for negative_index, (start_ms, end_ms) in enumerate(
        iter_negative_windows(
            timestamp_data=timestamp_data,
            imu_points=imu_points,
            window_ms=args.window_ms,
            step_ms=args.negative_step_ms,
        ),
        start=1,
    ):
        raw_points = extract_points(imu_points, start_ms, end_ms)
        if not raw_points:
            skipped_reasons["empty_negative_window"] += 1
            continue
        sample_points, sample_start_ms = prepare_window_points(
            raw_points,
            start_ms,
            args,
            rng,
        )
        if not sample_points:
            skipped_reasons["failed_to_prepare_negative_window"] += 1
            continue
        write_labeled_window(
            person_output_dir=person_output_dir,
            label=NEGATIVE_LABEL,
            label_counts=label_counts,
            session_dir=session_dir,
            sample_tag=f"negative_{negative_index:04d}",
            points=sample_points,
            window_start_ms=sample_start_ms,
        )
        written += 1
    return written


def build_quest_labeled_windows(
    session_dir,
    timestamp_data,
    imu_points,
    quest_events,
    person_output_dir,
    args,
    rng,
    label_counts,
    skipped_reasons,
):
    session_labels = [event["label"] for event in quest_events]
    session_label = dominant_label(session_labels)
    if session_label is None:
        return 0

    written = 0
    for window in parse_prompt_windows(timestamp_data, args.window_ms):
        window_events = matched_quest_events(
            events=quest_events,
            window=window,
            quest_tail_ms=args.quest_tail_ms,
        )
        if not window_events:
            skipped_reasons["quest_window_without_event"] += 1
            continue

        window_label = choose_window_label(
            window_events,
            anchor_ms=window["originalEndMs"],
            preferred_label=session_label,
        )
        if window_label != session_label:
            skipped_reasons["quest_window_not_session_majority"] += 1
            continue

        raw_points = extract_points(imu_points, window["startMs"], window["endMs"])
        if not raw_points:
            skipped_reasons["empty_quest_window"] += 1
            continue

        sample_points, sample_start_ms = prepare_window_points(
            raw_points,
            window["startMs"],
            args,
            rng,
        )
        if not sample_points:
            skipped_reasons["failed_to_prepare_quest_window"] += 1
            continue

        write_labeled_window(
            person_output_dir=person_output_dir,
            label=session_label,
            label_counts=label_counts,
            session_dir=session_dir,
            sample_tag=f"prompt_{window['promptIndex']:03d}",
            points=sample_points,
            window_start_ms=sample_start_ms,
        )
        written += 1

    return written


def process_session(
    session_dir: Path,
    person_output_dir: Path,
    args,
    rng,
    label_counts,
    skipped_reasons,
    session_mode_counts,
):
    timestamp_data = load_json(session_dir / "timestamp.json")
    imu_points = load_sorted_imu_points(session_dir)
    if not imu_points:
        skipped_reasons["session_without_imu"] += 1
        return 0

    quest_events = load_quest_events(session_dir)
    if quest_events is None:
        label = timestamp_session_label(timestamp_data)
        if label == NEGATIVE_LABEL:
            session_mode_counts["timestamp_negative_session"] += 1
            return build_negative_session_windows(
                session_dir=session_dir,
                timestamp_data=timestamp_data,
                imu_points=imu_points,
                person_output_dir=person_output_dir,
                args=args,
                rng=rng,
                label_counts=label_counts,
                skipped_reasons=skipped_reasons,
            )

        session_mode_counts["timestamp_positive_session"] += 1
        return build_positive_timestamp_windows(
            session_dir=session_dir,
            timestamp_data=timestamp_data,
            imu_points=imu_points,
            person_output_dir=person_output_dir,
            label=label,
            args=args,
            rng=rng,
            label_counts=label_counts,
            skipped_reasons=skipped_reasons,
        )

    if not quest_events:
        session_mode_counts["quest_negative_session"] += 1
        return build_negative_session_windows(
            session_dir=session_dir,
            timestamp_data=timestamp_data,
            imu_points=imu_points,
            person_output_dir=person_output_dir,
            args=args,
            rng=rng,
            label_counts=label_counts,
            skipped_reasons=skipped_reasons,
        )

    session_mode_counts["quest_positive_session"] += 1
    return build_quest_labeled_windows(
        session_dir=session_dir,
        timestamp_data=timestamp_data,
        imu_points=imu_points,
        quest_events=quest_events,
        person_output_dir=person_output_dir,
        args=args,
        rng=rng,
        label_counts=label_counts,
        skipped_reasons=skipped_reasons,
    )


def build_person_dataset(prompt_person_dir: Path, output_dir: Path, args, rng):
    person_name = strip_prompt_suffix(prompt_person_dir.name)
    person_output_dir = output_dir / person_name
    ensure_person_label_dirs(person_output_dir, DEFAULT_LABELS)

    label_counts = Counter()
    skipped_reasons = defaultdict(int)
    session_mode_counts = Counter()
    session_dirs = list_session_dirs(prompt_person_dir)
    if not session_dirs:
        return {
            "person": person_name,
            "promptFolder": prompt_person_dir.name,
            "sessionCount": 0,
            "labels": {},
            "totalSamples": 0,
            "sessionModes": {},
            "skipped": {},
        }

    for session_dir in session_dirs:
        process_session(
            session_dir=session_dir,
            person_output_dir=person_output_dir,
            args=args,
            rng=rng,
            label_counts=label_counts,
            skipped_reasons=skipped_reasons,
            session_mode_counts=session_mode_counts,
        )

    return {
        "person": person_name,
        "promptFolder": prompt_person_dir.name,
        "sessionCount": len(session_dirs),
        "outputDir": str(person_output_dir),
        "labels": dict(sorted(label_counts.items())),
        "totalSamples": int(sum(label_counts.values())),
        "sessionModes": dict(sorted(session_mode_counts.items())),
        "skipped": dict(sorted(skipped_reasons.items())),
    }


def write_summary(output_dir: Path, dataset_dir: Path, args, person_summaries):
    discovered_labels = sorted(
        {
            label
            for summary in person_summaries
            for label in summary.get("labels", {}).keys()
        }
        | set(DEFAULT_LABELS)
    )
    summary = {
        "datasetDir": str(dataset_dir),
        "outputDir": str(output_dir),
        "peopleFilter": args.people,
        "windowMs": args.window_ms,
        "questTailMs": args.quest_tail_ms,
        "negativeStepMs": args.negative_step_ms,
        "targetPoints": args.target_points,
        "padToPoints": args.pad_to_points,
        "padWindowMs": args.pad_window_ms if args.pad_to_points is not None else None,
        "padMode": args.pad_mode if args.pad_to_points is not None else None,
        "seed": args.seed,
        "labels": discovered_labels,
        "people": person_summaries,
    }
    summary_path = output_dir / "summary_by_person.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main():
    args = parse_args()
    validate_args(args)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    people_allowlist = parse_people_allowlist(args.people)
    rng = np.random.default_rng(args.seed)

    prompt_person_dirs = []
    for prompt_person_dir in list_prompt_person_dirs(dataset_dir):
        person_name = strip_prompt_suffix(prompt_person_dir.name)
        if people_allowlist is not None and person_name not in people_allowlist and prompt_person_dir.name not in people_allowlist:
            continue
        prompt_person_dirs.append(prompt_person_dir)

    if not prompt_person_dirs:
        raise SystemExit("No prompt people matched the current dataset path / --people filter")

    prepare_output_dir(output_dir, args.overwrite)

    person_summaries = []
    for prompt_person_dir in prompt_person_dirs:
        person_summaries.append(
            build_person_dataset(
                prompt_person_dir=prompt_person_dir,
                output_dir=output_dir,
                args=args,
                rng=rng,
            )
        )

    summary_path = write_summary(output_dir, dataset_dir, args, person_summaries)
    total_samples = sum(summary["totalSamples"] for summary in person_summaries)
    print(f"Wrote {total_samples} windows across {len(person_summaries)} people.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
