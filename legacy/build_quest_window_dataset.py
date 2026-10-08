import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


VECTOR_FIELDS = ("accelerometer", "gyroscope", "userAcceleration", "rotationRate")
RELABELED_DOUBLE_PINCH_PRE_MS = 700
RELABELED_DOUBLE_PINCH_POST_MS = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean label-folder dataset by cutting fixed IMU windows before Quest events."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help=(
            "Quest-style dataset root, a nested dataset root, or one session folder. "
            "Session folders are discovered by looking for imudata.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="clean_quest_windows",
        help="Output root. Each label gets one folder.",
    )
    parser.add_argument(
        "--pre-ms",
        type=int,
        default=1000,
        help="Milliseconds of IMU data to keep before each Quest event.",
    )
    parser.add_argument(
        "--post-ms",
        type=int,
        default=0,
        help="Milliseconds of IMU data to keep after each Quest event.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=80,
        help="Skip windows with fewer IMU samples. Use 1 to keep partial windows.",
    )
    parser.add_argument(
        "--target-points",
        type=int,
        default=100,
        help="Resample every output window to exactly this many IMU points.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional max samples per label. 0 means no limit.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated label allowlist, e.g. swipe_left,thumb_tap.",
    )
    parser.add_argument(
        "--session-label-filter",
        choices=("majority", "none"),
        default="majority",
        help=(
            "majority keeps only the most frequent Quest label in each session, "
            "so accidental detections from other gestures are ignored. Use none "
            "for intentionally mixed-label sessions."
        ),
    )
    parser.add_argument(
        "--include-end-events",
        action="store_true",
        help="Include *_end/release events. Default skips them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-dir before writing.",
    )
    parser.add_argument(
        "--flat-json",
        action="store_true",
        help="Write one JSON object with samples array instead of sample JSONL.",
    )
    parser.add_argument(
        "--no-negative",
        action="store_true",
        help="Do not generate negative windows.",
    )
    parser.add_argument(
        "--negative-label",
        default="negative",
        help="Folder/label name for generated negative examples.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=1.0,
        help=(
            "Target total negative samples as a ratio of total positive samples. "
            "Negatives are drawn only from sessions without Quest events."
        ),
    )
    parser.add_argument(
        "--negative-max-samples",
        type=int,
        default=0,
        help="Optional maximum total negative samples. 0 means no separate negative limit.",
    )
    parser.add_argument(
        "--negative-step-ms",
        type=int,
        default=1000,
        help="Stride for scanning candidate negative windows.",
    )
    parser.add_argument(
        "--negative-guard-ms",
        type=int,
        default=0,
        help="Extra margin around each Quest positive window that negative windows must avoid.",
    )
    return parser.parse_args()


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


def list_session_dirs(dataset_dir: Path):
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    if (dataset_dir / "imudata.jsonl").exists():
        return [dataset_dir]

    return sorted({path.parent for path in dataset_dir.rglob("imudata.jsonl")})


def normalize_label(value):
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def quest_label(row):
    gesture_event = row.get("gestureEvent")
    if isinstance(gesture_event, str) and gesture_event:
        normalized = normalize_label(gesture_event)
        if normalized != "quest_gesture_event":
            return normalized

    label = row.get("label")
    phase = row.get("phase")
    if isinstance(label, str) and label:
        normalized_label = normalize_label(label)
        normalized_phase = normalize_label(phase) if isinstance(phase, str) and phase else ""
        if normalized_label == "pinch" and normalized_phase == "start":
            return "pinch_start"
        if normalized_phase and normalized_phase not in ("detected", "none"):
            return f"{normalized_label}_{normalized_phase}"
        return normalized_label

    for field in ("pinchEvent", "microgesture", "eventType"):
        value = row.get(field)
        if isinstance(value, str) and value:
            normalized = normalize_label(value)
            if normalized != "quest_gesture_event":
                return normalized

    return "quest_event"


def is_end_event(row, label):
    phase = row.get("phase")
    if isinstance(phase, str) and normalize_label(phase) == "end":
        return True
    return label.endswith("_end")


def numeric_unix_time(row):
    value = row.get("unixTimeMs")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def sorted_imu_rows(rows):
    valid_rows = []
    for row in rows:
        timestamp = numeric_unix_time(row)
        if timestamp is not None:
            valid_rows.append((timestamp, row))
    return sorted(valid_rows, key=lambda item: item[0])


def extract_window(imu_rows, start_ms, end_ms):
    return [row for timestamp, row in imu_rows if start_ms <= timestamp <= end_ms]


def clean_vector(value):
    if not isinstance(value, dict):
        return None
    cleaned = {}
    for axis in ("x", "y", "z"):
        raw = value.get(axis)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            cleaned[axis] = float(raw)
    return cleaned if cleaned else None


def interpolate_vector_field(rows, event_ts_ms, relative_times_ms, field):
    source_times = []
    source_values = []
    for row in rows:
        timestamp = numeric_unix_time(row)
        value = clean_vector(row.get(field))
        if timestamp is None or value is None:
            continue
        if not all(axis in value for axis in ("x", "y", "z")):
            continue
        source_times.append(timestamp - event_ts_ms)
        source_values.append([value["x"], value["y"], value["z"]])

    if len(source_times) < 2:
        return None

    source_times = np.asarray(source_times, dtype=np.float64)
    source_values = np.asarray(source_values, dtype=np.float64)
    order = np.argsort(source_times)
    source_times = source_times[order]
    source_values = source_values[order]

    interpolated = {}
    for axis_index, axis in enumerate(("x", "y", "z")):
        interpolated[axis] = np.interp(relative_times_ms, source_times, source_values[:, axis_index])
    return interpolated


def resample_window(rows, event_ts_ms, pre_ms, post_ms, target_points):
    if target_points <= 0:
        raise SystemExit("--target-points must be > 0")

    relative_times_ms = np.linspace(-pre_ms, post_ms, target_points, endpoint=False, dtype=np.float64)
    field_values = {}
    for field in VECTOR_FIELDS:
        interpolated = interpolate_vector_field(rows, event_ts_ms, relative_times_ms, field)
        if interpolated is not None:
            field_values[field] = interpolated

    samples = []
    for index, relative_time in enumerate(relative_times_ms):
        sample = {
            "sampleIndex": index,
            "tMs": int(round(float(relative_time))),
        }
        for field, values in field_values.items():
            sample[field] = {
                axis: float(values[axis][index])
                for axis in ("x", "y", "z")
            }
        samples.append(sample)
    return samples


def write_sample_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_sample_json(path, label, samples, pre_ms, post_ms):
    payload = {
        "label": label,
        "preMs": pre_ms,
        "postMs": post_ms,
        "sampleCount": len(samples),
        "samples": samples,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def parse_label_allowlist(raw_labels):
    if not raw_labels:
        return None
    return {normalize_label(item) for item in raw_labels.split(",") if item.strip()}


def is_pinch_like_label(label):
    return label == "pinch" or label.startswith("pinch_")


def session_contains_double_pinch(quest_rows):
    for quest_row in quest_rows:
        if quest_label(quest_row) == "double_pinch":
            return True
    return False


def remap_session_label(label, session_has_double_pinch):
    if session_has_double_pinch and is_pinch_like_label(label):
        return "double_pinch"
    return label


def event_window_config(event_ts_ms, args, relabeled_as_double_pinch=False):
    if relabeled_as_double_pinch:
        pre_ms = RELABELED_DOUBLE_PINCH_PRE_MS
        post_ms = RELABELED_DOUBLE_PINCH_POST_MS
    else:
        pre_ms = args.pre_ms
        post_ms = args.post_ms

    return (
        event_ts_ms - pre_ms,
        event_ts_ms + post_ms,
        pre_ms,
        post_ms,
    )


def is_usable_quest_event(row, label, args, label_allowlist):
    if numeric_unix_time(row) is None:
        return False
    if not args.include_end_events and is_end_event(row, label):
        return False
    if label_allowlist is not None and label not in label_allowlist:
        return False
    return True


def dominant_session_label(quest_rows, args, label_allowlist, session_has_double_pinch):
    if args.session_label_filter == "none":
        return None

    label_counts = Counter()
    for quest_row in quest_rows:
        label = remap_session_label(quest_label(quest_row), session_has_double_pinch)
        if is_usable_quest_event(quest_row, label, args, label_allowlist):
            label_counts[label] += 1

    if not label_counts:
        return None
    return label_counts.most_common(1)[0][0]


def prepare_output_dir(output_dir, overwrite):
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_output_sample(output_dir, label, sample_id, samples, args, pre_ms=None, post_ms=None):
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    extension = ".json" if args.flat_json else ".jsonl"
    output_path = label_dir / f"{label}_{sample_id:06d}{extension}"
    actual_pre_ms = args.pre_ms if pre_ms is None else pre_ms
    actual_post_ms = args.post_ms if post_ms is None else post_ms
    if args.flat_json:
        write_sample_json(output_path, label, samples, actual_pre_ms, actual_post_ms)
    else:
        write_sample_jsonl(output_path, samples)
    return output_path


def intervals_overlap(start_a, end_a, start_b, end_b):
    return start_a < end_b and start_b < end_a


def candidate_overlaps_exclusions(start_ms, end_ms, exclusions):
    for exclusion_start, exclusion_end in exclusions:
        if intervals_overlap(start_ms, end_ms, exclusion_start, exclusion_end):
            return True
    return False


def quest_exclusion_intervals(quest_rows, args):
    exclusions = []
    guard_ms = max(0, args.negative_guard_ms)
    for quest_row in quest_rows:
        event_ts_ms = numeric_unix_time(quest_row)
        if event_ts_ms is None:
            continue
        start_ms = event_ts_ms - args.pre_ms - guard_ms
        end_ms = event_ts_ms + args.post_ms + guard_ms
        exclusions.append((start_ms, end_ms))
    return sorted(exclusions)


def choose_evenly_spaced(items, target_count):
    if target_count <= 0 or len(items) <= target_count:
        return items
    indices = np.linspace(0, len(items) - 1, target_count, dtype=int)
    return [items[index] for index in indices]


def generate_negative_candidates(imu_rows, exclusions, args):
    if not imu_rows:
        return []

    first_ts = imu_rows[0][0]
    last_ts = imu_rows[-1][0]
    total_window_ms = args.pre_ms + args.post_ms
    if total_window_ms <= 0:
        raise SystemExit("--pre-ms + --post-ms must be > 0")
    if args.negative_step_ms <= 0:
        raise SystemExit("--negative-step-ms must be > 0")

    candidates = []
    start_ms = first_ts
    last_start_ms = last_ts - total_window_ms
    while start_ms <= last_start_ms:
        event_ts_ms = start_ms + args.pre_ms
        end_ms = event_ts_ms + args.post_ms
        if not candidate_overlaps_exclusions(start_ms, end_ms, exclusions):
            candidates.append((start_ms, event_ts_ms, end_ms))
        start_ms += args.negative_step_ms
    return candidates


def write_negative_samples_from_empty_sessions(
    negative_source_sessions,
    output_dir,
    args,
    label_counts,
    skipped_reasons,
    positive_sample_count,
):
    if args.no_negative or args.negative_ratio <= 0:
        return
    if not negative_source_sessions:
        skipped_reasons["no_negative_source_sessions"] += 1
        return

    negative_label = normalize_label(args.negative_label)
    if args.negative_max_samples > 0:
        remaining_negative_budget = args.negative_max_samples - label_counts[negative_label]
        if remaining_negative_budget <= 0:
            skipped_reasons["negative_max_samples"] += 1
            return
    else:
        remaining_negative_budget = None

    if positive_sample_count <= 0:
        if remaining_negative_budget is None:
            skipped_reasons["negative_no_positive_samples"] += 1
            return
        target_count = remaining_negative_budget
    else:
        target_count = int(round(positive_sample_count * args.negative_ratio))
    if remaining_negative_budget is not None:
        target_count = min(target_count, remaining_negative_budget)
    if target_count <= 0:
        return

    candidates = []
    for imu_rows in negative_source_sessions:
        for start_ms, event_ts_ms, end_ms in generate_negative_candidates(imu_rows, [], args):
            candidates.append((imu_rows, start_ms, event_ts_ms, end_ms))

    candidates = choose_evenly_spaced(candidates, target_count)
    if not candidates:
        skipped_reasons["no_negative_candidates"] += 1
        return

    for imu_rows, start_ms, event_ts_ms, end_ms in candidates:
        window_rows = extract_window(imu_rows, start_ms, end_ms)
        if len(window_rows) < args.min_samples:
            skipped_reasons["negative_too_few_imu_samples"] += 1
            continue

        samples = resample_window(window_rows, event_ts_ms, args.pre_ms, args.post_ms, args.target_points)
        label_counts[negative_label] += 1
        write_output_sample(
            output_dir,
            negative_label,
            label_counts[negative_label],
            samples,
            args,
            pre_ms=args.pre_ms,
            post_ms=args.post_ms,
        )


def process_session(
    session_dir,
    output_dir,
    args,
    label_allowlist,
    label_counts,
    skipped_reasons,
):
    imu_rows = sorted_imu_rows(load_jsonl(session_dir / "imudata.jsonl"))
    quest_rows = load_jsonl(session_dir / "questlog.jsonl")

    if not imu_rows:
        skipped_reasons["no_imu_rows"] += 1
        return None
    if not quest_rows:
        return imu_rows

    session_has_double_pinch = session_contains_double_pinch(quest_rows)
    session_label = dominant_session_label(
        quest_rows,
        args,
        label_allowlist,
        session_has_double_pinch,
    )
    session_positive_count = 0
    for quest_row in quest_rows:
        raw_label = quest_label(quest_row)
        label = remap_session_label(raw_label, session_has_double_pinch)
        if not is_usable_quest_event(quest_row, label, args, label_allowlist):
            if numeric_unix_time(quest_row) is None:
                skipped_reasons["quest_event_missing_time"] += 1
            elif not args.include_end_events and is_end_event(quest_row, label):
                skipped_reasons["end_event"] += 1
            elif label_allowlist is not None and label not in label_allowlist:
                skipped_reasons["label_not_allowed"] += 1
            else:
                skipped_reasons["quest_event_unusable"] += 1
            continue

        if session_label is not None and label != session_label:
            skipped_reasons["non_dominant_session_label"] += 1
            continue

        event_ts_ms = numeric_unix_time(quest_row)
        if event_ts_ms is None:
            skipped_reasons["quest_event_missing_time"] += 1
            continue

        if args.max_samples > 0 and label_counts[label] >= args.max_samples:
            skipped_reasons["max_samples_per_label"] += 1
            continue

        relabeled_as_double_pinch = (
            session_has_double_pinch
            and raw_label != "double_pinch"
            and label == "double_pinch"
        )
        window_start_ms, window_end_ms, window_pre_ms, window_post_ms = event_window_config(
            event_ts_ms,
            args,
            relabeled_as_double_pinch=relabeled_as_double_pinch,
        )
        window_rows = extract_window(imu_rows, window_start_ms, window_end_ms)
        if len(window_rows) < args.min_samples:
            skipped_reasons["too_few_imu_samples"] += 1
            continue

        samples = resample_window(
            window_rows,
            event_ts_ms,
            window_pre_ms,
            window_post_ms,
            args.target_points,
        )
        label_counts[label] += 1
        session_positive_count += 1
        write_output_sample(
            output_dir,
            label,
            label_counts[label],
            samples,
            args,
            pre_ms=window_pre_ms,
            post_ms=window_post_ms,
        )

    if session_positive_count == 0:
        skipped_reasons["no_positive_samples_in_quest_session"] += 1
    return None


def write_summary(output_dir, args, label_counts, skipped_reasons, negative_source_session_count):
    summary = {
        "preMs": args.pre_ms,
        "postMs": args.post_ms,
        "minSamples": args.min_samples,
        "targetPoints": args.target_points,
        "sessionLabelFilter": args.session_label_filter,
        "negativeEnabled": not args.no_negative,
        "negativeLabel": normalize_label(args.negative_label),
        "negativeSource": "sessions_without_quest_events",
        "negativeSourceSessions": negative_source_session_count,
        "negativeRatio": args.negative_ratio,
        "negativeStepMs": args.negative_step_ms,
        "negativeGuardMs": args.negative_guard_ms,
        "pinchRelabeling": {
            "sessionDoublePinchPromotesPinchToDoublePinch": True,
            "relabeledPinchPreMs": RELABELED_DOUBLE_PINCH_PRE_MS,
            "relabeledPinchPostMs": RELABELED_DOUBLE_PINCH_POST_MS,
        },
        "labels": dict(sorted(label_counts.items())),
        "totalSamples": int(sum(label_counts.values())),
        "skipped": dict(sorted(skipped_reasons.items())),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    label_allowlist = parse_label_allowlist(args.labels)

    if args.pre_ms < 0 or args.post_ms < 0:
        raise SystemExit("--pre-ms and --post-ms must be >= 0")
    if args.min_samples < 1:
        raise SystemExit("--min-samples must be >= 1")
    if args.target_points < 1:
        raise SystemExit("--target-points must be >= 1")
    if args.negative_ratio < 0:
        raise SystemExit("--negative-ratio must be >= 0")
    if args.negative_max_samples < 0:
        raise SystemExit("--negative-max-samples must be >= 0")
    if args.negative_guard_ms < 0:
        raise SystemExit("--negative-guard-ms must be >= 0")

    sessions = list_session_dirs(dataset_dir)
    if not sessions:
        raise SystemExit(f"No Quest sessions found in {dataset_dir}")

    prepare_output_dir(output_dir, args.overwrite)
    label_counts = Counter()
    skipped_reasons = defaultdict(int)
    negative_source_sessions = []

    for session_dir in sessions:
        negative_imu_rows = process_session(
            session_dir=session_dir,
            output_dir=output_dir,
            args=args,
            label_allowlist=label_allowlist,
            label_counts=label_counts,
            skipped_reasons=skipped_reasons,
        )
        if negative_imu_rows is not None:
            negative_source_sessions.append(negative_imu_rows)

    positive_sample_count = sum(label_counts.values())
    write_negative_samples_from_empty_sessions(
        negative_source_sessions=negative_source_sessions,
        output_dir=output_dir,
        args=args,
        label_counts=label_counts,
        skipped_reasons=skipped_reasons,
        positive_sample_count=positive_sample_count,
    )

    summary_path = write_summary(
        output_dir,
        args,
        label_counts,
        skipped_reasons,
        len(negative_source_sessions),
    )

    print(f"Wrote clean dataset: {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Total samples: {sum(label_counts.values())}")
    print(f"Negative source sessions: {len(negative_source_sessions)}")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    if skipped_reasons:
        print("Skipped:")
        for reason, count in sorted(skipped_reasons.items()):
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
