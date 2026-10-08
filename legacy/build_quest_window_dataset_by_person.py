import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_quest_window_dataset import (
    list_session_dirs,
    parse_label_allowlist,
    prepare_output_dir,
    process_session,
    write_negative_samples_from_empty_sessions,
    write_summary,
)


ROOT_SESSION_GROUP = "root_sessions"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build one clean label-folder Quest window dataset per person from a "
            "dataset root laid out like dataset/<person>/<session>/..."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default="IMUGestureDC",
        help=(
            "Root containing person folders. Each person folder can contain one or "
            "more session folders discovered by looking for imudata.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="clean_quest_windows_by_person",
        help=(
            "Output root. Each person gets one folder, and inside each person folder "
            "each label gets one folder."
        ),
    )
    parser.add_argument(
        "--people",
        default=None,
        help=(
            "Optional comma-separated person folder allowlist, e.g. Jiawei,Yiquan. "
            "Names are matched exactly against folder names under dataset-dir."
        ),
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


def validate_args(args):
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


def parse_people_allowlist(raw_people):
    if not raw_people:
        return None
    return {item.strip() for item in raw_people.split(",") if item.strip()}


def session_person_name(dataset_dir: Path, session_dir: Path):
    relative = session_dir.relative_to(dataset_dir)
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return ROOT_SESSION_GROUP


def group_sessions_by_person(dataset_dir: Path, session_dirs, people_allowlist):
    grouped = defaultdict(list)
    for session_dir in session_dirs:
        person_name = session_person_name(dataset_dir, session_dir)
        if people_allowlist is not None and person_name not in people_allowlist:
            continue
        grouped[person_name].append(session_dir)
    return dict(sorted(grouped.items()))


def build_person_dataset(person_name, session_dirs, output_root, args, label_allowlist):
    person_output_dir = output_root / person_name
    prepare_output_dir(person_output_dir, args.overwrite)

    label_counts = Counter()
    skipped_reasons = defaultdict(int)
    negative_source_sessions = []

    for session_dir in sorted(session_dirs):
        negative_imu_rows = process_session(
            session_dir=session_dir,
            output_dir=person_output_dir,
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
        output_dir=person_output_dir,
        args=args,
        label_counts=label_counts,
        skipped_reasons=skipped_reasons,
        positive_sample_count=positive_sample_count,
    )

    summary_path = write_summary(
        person_output_dir,
        args,
        label_counts,
        skipped_reasons,
        len(negative_source_sessions),
    )
    return {
        "person": person_name,
        "sessionCount": len(session_dirs),
        "outputDir": str(person_output_dir),
        "summaryPath": str(summary_path),
        "labels": dict(sorted(label_counts.items())),
        "totalSamples": int(sum(label_counts.values())),
        "negativeSourceSessions": len(negative_source_sessions),
        "skipped": dict(sorted(skipped_reasons.items())),
    }


def write_master_summary(output_dir: Path, dataset_dir: Path, args, person_summaries):
    summary = {
        "datasetDir": str(dataset_dir),
        "outputDir": str(output_dir),
        "peopleFilter": args.people,
        "personCount": len(person_summaries),
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
    label_allowlist = parse_label_allowlist(args.labels)
    people_allowlist = parse_people_allowlist(args.people)

    session_dirs = list_session_dirs(dataset_dir)
    if not session_dirs:
        raise SystemExit(f"No Quest sessions found in {dataset_dir}")

    grouped_sessions = group_sessions_by_person(dataset_dir, session_dirs, people_allowlist)
    if not grouped_sessions:
        raise SystemExit("No person folders matched the requested --people filter.")

    prepare_output_dir(output_dir, args.overwrite)

    person_summaries = []
    for person_name, person_sessions in grouped_sessions.items():
        person_summary = build_person_dataset(
            person_name=person_name,
            session_dirs=person_sessions,
            output_root=output_dir,
            args=args,
            label_allowlist=label_allowlist,
        )
        person_summaries.append(person_summary)

        print(f"Wrote {person_name}: {output_dir / person_name}")
        print(f"  Sessions: {person_summary['sessionCount']}")
        print(f"  Total samples: {person_summary['totalSamples']}")
        for label, count in person_summary["labels"].items():
            print(f"  {label}: {count}")

    master_summary_path = write_master_summary(output_dir, dataset_dir, args, person_summaries)
    print(f"Master summary: {master_summary_path}")


if __name__ == "__main__":
    main()
