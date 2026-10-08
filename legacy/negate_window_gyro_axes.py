#!/usr/bin/env python3
"""Negate gyro axes in a by-person JSONL window dataset.

The default mode creates a copied dataset, preserving the original windows.
It negates x/y/z for both ``gyroscope`` and ``rotationRate`` when present.
"""

import argparse
import json
import shutil
from pathlib import Path


AXES = ("x", "y", "z")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Copy a prompt-window dataset and negate x/y/z in its gyro vector fields."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="prompt_1000ms_50hz_windows_by_person",
        help="Source root laid out as input/<person>/<label>/*.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default="prompt_1000ms_50hz_windows_by_person_gyro_negated",
        help="Copied dataset root. Ignored by --in-place.",
    )
    parser.add_argument(
        "--fields",
        default="gyroscope,rotationRate",
        help="Comma-separated gyro fields to transform. Default: gyroscope,rotationRate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing --output-dir before copying.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Modify --input-dir directly instead of creating --output-dir.",
    )
    parser.add_argument(
        "--confirm-in-place",
        action="store_true",
        help="Required together with --in-place because the source files will be overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect the dataset and report counts without copying or modifying files.",
    )
    args = parser.parse_args()

    args.fields = tuple(field.strip() for field in args.fields.split(",") if field.strip())
    if not args.fields:
        raise SystemExit("--fields must contain at least one field name.")
    if len(set(args.fields)) != len(args.fields):
        raise SystemExit(f"--fields contains duplicates: {args.fields}")
    if args.in_place and not args.confirm_in_place:
        raise SystemExit("--in-place requires --confirm-in-place.")
    if args.in_place and args.overwrite:
        raise SystemExit("--overwrite only applies when creating --output-dir.")
    return args


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    return rows


def negate_gyro_rows(rows, fields):
    changed_values = 0
    rows_with_field = {field: 0 for field in fields}

    for row in rows:
        for field in fields:
            vector = row.get(field)
            if vector is None:
                continue
            if not isinstance(vector, dict):
                raise ValueError(f"{field} is not an object")
            rows_with_field[field] += 1
            for axis in AXES:
                value = vector.get(axis)
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"Missing numeric value for {field}.{axis}")
                vector[axis] = -float(value)
                changed_values += 1
    return changed_values, rows_with_field


def write_jsonl_atomic(path, rows):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary_path.replace(path)


def prepare_output(input_dir, output_dir, overwrite):
    if output_dir.resolve() == input_dir.resolve():
        raise SystemExit("--output-dir must differ from --input-dir unless --in-place is used.")
    if output_dir.exists():
        if not overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    return output_dir


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    target_dir = input_dir if args.in_place else output_dir
    if not args.dry_run and not args.in_place:
        target_dir = prepare_output(input_dir, output_dir, args.overwrite)

    source_paths = sorted(input_dir.rglob("*.jsonl"))
    if not source_paths:
        raise SystemExit(f"No JSONL windows found in {input_dir}")

    summary = {
        "operation": "negate_gyro_axes",
        "inputDir": str(input_dir),
        "outputDir": str(target_dir),
        "inPlace": bool(args.in_place),
        "fields": list(args.fields),
        "axes": list(AXES),
        "files": 0,
        "rows": 0,
        "changedValues": 0,
        "rowsWithField": {field: 0 for field in args.fields},
        "errors": [],
    }

    for file_index, source_path in enumerate(source_paths, start=1):
        relative_path = source_path.relative_to(input_dir)
        target_path = source_path if args.in_place else target_dir / relative_path
        try:
            rows = load_jsonl(source_path)
            changed_values, rows_with_field = negate_gyro_rows(rows, args.fields)
        except ValueError as exc:
            summary["errors"].append({"path": str(source_path), "error": str(exc)})
            continue

        summary["files"] += 1
        summary["rows"] += len(rows)
        summary["changedValues"] += changed_values
        for field, count in rows_with_field.items():
            summary["rowsWithField"][field] += count

        if not args.dry_run:
            write_jsonl_atomic(target_path, rows)
        if file_index % 1000 == 0 or file_index == len(source_paths):
            print(f"processed={file_index}/{len(source_paths)}", flush=True)

    if summary["errors"]:
        error_path = target_dir / "gyro_axis_transform_errors.json"
        if not args.dry_run:
            error_path.write_text(json.dumps(summary["errors"], indent=2), encoding="utf-8")
        print(f"Skipped malformed files: {len(summary['errors'])}", flush=True)

    if not args.dry_run:
        (target_dir / "gyro_axis_transform_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
