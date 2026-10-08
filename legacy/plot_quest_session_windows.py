import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


AXES = ("x", "y", "z")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one Quest session and highlight fixed IMU windows before Quest events."
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Dataset root containing per-session folders.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session id or path to one session folder. If omitted, an interactive picker is shown.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to <session>/plots/quest_windows_plot.png.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    parser.add_argument("--pre-ms", type=int, default=1000, help="Milliseconds before each Quest event to highlight.")
    parser.add_argument("--post-ms", type=int, default=0, help="Milliseconds after each Quest event to highlight.")
    parser.add_argument(
        "--labels",
        default=None,
        help="Optional comma-separated label allowlist, e.g. swipe_left,thumb_tap.",
    )
    parser.add_argument(
        "--include-end-events",
        action="store_true",
        help="Include *_end/release events. Default skips them.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Maximum Quest events to draw after filtering. 0 means all.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Maximum IMU points plotted per trace after downsampling.",
    )
    parser.add_argument(
        "--acc-field",
        default="accelerometer",
        choices=("accelerometer", "userAcceleration"),
        help="Acceleration field to plot.",
    )
    parser.add_argument(
        "--gyro-field",
        default="gyroscope",
        choices=("gyroscope", "rotationRate"),
        help="Gyroscope field to plot.",
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


def load_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def numeric_unix_time(row):
    value = row.get("unixTimeMs")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def list_session_dirs(dataset_dir: Path):
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    sessions = []
    for path in sorted(dataset_dir.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_dir() and (path / "imudata.jsonl").exists():
            sessions.append(path)
    return sessions


def count_jsonl_lines(path: Path):
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def choose_session(dataset_dir: Path):
    sessions = list_session_dirs(dataset_dir)
    if not sessions:
        raise SystemExit(f"No session folders with imudata.jsonl found in {dataset_dir}")

    if len(sessions) == 1:
        print(f"Using only session: {sessions[0].name}")
        return sessions[0]

    print("Available sessions:")
    for index, session_dir in enumerate(sessions, start=1):
        timestamp = load_json(session_dir / "timestamp.json")
        gesture = timestamp.get("gestureType", "unknown")
        imu_count = count_jsonl_lines(session_dir / "imudata.jsonl")
        quest_count = count_jsonl_lines(session_dir / "questlog.jsonl")
        print(f"{index:2d}. {session_dir.name} | gesture={gesture} | imu={imu_count} | quest={quest_count}")

    raw = input("Select session number: ").strip()
    try:
        selected_index = int(raw)
    except ValueError as exc:
        raise SystemExit(f"Bad session selection: {raw}") from exc

    if selected_index < 1 or selected_index > len(sessions):
        raise SystemExit(f"Session selection out of range: {selected_index}")
    return sessions[selected_index - 1]


def resolve_session_dir(dataset_dir: Path, session_arg):
    if session_arg is None:
        return choose_session(dataset_dir)

    session_path = Path(session_arg)
    if session_path.exists():
        return session_path

    session_path = dataset_dir / session_arg
    if session_path.exists():
        return session_path

    raise SystemExit(f"Session not found as path or id: {session_arg}")


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


def parse_label_allowlist(raw_labels):
    if not raw_labels:
        return None
    return {normalize_label(item) for item in raw_labels.split(",") if item.strip()}


def vector_value(row, field, axis):
    value = row.get(field)
    if isinstance(value, dict):
        raw = value.get(axis)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    return np.nan


def build_imu_arrays(rows, acc_field, gyro_field):
    valid_rows = [row for row in rows if numeric_unix_time(row) is not None]
    valid_rows = sorted(valid_rows, key=lambda row: numeric_unix_time(row))
    if not valid_rows:
        raise SystemExit("No IMU rows with unixTimeMs found.")

    timestamps = np.asarray([numeric_unix_time(row) for row in valid_rows], dtype=np.float64)
    arrays = {
        acc_field: np.asarray(
            [[vector_value(row, acc_field, axis) for axis in AXES] for row in valid_rows],
            dtype=np.float64,
        ),
        gyro_field: np.asarray(
            [[vector_value(row, gyro_field, axis) for axis in AXES] for row in valid_rows],
            dtype=np.float64,
        ),
    }
    return timestamps, arrays


def downsample_for_plot(timestamps, arrays, max_points):
    if max_points <= 0 or len(timestamps) <= max_points:
        return timestamps, arrays

    stride = max(1, math.ceil(len(timestamps) / max_points))
    return timestamps[::stride], {field: values[::stride] for field, values in arrays.items()}


def pick_origin_ts(timestamps, timestamp):
    started_at = timestamp.get("startedAtUnixMs")
    if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
        return float(started_at)
    return float(timestamps[0])


def relative_seconds(unix_ms, origin_ts):
    return (float(unix_ms) - float(origin_ts)) / 1000.0


def select_quest_events(quest_rows, label_allowlist, include_end_events, max_events):
    events = []
    for row in quest_rows:
        event_ts = numeric_unix_time(row)
        if event_ts is None:
            continue

        label = quest_label(row)
        if not include_end_events and is_end_event(row, label):
            continue
        if label_allowlist is not None and label not in label_allowlist:
            continue
        events.append({"row": row, "label": label, "unixTimeMs": event_ts})

    events.sort(key=lambda event: event["unixTimeMs"])
    if max_events > 0:
        events = events[:max_events]
    return events


def draw_vector_axis(axis, time_s, values, title, ylabel):
    colors = ("tab:blue", "tab:orange", "tab:green")
    for index, name in enumerate(AXES):
        axis.plot(time_s, values[:, index], label=name, linewidth=0.85, color=colors[index])
    axis.set_title(title, loc="left", fontsize=10)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(loc="upper right", ncol=3, fontsize=8)


def draw_windows_and_points(axes, events, origin_ts, pre_ms, post_ms):
    if not events:
        return Counter(), {}

    labels = sorted({event["label"] for event in events})
    cmap = plt.get_cmap("tab20")
    label_to_color = {label: cmap(index % cmap.N) for index, label in enumerate(labels)}
    counts = Counter(event["label"] for event in events)

    for event in events:
        event_x = relative_seconds(event["unixTimeMs"], origin_ts)
        start_x = relative_seconds(event["unixTimeMs"] - pre_ms, origin_ts)
        end_x = relative_seconds(event["unixTimeMs"] + post_ms, origin_ts)
        color = label_to_color[event["label"]]

        for axis in axes:
            axis.axvspan(start_x, end_x, color=color, alpha=0.13, linewidth=0)
            axis.axvline(event_x, color=color, alpha=0.78, linewidth=1.15)

    return counts, label_to_color


def draw_event_axis(axis, events, origin_ts, label_to_color, pre_ms, post_ms):
    if not events:
        axis.text(0.01, 0.5, "No Quest events", transform=axis.transAxes, va="center")
        axis.set_yticks([])
        return

    labels = sorted(label_to_color)
    label_to_y = {label: index for index, label in enumerate(labels)}
    for event in events:
        event_x = relative_seconds(event["unixTimeMs"], origin_ts)
        start_x = relative_seconds(event["unixTimeMs"] - pre_ms, origin_ts)
        end_x = relative_seconds(event["unixTimeMs"] + post_ms, origin_ts)
        y = label_to_y[event["label"]]
        color = label_to_color[event["label"]]
        axis.hlines(y, start_x, end_x, color=color, linewidth=5, alpha=0.25)
        axis.scatter([event_x], [y], color=[color], s=36, zorder=5, edgecolors="black", linewidths=0.4)

    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels)
    axis.set_ylim(-0.7, len(labels) - 0.3)
    axis.set_ylabel("Quest")
    axis.grid(axis="x", alpha=0.25, linewidth=0.5)


def title_for(session_dir, timestamp, events, pre_ms, post_ms, counts):
    session_id = timestamp.get("sessionId", session_dir.name)
    quest_summary = ", ".join(f"{label}:{count}" for label, count in sorted(counts.items()))
    if not quest_summary:
        quest_summary = "no Quest events"
    return (
        f"{session_id} | Quest point + window [{-pre_ms}ms, +{post_ms}ms] | events={len(events)}\n"
        f"{quest_summary}"
    )


def plot_session(session_dir, output_path, show, args):
    imu_rows = load_jsonl(session_dir / "imudata.jsonl")
    quest_rows = load_jsonl(session_dir / "questlog.jsonl")
    timestamp = load_json(session_dir / "timestamp.json")

    raw_timestamps, raw_arrays = build_imu_arrays(imu_rows, args.acc_field, args.gyro_field)
    timestamps, arrays = downsample_for_plot(raw_timestamps, raw_arrays, args.max_points)
    origin_ts = pick_origin_ts(raw_timestamps, timestamp)
    time_s = (timestamps - origin_ts) / 1000.0

    events = select_quest_events(
        quest_rows,
        label_allowlist=parse_label_allowlist(args.labels),
        include_end_events=args.include_end_events,
        max_events=args.max_events,
    )

    fig, axes = plt.subplots(3, 1, figsize=(17, 8.5), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    draw_vector_axis(
        axes[0],
        time_s,
        arrays[args.acc_field],
        title=f"{args.acc_field} (acc)",
        ylabel="m/s^2",
    )
    draw_vector_axis(
        axes[1],
        time_s,
        arrays[args.gyro_field],
        title=f"{args.gyro_field} (gyro)",
        ylabel="rad/s",
    )
    counts, label_to_color = draw_windows_and_points(axes[:2], events, origin_ts, args.pre_ms, args.post_ms)
    draw_event_axis(axes[2], events, origin_ts, label_to_color, args.pre_ms, args.post_ms)
    axes[2].set_xlabel("Seconds from recording start")

    legend_items = [
        Patch(facecolor="grey", alpha=0.18, label=f"highlighted window: -{args.pre_ms}ms to +{args.post_ms}ms"),
        Line2D([0], [0], color="black", linewidth=1.15, label="Quest event time"),
    ]
    axes[0].legend(handles=[*axes[0].get_legend_handles_labels()[0], *legend_items], loc="upper right", ncol=5, fontsize=8)

    fig.suptitle(title_for(session_dir, timestamp, events, args.pre_ms, args.post_ms, counts), fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    print(f"Wrote plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    args = parse_args()
    if args.pre_ms < 0 or args.post_ms < 0:
        raise SystemExit("--pre-ms and --post-ms must be >= 0")

    dataset_dir = Path(args.dataset_dir)
    session_dir = resolve_session_dir(dataset_dir, args.session)

    if not (session_dir / "imudata.jsonl").exists():
        raise SystemExit(f"Missing imudata.jsonl in {session_dir}")

    output_path = Path(args.output) if args.output else session_dir / "plots" / "quest_windows_plot.png"
    plot_session(session_dir, output_path, args.show, args)


if __name__ == "__main__":
    main()
