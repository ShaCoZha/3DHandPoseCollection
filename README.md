# 3DHandPoseCollection

Four-camera RGB capture controlled by an iPhone, with Apple Watch IMU recording, WiLoR hand pose estimation, Anipose triangulation, per-joint reliability weighting, and synchronized replay with EIT measurement frames.

The local project directory is `HandPoseCollection`; the GitHub repository is named `3DHandPoseCollection`.

## What is included

- iPhone prepare / commit / stop protocol and a receiver that supports either multicamera or Quest pose capture.
- Four FLIR GigE cameras through the Spinnaker/PySpin SDK, with per-frame hardware timestamps and camera-to-PC clock mapping.
- ChArUco recording, corner detection, camera calibration, and held-out validation reports.
- WiLoR 2D predictions, Anipose triangulation, bone-length and actual-time temporal constraints.
- Crop-perturbation stability, predicted-mesh self-occlusion hints, cross-view consistency, and weighted 2D reprojection fitting.
- English interactive RGB / 2D / 3D / Watch IMU replay, EIT measurement heatmaps, and MP4 export.
- Tests for the capture protocol, timestamp matching, constraints, reliability geometry, and EIT handling.

The iPhone, Watch and Quest applications, EIT acquisition firmware, proprietary SDK, model weights, and virtual environments are external dependencies. POEM experiments remain in the separate `HandPoseComparison` project; this repository contains the WiLoR collection pipeline.

## Repository layout

```text
pc_receiver/       Capture, calibration, reconstruction and visualization modules
scripts/run.py     Launcher that selects a configured Python environment
configs/           Portable configuration examples; local copies are Git-ignored
tests/             Unit and protocol tests
docs/              Synchronization protocol and weighted reconstruction explanation
legacy/            Quest dataset tools and the older WiLoR image-folder adapter
data/              Local recordings/calibrations or links to them (Git-ignored)
logs/              Local logs and migration reports (Git-ignored)
```

Modules and HTML/JS templates stay together in `pc_receiver/` so subprocess launches and imports resolve consistently.

## Environment setup

Use separate Python environments for the SDK and inference stacks. The existing laptop environments can be reused; do not move virtual environments, since their scripts may contain absolute paths.

| Role | Required components |
| --- | --- |
| Receiver / launcher | Python 3.10+, `websockets` for WebSocket clients |
| Camera worker | Compatible PySpin/Spinnaker installation, NumPy, OpenCV, `toml` |
| WiLoR | CUDA-capable PyTorch, `wilor_mini`, detector and MANO assets |
| Triangulation | `aniposelib` (the existing pipeline uses 0.7.2), NumPy |
| Calibration / weighted fit / visualization | NumPy, SciPy, OpenCV with `aruco`, `toml`; WiLoR dependencies for reliability inference |
| Video export | System `ffmpeg` and `ffprobe` |

These are dependency roles, not a universal environment lock file. PySpin must match its SDK and Python version; WiLoR/MANO assets must be obtained under their respective terms. Upstream implementations and weights are not vendored here.

On a new checkout:

```bash
cp configs/multicamera.example.json configs/multicamera.local.json
cp configs/environments.example.json configs/environments.local.json
mkdir -p data/dataset_multicamera data/calibration_captures data/calibration logs
```

Edit the local JSON files to supply executable paths, camera serial numbers, and a calibration for the **current camera geometry and image settings**. Relative paths resolve from the configuration file. The board implementation is 10 × 8 squares, 22 mm squares, 16 mm markers, `DICT_4X4_50`.

On the original laptop, ignored local configs and links already reference the existing environments and data. The migration did not activate an October 7 candidate calibration. The configured October 2 calibration is historical and must not be assumed valid for a moved rig. See [local deployment status](docs/LOCAL_DEPLOYMENT.md).

```bash
python scripts/run.py check
python scripts/run.py --dry-run collect
```

`check` checks configured paths only. It does not open cameras or certify calibration, clock synchronization, or model availability.

## Start data collection

Run from this repository root:

```bash
python scripts/run.py collect
```

The local collection config sets `fps: 30` and `exposureTimeUs: 2000` (2 ms). The camera worker disables automatic exposure, selects Timed mode, checks the camera range, and verifies the readback before acquisition. Per-frame chunk exposure values remain recorded in the frame JSONL files. Gain is unchanged; provide enough illumination for the shorter exposure. Omit the field to inherit camera exposure settings.

The receiver listens on port **8765** and writes to `data/dataset_multicamera/<sessionId>/`. Start and stop synchronized acquisition from the iPhone client. The default local configuration enables the weighted hand pose pipeline after capture.

To use another port:

```bash
python scripts/run.py collect --port 8769
```

An existing receiver may already own port 8765. Stop the instance you intend to replace before starting another one; this launcher does not terminate other services.

## Record and solve a calibration

Keep cameras and lenses fixed. Move the board through the hand capture volume, exposing its printed face to multiple cameras at once, and hold each pose for 3–5 seconds. Use a rigid, flat board.

```bash
python scripts/run.py record-calibration --seconds 120 --confirm-board
python scripts/run.py detect-board data/calibration_captures/<take>
python scripts/run.py solve-calibration data/calibration_captures/<take>
```

Use `--confirm-board` only after checking the physical board dimensions. Calibration recording uses the PC clock and is explicitly marked as having no iPhone clock synchronization.

The solver writes `calibration/calibration_candidate.toml` and `calibration/calibration_report.json`. It does **not** deploy candidates automatically. It supports `--refine-intrinsics` and `--pairwise-motion-budget` for explicit experiments; the latter selects close pairwise exposures using measured board motion and records its assumptions. Inspect held-out errors, per-camera coverage, and rig stability before updating a local configuration. New camera positions cannot automatically repair recordings from different positions.

## Process an existing recording

```bash
python scripts/run.py process data/dataset_multicamera/<sessionId>
```

This full pipeline aligns camera timestamps, runs WiLoR, triangulates, applies configured constraints, and optionally runs weighted fitting. It overwrites derived outputs; back them up before comparing parameter changes.

To reuse existing 2D predictions after changing a session's calibration:

```bash
python scripts/run.py process data/dataset_multicamera/<sessionId> --stage triangulate
python scripts/run.py process data/dataset_multicamera/<sessionId> --stage regularize
python scripts/run.py weighted data/dataset_multicamera/<sessionId>
```

On a GPU with sufficient memory, `weighted ... --inference-workers 2` (up to 4)
assigns disjoint inference chunks to separate processes. All frames and the same
model/weighting logic are retained; existing compatible chunks are reused.

The session's `multicamera/config.json` controls its processing environments. Old sessions retain their original settings. Reliability caches include input hashes; changing calibration or frame pairing requires a fresh cache/output directory. Do not overwrite a candidate's provenance or silently reuse an incompatible cache.

## Visualize RGB, skeletons, IMU and EIT

After weighted processing completes:

```bash
python scripts/run.py viz data/dataset_multicamera/<sessionId> --method weighted
python scripts/run.py eit data/dataset_multicamera/<sessionId> --attach weighted
python scripts/run.py export data/dataset_multicamera/<sessionId>
python scripts/run.py serve data/dataset_multicamera/<sessionId>/visualization --port 8768
```

Open `http://localhost:8768/weighted/viewer.html`. On a headless remote machine, forward port 8768 through SSH or VS Code. The exported video is `visualization/weighted/rgb_pose_imu_eit.mp4`.

For the baseline output, use `--method wilor` with `viz` and `export`, and `--attach wilor` with `eit`. EIT preparation requires `eit_frames.jsonl`; the export also requires aligned IMU data. Use `--diagnostic` on video export for an unvalidated calibration experiment.

## Interpretation and limitations

- Matching Unix timestamps does not establish clock agreement or simultaneous exposures. Camera and device clock mappings are recorded separately.
- EIT clock alignment defaults to **unverified**. A user-adjusted time offset is not clock calibration. Missing EIT intervals remain blank.
- The EIT heatmap is complex-measurement magnitude, not a reconstructed conductivity image.
- Per-joint weights are heuristic reliability estimates, not trained and calibrated confidence probabilities.
- Bone and temporal constraints can reduce jitter while increasing disagreement with individual 2D predictions. Neither coverage nor low reprojection error proves 3D accuracy.
- Latest candidate experiments may be viewable while formal calibration validation remains failed. Preserve that distinction in reports and exports.

## Tests

```bash
python scripts/run.py test
```

These tests do not record from physical cameras. They do not substitute for a live capture check or calibration validation.

## Documentation and provenance

- [Mechanics: detection, weighting and 3D consistency](MECHANICS.md)
- [Weighted hand pose pipeline](docs/WEIGHTED_HANDPOSE_PIPELINE.md)
- [Synchronization protocol and historical operating notes](docs/SYNCHRONIZED_CAPTURE.md)
- [Local deployment and migration notes](docs/LOCAL_DEPLOYMENT.md)

This repository was organized from the existing `pc_receiver` workspace. The original workspace remains available for running services and historical data. No raw participant recordings, model weights, local credentials, or virtual environments are included in Git.
