# """
# Extract 21 2D hand keypoints per frame with WiLoR-mini and write DeepLabCut-style
# HDF5 files that Anipose reads as pose-2d.

# Input layout:
#     <img_root>/
#         take0-cam01/   frame_000001.jpg ...
#         take0-cam02/   ...
#         take0-cam03/   ...
#         take0-cam04/   ...

# Output: one <folder_name>.h5 per subfolder, in <out_dir>.

# Guarantees one row per frame in sorted order; frames with no detection get NaN
# coordinates and likelihood 0.0, so all cameras stay aligned for triangulation.

# Usage:
#     python wilor_to_pose2d.py --img_root frames --out_dir pose-2d
# """

# import argparse
# import re
# from pathlib import Path

# import cv2
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm

# from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
#     WiLorHandPose3dEstimationPipeline,
# )

# # Order matches the MANO / WiLoR keypoint convention:
# # 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
# # Names match the [labeling] scheme in config.toml.
# HAND_PARTS = [
#     'base',
#     'thumb_CMC', 'thumb_MCP', 'thumb_DIP', 'thumb_tip',
#     'index_MCP', 'index_PIP', 'index_DIP', 'index_tip',
#     'middle_MCP', 'middle_PIP', 'middle_DIP', 'middle_tip',
#     'ring_MCP', 'ring_PIP', 'ring_DIP', 'ring_tip',
#     'pinky_MCP', 'pinky_PIP', 'pinky_DIP', 'pinky_tip',
# ]


# def natural_key(path):
#     """frame_2.jpg sorts before frame_10.jpg even without zero padding."""
#     return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


# def side_name(is_right):
#     if is_right is None:
#         return 'none'
#     return 'right' if float(is_right) > 0.5 else 'left'


# def save_pose2d(kp, outpath, scorer='wilor'):
#     """kp: (n_frames, 21, 3) array of x, y, likelihood."""
#     assert kp.ndim == 3 and kp.shape[1:] == (21, 3), kp.shape
#     cols = pd.MultiIndex.from_product(
#         [[scorer], HAND_PARTS, ['x', 'y', 'likelihood']],
#         names=['scorer', 'bodyparts', 'coords'])
#     df = pd.DataFrame(kp.reshape(len(kp), -1), columns=cols)
#     Path(outpath).parent.mkdir(parents=True, exist_ok=True)
#     df.to_hdf(outpath, key='df_with_missing', mode='w')
#     return df


# def process_folder(pipe, img_folder, file_types):
#     """
#     Run WiLoR over one camera folder.
#     Returns (kp array, per-frame sides, events dict) — nothing is printed here
#     except the progress bar; callers report the collected events.
#     """
#     img_paths = sorted(
#         (p for pat in file_types for p in Path(img_folder).glob(pat)),
#         key=natural_key,
#     )
#     n = len(img_paths)

#     kp_all = np.full((n, 21, 3), np.nan, dtype=np.float32)
#     kp_all[:, :, 2] = 0.0            # likelihood 0 = not detected
#     sides = []                        # per-frame handedness, None if no detection

#     events = {'unreadable': [], 'multi_hand': [], 'side_flips': []}

#     prev_side = None
#     for i, img_path in enumerate(tqdm(img_paths, desc=f'  {img_folder.name}',
#                                       unit='img', leave=False)):
#         img = cv2.imread(str(img_path))
#         if img is None:
#             events['unreadable'].append((i, img_path.name))
#             sides.append(None)
#             continue

#         outputs = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#         if len(outputs) == 0:
#             sides.append(None)
#             continue

#         # One hand expected. If several, take the largest bbox.
#         if len(outputs) > 1:
#             areas = []
#             for o in outputs:
#                 x1, y1, x2, y2 = o['hand_bbox']
#                 areas.append((x2 - x1) * (y2 - y1))
#             out = outputs[int(np.argmax(areas))]
#             events['multi_hand'].append((i, img_path.name, len(outputs)))
#         else:
#             out = outputs[0]

#         kp2d = np.asarray(out['wilor_preds']['pred_keypoints_2d'][0], dtype=np.float32)
#         kp_all[i, :, :2] = kp2d
#         kp_all[i, :, 2] = 1.0

#         this_side = side_name(out.get('is_right'))
#         sides.append(this_side)

#         if prev_side is not None and this_side != prev_side:
#             events['side_flips'].append((i, img_path.name, prev_side, this_side))
#         prev_side = this_side

#     return kp_all, sides, events


# def report_folder(name, kp, sides, events, outpath, max_listed=10):
#     """Print one compact block summarising a finished folder."""
#     n = len(kp)
#     found = int((kp[:, 0, 2] > 0).sum())
#     pct = 100.0 * found / max(n, 1)
#     print(f'{name}: {n} frames, {found} detected ({pct:.1f}%) -> {outpath}')

#     seen = [s for s in sides if s is not None]
#     if not seen:
#         print('  handedness: no detections')
#         return 'none'

#     vals, counts = np.unique(seen, return_counts=True)
#     dominant = vals[int(np.argmax(counts))]
#     breakdown = ', '.join(f'{v}={c}' for v, c in zip(vals, counts))
#     flips = events['side_flips']

#     if len(vals) > 1:
#         print(f'  handedness: MIXED ({breakdown}), {len(flips)} flip(s)')
#         for i, fname, a, b in flips[:max_listed]:
#             print(f'    frame {i} ({fname}): {a} -> {b}')
#         if len(flips) > max_listed:
#             print(f'    ... and {len(flips) - max_listed} more')
#     else:
#         print(f'  handedness: {dominant} (consistent)')

#     if events['multi_hand']:
#         print(f'  {len(events["multi_hand"])} frame(s) with >1 hand, took largest '
#               f'(first: frame {events["multi_hand"][0][0]})')
#     if events['unreadable']:
#         print(f'  {len(events["unreadable"])} unreadable file(s) '
#               f'(first: {events["unreadable"][0][1]})')

#     return dominant


# def main():
#     ap = argparse.ArgumentParser(
#         description='WiLoR hand keypoints -> Anipose pose-2d h5 files')
#     ap.add_argument('--img_root', required=True,
#                     help='Folder containing one subfolder of frames per camera')
#     ap.add_argument('--out_dir', required=True,
#                     help='Where to write the .h5 files (Anipose pose-2d folder)')
#     ap.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
#     ap.add_argument('--scorer', default='wilor',
#                     help="'scorer' level name inside the h5")
#     args = ap.parse_args()

#     folders = sorted(p for p in Path(args.img_root).iterdir() if p.is_dir())
#     if not folders:
#         raise SystemExit(f'No subfolders found in {args.img_root}')
#     print(f'{len(folders)} camera folders: {[f.name for f in folders]}\n')

#     # Load the pipeline ONCE — it is ~2.4 GB and slow to initialise.
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     dtype = torch.float16 if device.type == 'cuda' else torch.float32
#     print(f'Loading WiLoR pipeline on {device}...')
#     pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

#     summary = {}
#     dominant_sides = {}

#     for folder in folders:
#         kp, sides, events = process_folder(pipe, folder, args.file_type)

#         outpath = Path(args.out_dir) / f'{folder.name}.h5'
#         save_pose2d(kp, outpath, scorer=args.scorer)

#         print()
#         dominant_sides[folder.name] = report_folder(
#             folder.name, kp, sides, events, outpath)
#         summary[folder.name] = (len(kp), int((kp[:, 0, 2] > 0).sum()))

#     print('\n' + '=' * 60)
#     print('Summary')
#     counts = [f for f, _ in summary.values()]
#     for name, (n_frames, found) in summary.items():
#         print(f'  {name}: {n_frames} frames, {found} detected, '
#               f'hand={dominant_sides[name]}')

#     if len(set(counts)) > 1:
#         print('\n  WARNING: frame counts differ across cameras. Anipose needs '
#               'identical row counts per camera for triangulation.')

#     distinct = {s for s in dominant_sides.values() if s != 'none'}
#     if len(distinct) > 1:
#         print('\n  WARNING: cameras disagree on handedness: '
#               + ', '.join(f'{k}={v}' for k, v in dominant_sides.items()))
#         print('  Triangulation assumes all cameras see the same hand.')


# if __name__ == '__main__':
#     main()
"""
Extract 21 2D hand keypoints per frame with WiLoR-mini and write DeepLabCut-style
HDF5 files that Anipose reads as pose-2d.

Input layout:
    <img_root>/
        take0-cam01/   frame_000001.jpg ...
        take0-cam02/   ...
        take0-cam03/   ...
        take0-cam04/   ...

Output: one <folder_name>.h5 per subfolder, in <out_dir>, PLUS:
    - detection_log.csv: per-frame, per-camera hand-detected yes/no
    - keypoint_visibility_log.csv: per-frame, per-keypoint count of cameras
      whose predicted location for that keypoint fell inside WiLoR's own
      hand bounding box (a real "was this landmark plausibly tracked" signal,
      since WiLoR gives no native per-keypoint confidence).

Usage:
    python wilor_to_pose2d.py --img_root frames --out_dir pose-2d
"""

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
    WiLorHandPose3dEstimationPipeline,
)

HAND_PARTS = [
    'base',
    'thumb_CMC', 'thumb_MCP', 'thumb_DIP', 'thumb_tip',
    'index_MCP', 'index_PIP', 'index_DIP', 'index_tip',
    'middle_MCP', 'middle_PIP', 'middle_DIP', 'middle_tip',
    'ring_MCP', 'ring_PIP', 'ring_DIP', 'ring_tip',
    'pinky_MCP', 'pinky_PIP', 'pinky_DIP', 'pinky_tip',
]


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', path.name)]


def side_name(is_right):
    if is_right is None:
        return 'none'
    return 'right' if float(is_right) > 0.5 else 'left'


def save_pose2d(kp, outpath, scorer='wilor'):
    assert kp.ndim == 3 and kp.shape[1:] == (21, 3), kp.shape
    cols = pd.MultiIndex.from_product(
        [[scorer], HAND_PARTS, ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])
    df = pd.DataFrame(kp.reshape(len(kp), -1), columns=cols)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(outpath, key='df_with_missing', mode='w')
    return df


def keypoint_in_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_in_image(bbox, img_shape):
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    return x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h


def detect_one_image(pipe, img_path, events, frame_idx):
    """Run WiLoR on a single image. Returns (kp21x3, side_str_or_None, hand_bbox_or_None)."""
    kp = np.full((21, 3), np.nan, dtype=np.float32)
    kp[:, 2] = 0.0

    img = cv2.imread(str(img_path))
    if img is None:
        events['unreadable'].append((frame_idx, img_path.name))
        return kp, None, None

    outputs = pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(outputs) == 0:
        return kp, None, None

    if len(outputs) > 1:
        areas = []
        for o in outputs:
            x1, y1, x2, y2 = o['hand_bbox']
            areas.append((x2 - x1) * (y2 - y1))
        out = outputs[int(np.argmax(areas))]
        events['multi_hand'].append((frame_idx, img_path.name, len(outputs)))
    else:
        out = outputs[0]

    kp2d = np.asarray(out['wilor_preds']['pred_keypoints_2d'][0], dtype=np.float32)
    kp[:, :2] = kp2d
    kp[:, 2] = 1.0

    hand_bbox = out['hand_bbox']   # (x1, y1, x2, y2)

    return kp, side_name(out.get('is_right')), hand_bbox


def process_synchronized_frames(pipe, cam_names, img_lists, n_frames):
    """
    Run WiLoR across all cameras, frame-by-frame (synchronized).
    Returns:
        kp_per_cam: dict[cam_name] -> (n_frames, 21, 3) array
        detection_log: list of dicts, one per frame, with per-cam detection flags
        keypoint_log: list of dicts, one per frame, with per-keypoint camera counts
        events_per_cam: dict[cam_name] -> events dict
    """
    kp_per_cam = {c: np.full((n_frames, 21, 3), np.nan, dtype=np.float32) for c in cam_names}
    for c in cam_names:
        kp_per_cam[c][:, :, 2] = 0.0

    events_per_cam = {
        c: {'unreadable': [], 'multi_hand': [], 'side_flips': [], 'bbox_clipped': []}
        for c in cam_names
    }
    prev_side = {c: None for c in cam_names}
    detection_log = []
    keypoint_log = []

    for i in tqdm(range(n_frames), desc='frames', unit='frame'):
        row = {'frame': i}
        n_detected = 0
        kp_visible_count = np.zeros(21, dtype=int)

        for cam_idx, cam in enumerate(cam_names):
            img_path = img_lists[cam][i]
            img = cv2.imread(str(img_path))
            img_shape = img.shape if img is not None else None

            kp, side, hand_bbox = detect_one_image(pipe, img_path, events_per_cam[cam], i)

            detected = kp[0, 2] > 0
            row[cam] = 'yes' if detected else 'no'
            if detected:
                n_detected += 1

                if hand_bbox is not None and img_shape is not None:
                    if not bbox_in_image(hand_bbox, img_shape):
                        events_per_cam[cam]['bbox_clipped'].append((i, img_path.name))

                    for k in range(21):
                        x, y = kp[k, 0], kp[k, 1]
                        if np.isnan(x) or not keypoint_in_bbox(x, y, hand_bbox):
                            kp[k, 2] = 0.0
                        else:
                            kp_visible_count[k] += 1

            kp_per_cam[cam][i] = kp

            if side is not None:
                if prev_side[cam] is not None and side != prev_side[cam]:
                    events_per_cam[cam]['side_flips'].append(
                        (i, img_path.name, prev_side[cam], side))
                prev_side[cam] = side

        row['n_cams_detected'] = n_detected
        detection_log.append(row)

        kp_row = {'frame': i}
        for part_idx, part_name in enumerate(HAND_PARTS):
            kp_row[part_name] = int(kp_visible_count[part_idx])
        keypoint_log.append(kp_row)

    return kp_per_cam, detection_log, keypoint_log, events_per_cam


def report_folder(name, kp, events, max_listed=10):
    n = len(kp)
    found = int((kp[:, 0, 2] > 0).sum())
    pct = 100.0 * found / max(n, 1)
    print(f'{name}: {n} frames, {found} detected ({pct:.1f}%)')

    flips = events['side_flips']
    if flips:
        print(f'  {len(flips)} handedness flip(s)')
        for i, fname, a, b in flips[:max_listed]:
            print(f'    frame {i} ({fname}): {a} -> {b}')
        if len(flips) > max_listed:
            print(f'    ... and {len(flips) - max_listed} more')

    if events['multi_hand']:
        print(f'  {len(events["multi_hand"])} frame(s) with >1 hand, took largest '
              f'(first: frame {events["multi_hand"][0][0]})')
    if events['unreadable']:
        print(f'  {len(events["unreadable"])} unreadable file(s) '
              f'(first: {events["unreadable"][0][1]})')
    if events['bbox_clipped']:
        print(f'  {len(events["bbox_clipped"])} frame(s) with hand bbox clipped at image edge '
              f'(first: frame {events["bbox_clipped"][0][0]})')


def write_detection_log(detection_log, cam_names, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['frame'] + cam_names + ['n_cams_detected']
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detection_log)
    print(f'Per-frame cross-camera detection log -> {outpath}')


def write_keypoint_log(keypoint_log, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['frame'] + HAND_PARTS
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(keypoint_log)
    print(f'Per-keypoint cross-camera visibility log -> {outpath}')


def summarize_coverage(detection_log, cam_names):
    n_frames = len(detection_log)
    counts = {c: sum(1 for r in detection_log if r[c] == 'yes') for c in cam_names}
    coverage_dist = {}
    for r in detection_log:
        k = r['n_cams_detected']
        coverage_dist[k] = coverage_dist.get(k, 0) + 1

    print('\nPer-camera detection rate:')
    for c in cam_names:
        pct = 100.0 * counts[c] / max(n_frames, 1)
        print(f'  {c}: {counts[c]}/{n_frames} ({pct:.1f}%)')

    print('\nFrames by # cameras with a detection:')
    for k in sorted(coverage_dist):
        print(f'  {k} camera(s): {coverage_dist[k]} frames')

    zero_cam_frames = coverage_dist.get(0, 0)
    if zero_cam_frames:
        print(f'\n  NOTE: {zero_cam_frames} frame(s) had NO detections in any camera.')


def main():
    ap = argparse.ArgumentParser(
        description='WiLoR hand keypoints -> Anipose pose-2d h5 files, synchronized across cameras')
    ap.add_argument('--img_root', required=True,
                    help='Folder containing one subfolder of frames per camera')
    ap.add_argument('--out_dir', required=True,
                    help='Where to write the .h5 files (Anipose pose-2d folder)')
    ap.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
    ap.add_argument('--scorer', default='wilor',
                    help="'scorer' level name inside the h5")
    args = ap.parse_args()

    folders = sorted(p for p in Path(args.img_root).iterdir() if p.is_dir())
    if not folders:
        raise SystemExit(f'No subfolders found in {args.img_root}')

    cam_names = [f.name for f in folders]
    img_lists = {
        f.name: sorted(
            (p for pat in args.file_type for p in f.glob(pat)),
            key=natural_key,
        )
        for f in folders
    }

    print(f'{len(folders)} camera folders: {cam_names}')
    frame_counts = {c: len(imgs) for c, imgs in img_lists.items()}
    print(f'Frame counts: {frame_counts}')

    n_frames = min(frame_counts.values())
    if len(set(frame_counts.values())) > 1:
        print(f'  WARNING: frame counts differ across cameras. Using min = {n_frames}. '
              f'Confirm cameras are actually synchronized frame-for-frame.')
    print()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    print(f'Loading WiLoR pipeline on {device}...')
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

    kp_per_cam, detection_log, keypoint_log, events_per_cam = process_synchronized_frames(
        pipe, cam_names, img_lists, n_frames)

    print()
    for cam in cam_names:
        outpath = Path(args.out_dir) / f'{cam}.h5'
        save_pose2d(kp_per_cam[cam], outpath, scorer=args.scorer)
        report_folder(cam, kp_per_cam[cam], events_per_cam[cam])
        print()

    log_path = Path(args.out_dir) / 'detection_log.csv'
    write_detection_log(detection_log, cam_names, log_path)
    summarize_coverage(detection_log, cam_names)

    kp_log_path = Path(args.out_dir) / 'keypoint_visibility_log.csv'
    write_keypoint_log(keypoint_log, kp_log_path)


if __name__ == "__main__":
    main()
