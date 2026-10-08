"""Fit rigid camera extrinsics to stationary ChArUco observations.

Intrinsics can be fixed or refined with soft priors. Time-block held-out observations are never included in
the optimizer. Cross-camera validation predicts each target view from a board
pose estimated in a different source view, rather than fitting the target view.
"""
import argparse
import bisect
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import toml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from align_multicamera import pair_frames
from calibrate_multicamera_recording import board, read_rows


OBJ = board().getChessboardCorners().astype(float)


def pose(row, K, D):
    obj, xy = OBJ[row['ids']], np.array(row['xy'], dtype=float)
    found, rs, ts, _ = cv2.solvePnPGeneric(obj, xy, K, D, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for r, t in zip(rs, ts):
        if t[2, 0] <= 0:
            continue
        error = np.linalg.norm(cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2) - xy, axis=1)
        candidates.append((float(np.mean(error)), r, t))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    # A near-fronto-parallel board can have ambiguous planar pose solutions.
    if len(candidates) > 1 and candidates[1][0] < 1.5 * best[0]:
        return None
    r, t = cv2.solvePnPRefineLM(obj, xy, K, D, best[1], best[2])
    errors = np.linalg.norm(cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2) - xy, axis=1)
    if np.median(errors) > 1.5 or np.percentile(errors, 90) > 3:
        return None
    return cv2.Rodrigues(r)[0], t.ravel(), float(np.median(errors))


def stats(values):
    a = np.array(values, dtype=float)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                p95=float(np.percentile(a, 95)), rms=float(np.sqrt(np.mean(a * a))))


def initialize(frames):
    edges = {}
    report = {}
    for a, b in itertools.combinations(range(4), 2):
        relative = []
        for f in frames:
            if a not in f['poses'] or b not in f['poses']:
                continue
            Ra, ta, _ = f['poses'][a]
            Rb, tb, _ = f['poses'][b]
            R = Rb @ Ra.T
            relative.append((R, tb - R @ ta))
        if not relative:
            continue
        Rs = np.stack([x[0] for x in relative]); ts = np.stack([x[1] for x in relative])
        R = Rotation.from_matrix(Rs).mean().as_matrix()
        t = np.median(ts, axis=0)
        angles = Rotation.from_matrix(Rs @ R.T).magnitude()
        distances = np.linalg.norm(ts - t, axis=1)
        good = (angles < np.deg2rad(3)) & (distances < .025)
        report[f'cam{a+1:02d}-cam{b+1:02d}'] = dict(observations=len(relative), inliers=int(good.sum()))
        if good.sum() < 5:
            continue
        R = Rotation.from_matrix(Rs[good]).mean().as_matrix()
        t = np.median(ts[good], axis=0)
        edges[a, b] = (R, t, int(good.sum()))
        edges[b, a] = (R.T, -R.T @ t, int(good.sum()))
    known = {0: (np.eye(3), np.zeros(3))}
    while len(known) < 4:
        options = [(v[2], a, b) for (a, b), v in edges.items() if a in known and b not in known]
        if not options:
            raise RuntimeError(f'Stationary board observations do not connect all cameras: {report}')
        _, a, b = max(options)
        R, t, _ = edges[a, b]
        known[b] = (R @ known[a][0], R @ known[a][1] + t)
    return [known[i] for i in range(4)], report


def optimize(frames, cameras, Ks, Ds, refine_intrinsics=False):
    x = []
    for R, t in cameras[1:]:
        x.extend(cv2.Rodrigues(R)[0].ravel()); x.extend(t)
    for f in frames:
        c = min(f['poses'], key=lambda c: f['poses'][c][2])
        Rc, tc = cameras[c]; Rb, tb, _ = f['poses'][c]
        x.extend(cv2.Rodrigues(Rc.T @ Rb)[0].ravel()); x.extend(Rc.T @ (tb - tc))
    intrinsic_start = len(x)
    intrinsic_initial = np.array([[K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D.ravel()]
                                  for K, D in zip(Ks, Ds)])
    intrinsic_scale = np.array([40., 40., 20., 20., .05, .1, .01, .01, .1])
    if refine_intrinsics:
        x.extend(intrinsic_initial.ravel())
    observations = [(fi, ci, OBJ[row['ids']], np.array(row['xy']))
                    for fi, f in enumerate(frames) for ci, row in f['rows'].items()]
    size = sum(2 * len(o[2]) for o in observations)
    sparsity = lil_matrix((size + (36 if refine_intrinsics else 0), len(x)), dtype=int)
    offset = 0
    for fi, ci, obj, xy in observations:
        n = len(obj) * 2
        if ci:
            sparsity[offset:offset+n, (ci-1)*6:ci*6] = 1
        sparsity[offset:offset+n, 18+fi*6:24+fi*6] = 1
        if refine_intrinsics:
            sparsity[offset:offset+n, intrinsic_start+ci*9:intrinsic_start+(ci+1)*9] = 1
        offset += n
    if refine_intrinsics:
        for j in range(36):
            sparsity[size+j, intrinsic_start+j] = 1

    def intrinsics(x):
        if not refine_intrinsics:
            return Ks, Ds
        ks, ds = [], []
        for v in x[intrinsic_start:].reshape(4, 9):
            ks.append(np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]]))
            ds.append(v[4:])
        return ks, ds

    def unpack(x):
        cams = [(np.eye(3), np.zeros(3))]
        for i in range(3):
            v = x[i*6:i*6+6]; cams.append((cv2.Rodrigues(v[:3])[0], v[3:]))
        boards = []
        for i in range(len(frames)):
            v = x[18+i*6:24+i*6]; boards.append((cv2.Rodrigues(v[:3])[0], v[3:]))
        return cams, boards

    def fun(x):
        cams, boards = unpack(x); residual = []
        ks, ds = intrinsics(x)
        for fi, ci, obj, xy in observations:
            Rc, tc = cams[ci]; Rb, tb = boards[fi]
            r = cv2.Rodrigues(Rc @ Rb)[0]; t = Rc @ tb + tc
            projected = cv2.projectPoints(obj, r, t, ks[ci], ds[ci])[0].reshape(-1, 2)
            residual.append((projected - xy).ravel())
        if refine_intrinsics:
            residual.append(((x[intrinsic_start:].reshape(4, 9) - intrinsic_initial) / intrinsic_scale).ravel())
        return np.concatenate(residual)

    fit = least_squares(fun, np.array(x), jac_sparsity=sparsity.tocsr(),
                        loss='soft_l1', f_scale=1, x_scale='jac', max_nfev=250,
                        ftol=1e-7, xtol=1e-8, gtol=1e-7)
    cameras, _ = unpack(fit.x)
    ks, ds = intrinsics(fit.x)
    return cameras, ks, ds, dict(success=bool(fit.success), message=fit.message, evaluations=fit.nfev,
                        intrinsicPriorScales=intrinsic_scale.tolist() if refine_intrinsics else None,
                        cost=float(fit.cost), reprojectionPixels=stats(np.linalg.norm(fun(fit.x)[:size].reshape(-1, 2), axis=1)))


def validate(frames, cameras, Ks, Ds):
    errors, per_camera, per_pair, spacing = [], {i: [] for i in range(4)}, {}, []
    for f in frames:
        # Re-estimate source-only poses using the intrinsics under evaluation.
        board_poses = {}
        for c, row in f['rows'].items():
            ok, r, t = cv2.solvePnP(OBJ[row['ids']], np.array(row['xy']), Ks[c], Ds[c])
            if not ok:
                raise RuntimeError('Held-out source PnP failed')
            board_poses[c] = (cv2.Rodrigues(r)[0], t.ravel())
        for a, b in itertools.permutations(f['rows'], 2):
            Ra, ta = cameras[a]; Rb, tb = cameras[b]
            Rp, tp = board_poses[a]
            R = Rb @ Ra.T @ Rp
            t = Rb @ Ra.T @ (tp - ta) + tb
            row = f['rows'][b]
            predicted = cv2.projectPoints(OBJ[row['ids']], cv2.Rodrigues(R)[0], t, Ks[b], Ds[b])[0].reshape(-1, 2)
            e = np.linalg.norm(predicted - np.array(row['xy']), axis=1).tolist()
            errors.extend(e); per_camera[b].extend(e)
            per_pair.setdefault(f'cam{a+1:02d}->cam{b+1:02d}', []).extend(e)
        # Known board spacing on HELD-OUT data: unconstrained DLT triangulation.
        normalized = {}
        for c, row in f['rows'].items():
            pts = cv2.undistortPoints(np.array(row['xy']).reshape(-1, 1, 2), Ks[c], Ds[c]).reshape(-1, 2)
            normalized[c] = dict(zip(row['ids'], pts))
        reconstructed = {}
        for idx in range(len(OBJ)):
            A = []
            for c, points in normalized.items():
                if idx not in points:
                    continue
                P = np.column_stack(cameras[c]); u, v = points[idx]
                A.extend([u * P[2] - P[0], v * P[2] - P[1]])
            if len(A) < 4:
                continue
            _, _, vt = np.linalg.svd(A); h = vt[-1]
            if abs(h[3]) > 1e-10:
                point = h[:3] / h[3]
                if all((R @ point + t)[2] > 0 for c, (R, t) in enumerate(cameras) if c in normalized and idx in normalized[c]):
                    reconstructed[idx] = point
        for i, point in reconstructed.items():
            for j in (i + 1, i + 9):
                if j not in reconstructed or (j == i + 1 and i // 9 != j // 9):
                    continue
                spacing.append(abs(np.linalg.norm(point - reconstructed[j]) - .022) * 1000)
    return dict(frames=len(frames), crossViewPixels=stats(errors),
                perTargetCamera={f'cam{i+1:02d}': stats(v) for i, v in per_camera.items()},
                directedCameraPairs={k: stats(v) for k, v in per_pair.items()},
                absolute22mmSpacingErrorMm=stats(spacing))


def pairwise_motion_observations(streams, origin, Ks, Ds):
    """Use close exposures with a bounded measured image-motion discrepancy.

    No corner interpolation: retain original detections and timestamps. Whole
    two-second blocks remain disjoint between training and validation.
    """
    frames = []
    for a, b in itertools.combinations(range(4), 2):
        left, right = streams[f'cam{a+1:02d}'], streams[f'cam{b+1:02d}']
        times = [r['pcMonotonicMs'] for r in right]
        last = -1
        for row in left:
            t = row['pcMonotonicMs']; k = bisect.bisect_left(times, t)
            options = [i for i in (k-1, k) if last < i < len(right)]
            if not options:
                continue
            k = min(options, key=lambda i: abs(times[i]-t)); other = right[k]
            skew = abs(times[k]-t)
            if skew > 35:
                continue
            speeds = [r['motionPixelsPerSecond'] for r in (row, other)]
            if any(s is None or s > 90 for s in speeds) or max(speeds)*skew/1000 > .75:
                continue
            if min(len(r['ids']) for r in (row, other)) < 16:
                continue
            blocks = [int((r['pcMonotonicMs']-origin)/2000) for r in (row, other)]
            if blocks[0] != blocks[1]:
                continue
            poses = {i: pose(r, Ks[i], Ds[i]) for i, r in ((a, row), (b, other))}
            if any(p is None for p in poses.values()):
                continue
            last = k
            frames.append(dict(time=(t-origin)/1000, skewMs=skew,
                rows={a: row, b: other}, poses=poses, heldOut=blocks[0] % 5 == 2))
    return sorted(frames, key=lambda f: f['time'])


def main(session, refine_intrinsics=False, pairwise_motion_budget=False):
    cv2.setNumThreads(2)
    out = session / 'calibration'; root = session / 'multicamera'
    calibration = toml.load(root / 'calibration.toml')
    Ks = [np.array(calibration[f'cam_{i}']['matrix']) for i in range(4)]
    Ds = [np.array(calibration[f'cam_{i}']['distortions']) for i in range(4)]
    old = [(cv2.Rodrigues(np.array(calibration[f'cam_{i}']['rotation']))[0],
            np.array(calibration[f'cam_{i}']['translation'])) for i in range(4)]
    streams = {}
    for i in range(4):
        name = f'cam{i+1:02d}'; streams[name] = read_rows(out / f'{name}_detections_stride1.jsonl')
        for row in streams[name]:
            row['unixTimeMs'] = row['pcMonotonicMs']
    origin = min(x[0]['pcMonotonicMs'] for x in streams.values())
    frames = []; last_time = -float('inf'); mono_errors = {i: [] for i in range(4)}
    for pair in pair_frames(streams, 35):
        # Up to 10 matched sets per second; neighboring images are not independent poses.
        if pair['unixTimeMs'] - last_time < 95:
            continue
        rows, poses = {}, {}
        for i in range(4):
            row = pair['frames'][f'cam{i+1:02d}']
            speed = row['motionPixelsPerSecond']
            if len(row['ids']) < 16 or speed is None or speed > 30:
                continue
            p = pose(row, Ks[i], Ds[i])
            if p is not None:
                rows[i] = row; poses[i] = p; mono_errors[i].append(p[2])
        if len(rows) < 2:
            continue
        last_time = pair['unixTimeMs']
        time_sec = (last_time - origin) / 1000
        frames.append(dict(time=time_sec, skewMs=pair['cameraSkewMs'], rows=rows, poses=poses,
                           heldOut=int(time_sec // 2) % 5 == 2))
    if pairwise_motion_budget:
        frames = pairwise_motion_observations(streams, origin, Ks, Ds)
        mono_errors = {i: [f['poses'][i][2] for f in frames if i in f['poses']] for i in range(4)}
    train = [f for f in frames if not f['heldOut']]; held = [f for f in frames if f['heldOut']]
    print(f'Stationary multi-camera sets: {len(frames)}; training {len(train)}; held-out {len(held)}', flush=True)
    initial, edges = initialize(train)
    print('Training edges: '+json.dumps(edges), flush=True)
    print(f'Optimizing rigid camera and per-frame board poses; refine intrinsics={refine_intrinsics}', flush=True)
    fitted, new_Ks, new_Ds, fit_report = optimize(train, initial, Ks, Ds, refine_intrinsics)
    before = validate(held, old, Ks, Ds); after = validate(held, fitted, new_Ks, new_Ds)
    training_validation = validate(train, fitted, new_Ks, new_Ds)
    report = dict(board=dict(squares=[10, 8], squareMetres=.022, markerMetres=.016,
                             dictionary='DICT_4X4_50', physicalDimensionsConfirmedByUser=
                             json.loads((session / 'recording.json').read_text()).get('boardDimensionsConfirmedByUser', False)),
        method='Robust joint camera/board pose optimization; cam01 is world origin; optional intrinsic priors',
        intrinsicsSource=str(root / 'calibration.toml'), intrinsicsRefitted=refine_intrinsics,
        sampling=dict(maxCameraSkewMs=35, maxMotionPixelsPerSecond=30, minCornersPerView=16),
        validationSplit='Hold out complete 2-second time blocks where floor(timeSeconds/2) mod 5 == 2; same recording, not a separate capture',
        validationCaveat='If used to compare model choices, these held-out blocks are validation data, not an untouched independent test capture.',
        selectedFrames=len(frames), trainFrames=len(train), heldOutFrames=len(held), edges=edges,
        singleCameraPoseFitMedianPixels={f'cam{i+1:02d}': stats(x) for i, x in mono_errors.items()},
        optimizer=fit_report, oldHeldOut=before, candidateHeldOut=after,
        candidateTrainingCrossView=training_validation)
    supported = len(held) >= 10 and all(x.get('count', 0) >= 100 for x in after['perTargetCamera'].values())
    if pairwise_motion_budget:
        report['sampling'] = dict(pairwise=True, maxCameraSkewMs=35,
            maxMotionPixelsPerSecond=90, maxEstimatedMotionDiscrepancyPixels=.75,
            minCornersPerView=16, interpolation=False,
            note='Motion discrepancy is a local estimate, not a guaranteed error bound; neighboring observations are correlated.')
    passes = supported and fit_report['success'] and all(
        x.get('median', float('inf')) < 3 and x.get('p90', float('inf')) < 5
        for x in after['perTargetCamera'].values())
    report['passesInitialValidation'] = bool(passes)
    report['limitations'] = ['No independent validation recording',
        'Free-running exposures; use only low-motion observations',
        'Board size assumed from physically confirmed configuration',
        'New calibration applies to historical recordings only if rig and lens settings were unchanged']
    for i, (R, t) in enumerate(fitted):
        calibration[f'cam_{i}']['matrix'] = new_Ks[i].tolist()
        calibration[f'cam_{i}']['distortions'] = new_Ds[i].ravel().tolist()
        calibration[f'cam_{i}']['rotation'] = cv2.Rodrigues(R)[0].ravel().tolist()
        calibration[f'cam_{i}']['translation'] = t.tolist()
    calibration['metadata'] = dict(adjusted=True, error=fit_report['reprojectionPixels']['rms'],
        method=report['method'], held_out_validated=bool(passes))
    (out / 'calibration_candidate.toml').write_text(toml.dumps(calibration))
    (out / 'calibration_report.json').write_text(json.dumps(report, indent=2))
    # Keep exact source frames so validation can be reproduced and visualized.
    selected = [dict(time=f['time'], skewMs=f['skewMs'], heldOut=f['heldOut'],
                     rows={f'cam{i+1:02d}': row for i, row in f['rows'].items()}) for f in frames]
    (out / 'selected_frames.json').write_text(json.dumps(selected))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session', type=Path)
    ap.add_argument('--refine-intrinsics', action='store_true')
    ap.add_argument('--pairwise-motion-budget', action='store_true')
    args = ap.parse_args()
    main(args.session, args.refine_intrinsics, args.pairwise_motion_budget)
