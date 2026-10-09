"""Direct multiview joint reconstruction from weighted 2D observations.

Pair triangulations are numerical seeds, not fixed 3D targets. No baseline pose
or pre-smoothed skeleton is read. Camera calibration and reliability weights
remain fixed; this is joint pose optimization, not camera bundle adjustment.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import toml

from fit_weighted_handpose import SETTINGS, ReprojectionProblem, errors, fit_windows
from hand_reliability import cameras, leave_one_out, project, triangulate_pair
from regularize_handpose import diagnostics, describe, estimate_lengths


JOINT_SETTINGS = dict(SETTINGS, initializationMinimumWeight=.15,
                      initializationPairMaxErrorPixels=15., minimumRayAngleDegrees=2.)


class JointProblem(ReprojectionProblem):
    """One objective: weighted image errors + bones + robust acceleration.

Reliability multiplies the robust loss itself. It must not move the robust
loss transition as happens with weights inside a Cauchy loss. Anchor residuals
are zero: changing the seed cannot change the objective at the same 3D point.
"""
    def __init__(self, xyz, times, xy, weights, cams, lengths, settings):
        super().__init__(xyz, times, xy, weights, cams, lengths, settings)
        self.sigma[:] = np.inf
        self.loss_weights = np.concatenate([np.repeat((w*settings['pixelSigma'])**2, 2)
                                            for _, _, w, _ in self.obs])
        self.obs = [(ids, uv, np.full(len(ids), 1/settings['pixelSigma']), offset)
                    for ids, uv, _, offset in self.obs]

    def loss(self, z):
        rho = super().loss(z)
        # Soft-L1 image loss retains a bounded pull from inconsistent views.
        a = z[self.nres:]
        rho[:, self.nres:] = np.array([2*(np.sqrt(1+a)-1),
                                       1/np.sqrt(1+a), -.5/(1+a)**1.5]) * self.loss_weights
        return rho


def initialize(xy, weights, cams, settings=JOINT_SETTINGS):
    """Rank positive-depth, nondegenerate pair seeds using ALL weighted views."""
    shape = xy.shape[1:-1]
    result = np.full((*shape, 3), np.nan)
    score = np.full(shape, np.inf)
    selected = np.full((*shape, 2), -1, dtype=np.int16)
    weights = np.where(np.isfinite(xy).all(-1), weights, 0.)
    for a, b in itertools.combinations(range(len(cams)), 2):
        x = triangulate_pair(xy[a], xy[b], cams[a], cams[b])
        valid = np.isfinite(x).all(-1)
        centers = [-c['R'].T @ c['t'] for c in (cams[a], cams[b])]
        rays = [x-c for c in centers]
        cos = np.sum(rays[0]*rays[1], axis=-1) / np.maximum(
            np.linalg.norm(rays[0], axis=-1)*np.linalg.norm(rays[1], axis=-1), 1e-12)
        angle = np.degrees(np.arccos(np.clip(np.abs(cos), 0, 1)))
        e = errors(x, xy, cams)
        valid &= (weights[a] >= settings['initializationMinimumWeight'])
        valid &= (weights[b] >= settings['initializationMinimumWeight'])
        valid &= angle >= settings['minimumRayAngleDegrees']
        valid &= np.maximum(e[a], e[b]) <= settings['initializationPairMaxErrorPixels']
        for c, camera in enumerate(cams):
            depth = x @ camera['R'][2] + camera['t'][2]
            valid &= (weights[c] == 0) | (depth > .05)
        # Match the optimizer's per-coordinate weighted soft-L1 image loss.
        cost = np.zeros(shape)
        for c, camera in enumerate(cams):
            pred = np.full(xy[c].shape, np.nan)
            good = np.isfinite(x).all(-1)
            pred[good] = project(x[good], camera)
            residual = (pred-xy[c]) / settings['pixelSigma']
            penalty = (2*(np.sqrt(1+residual**2)-1)).sum(-1)
            cost += np.where(weights[c] > 0, weights[c]*penalty, 0.)
        choose = valid & np.isfinite(cost) & (cost < score)
        result[choose] = x[choose]
        score[choose] = cost[choose]
        selected[choose] = [a, b]
    if not np.isfinite(result).any():
        raise ValueError('No reliable nondegenerate two-view seeds; check calibration and observations')
    return result, selected


def objective(problem, xyz):
    r = problem.residual(xyz[problem.valid].ravel())
    rho = problem.loss(r*r)[0] / 2
    n = problem.observed.size
    return dict(total=float(rho.sum()), anchor=float(rho[:n].sum()),
                bones=float(rho[n:n+len(problem.ba)].sum()),
                motion=float(rho[n+len(problem.ba):problem.nres].sum()),
                reprojection=float(rho[problem.nres:].sum()))


def validate_solution(initial, candidate, times, xy, weights, cams, lengths, settings):
    """Check the SAME full objective after blending, not a conflicting 3D prior.

This is a numerical safeguard, not a guarantee of anatomical or 3D accuracy.
View disagreements are reported separately instead of rejecting a whole frame
because an incompatible view's P90 increased.
"""
    p = JointProblem(initial, times, xy, weights, cams, lengths, settings)
    before = objective(p, initial)
    finite = np.isfinite(candidate[p.valid]).all()
    positive = all(np.all((candidate[p.valid] @ c['R'][2] + c['t'][2]) > .05)
                   for c in cams)
    after = objective(p, candidate) if finite else None
    accept = finite and positive and after['total'] <= before['total'] + max(1e-6, before['total']*1e-6)
    result = candidate.copy() if accept else initial.copy()
    result[~p.valid] = np.nan
    return result, dict(accepted=bool(accept), finite=bool(finite), positiveDepth=bool(positive),
                       before=before, candidate=after,
                       policy='Accept blended solution only if the same joint objective does not increase; otherwise keep numerical seeds',
                       caveat='Objective agreement does not validate calibration, weights, anatomy or exposure synchronization')


def run(session, mode="joint"):
    session = Path(session).resolve()
    root = session/'multicamera'
    out = root/'weighted_handpose'
    obs_path = out/'observations.npz'
    obs = np.load(obs_path)
    stamps = obs['unixTimeMs']
    times = (stamps-stamps[0])/1000
    pairs = [json.loads(line) for line in (root/'frame_sets.jsonl').read_text().splitlines()]
    np.testing.assert_allclose(stamps, [p['unixTimeMs'] for p in pairs], rtol=0, atol=.001)
    xy = obs['augment_xy'][:, :, 0].copy()
    xy[obs['weight_image'] == 0] = np.nan
    cams = cameras(toml.load(root/'calibration.toml'))
    extras = {}
    if mode == 'trajectory':
        from fit_hand_trajectory import TRAJECTORY_SETTINGS, prepare, fit, evaluate, basis
        settings = TRAJECTORY_SETTINGS
        view_times = np.array([[p['frames'][f'cam{c+1:02d}']['unixTimeMs'] for p in pairs] for c in range(4)])
        view_times = (view_times-stamps[0])/1000
        if not np.isfinite(view_times).all() or np.any(np.diff(view_times,axis=1)<=0):
            raise ValueError('Native camera timestamps must be finite and strictly increasing')
        initial, seed, inferred, seed_pairs, weights, geometry, geoerr, refquality, aligned, aligned_w, seed_fallback = prepare(xy,obs['weight_image'],cams,times,view_times)
        seed_error = errors(seed, aligned, cams)
        seed_support = ((aligned_w >= SETTINGS['reliabilityThreshold']) & (seed_error <= 15)).sum(0)
        lengths, bone_report = estimate_lengths(seed, seed_support)
        _, before_error = evaluate(initial,times,view_times,xy,cams)
        result,candidate,windows,quality = fit(initial,times,xy,weights,cams,lengths,view_times)
        inferred &= np.isfinite(result).all(-1)
        per_view_xyz, after_error = evaluate(result,times,view_times,xy,cams)
        view_inferred = np.zeros(per_view_xyz.shape[:-1],bool)
        for c in range(4):
            lo,hi,_ = basis(times,np.isfinite(result).all(-1),view_times[c],settings['trajectoryMaxGapSeconds'],True)
            rr,jj = np.where(lo>=0)
            view_inferred[c,rr,jj] = inferred[lo[rr,jj],jj] | inferred[hi[rr,jj],jj]
        extras = dict(perViewJoints=per_view_xyz, perViewTemporalInferred=view_inferred, viewUnixTimeMs=view_times*1000+stamps[0],
                      temporalInferred=inferred, seedValid=np.isfinite(seed).all(-1),
                      trajectoryReconstruction=True, nativeSeedFallback=seed_fallback)
    else:
        settings = JOINT_SETTINGS
        refs, refquality = leave_one_out(xy, cams)
        geoerr = np.linalg.norm(refs-xy, axis=-1)
        geometry = np.ones_like(geoerr)
        known = np.isfinite(geoerr)
        geometry[known] = np.maximum(.02, 1/(1+(geoerr[known]/SETTINGS['geometryScalePixels'])**2))
        weights = obs['weight_image']*geometry
        initial, seed_pairs = initialize(xy, weights, cams)
        before_error = errors(initial, xy, cams)
        seed_support = ((weights >= SETTINGS['reliabilityThreshold']) & (before_error <= 15)).sum(0)
        lengths, bone_report = estimate_lengths(initial, seed_support)
        candidate, windows = fit_windows(initial, times, xy, weights, cams, lengths, settings,
                                         problem_class=JointProblem)
        result, quality = validate_solution(initial, candidate, times, xy, weights, cams, lengths, settings)
        after_error = errors(result, xy, cams)
    known = np.isfinite(geoerr)
    valid = np.isfinite(result).all(-1)
    support = ((weights >= SETTINGS['reliabilityThreshold']) & (after_error <= 15)).sum(0)
    high = weights >= .5
    report = dict(method='Direct joint reconstruction: fixed weighted 2D observations + bones + robust timestamp motion; no 3D anchor',
        fitMode=mode, settings=settings, frames=len(times), windows=windows,
        initialization='Best positive-depth, nondegenerate camera-pair seed ranked by all-view weighted soft-L1 reprojection; numerical seed only',
        baselinePoseUsed=False, boneLengthEstimation=bone_report, qualityGuard=quality,
        objectiveLoss=dict(reprojection='weight * soft-L1', anchors='none', bones='quadratic', acceleration='soft-L1'),
        calibrationSha256=hashlib.sha256((root/'calibration.toml').read_bytes()).hexdigest(),
        observationsSha256=hashlib.sha256(obs_path.read_bytes()).hexdigest(),
        before=diagnostics(initial, times, lengths, SETTINGS['maxGapSeconds']),
        after=diagnostics(result, times, lengths, SETTINGS['maxGapSeconds']),
        correctionMm=describe(np.linalg.norm(result-initial, axis=-1)*1000),
        highWeightReprojectionPixels=dict(before=describe(before_error[high]), after=describe(after_error[high])),
        perCameraReprojectionPixels={f'cam{c+1:02d}':dict(before=describe(before_error[c]), after=describe(after_error[c])) for c in range(4)},
        weights=describe(weights[obs['weight_image']>0]), leaveOneOutReferenceFraction=float(known.mean()),
        underTwoSupportedViewsFraction=float((support[valid]<2).mean()), validJointFraction=float(valid.mean()),
        missingPolicy='Independent two-view initialization mask; no single-view depth recovery or gap interpolation',
        caveats=['Calibration and weights fixed; not camera bundle adjustment',
                 'Camera exposure skew remains; same nominal time does not guarantee same pose',
                 'Heuristic weights and estimated bone lengths are not ground truth',
                 'No learned anatomical joint-angle or palm-shape constraint',
                 'Missing joints differ from legacy 5px subset-filter outputs; compare common joints separately'])
    if mode == 'trajectory':
        report.update(method='Asynchronous 3D trajectory: each native 2D observation constrains its own camera timestamp',
            timestampConvention='Recorded camera timestamp mapped to Unix time; exposure midpoint offset is not assumed or corrected',
            trajectoryModel='Piecewise-linear 3D knots with robust real-time acceleration and bone constraints',
            temporalInferredJoints=int(inferred.sum()), nativeSeedFallbackJoints=int(seed_fallback.sum()), seedValidJointFraction=float(np.isfinite(seed).all(-1).mean()),
            missingPolicy='Fill only interior seed gaps with bracketing observations <=150 ms apart; no endpoint extrapolation; label temporal estimates',
            supportMeaning='Number of nearby native camera observations explained by the fitted trajectory, not simultaneous views at the knot',
            initialization='Time-aligned 2D seeds preferred; valid native camera-pair seeds used where alignment unavailable. Seeds are guesses, not 3D anchors. Native 2D observations used in optimization',
            caveats=['Recorded camera times are estimates; unknown clock bias and exposure timestamp semantics remain',
                     'Exposure integration and motion blur are not modeled',
                     'Short-gap estimates are prior-assisted positions, not independent measured ground truth',
                     'Piecewise-linear motion can miss within-frame nonlinear motion',
                     'Calibration, heuristic weights and estimated bone lengths remain uncertain'])
    np.savez_compressed(out/'pose_3d.npz', joints=result, valid=valid, unixTimeMs=stamps,
        joint_names=obs['joint_names'], units='metres', camerasUsed=support,
        perCameraReprojectionErrorPixels=after_error, reprojectionErrorPixels=np.nanmedian(after_error, axis=0),
        weights=weights, imageWeights=obs['weight_image'], geometryWeights=geometry,
        leaveOneOutErrorPixels=geoerr, leaveOneOutReferenceQualityPixels=refquality,
        boneLengthsMetres=lengths, weightedReprojection=True, jointReconstruction=True,
        qualityMeaning='Two-view support is diagnostic consistency, not verified 3D accuracy', **extras)
    np.savez_compressed(out/'joint_initialization.npz', joints=initial, seedPairs=seed_pairs,
                        unixTimeMs=stamps, boneLengthsMetres=lengths)
    np.savez_compressed(out/'optimization_candidate.npz', joints=candidate, unixTimeMs=stamps)
    (out/'weighted_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    with (out/'hand_pose_aligned.jsonl').open('w') as f:
        for i, pair in enumerate(pairs):
            row = dict(type=f'{mode}_multicamera_hand_pose', sampleIndex=i, unixTimeMs=float(stamps[i]),
                coordinateSpace='calibration world', units='metres', cameraSkewMs=pair['cameraSkewMs'],
                joints=[dict(name=str(name), valid=bool(valid[i,j]),
                    position=result[i,j].tolist() if valid[i,j] else None,
                    supportedViews=int(support[i,j]), priorDominated=bool(support[i,j]<2),
                    viewWeights=weights[:,i,j].tolist(),
                    temporalInferred=bool(extras['temporalInferred'][i,j]) if mode=='trajectory' else False) for j,name in enumerate(obs['joint_names'])])
            f.write(json.dumps(row, allow_nan=False)+'\n')
    (out/'processing_status.json').write_text(json.dumps(dict(status='fit_complete',fitMode=mode,frames=len(times))))
    print(json.dumps({k:report[k] for k in ('qualityGuard','highWeightReprojectionPixels','validJointFraction','underTwoSupportedViewsFraction')}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session')
    ap.add_argument('--mode', choices=('joint','trajectory'), default='joint')
    args=ap.parse_args()
    run(args.session,args.mode)
