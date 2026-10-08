"""Robust bone-length and real-time acceleration constraints for hand trajectories.

Keeps unregularized results, never fills missing joints, and breaks temporal
constraints at gaps. Bone lengths are estimated per recording, not population
averages. This does not resolve cross-view person identity or exposure skew.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import toml
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

CHAINS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
EDGES = np.array([(c[i],c[i+1]) for c in CHAINS for i in range(4)])
DEFAULTS = dict(windowFrames=80, overlapFrames=20, maxGapSeconds=.12,
                accelerationScale=3., boneRelativeSigma=.05, boneMinimumSigmaMetres=.002,
                anchorThreeViewSigmaMetres=.003, anchorTwoViewSigmaMetres=.007,
                maxEvaluations=200)


def describe(values):
    a = np.asarray(values); a = a[np.isfinite(a)]
    if not len(a): return dict(count=0)
    return dict(count=int(len(a)),median=float(np.median(a)),p90=float(np.percentile(a,90)),
                p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)))


def project_errors(xyz, points, calibration):
    valid = np.isfinite(xyz).all(-1)
    error = np.full(points.shape[:-1], np.nan)
    for i in range(4):
        c = calibration[f'cam_{i}']
        if not valid.any(): continue
        uv = cv2.projectPoints(xyz[valid],np.array(c['rotation']),np.array(c['translation']),
                               np.array(c['matrix']),np.array(c['distortions']))[0].reshape(-1,2)
        error[i][valid] = np.linalg.norm(uv-points[i,...,:2][valid],axis=-1)
        error[i][points[i,...,2]<.6] = np.nan
    return error


def estimate_lengths(xyz, support):
    lengths, details = [], []
    for a,b in EDGES:
        v = np.linalg.norm(xyz[:,a]-xyz[:,b],axis=-1)
        finite = np.isfinite(v) & (v>.003) & (v<.20)
        reliable = finite & (support[:,a]>=3) & (support[:,b]>=3)
        selected = reliable if reliable.sum()>=20 else finite
        if selected.sum()<20:
            raise ValueError(f'Insufficient observations to estimate bone {a}-{b}')
        sample = v[selected]; median = np.median(sample)
        mad = np.median(np.abs(sample-median))
        kept = sample[np.abs(sample-median)<=max(3*1.4826*mad,.002)]
        value = float(np.median(kept)); lengths.append(value)
        details.append(dict(joints=[int(a),int(b)],lengthMetres=value,samples=int(len(kept)),
                            source='three-view support at both ends' if reliable.sum()>=20 else 'robust finite observations fallback'))
    return np.array(lengths), details


class WindowProblem:
    def __init__(self, xyz, times, support, lengths, settings):
        self.valid = np.isfinite(xyz).all(-1)
        self.observed = xyz[self.valid].copy()
        self.shape = xyz.shape
        self.ids = np.full(self.valid.shape,-1,dtype=int)
        self.ids[self.valid] = np.arange(self.valid.sum())
        self.sigma = np.where(support[self.valid]>=3,settings['anchorThreeViewSigmaMetres'],settings['anchorTwoViewSigmaMetres'])
        aa,bb = self.ids[:,EDGES[:,0]],self.ids[:,EDGES[:,1]]
        good = (aa>=0)&(bb>=0)
        self.ba,self.bb = aa[good],bb[good]
        self.lengths = np.broadcast_to(lengths,aa.shape)[good]
        self.bsigma = np.maximum(settings['boneMinimumSigmaMetres'],settings['boneRelativeSigma']*self.lengths)
        dt = np.diff(times)
        if (dt<=0).any(): raise ValueError('Timestamps must increase strictly')
        triples, first_intervals, second_intervals = [], [], []
        for j in range(self.valid.shape[1]):
            f = np.flatnonzero(self.valid[:,j])
            for a,b,c in zip(f[:-2],f[1:-1],f[2:]):
                d1,d2=times[b]-times[a],times[c]-times[b]
                if d1<=settings['maxGapSeconds'] and d2<=settings['maxGapSeconds']:
                    triples.append([self.ids[a,j],self.ids[b,j],self.ids[c,j]])
                    first_intervals.append(d1);second_intervals.append(d2)
        # Connect consecutive OBSERVATIONS even when a short missing sample is
        # between them. The missing sample itself stays missing in the output.
        self.tids = np.array(triples,dtype=int).reshape(-1,3)
        d1,d2 = np.array(first_intervals),np.array(second_intervals)
        # Difference of adjacent velocities divided by the separation of their
        # temporal midpoints: true acceleration on nonuniform sample intervals.
        self.coeff = np.column_stack([1/d1,-1/d1-1/d2,1/d2])*(2/(d1+d2)/settings['accelerationScale'])[:,None]
        n = self.observed.size; nb = len(self.ba); nt = len(self.tids)
        self.nres = n+nb+nt*3
        rr = [np.arange(n)]; cc = [np.arange(n)]
        rr.append(np.repeat(n+np.arange(nb),6))
        cc.append(np.column_stack([self.ba[:,None]*3+np.arange(3),self.bb[:,None]*3+np.arange(3)]).ravel())
        tr = n+nb+np.arange(nt*3).reshape(nt,3)
        rr.append(np.repeat(tr,3,axis=1).ravel())
        cc.append((self.tids[:,None,:]*3+np.arange(3)[None,:,None]).ravel())
        self.jrows = np.concatenate(rr); self.jcols = np.concatenate(cc)

    def residual(self, flat):
        x = flat.reshape(-1,3)
        anchor = ((x-self.observed)/self.sigma[:,None]).ravel()
        bone = (np.linalg.norm(x[self.ba]-x[self.bb],axis=1)-self.lengths)/self.bsigma
        temporal = (x[self.tids]*self.coeff[:,:,None]).sum(axis=1).ravel()
        return np.concatenate([anchor,bone,temporal])

    def jacobian(self, flat):
        x = flat.reshape(-1,3); d = x[self.ba]-x[self.bb]
        gradient = d/np.maximum(np.linalg.norm(d,axis=1),1e-12)[:,None]/self.bsigma[:,None]
        data = np.concatenate([np.repeat(1/self.sigma,3),np.column_stack([gradient,-gradient]).ravel(),
                               np.broadcast_to(self.coeff[:,None,:],(len(self.tids),3,3)).ravel()])
        return coo_matrix((data,(self.jrows,self.jcols)),shape=(self.nres,x.size)).tocsr()


def constrain(xyz, times, support, lengths, settings):
    n = len(xyz); window = settings['windowFrames']; overlap = settings['overlapFrames']
    if not 0<=overlap<window: raise ValueError('Invalid overlap/window')
    accum = np.zeros_like(xyz); weights = np.zeros(xyz.shape[:-1]); reports=[]
    for start in range(0,n,window-overlap):
        end = min(n,start+window)
        problem = WindowProblem(xyz[start:end],times[start:end],support[start:end],lengths,settings)
        if len(problem.observed):
            fit = least_squares(problem.residual,problem.observed.ravel(),jac=problem.jacobian,
                loss='soft_l1',f_scale=1.,x_scale='jac',max_nfev=settings['maxEvaluations'],
                ftol=1e-5,xtol=1e-6,gtol=1e-5)
            if not np.isfinite(fit.x).all(): raise RuntimeError(f'Non-finite optimization at {start}')
            value=np.full_like(xyz[start:end],np.nan);value[problem.valid]=fit.x.reshape(-1,3)
            taper=np.ones(end-start)
            if start: taper[:min(overlap,len(taper))]=np.linspace(1/(overlap+1),1,min(overlap,len(taper)))
            if end<n: taper[-overlap:]=np.linspace(1,1/(overlap+1),overlap)
            wt=taper[:,None]*problem.valid
            accum[start:end]+=np.nan_to_num(value)*wt[...,None]; weights[start:end]+=wt
            reports.append(dict(start=start,end=end,converged=bool(fit.success),evaluations=fit.nfev,cost=float(fit.cost)))
            print(f'Constraints {end}/{n}, evaluations={fit.nfev}, converged={fit.success}',flush=True)
        if end==n: break
    result=np.full_like(xyz,np.nan);valid=weights>0;result[valid]=accum[valid]/weights[valid,None]
    return result,reports


def diagnostics(xyz, times, lengths, max_gap):
    bone = np.linalg.norm(xyz[:,EDGES[:,0]]-xyz[:,EDGES[:,1]],axis=-1)
    dt=np.diff(times);velocity=np.diff(xyz,axis=0)/dt[:,None,None]
    acceleration=np.diff(velocity,axis=0)/(0.5*(dt[1:]+dt[:-1]))[:,None,None]
    acceleration[(dt[1:]>max_gap)|(dt[:-1]>max_gap)] = np.nan
    return dict(boneLengthAbsoluteErrorMm=describe(np.abs(bone-lengths)*1000),
                accelerationMetresPerSecondSquared=describe(np.linalg.norm(acceleration,axis=-1)),
                stepDistanceMm=describe(np.linalg.norm(np.diff(xyz,axis=0),axis=-1)*1000),
                speedMetresPerSecond=describe(np.linalg.norm(velocity,axis=-1)),
                finiteJoints=int(np.isfinite(xyz).all(-1).sum()))


def main(session):
    session=Path(session).resolve();root=session/'multicamera';out=root/'handpose'
    config=json.loads((root/'config.json').read_text());settings=dict(DEFAULTS)
    settings.update(config.get('regularization',{}))
    raw_path=out/'pose_3d_unconstrained.npz'
    if not raw_path.exists(): shutil.copy2(out/'pose_3d.npz',raw_path)
    report_raw=out/'pose_report_unconstrained.json'
    if not report_raw.exists(): shutil.copy2(out/'pose_report.json',report_raw)
    lines_raw=session/'multicamera_hand_pose_unconstrained.jsonl'
    if not lines_raw.exists(): shutil.copy2(session/'multicamera_hand_pose_aligned.jsonl',lines_raw)
    p=np.load(raw_path);raw=p['joints'].copy();times=(p['unixTimeMs']-p['unixTimeMs'][0])/1000
    points=np.load(out/'wilor_2d.npz')['keypoints'];calibration=toml.load(root/'calibration.toml')
    before_error=project_errors(raw,points,calibration);support=(before_error<=10).sum(0)
    lengths,length_report=estimate_lengths(raw,support)
    # Reject only gross spatial outliers relative to this recording's supported
    # capture volume. A 25 cm margin keeps ordinary finger articulation/motion.
    trusted=raw[(support>=3)&np.isfinite(raw).all(-1)]
    if len(trusted)<100: raise ValueError('Too few three-view-supported joints to establish capture volume')
    lower=np.percentile(trusted,.5,axis=0)-.25;upper=np.percentile(trusted,99.5,axis=0)+.25
    outside=np.isfinite(raw).all(-1)&((raw<lower)|(raw>upper)).any(-1)
    initial=raw.copy();initial[outside]=np.nan
    result,windows=constrain(initial,times,support,lengths,settings)
    after_error=project_errors(result,points,calibration)
    valid=np.isfinite(result).all(-1)
    assert not (valid & ~p['valid']).any(), 'Constraints must not invent missing joints'
    original_trusted=(before_error<=10)&np.isfinite(after_error)
    report=dict(method='Robust 3D anchors + recording-specific bone lengths + acceleration using actual timestamp intervals',
        settings=settings,bones=length_report,rawInputSha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        calibrationSha256=hashlib.sha256((root/'calibration.toml').read_bytes()).hexdigest(),
        grossOutliersRejected=int(outside.sum()),captureVolumeMetres=dict(lower=lower.tolist(),upper=upper.tolist()),
        gapsFilled=0,windowReports=windows,
        before=diagnostics(raw,times,lengths,settings['maxGapSeconds']),
        beforeOnKeptJoints=diagnostics(initial,times,lengths,settings['maxGapSeconds']),
        after=diagnostics(result,times,lengths,settings['maxGapSeconds']),
        correctionMm=describe(np.linalg.norm(result-raw,axis=-1)*1000),
        originalSupportedViewsReprojectionPixels=dict(before=describe(before_error[original_trusted]),after=describe(after_error[original_trusted])),
        perCameraReprojectionPixels={f'cam{i+1:02d}':dict(before=describe(before_error[i]),after=describe(after_error[i])) for i in range(4)},
        caveats=['Bone lengths estimated from predictions, not measured anatomy','Constraints do not fix identity errors or exposure mismatch',
                 'No 3D ground truth; reduced jitter is not evidence of improved absolute accuracy','Missing points and temporal gaps remain missing'])
    (out/'regularization_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    # Keep original subset residuals explicitly named; they do not describe the
    # constrained coordinates. Store new residuals in all views separately.
    np.savez_compressed(out/'pose_3d.npz',joints=result,valid=valid,unixTimeMs=p['unixTimeMs'],
        joint_names=p['joint_names'],units='metres',camerasUsed=p['camerasUsed'],
        rawSubsetReprojectionErrorPixels=p['reprojectionErrorPixels'],
        reprojectionErrorPixels=np.nanmedian(after_error,axis=0),
        perCameraReprojectionErrorPixels=after_error,boneLengthsMetres=lengths,
        regularized=True,qualityMeaning='Constrained prediction, not newly validated or measured ground truth')
    with lines_raw.open() as source,(session/'multicamera_hand_pose_aligned.jsonl').open('w') as target:
        for f,line in enumerate(source):
            row=json.loads(line);row['regularized']=True;row['constraints']=dict(boneLength=True,realTimestampAcceleration=True)
            for j,joint in enumerate(row['joints']):
                joint['rawSubsetReprojectionErrorPixels']=joint.get('reprojectionErrorPixels')
                joint.update(valid=bool(valid[f,j]),position=result[f,j].tolist() if valid[f,j] else None,
                    reprojectionErrorPixels=float(np.nanmedian(after_error[:,f,j])) if valid[f,j] and np.isfinite(after_error[:,f,j]).any() else None,
                    reprojectionErrorMeaning='median across available WiLoR views after constraints',
                    grossOutlierRejected=bool(outside[f,j]))
            target.write(json.dumps(row,allow_nan=False)+'\n')
    quality=json.loads(report_raw.read_text());quality.update(validJointFraction=float(valid.mean()),validFrameFraction=float(valid.all(1).mean()),
        algorithm='WiLoR 2D + Anipose RANSAC + robust bone/real-time acceleration constraints',
        regularizationReport='regularization_report.json',reprojectionThresholdPixels=None,
        validityMeaning='Raw subset acceptance minus gross outliers; constrained coordinates have not been reaccepted under the original 5px threshold')
    (out/'pose_report.json').write_text(json.dumps(quality,indent=2))
    print(json.dumps({k:report[k] for k in ('grossOutliersRejected','beforeOnKeptJoints','after','correctionMm','originalSupportedViewsReprojectionPixels')},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session',type=Path)
    main(parser.parse_args().session)
