"""Weighted calibrated 2D reprojection + bone lengths + real-timestamp acceleration.

Writes a separate result; never overwrites the baseline and never fills missing
baseline joints. Reliability scores are heuristics, not learned probabilities.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import toml
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix,vstack
from regularize_handpose import WindowProblem,DEFAULTS,diagnostics,describe
from hand_reliability import cameras,project,leave_one_out

SETTINGS=dict(DEFAULTS,pixelSigma=3.,weakAnchorSigmaMetres=.020,
    lowSupportAnchorSigmaMetres=.010,geometryScalePixels=15.,reliabilityThreshold=.15,
    boneMinimumSigmaMetres=.0015,boneRelativeSigma=.04,accelerationScale=10.,maxEvaluations=400,
    qualityWeightThreshold=.5,qualityMinimumObservations=6,
    qualityAbsoluteTolerancePixels=2.,qualityRelativeTolerance=.10,
    qualityJointAbsoluteTolerancePixels=5.,qualityJointRelativeTolerance=.25)


class ReprojectionProblem(WindowProblem):
    def __init__(self,xyz,times,xy,weights,cams,lengths,settings):
        support=(weights>=settings['reliabilityThreshold']).sum(0)
        super().__init__(xyz,times,support,lengths,settings)
        self.sigma=np.where(support[self.valid]>=2,settings['weakAnchorSigmaMetres'],settings['lowSupportAnchorSigmaMetres'])
        self.cams=cams;self.obs=[];count=0
        for c in range(len(cams)):
            valid=self.valid&np.isfinite(xy[c]).all(-1)&(weights[c]>0)
            ids=self.ids[valid];n=len(ids)
            self.obs.append((ids,xy[c][valid],np.sqrt(weights[c][valid])/settings['pixelSigma'],count))
            count+=n*2
        self.obsres=count

    def loss(self,z):
        # A very wrong 2D view must not stretch bones or overwhelm motion
        # continuity. Acceleration and anchors use soft-L1 so real fast motion
        # cannot dominate the robust image residuals. Bones remain quadratic.
        rho=np.vstack([z.copy(),np.ones_like(z),np.zeros_like(z)])
        anchor=self.observed.size;a=z[:anchor]
        rho[0,:anchor]=2*(np.sqrt(1+a)-1);rho[1,:anchor]=1/np.sqrt(1+a);rho[2,:anchor]=-.5/(1+a)**1.5
        motion=anchor+len(self.ba);a=z[motion:self.nres]
        rho[0,motion:self.nres]=2*(np.sqrt(1+a)-1)
        rho[1,motion:self.nres]=1/np.sqrt(1+a)
        rho[2,motion:self.nres]=-.5/(1+a)**1.5
        a=z[self.nres:];rho[0,self.nres:]=np.log1p(a);rho[1,self.nres:]=1/(1+a);rho[2,self.nres:]=-1/(1+a)**2
        return rho

    def residual(self,flat):
        x=flat.reshape(-1,3);base=super().residual(flat);extra=[]
        for c,(ids,xy,w,_) in enumerate(self.obs):extra.append(((project(x[ids],self.cams[c])-xy)*w[:,None]).ravel())
        return np.concatenate([base,*extra])

    def jacobian(self,flat):
        x=flat.reshape(-1,3);base=super().jacobian(flat);rr=[];cc=[];vv=[]
        for c,(ids,xy,w,offset) in enumerate(self.obs):
            _,j=project(x[ids],self.cams[c],True)
            rr.append(np.repeat(offset+np.arange(len(ids)*2),3))
            cc.append(np.broadcast_to(ids[:,None,None]*3+np.arange(3),(len(ids),2,3)).ravel())
            vv.append((j*w[:,None,None]).ravel())
        extra=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(self.obsres,x.size)).tocsr()
        return vstack([base,extra],format='csr')


def fit_windows(initial,times,xy,weights,cams,lengths,settings,problem_class=ReprojectionProblem):
    n=len(initial);window=settings['windowFrames'];overlap=settings['overlapFrames']
    accum=np.zeros_like(initial);total=np.zeros(initial.shape[:-1]);reports=[]
    for start in range(0,n,window-overlap):
        end=min(n,start+window)
        p=problem_class(initial[start:end],times[start:end],xy[:,start:end],weights[:,start:end],cams,lengths,settings)
        if p.valid.any():
            f=least_squares(p.residual,p.observed.ravel(),jac=p.jacobian,loss=p.loss,f_scale=1.,x_scale='jac',
                max_nfev=settings['maxEvaluations'],ftol=1e-4,xtol=1e-5,gtol=1e-5)
            if not np.isfinite(f.x).all():raise ValueError(f'Nonfinite solution at {start}')
            taper=np.ones(end-start)
            if start:taper[:min(overlap,len(taper))]=np.linspace(1/(overlap+1),1,min(overlap,len(taper)))
            if end<n and overlap:taper[-overlap:]=np.linspace(1,1/(overlap+1),overlap)
            x=np.zeros_like(initial[start:end])
            x[p.valid]=f.x.reshape(-1,3) if f.success else p.observed
            w=taper[:,None]*p.valid
            accum[start:end]+=x*w[...,None];total[start:end]+=w
            reports.append(dict(start=start,end=end,converged=bool(f.success),fallbackToInput=not bool(f.success),evaluations=f.nfev,cost=float(f.cost),termination=f.message,optimality=float(f.optimality)))
            print(f'Weighted fit {end}/{n}, evaluations={f.nfev}, converged={f.success}',flush=True)
        if end==n:break
    result=np.full_like(initial,np.nan);valid=total>0;result[valid]=accum[valid]/total[valid,None]
    return result,reports


def errors(xyz,xy,cams):
    result=np.full(xy.shape[:-1],np.nan);valid=np.isfinite(xyz).all(-1)
    for c in range(len(cams)):result[c][valid]=np.linalg.norm(project(xyz[valid],cams[c])-xy[c][valid],axis=-1)
    return result


def guard_reprojection(initial,candidate,xy,weights,cams,settings):
    """Check fixed observations AFTER overlap blending; roll back whole frames.

    A rollback preserves the input hand's bones instead of mixing individual
    joints from different fits. This checks 2D agreement, not 3D ground truth.
    """
    before=errors(initial,xy,cams);after=errors(candidate,xy,cams)
    valid=np.isfinite(initial).all(-1);result=candidate.copy()
    result[~valid]=np.nan
    fallback=np.zeros(len(initial),bool);checked=np.zeros(len(initial),bool);details=[]
    for t in range(len(initial)):
        reliable=(weights[:,t]>=settings['qualityWeightThreshold'])&np.isfinite(before[:,t])
        reasons=[];metrics={}
        if (valid[t]&~np.isfinite(candidate[t]).all(-1)).any():reasons.append('nonfinite_candidate')
        # The same input-defined observations are used on both sides, even
        # when the candidate has a very large error. Never reselect inliers.
        if reliable.sum()>=settings['qualityMinimumObservations'] and np.count_nonzero(reliable.any(1))>=2:
            checked[t]=True
            old=before[:,t][reliable];new=after[:,t][reliable]
            new=np.where(np.isfinite(new),new,np.inf)
            for q,label in ((50,'median'),(90,'p90')):
                b=float(np.percentile(old,q));a=float(np.percentile(new,q))
                limit=b+max(settings['qualityAbsoluteTolerancePixels'],settings['qualityRelativeTolerance']*b)
                metrics[label]=dict(before=b,candidate=a if np.isfinite(a) else None,limit=limit)
                if a>limit:reasons.append(label+'_regressed')
        for j in np.flatnonzero(reliable.sum(0)>=2):
            checked[t]=True;use=reliable[:,j]
            b=float(np.median(before[:,t,j][use]));a=after[:,t,j][use]
            a=float(np.median(np.where(np.isfinite(a),a,np.inf)))
            limit=b+max(settings['qualityJointAbsoluteTolerancePixels'],settings['qualityJointRelativeTolerance']*b)
            if a>limit:reasons.append(f'joint_{j}_regressed')
        if reasons:
            fallback[t]=True;result[t]=initial[t]
            details.append(dict(frame=t,reasons=reasons,metrics=metrics))
    report=dict(reference='Baseline constrained pose; fixed observations with weight >= 0.5',
        policy='After window blending, restore the whole baseline frame on material median/P90 or multi-view joint regression',
        checkedFrames=int(checked.sum()),uncheckedFrames=int((~checked).sum()),
        fallbackFrames=int(fallback.sum()),details=details,
        caveat='Passing preserves agreement within configured tolerances; it does not validate calibration or recover missing joints. Rollback boundaries may introduce motion discontinuities.')
    return result,fallback,checked,report


def run(session,limit=None):
    session=Path(session).resolve();root=session/'multicamera';out=root/'weighted_handpose';out.mkdir(exist_ok=True)
    backup=out/'baseline_pose_3d.npz'
    if not backup.exists():shutil.copy2(root/'handpose/pose_3d.npz',backup)
    src=np.load(backup);obs=np.load(out/('observations_smoke.npz' if limit else 'observations.npz'))
    n=len(obs['unixTimeMs']);xyz=src['joints'][:n].copy();times=(obs['unixTimeMs']-obs['unixTimeMs'][0])/1000
    if limit is not None and n!=limit:raise ValueError(f'Smoke observation cache has {n} frames, requested {limit}')
    np.testing.assert_array_equal(obs['unixTimeMs'],src['unixTimeMs'][:n])
    xy=obs['augment_xy'][:,:,0].copy();xy[obs['weight_image']==0]=np.nan
    cams=cameras(toml.load(root/'calibration.toml'));refs,refquality=leave_one_out(xy,cams)
    geoerr=np.linalg.norm(refs-xy,axis=-1)
    geometry=np.ones_like(geoerr);known=np.isfinite(geoerr);geometry[known]=np.maximum(.02,1/(1+(geoerr[known]/SETTINGS['geometryScalePixels'])**2))
    # Fixed weights independent of the fitted output: no residual feedback loop.
    weights=obs['weight_image']*geometry;lengths=src['boneLengthsMetres']
    candidate,windows=fit_windows(xyz,times,xy,weights,cams,lengths,SETTINGS)
    result,fallback,checked,quality=guard_reprojection(xyz,candidate,xy,weights,cams,SETTINGS)
    before=errors(xyz,xy,cams);after=errors(result,xy,cams);valid=np.isfinite(result).all(-1)
    assert np.array_equal(valid,np.isfinite(xyz).all(-1))
    support=((weights>=SETTINGS['reliabilityThreshold'])&(after<=15)).sum(0)
    report=dict(method='Fixed per-joint per-camera heuristic weights + robust calibrated 2D reprojection + bones + timestamp acceleration',
        settings=SETTINGS,frames=n,windows=windows,missingJointsFilled=0,
        objectiveLoss=dict(reprojection='Cauchy',anchors='soft-L1',bones='quadratic',acceleration='soft-L1'),
        qualityGuard=quality,
        baselinePoseSha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
        calibrationSha256=hashlib.sha256((root/'calibration.toml').read_bytes()).hexdigest(),
        observationsSha256=hashlib.sha256((out/('observations_smoke.npz' if limit else 'observations.npz')).read_bytes()).hexdigest(),
        before=diagnostics(xyz,times,lengths,SETTINGS['maxGapSeconds']),after=diagnostics(result,times,lengths,SETTINGS['maxGapSeconds']),
        correctionMm=describe(np.linalg.norm(result-xyz,axis=-1)*1000),
        highWeightReprojectionPixels=dict(before=describe(before[weights>=.5]),after=describe(after[weights>=.5])),
        perCameraReprojectionPixels={f'cam{c+1:02d}':dict(before=describe(before[c]),after=describe(after[c])) for c in range(4)},
        weights=describe(weights[obs['weight_image']>0]),leaveOneOutReferenceFraction=float(known.mean()),
        underTwoSupportedViewsFraction=float((support[valid]<2).mean()),
        caveats=['Heuristic weights are not calibrated joint confidence or visibility probabilities',
            'Mesh self-occlusion cannot detect external-object occlusion and can inherit model mistakes',
            'Crop stability can be consistently wrong; leave-one-out references require agreement of all other cameras',
            'Detector confidence is per hand, not per joint; target association uses other-view guide then original selection fallback',
            'Three-view geometric agreement is a proxy, not ground truth; weights are fixed before optimization',
            'Free-running camera exposure skew remains; rapid motion can conflict with a common pose timestamp',
            'One or zero supported views rely on baseline depth, bones and temporal priors',
            'Bone lengths inferred from this recording; no independent 3D ground truth'])
    np.savez_compressed(out/('pose_smoke.npz' if limit else 'pose_3d.npz'),joints=result,valid=valid,
        unixTimeMs=obs['unixTimeMs'],joint_names=src['joint_names'],units='metres',camerasUsed=support,
        perCameraReprojectionErrorPixels=after,reprojectionErrorPixels=np.nanmedian(after,axis=0),
        weights=weights,imageWeights=obs['weight_image'],geometryWeights=geometry,
        leaveOneOutErrorPixels=geoerr,leaveOneOutReferenceQualityPixels=refquality,boneLengthsMetres=lengths,
        qualityFallback=fallback,qualityChecked=checked,
        weightedReprojection=True,qualityMeaning='Heuristic reliability; support is not accuracy')
    np.savez_compressed(out/('optimization_candidate_smoke.npz' if limit else 'optimization_candidate.npz'),
        joints=candidate,unixTimeMs=obs['unixTimeMs'],qualityFallback=fallback,qualityChecked=checked)
    (out/('weighted_report_smoke.json' if limit else 'weighted_report.json')).write_text(json.dumps(report,indent=2,allow_nan=False))
    if not limit:
        pairs=[json.loads(s) for s in (root/'frame_sets.jsonl').read_text().splitlines()]
        with (out/'hand_pose_aligned.jsonl').open('w') as f:
            for i,pair in enumerate(pairs):
                row=dict(type='weighted_multicamera_hand_pose',sampleIndex=i,unixTimeMs=float(obs['unixTimeMs'][i]),
                    coordinateSpace='calibration world',units='metres',cameraSkewMs=pair['cameraSkewMs'],
                    qualityFallback=bool(fallback[i]),qualityChecked=bool(checked[i]),
                    joints=[dict(name=str(name),valid=bool(valid[i,j]),position=result[i,j].tolist() if valid[i,j] else None,
                        supportedViews=int(support[i,j]),priorDominated=bool(support[i,j]<2),
                        viewWeights=weights[:,i,j].tolist()) for j,name in enumerate(src['joint_names'])])
                f.write(json.dumps(row,allow_nan=False)+'\n')
        (out/'processing_status.json').write_text(json.dumps(dict(status='fit_complete',frames=n)))
    print(json.dumps({k:report[k] for k in ('highWeightReprojectionPixels','correctionMm','underTwoSupportedViewsFraction')},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('session');ap.add_argument('--limit',type=int);a=ap.parse_args();run(a.session,a.limit)
