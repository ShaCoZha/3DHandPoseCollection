"""Asynchronous weighted 3D trajectories with bounded, labelled gap recovery.

Uses recorded camera timestamps, without inventing an exposure-midpoint offset.
Piecewise-linear 3D knots are jointly fitted; 2D interpolation is only used for
initialization and independent geometric reliability, never as fitting data.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, vstack
from fit_joint_handpose import JointProblem, JOINT_SETTINGS, initialize, objective
from regularize_handpose import WindowProblem
from hand_reliability import project, leave_one_out

TRAJECTORY_SETTINGS = dict(JOINT_SETTINGS, trajectoryMaxGapSeconds=.15,
                           observationInterpolationMaxGapSeconds=.12)


def basis(times, valid, queries, max_gap, require_adjacent=False):
    """Bracketing valid knots, no extrapolation or interpolation across long gaps."""
    times=np.asarray(times);queries=np.asarray(queries)
    if np.any(np.diff(times)<=0):raise ValueError('Trajectory times must increase')
    lo=np.full((len(queries),valid.shape[1]),-1,int);hi=lo.copy();alpha=np.zeros(lo.shape)
    for j in range(valid.shape[1]):
        indices=np.flatnonzero(valid[:,j])
        if not len(indices):continue
        t=times[indices];right=np.searchsorted(t,queries)
        clipped=np.minimum(right,len(t)-1)
        exact=np.abs(t[clipped]-queries)<1e-8
        lo[exact,j]=hi[exact,j]=indices[clipped[exact]]
        good=~exact&(right>0)&(right<len(t))
        rows=np.flatnonzero(good);r=right[rows];span=t[r]-t[r-1]
        keep=span<=max_gap+1e-8
        if require_adjacent:keep &= indices[r]-indices[r-1]==1
        rows=rows[keep];r=right[rows]
        lo[rows,j]=indices[r-1];hi[rows,j]=indices[r]
        alpha[rows,j]=(queries[rows]-t[r-1])/(t[r]-t[r-1])
    return lo,hi,alpha


def interpolate(values,times,queries,max_gap,valid=None,require_adjacent=False):
    if valid is None:valid=np.isfinite(values).all(-1)
    lo,hi,a=basis(times,valid,queries,max_gap,require_adjacent)
    out=np.full((len(queries),values.shape[1],values.shape[2]),np.nan)
    r,j=np.where(lo>=0)
    out[r,j]=(1-a[r,j,None])*values[lo[r,j],j]+a[r,j,None]*values[hi[r,j],j]
    return out,(lo,hi,a)


def align_observations(xy,weights,view_times,queries,settings=TRAJECTORY_SETTINGS):
    aligned=[];confidence=[]
    for c in range(len(xy)):
        valid=np.isfinite(xy[c]).all(-1)&(weights[c]>0)
        uv,(lo,hi,_) = interpolate(xy[c],view_times[c],queries,
            settings['observationInterpolationMaxGapSeconds'],valid)
        w=np.zeros(uv.shape[:-1]);r,j=np.where(lo>=0)
        w[r,j]=np.minimum(weights[c,lo[r,j],j],weights[c,hi[r,j],j])
        aligned.append(uv);confidence.append(w)
    return np.array(aligned),np.array(confidence)


def prepare(xy,image_weights,cams,times,view_times,settings=TRAJECTORY_SETTINGS):
    # Each held-out view is assessed at ITS time, using the other views only.
    refs=np.full_like(xy,np.nan,dtype=float);refquality=np.full(xy.shape[:-1],np.nan)
    for held in range(len(cams)):
        uv,_=align_observations(xy,image_weights,view_times,view_times[held],settings)
        rr,qq=leave_one_out(uv,cams)
        refs[held]=rr[held];refquality[held]=qq[held]
    geoerr=np.linalg.norm(refs-xy,axis=-1);geometry=np.ones_like(geoerr)
    known=np.isfinite(geoerr)
    geometry[known]=np.maximum(.02,1/(1+(geoerr[known]/settings['geometryScalePixels'])**2))
    weights=image_weights*geometry
    aligned,aligned_w=align_observations(xy,weights,view_times,times,settings)
    def seeds_or_empty(points,confidence):
        try:return initialize(points,confidence,cams,settings)
        except ValueError as exc:
            if not str(exc).startswith('No reliable nondegenerate'):raise
            return np.full((*points.shape[1:-1],3),np.nan),np.full((*points.shape[1:-1],2),-1,dtype=np.int16)
    seed,pairs=seeds_or_empty(aligned,aligned_w)
    # Interpolation can lose a view at an endpoint or adjacent missed detection.
    # A valid native pair remains a numerical guess, not a fixed 3D target.
    native,native_pairs=seeds_or_empty(xy,weights)
    fallback=~np.isfinite(seed).all(-1)&np.isfinite(native).all(-1)
    seed[fallback]=native[fallback];pairs[fallback]=native_pairs[fallback]
    if not np.isfinite(seed).any():raise ValueError('No usable trajectory seeds')
    initial,inferred=fill_short_gaps(seed,times,settings['trajectoryMaxGapSeconds'])
    return initial,seed,inferred,pairs,weights,geometry,geoerr,refquality,aligned,aligned_w,fallback


def fill_short_gaps(seed,times,max_gap):
    initial,_=interpolate(seed,times,times,max_gap)
    inferred=np.isfinite(initial).all(-1)&~np.isfinite(seed).all(-1)
    return initial,inferred


class TrajectoryProblem(JointProblem):
    def __init__(self,xyz,times,xy,weights,cams,lengths,settings,view_times):
        super().__init__(xyz,times,xy,weights,cams,lengths,settings)
        self.obs=[];count=0;loss_weights=[]
        for c in range(len(cams)):
            lo,hi,a=basis(times,self.valid,view_times[c],settings['trajectoryMaxGapSeconds'],True)
            good=(lo>=0)&np.isfinite(xy[c]).all(-1)&(weights[c]>0)
            r,j=np.where(good);left=self.ids[lo[r,j],j];right=self.ids[hi[r,j],j]
            self.obs.append((left,right,a[r,j],xy[c,r,j],count,r,j))
            loss_weights.append(np.repeat(weights[c,r,j],2));count+=len(r)*2
        self.obsres=count;self.loss_weights=np.concatenate(loss_weights)
        self.pixel_sigma=settings['pixelSigma']

    def residual(self,flat):
        x=flat.reshape(-1,3);base=WindowProblem.residual(self,flat);extra=[]
        for c,(lo,hi,a,uv,_,_,_) in enumerate(self.obs):
            xyz=(1-a[:,None])*x[lo]+a[:,None]*x[hi]
            extra.append(((project(xyz,self.cams[c])-uv)/self.pixel_sigma).ravel())
        return np.concatenate([base,*extra])

    def jacobian(self,flat):
        x=flat.reshape(-1,3);base=WindowProblem.jacobian(self,flat);rr=[];cc=[];vv=[]
        for c,(lo,hi,a,uv,offset,_,_) in enumerate(self.obs):
            xyz=(1-a[:,None])*x[lo]+a[:,None]*x[hi]
            _,jac=project(xyz,self.cams[c],True);jac/=self.pixel_sigma
            for ids,factor in [(lo,1-a),(hi,a)]:
                rr.append(np.repeat(offset+np.arange(len(ids)*2),3))
                cc.append(np.broadcast_to(ids[:,None,None]*3+np.arange(3),(len(ids),2,3)).ravel())
                vv.append((jac*factor[:,None,None]).ravel())
        extra=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(self.obsres,x.size)).tocsr()
        return vstack([base,extra],format='csr')


def evaluate(xyz,times,view_times,xy,cams,settings=TRAJECTORY_SETTINGS):
    positions=[];errors=[]
    for c in range(len(cams)):
        x,_=interpolate(xyz,times,view_times[c],settings['trajectoryMaxGapSeconds'],require_adjacent=True)
        uv=np.full_like(xy[c],np.nan,dtype=float);valid=np.isfinite(x).all(-1)
        uv[valid]=project(x[valid],cams[c]);positions.append(x)
        errors.append(np.linalg.norm(uv-xy[c],axis=-1))
    return np.array(positions),np.array(errors)


def fit(initial,times,xy,weights,cams,lengths,view_times,settings=TRAJECTORY_SETTINGS):
    n=len(times);window=settings['windowFrames'];overlap=settings['overlapFrames']
    accum=np.zeros_like(initial);total=np.zeros(initial.shape[:-1]);reports=[]
    for start in range(0,n,window-overlap):
        end=min(n,start+window)
        p=TrajectoryProblem(initial[start:end],times[start:end],xy[:,start:end],weights[:,start:end],
                            cams,lengths,settings,view_times[:,start:end])
        if p.valid.any():
            f=least_squares(p.residual,p.observed.ravel(),jac=p.jacobian,loss=p.loss,
                x_scale='jac',max_nfev=settings['maxEvaluations'],ftol=1e-4,xtol=1e-5,gtol=1e-5)
            accepted=bool(f.success and np.isfinite(f.x).all())
            x=np.zeros_like(initial[start:end]);x[p.valid]=f.x.reshape(-1,3) if accepted else p.observed
            taper=np.ones(end-start)
            if start:taper[:min(overlap,len(taper))]=np.linspace(1/(overlap+1),1,min(overlap,len(taper)))
            if end<n:taper[-overlap:]=np.linspace(1,1/(overlap+1),overlap)
            w=taper[:,None]*p.valid;accum[start:end]+=x*w[...,None];total[start:end]+=w
            reports.append(dict(start=start,end=end,converged=bool(f.success),fallbackToInput=not accepted,evaluations=f.nfev,cost=float(f.cost)))
            print(f'Trajectory fit {end}/{n}, evaluations={f.nfev}, accepted={accepted}',flush=True)
        if end==n:break
    candidate=np.full_like(initial,np.nan);valid=total>0;candidate[valid]=accum[valid]/total[valid,None]
    # A few unconstrained knots can escape behind a camera. Reject those
    # coordinates explicitly instead of reverting the entire recording.
    output,quality=validate_trajectory(initial,candidate,times,xy,weights,cams,lengths,view_times,settings)
    return output,candidate,reports,quality


def validate_trajectory(initial,candidate,times,xy,weights,cams,lengths,view_times,settings=TRAJECTORY_SETTINGS):
    originally_valid=np.isfinite(initial).all(-1)
    admissible=np.isfinite(candidate).all(-1)
    for c in cams:admissible &= candidate@c['R'][2]+c['t'][2]>.05
    rejected=originally_valid&~admissible
    reference=initial.copy();reference[rejected]=np.nan
    checked=candidate.copy();checked[~np.isfinite(reference).all(-1)]=np.nan
    if not np.isfinite(reference).any():raise ValueError('No physically admissible trajectory knots')
    # Evaluate both candidates on exactly the SAME retained variables and
    # observations. Report removed knots; do not count them as improvements.
    p=TrajectoryProblem(reference,times,xy,weights,cams,lengths,settings,view_times)
    before=objective(p,reference);after=objective(p,checked)
    accepted=after['total']<=before['total']+max(1e-6,before['total']*1e-6)
    output=checked if accepted else reference
    quality=dict(accepted=bool(accepted),finite=True,positiveDepth=True,before=before,candidate=after,
                 rejectedKnots=int(rejected.sum()),rejectedIndices=np.argwhere(rejected).tolist(),
                 policy='Reject nonfinite/behind-camera knots, then compare the same asynchronous objective on retained support; no baseline pose rollback',
                 caveat='Rejections stay missing. Objective convergence is not a 3D accuracy or synchronization guarantee')
    return output,quality
