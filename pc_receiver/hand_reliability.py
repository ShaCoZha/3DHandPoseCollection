"""Heuristic joint reliability and calibrated multiview geometry (no learned confidence)."""
import itertools
import cv2
import numpy as np


def fast_crop_gaussian(image,sigma,channel_axis=2,preserve_range=True):
    """Equivalent to the pipeline's skimage Gaussian defaults, using OpenCV.

    Same float64 input conversion, 4-sigma truncation and nearest-edge extension.
    Restricted signature intentionally prevents silent use with other settings.
    """
    if image.dtype!=np.uint8 or image.ndim!=3 or channel_axis!=2 or not preserve_range:
        raise ValueError('Unexpected WiLoR antialias input')
    radius=int(4*sigma+.5)
    return cv2.GaussianBlur(image.astype(np.float64),(2*radius+1,2*radius+1),
        sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REPLICATE)


def cameras(calibration):
    result=[]
    for i in range(4):
        c=calibration[f'cam_{i}'];r=cv2.Rodrigues(np.array(c['rotation']))[0]
        result.append(dict(R=r,t=np.array(c['translation']),K=np.array(c['matrix']),d=np.array(c['distortions']),P=np.column_stack([r,c['translation']])))
    return result


def project(xyz, c, jacobian=False):
    """OpenCV distortion model; derivative wrt world xyz via translation columns."""
    x=np.asarray(xyz).reshape(-1,3)
    if not len(x):return (np.empty((0,2)),np.empty((0,2,3))) if jacobian else np.empty((0,2))
    uv,j=cv2.projectPoints(x,cv2.Rodrigues(c['R'])[0],c['t'],c['K'],c['d'])
    if jacobian:return uv[:,0],j[:,3:6].reshape(-1,2,3)@c['R']
    return uv[:,0]


def triangulate_pair(a,b,ca,cb):
    shape=a.shape[:-1];aa=a.reshape(-1,2);bb=b.reshape(-1,2)
    good=np.isfinite(aa).all(1)&np.isfinite(bb).all(1);out=np.full((len(aa),3),np.nan)
    if good.any():
        u=cv2.undistortPoints(aa[good,None],ca['K'],ca['d'])[:,0]
        v=cv2.undistortPoints(bb[good,None],cb['K'],cb['d'])[:,0]
        h=cv2.triangulatePoints(ca['P'],cb['P'],u.T,v.T).T
        ok=np.abs(h[:,3])>1e-10
        val=np.full((len(h),3),np.nan);val[ok]=h[ok,:3]/h[ok,3,None]
        deptha=val@ca['R'][2]+ca['t'][2];depthb=val@cb['R'][2]+cb['t'][2]
        val[(deptha<=.05)|(depthb<=.05)]=np.nan;out[good]=val
    return out.reshape(*shape,3)


def leave_one_out(xy, cams):
    """Predict each view ONLY from the other three. Require all 3 to agree.

    Unknown/inconsistent references remain NaN, not evidence of occlusion.
    """
    refs=np.full(xy.shape,np.nan,dtype=np.float64);quality=np.full(xy.shape[:-1],np.nan)
    shape=xy.shape[1:-1]
    for held in range(4):
        others=[c for c in range(4) if c!=held];candidates=[];costs=[]
        for a,b in itertools.combinations(others,2):
            x=triangulate_pair(xy[a],xy[b],cams[a],cams[b]);valid=np.isfinite(x).all(-1)
            err=np.full((3,*shape),np.inf)
            for k,c in enumerate(others):
                pred=np.full(xy[c].shape,np.nan,dtype=np.float64);pred[valid]=project(x[valid],cams[c])
                e=np.linalg.norm(pred-xy[c],axis=-1);err[k]=np.where(np.isfinite(e),e,np.inf)
            costs.append(np.max(err,axis=0));candidates.append(x)
        costs=np.array(costs);best=np.argmin(costs,axis=0)
        flat=np.array(candidates).reshape(3,-1,3);idx=best.ravel()
        x=flat[idx,np.arange(len(idx))].reshape(*shape,3)
        q=np.min(costs,axis=0);depth=x@cams[held]['R'][2]+cams[held]['t'][2]
        valid=(q<=12)&np.isfinite(x).all(-1)&(depth>.05)
        refs[held][valid]=project(x[valid],cams[held]);quality[held][valid]=q[valid]
    return refs,quality


def perturbed_boxes(box):
    box=np.asarray(box);center=(box[:2]+box[2:])/2;size=box[2:]-box[:2];s=max(size)
    result=[box]
    for dx,dy,scale in ((.035,0,1.04),(-.035,.025,.96),(0,-.035,1.02)):
        c=center+np.array([dx,dy])*s;result.append(np.r_[c-size*scale/2,c+size*scale/2])
    return np.array(result,dtype=np.float32)


def mesh_surface_visibility(vertices,joints,translation,faces,samples=16,surface_indices=None):
    """Visible fraction of nearest skin vertices around each joint.

    Ray casting uses WiLoR's virtual camera, not rig extrinsics. Internal skeleton
    centres are NOT tested against skin depth. This cannot see external objects.
    """
    v=np.asarray(vertices,dtype=float)+np.asarray(translation).reshape(1,3)
    j=np.asarray(joints,dtype=float)+np.asarray(translation).reshape(1,3)
    # Fixed anatomical patches from the rest mesh avoid jumping to a different
    # finger when two fingers touch or overlap in the posed mesh.
    nearest=(np.asarray(surface_indices) if surface_indices is not None else
             np.argsort(((j[:,None]-v[None])**2).sum(-1),axis=1)[:,:samples])
    samples=nearest.shape[1]
    points=v[nearest].reshape(-1,3)
    if not np.isfinite(v).all() or (v[:,2]<=0).any():return np.full(21,np.nan)
    tri=v[faces];uv=tri[...,:2]/tri[...,2,None];p=points[:,:2]/points[:,2,None]
    a,b,c=uv[:,0],uv[:,1],uv[:,2]
    den=(b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
    good=np.abs(den)>1e-14;den=np.where(good,den,np.nan)
    visible=[]
    for pp,depth in zip(np.array_split(p,4),np.array_split(points[:,2],4)):
        xx=pp[:,None,0]-c[None,:,0];yy=pp[:,None,1]-c[None,:,1]
        w0=((b[:,1]-c[:,1])*xx+(c[:,0]-b[:,0])*yy)/den
        w1=((c[:,1]-a[:,1])*xx+(a[:,0]-c[:,0])*yy)/den;w2=1-w0-w1
        inside=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)&good
        inv=w0/tri[:,0,2]+w1/tri[:,1,2]+w2/tri[:,2,2]
        z=np.divide(1.,inv,out=np.full_like(inv,np.inf),where=inside&(inv>0))
        surface=z.min(1);visible.append(depth<=surface+.001)
    return np.concatenate(visible).reshape(len(j),samples).mean(1)


def reliability(augment_xy,visibility,detector_score,image_size):
    # Stability is normalized to the hand extent, with a pixel floor.
    xy=augment_xy[...,0,:,:]
    deviation=np.sqrt(np.mean(np.sum((augment_xy-xy[...,None,:,:])**2,axis=-1),axis=-2))
    span=np.nanmax(xy,axis=-2)-np.nanmin(xy,axis=-2)
    scale=np.maximum(3.,np.max(span,axis=-1)*.025)
    stability=1/(1+(deviation/scale[...,None])**2)
    mesh=.35+.65*np.clip(np.nan_to_num(visibility,nan=.45)/.45,0,1)
    weight=stability*mesh*np.clip(detector_score[...,None],.3,1)
    inside=np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<image_size[0])&(xy[...,1]>=0)&(xy[...,1]<image_size[1])
    weight=np.where(inside,weight,0)
    return weight,deviation,stability
