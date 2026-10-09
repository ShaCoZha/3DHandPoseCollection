"""Resumeable WiLoR crop perturbation + mesh cache for the existing paired RGB frames."""
import argparse, fcntl, hashlib, json, os, time
from pathlib import Path
import cv2
import numpy as np
import toml
from hand_reliability import cameras,leave_one_out,perturbed_boxes,mesh_surface_visibility,reliability,fast_crop_gaussian
from visualize_multicamera import Reader,rows


def assigned_chunks(n, chunk_size, shard_index=0, shard_count=1):
    if chunk_size < 1 or shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('Invalid chunk size or shard assignment')
    return [start for i,start in enumerate(range(0,n,chunk_size)) if i % shard_count == shard_index]


def run(session,limit=None,chunk_size=64,shard_index=0,shard_count=1):
    # Legacy MANO pickle imports chumpy, which still references removed aliases.
    # Keep this compatibility shim local to the inference process.
    for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,
                       'object':object,'unicode':str,'str':str}.items():
        if name not in np.__dict__:setattr(np,name,value)
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    import wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline as pipeline_module
    # A process-local equivalent filter; never edits installed WiLoR code.
    pipeline_module.gaussian=fast_crop_gaussian
    session=Path(session).resolve();root=session/'multicamera';out=root/'weighted_handpose';out.mkdir(exist_ok=True)
    cache=out/('smoke_cache' if limit else 'cache');cache.mkdir(exist_ok=True)
    pairs=rows(root/'frame_sets.jsonl');pairs=pairs[:limit] if limit else pairs
    n=len(pairs);config=json.loads((root/'config.json').read_text());wanted=int(config['hand']=='right')
    detector_kwargs={}
    if 'detectorImageSize' in config:
        size=config['detectorImageSize']
        if type(size) is not int or not 64<=size<=2048 or size%32:
            raise ValueError('detectorImageSize must be an integer multiple of 32 between 64 and 2048')
        detector_kwargs['imgsz']=size
    original=np.load(root/'handpose/wilor_2d.npz');originalxy=original['keypoints'][:,:n,:,:2].copy()
    originalxy[original['keypoints'][:,:n,:,2]<.6]=np.nan
    cams=cameras(toml.load(root/'calibration.toml'));guide,_=leave_one_out(originalxy,cams)
    manifest=dict(version=2,frames=n,hand=config['hand'],perturbations=4,
        inputSha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'frame_sets.jsonl',root/'calibration.toml',root/'handpose/wilor_2d.npz']},
        identityGuide='Other-three-view-only agreement <=12px; fallback to original selected skeleton; no external identity labels',
        meshMeaning='Visible fraction of fixed anatomical skin patches selected on rest mesh; predicted virtual camera; self-occlusion hint only')
    if detector_kwargs:manifest['detectorImageSize']=detector_kwargs['imgsz']
    mp=cache/'manifest.json'
    with (cache/'manifest.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Cache provenance mismatch; use a new output directory')
        mp.write_text(json.dumps(manifest,indent=2))
    torch.set_num_threads(4);cv2.setNumThreads(2)
    model=WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'),dtype=torch.float16,verbose=False)
    faces=np.asarray(model.wilor_model.mano.faces,dtype=np.int32)
    if shard_index==0:np.save(out/'mano_faces.npy',faces)
    mano=model.wilor_model.mano
    rest=mano.v_template.detach().cpu().float().numpy()
    rest_joints=mano.J_regressor.detach().cpu().float().numpy()@rest
    rest_joints=np.concatenate([rest_joints,rest[mano.extra_joints_idxs.cpu().numpy()]])[mano.joint_map.cpu().numpy()]
    surface_indices=np.argsort(((rest_joints[:,None]-rest[None])**2).sum(-1),axis=1)[:,:16]
    if shard_index==0:np.save(out/'mano_joint_surface_indices.npy',surface_indices)
    readers=[Reader(root) for _ in range(4)];started=time.monotonic()
    try:
        for start in assigned_chunks(n,chunk_size,shard_index,shard_count):
            end=min(n,start+chunk_size);path=cache/f'{start:05d}.npz'
            if path.exists():continue
            count=end-start;shape=(4,count)
            d=dict(augment_xy=np.full((*shape,4,21,2),np.nan,np.float32),vertices=np.full((*shape,778,3),np.nan,np.float32),
                joints_local=np.full((*shape,21,3),np.nan,np.float32),camera_translation=np.full((*shape,3),np.nan,np.float32),
                focal=np.full(shape,np.nan,np.float32),visibility=np.full((*shape,21),np.nan,np.float32),
                boxes=np.full((*shape,4),np.nan,np.float32),detector_score=np.zeros(shape,np.float32),
                identity_source=np.zeros(shape,np.int8),candidate_count=np.zeros(shape,np.int8),selected_index=np.full(shape,-1,np.int8))
            candidates_log=[];params={}
            for f in range(start,end):
                for ci,spec in enumerate(config['cameras']):
                    k=f-start;image=cv2.cvtColor(readers[ci].read(pairs[f]['frames'][spec['name']]),cv2.COLOR_BGR2RGB)
                    det=model.hand_detector(image,conf=.3,verbose=False,**detector_kwargs)[0].boxes.data.cpu().numpy()
                    det=det[det[:,5].astype(int)==wanted];d['candidate_count'][ci,k]=len(det)
                    entry=dict(frame=f,camera=spec['name'],candidates=det.tolist(),selected=None);candidates_log.append(entry)
                    if not len(det):continue
                    guidepoints=guide[ci,f];valid=np.isfinite(guidepoints).all(1)
                    if valid.sum()>=5:target=np.median(guidepoints[valid],axis=0);source=1
                    else:
                        old=originalxy[ci,f];valid=np.isfinite(old).all(1)
                        target=np.median(old[valid],axis=0) if valid.any() else None;source=2
                    if target is not None:
                        centers=(det[:,:2]+det[:,2:4])/2;sides=np.max(det[:,2:4]-det[:,:2],axis=1)
                        distances=np.linalg.norm(centers-target,axis=1)/np.maximum(sides,30)
                        selected=int(np.argmin(distances))
                        if distances[selected]>1.5:
                            entry['rejected']='No candidate near target guide';continue
                    else:
                        selected=int(np.argmax(np.prod(det[:,2:4]-det[:,:2],axis=1)));source=3
                    entry['selected']=selected;entry['identitySource']=source
                    box=det[selected,:4];predictions=model.predict_with_bboxes(image,perturbed_boxes(box),np.full(4,wanted))
                    base=predictions[0]['wilor_preds'];d['augment_xy'][ci,k]=np.stack([p['wilor_preds']['pred_keypoints_2d'][0] for p in predictions])
                    v=base['pred_vertices'][0];j=base['pred_keypoints_3d'][0];tr=base['pred_cam_t_full'][0]
                    d['vertices'][ci,k]=v;d['joints_local'][ci,k]=j;d['camera_translation'][ci,k]=tr
                    d['focal'][ci,k]=base['scaled_focal_length'];d['visibility'][ci,k]=mesh_surface_visibility(v,j,tr,faces,surface_indices=surface_indices)
                    d['boxes'][ci,k]=box;d['detector_score'][ci,k]=det[selected,4]
                    d['identity_source'][ci,k]=source;d['selected_index'][ci,k]=selected
                    for name in ('global_orient','hand_pose','betas','pred_cam'):
                        if name not in params:params[name]=np.full((*shape,*base[name][0].shape),np.nan,np.float32)
                        params[name][ci,k]=base[name][0]
            d.update(params);d['sourceFrameIndex']=np.arange(start,end)
            temp=path.with_suffix('.pending.npz');np.savez_compressed(temp,**d);temp.replace(path)
            path.with_suffix('.json').write_text(json.dumps(candidates_log))
            elapsed=time.monotonic()-started
            state=out/('processing_status.json' if shard_count==1 else f'inference_worker_{shard_index}.json')
            state.write_text(json.dumps(dict(status='inference',completedFrames=end,totalFrames=n,elapsedSeconds=elapsed,
                shardIndex=shard_index,shardCount=shard_count,progressMeaning='Last completed frame index in this shard; not global progress')))
            print(f'Reliability inference {end}/{n}, {elapsed:.1f}s, shard {shard_index}/{shard_count}',flush=True)
    finally:
        for r in readers:r.close()
    if shard_count>1:
        (out/f'inference_worker_{shard_index}.json').write_text(json.dumps(dict(status='complete',shardIndex=shard_index,shardCount=shard_count)))
        return
    chunks=[np.load(cache/f'{s:05d}.npz') for s in range(0,n,chunk_size)]
    d={k:np.concatenate([p[k] for p in chunks],axis=1 if k!='sourceFrameIndex' else 0) for k in chunks[0].files}
    d['weight_image'],d['stability_pixels'],d['stability_weight']=reliability(d['augment_xy'],d['visibility'],d['detector_score'],(1920,1200))
    d['unixTimeMs']=np.array([p['unixTimeMs'] for p in pairs]);d['joint_names']=original['joint_names'];d['image_size']=np.array([1920,1200])
    np.savez_compressed(out/('observations_smoke.npz' if limit else 'observations.npz'),**d)
    (out/'inference_report.json').write_text(json.dumps(dict(**manifest,elapsedSeconds=time.monotonic()-started,
        detectionsPerCamera=(d['detector_score']>0).sum(1).tolist(),guideSelectedPerCamera=(d['identity_source']==1).sum(1).tolist()),indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('session');ap.add_argument('--limit',type=int)
    ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--shard-count',type=int,default=1)
    args=ap.parse_args();assigned_chunks(0,64,args.shard_index,args.shard_count)
    run(args.session,args.limit,shard_index=args.shard_index,shard_count=args.shard_count)
