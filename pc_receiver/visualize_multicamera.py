"""Export timestamp-based RGB/pose review video and an offline IMU/3D viewer."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import toml

CHAINS = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def label(image, value, pos, color=(245,245,245), scale=.6):
    cv2.putText(image, value, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(image, value, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def nearest(times, value):
    i = int(np.searchsorted(times,value))
    return min({max(0,i-1),min(len(times)-1,i)},key=lambda k:abs(times[k]-value))


class Reader:
    def __init__(self, root):
        self.root, self.path, self.cap, self.position = root, None, None, 0
        self.last = None
        self.last_index = None

    def read(self, row):
        path = self.root/row['video']; target=row['videoFrameIndex']
        if path != self.path:
            self.close();self.cap=cv2.VideoCapture(str(path));self.path=path;self.position=0;self.last_index=None
        if self.last_index == target:
            return self.last.copy()
        if target < self.position:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,target);self.position=target
        while self.position <= target:
            ok, self.last = self.cap.read()
            if not ok:raise ValueError(f'Cannot decode {path} frame {target}')
            self.last_index=self.position;self.position+=1
        return self.last.copy()

    def close(self):
        if self.cap is not None:self.cap.release()


def skeleton(image, xy, valid, color, radius=3):
    valid=valid & np.isfinite(xy).all(1) & (np.abs(xy).max(1)<10000)
    for chain in CHAINS:
        for a,b in zip(chain,chain[1:]):
            if valid[a] and valid[b]:cv2.line(image,tuple(xy[a].astype(int)),tuple(xy[b].astype(int)),color,2,cv2.LINE_AA)
    for point in xy[valid]:cv2.circle(image,tuple(point.astype(int)),radius,color,-1,cv2.LINE_AA)


def clean(array, digits=4):
    array=np.round(np.asarray(array),digits)
    return np.where(np.isfinite(array),array,None).tolist()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('session',type=Path)
    ap.add_argument('--method',choices=('wilor','poem','weighted'),default='wilor');args=ap.parse_args()
    session=args.session.resolve();root=session/'multicamera';out=session/'visualization'
    if args.method in ('poem','weighted'):out=out/args.method
    out.mkdir(parents=True,exist_ok=True)
    pose_dir={'poem':'poem','weighted':'weighted_handpose','wilor':'handpose'}[args.method]
    method_label={'poem':'POEM OakInk','weighted':'WiLoR + weighted 2D fit','wilor':'WiLoR + Anipose'}[args.method]
    # NPZ access decompresses on every key lookup; cache small pose arrays once
    # before rendering thousands of frames and per-joint weight dots.
    with np.load(root/pose_dir/'pose_3d.npz') as saved:p={k:saved[k] for k in saved.files}
    with np.load(root/'handpose/wilor_2d.npz') as saved:w={k:saved[k] for k in saved.files}
    weighted=args.method=='weighted'
    if weighted:
        obs=np.load(root/pose_dir/'observations.npz')
        w=dict(keypoints=np.concatenate([obs['augment_xy'][:,:,0],(obs['weight_image']>0)[...,None]],axis=-1))
        weighted_report=json.loads((root/pose_dir/'weighted_report.json').read_text())
    constrained=args.method=='wilor' and 'regularized' in p and bool(p['regularized'])
    regularization=json.loads((root/'handpose/regularization_report.json').read_text()) if constrained else None
    if constrained:method_label+=' + constraints'
    pairs=rows(root/'frame_sets.jsonl');config=json.loads((root/'config.json').read_text())
    capture=json.loads((session/'capture.json').read_text());meta=json.loads((session/'timestamp.json').read_text())
    calibration_meta=json.loads((root/'metadata.json').read_text())
    imu_path=session/'imudata_aligned.jsonl'
    if not imu_path.exists():
        from align_capture import align_file,calibration_points
        clocks=rows(session/'clock_sync.jsonl')
        align_file(session/'imudata.jsonl',imu_path,calibration_points(clocks,'watch'),capture['phoneEpochOffsetMs'])
    imu=rows(imu_path)
    origin=float(meta['startedAtUnixMs']);duration=(meta['stoppedAtUnixMs']-origin)/1000
    times=(p['unixTimeMs']-origin)/1000
    assert len(pairs)==len(times)==w['keypoints'].shape[1]
    cal=toml.load(root/'calibration.toml');cams=[cal[f'cam_{i}'] for i in range(4)]
    source_rows=[rows(root/f'{c["name"]}_frames_aligned.jsonl') for c in cams]
    source_times=[np.array([(r['unixTimeMs']-origin)/1000 for r in rr]) for rr in source_rows]
    reprojections=np.full((4,len(times),21,2),np.nan)
    for ci,c in enumerate(cams):
        xyz=p['joints'].reshape(-1,3);valid=np.isfinite(xyz).all(1)
        if valid.any():
            reprojections[ci].reshape(-1,2)[valid]=cv2.projectPoints(xyz[valid],np.array(c['rotation']),np.array(c['translation']),np.array(c['matrix']),np.array(c['distortions']))[0].reshape(-1,2)
    residuals=np.linalg.norm(reprojections-w['keypoints'][...,:2],axis=-1)
    residuals[w['keypoints'][...,2]<.6]=np.nan
    support=(residuals<=10).sum(axis=0)
    accepted=p['valid']
    audit=dict(method='Reproject finite 3D joints into every camera and compare with saved, in-image WiLoR observations; consistency, not ground-truth accuracy. Original in-box flags are not confidence scores.',
        calibration=calibration_meta['calibrationSource'],supportThresholdPixels=10,
        acceptedJoints=int(accepted.sum()),
        fractionAcceptedWithAtLeastThreeViewsWithin10px=float((support[accepted]>=3).mean()) if accepted.any() else None,
        fractionAcceptedWithAllFourViewsWithin10px=float((support[accepted]==4).mean()) if accepted.any() else None,
        perCamera={c['name']:dict(medianPixels=float(np.nanmedian(residuals[ci])),
                    p90Pixels=float(np.nanpercentile(residuals[ci],90)))
                    for ci,c in enumerate(cams) if np.isfinite(residuals[ci]).any()})
    (out/'all_view_reprojection_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    fps=20;count=math.ceil(duration*fps);cv2.setNumThreads(2)
    enc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x856',
        '-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','21','-pix_fmt','yuv420p',
        '-movflags','+faststart',str(out/'rgb_pose_review.pending.mp4')],stdin=subprocess.PIPE)
    readers=[Reader(root) for _ in range(4)]
    snapshot_frames={round(t*fps) for t in (5,25,50,80)}
    try:
        for frame in range(count):
            t=frame/fps;pi=nearest(times,t);paired=abs(times[pi]-t)<=.026
            canvas=np.full((856,1280,3),20,np.uint8)
            valid_count=int(p['valid'][pi].sum()) if paired else 0
            label(canvas,f'{method_label} | {t:6.2f} / {duration:.2f} s | reconstructed joints: {valid_count}/21',(16,23),scale=.63)
            legend='Orange: WiLoR ROI reference   Cyan: POEM 3D (unfiltered)' if args.method=='poem' else ('Orange: WiLoR 2D   Cyan: bone + temporal constrained 3D' if constrained else 'Orange: selected WiLoR 2D   Cyan: 3D reprojection (subset filter)')
            if weighted:legend='2D dots: red low / green high weight | Cyan: fitted 3D | Yellow: <2 supported views'
            label(canvas,legend+'   Inset: full view',(16,46),scale=.51)
            for ci,c in enumerate(cams):
                if paired:
                    record=pairs[pi]['frames'][c['name']]
                else:
                    record=source_rows[ci][nearest(source_times[ci],t)]
                image=readers[ci].read(record)
                xy=w['keypoints'][ci,pi,:,:2] if paired else np.full((21,2),np.nan)
                good=np.isfinite(xy).all(1)
                if good.any():
                    low,high=np.nanmin(xy,axis=0),np.nanmax(xy,axis=0)
                    center=(low+high)/2;side=max(380,float((high-low).max())*1.9)
                else:
                    center=np.array([960,600]);side=1920
                scale=640/side
                affine=np.array([[scale,0,320-scale*center[0]],[0,scale,200-scale*center[1]]])
                tile=cv2.warpAffine(image,affine,(640,400))
                if paired:
                    skeleton(tile,cv2.transform(xy[None],affine)[0],good,(30,170,255))
                    skeleton(tile,cv2.transform(reprojections[ci,pi][None],affine)[0],p['valid'][pi],(255,245,0),5)
                    if weighted:
                        fitted=cv2.transform(reprojections[ci,pi][None],affine)[0]
                        for j in range(21):
                            if p['valid'][pi,j] and p['camerasUsed'][pi,j]<2 and np.isfinite(fitted[j]).all() and np.abs(fitted[j]).max()<10000:
                                cv2.circle(tile,tuple(fitted[j].astype(int)),6,(0,220,255),1,cv2.LINE_AA)
                        dots=cv2.transform(xy[None],affine)[0]
                        for j in range(21):
                            if good[j] and np.abs(dots[j]).max()<10000:
                                weight=float(p['weights'][ci,pi,j]);color=(60,int(70+175*weight),int(245-170*weight))
                                cv2.circle(tile,tuple(dots[j].astype(int)),3,color,-1,cv2.LINE_AA)
                inset=cv2.resize(image,(160,100));tile[292:392,472:632]=inset
                cv2.rectangle(tile,(471,291),(632,392),(200,200,200),1)
                delta=record['unixTimeMs']-origin-t*1000
                label(tile,f'{c["name"]} | frame {record["sampleIndex"]} | dt {delta:+.1f} ms',(12,25),scale=.53)
                if not paired:label(tile,'NO MATCHED POSE AT THIS TIME',(12,54),(90,130,255),.57)
                elif not good.any():label(tile,'NO RIGHT-HAND DETECTION',(12,54),(90,130,255),.57)
                if weighted and paired:label(tile,f'Median joint weight {np.median(p["weights"][ci,pi]):.2f} (heuristic)',(12,78),scale=.48)
                y=56+(ci//2)*400;x=(ci%2)*640;canvas[y:y+400,x:x+640]=tile
            enc.stdin.write(canvas.tobytes())
            if frame in snapshot_frames:cv2.imwrite(str(out/f'preview_{round(t):03d}s.jpg'),canvas)
            if frame%200==0:print(f'Video {frame}/{count}',flush=True)
    finally:
        enc.stdin.close();code=enc.wait()
        for reader in readers:reader.close()
    if code:raise RuntimeError(f'ffmpeg exited {code}')
    (out/'rgb_pose_review.pending.mp4').replace(out/'rgb_pose_review.mp4')
    data=dict(session=session.name,duration=duration,origin=origin,times=clean(times,5),
        poseMethod=args.method,methodLabel=method_label,
        regularization=({k:regularization[k] for k in ('method','settings','grossOutliersRejected','gapsFilled','beforeOnKeptJoints','after','originalSupportedViewsReprojectionPixels')} if constrained else None),
        calibrationLabel=Path(calibration_meta['calibrationSource']).parent.parent.name,
        watchRttMs=[r['rttMs'] for r in rows(session/'clock_sync.jsonl') if r.get('device')=='watch'],
        joints=clean(p['joints'],5),counts=p['valid'].sum(1).tolist(),chains=CHAINS,
        skew=[round(row['cameraSkewMs'],2) for row in pairs],
        imu=[[(row['unixTimeMs']-origin)/1000]+[row[key][axis] for key in ('userAcceleration','rotationRate') for axis in 'xyz'] for row in imu],
        validFraction=float(p['valid'].mean()),names=p['joint_names'].tolist(),
        geometryAudit=audit,threeViewCounts=((support>=3)&accepted).sum(1).tolist())
    if weighted:
        data['weightedReport']=weighted_report
        data['jointViewSupport']=p['camerasUsed'].tolist()
        data['viewWeights']=clean(p['weights'],3)
        data['reliabilityDetails']=dict(stabilityPixels=clean(obs['stability_pixels'],2),
            meshSurfaceVisibility=clean(obs['visibility'],2),geometryWeights=clean(p['geometryWeights'],3))
    display_points=p['joints'][accepted]
    if len(display_points):
        low,high=np.percentile(display_points,[.5,99.5],axis=0)
        data['displayCenter']=((low+high)/2).tolist()
        data['displayRadius']=max(.12,float(((high-low)/2).max())*1.25)
        data['displayMode']='fixed center and scale for the whole recording'
    data['imu']=clean(data['imu'],5)
    stabilization_path=root/'handpose/stabilization_comparison.json'
    if constrained and stabilization_path.exists():
        candidate=json.loads(stabilization_path.read_text())
        if candidate.get('outputPoseSha256')==hashlib.sha256((root/pose_dir/'pose_3d.npz').read_bytes()).hexdigest():
            data['stabilization']=candidate
    data['constraintsComparisonAvailable'] = (out/'constraints/comparison.html').is_file()
    template=Path(__file__).with_name('multicamera_viewer.template.html').read_text()
    rendered=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    revision=calibration_meta['calibrationSha256'][:12]+'-'+hashlib.sha256((root/pose_dir/'pose_3d.npz').read_bytes()).hexdigest()[:12]
    rendered=rendered.replace('__REVISION__',revision)
    if args.method=='wilor' and (root/'weighted_handpose/pose_3d.npz').exists():
        nav='<p id="weightedLatest"><a href="weighted/comparison.html">Before / After Per-Joint Weighted Fit</a> · <a href="weighted/viewer.html?t=50">Weighted 3D Replay and Per-View Weights</a> · <a href="weighted/review.html">Raw-Image Review</a></p>'
        rendered=rendered.replace('</h1>','</h1>'+nav,1)
    (out/'viewer.pending.html').write_text(rendered)
    (out/'viewer.pending.html').replace(out/'viewer.html')
    summary=dict(videoFrames=count,fps=fps,durationSeconds=duration,poseFrames=len(times),method=method_label,
                 validJointFraction=float(p['valid'].mean()),completePoseFrames=int(p['valid'].all(1).sum()),
                 maxValidJointsPerFrame=int(p['valid'].sum(1).max()),imuSamples=len(imu),
                 calibration=calibration_meta['calibrationSource'],geometryAudit=audit,
                 revision=revision,regularized=constrained,
                 note=('Heuristic per-joint weights; yellow marks prior-dominated depth. Independent baseline preserved.' if weighted else ('POEM finite model outputs shown without confidence filtering; orange WiLoR is a crop/reference source, not ground truth.' if args.method=='poem' else 'Visualizes existing outputs without changing detections, calibration or rejected joints. Missing 3D is not interpolated.')))
    (out/'visualization_report.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
