"""Render four RGB/pose views, a 3D skeleton, EIT frames and Watch IMU plots."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from export_pose_presentation import CleanRenderer, sha256
from visualize_multicamera import label, nearest, rows


def draw_eit(metadata, values, times, t):
    image=np.full((480,600,3),(35,28,23),np.uint8)
    label(image,'EIT measurement magnitude',(20,28),scale=.62)
    label(image,'Recorded matrix, not a reconstructed image',(20,52),scale=.43)
    i=nearest(times,t);matched=abs(times[i]-t)<=metadata['matchToleranceSeconds']
    if matched:
        frame=values[i];lo,hi=metadata['magnitudeRange']
        x=np.clip((frame-lo)/(hi-lo),0,1)
        colored=cv2.applyColorMap(np.uint8(np.nan_to_num(x)*255),cv2.COLORMAP_VIRIDIS)
        colored[~np.isfinite(frame)]=(96,96,96)
        image[76:396,150:470]=cv2.resize(colored,(320,320),interpolation=cv2.INTER_NEAREST)
        label(image,'Injection',(30,225),scale=.45)
        label(image,'Measurement index',(216,418),scale=.44)
        label(image,f'Frame {metadata["frameIndices"][i]} | missing cells: {metadata["missingCells"][i]}',(20,444),scale=.45)
    else:
        label(image,'No EIT frame at this time',(125,236),(130,160,255),.66)
    state='EIT clock alignment unverified' if metadata['clockStatus']=='unverified' else 'EIT clock: '+metadata['clockStatus']
    label(image,state,(20,469),(130,180,255),.43)
    return image,matched


def imu_plot(samples,t,column,title,limit):
    image=np.full((200,960,3),(35,28,23),np.uint8)
    left=max(0,t-5);right=left+10
    start=np.searchsorted(samples[:,0],left);end=np.searchsorted(samples[:,0],right,side='right')
    segment=samples[start:end];width=890;height=125
    label(image,title,(14,24),scale=.58)
    for k,color in enumerate(((123,148,255),(160,213,119),(255,186,137))):
        label(image,'XYZ'[k],(650+k*75,24),color,.52)
    label(image,f'+/- {limit:.2f}',(14,47),scale=.4)
    cv2.line(image,(55,112),(945,112),(80,70,60),1)
    if len(segment):
        x=55+(segment[:,0]-left)/10*width
        cuts=np.r_[0,np.flatnonzero(np.diff(segment[:,0])>.05)+1,len(segment)]
        for k,color in enumerate(((123,148,255),(160,213,119),(255,186,137))):
            y=112-np.clip(segment[:,column+k]/limit,-1,1)*height/2
            uv=np.column_stack([x,y]).astype(np.int32)
            for a,b in zip(cuts,cuts[1:]):
                if b-a>1:cv2.polylines(image,[uv[a:b]],False,color,1,cv2.LINE_AA)
    marker=round(55+(t-left)/10*width)
    cv2.line(image,(marker,49),(marker,176),(245,245,245),1)
    for k in range(6):label(image,f'{left+2*k:.0f}s',(45+178*k,195),scale=.37)
    return image


def run(session,method='weighted',diagnostic=False):
    session=Path(session).resolve();root=session/'multicamera';out=session/'visualization'
    capture_manifest=session/'capture.json'
    if capture_manifest.exists():
        diagnostic=diagnostic or bool(json.loads(capture_manifest.read_text()).get('diagnosticCandidate'))
    if method=='weighted':out=out/'weighted'
    pose=root/('weighted_handpose' if method=='weighted' else 'handpose')/'pose_3d.npz'
    with np.load(pose) as p:
        xyz=p['joints'].copy();stamps=p['unixTimeMs'].copy();support=p['camerasUsed'].copy()
    capture=json.loads((session/'timestamp.json').read_text());origin=capture['startedAtUnixMs']
    duration=(capture['stoppedAtUnixMs']-origin)/1000;times=(stamps-origin)/1000
    pairs=rows(root/'frame_sets.jsonl')
    np.testing.assert_allclose(stamps,[p['unixTimeMs'] for p in pairs],rtol=0,atol=.001)
    points=xyz[np.isfinite(xyz).all(-1)];low,high=np.percentile(points,[.5,99.5],axis=0)
    center=(low+high)/2;radius=max(.12,float(((high-low)/2).max())*1.25)
    metadata=json.loads((session/'visualization/eit/metadata.json').read_text())
    eit=np.fromfile(session/'visualization/eit/magnitude.f32',dtype='<f4').reshape(-1,16,16)
    eit_times=np.array(metadata['rawRelativeSeconds'])+metadata['offsetMs']/1000
    if len(eit)!=len(eit_times):raise ValueError('EIT frame count mismatch')
    imu=np.array([[(r['unixTimeMs']-origin)/1000]+[r[k][a] for k in ('userAcceleration','rotationRate') for a in 'xyz'] for r in rows(session/'imudata_aligned.jsonl')])
    inside=(imu[:,0]>=0)&(imu[:,0]<=duration)
    limits=[max(.1,float(np.max(np.abs(imu[inside,c:c+3])))*1.05) for c in (1,4)]
    renderer=CleanRenderer(root,pairs,xyz,origin,method=method);fps=20;count=math.ceil(duration*fps)
    pending=out/'rgb_pose_imu_eit.pending.mp4';output=out/'rgb_pose_imu_eit.mp4'
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1280','-r','20','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(pending)],stdin=subprocess.PIPE)
    cv2.setNumThreads(2);matched_eit=matched_pose=0
    try:
        for f in range(count):
            t=f/fps;pi=nearest(times,t);paired=abs(times[pi]-t)<=.026;matched_pose+=int(paired)
            canvas=np.full((1280,1920,3),(30,21,16),np.uint8)
            canvas[:1080]=renderer.draw(t,pi,paired,center,radius,support[pi])
            if diagnostic:
                label(canvas,'Unvalidated 3D: candidate camera calibration',(980,38),(130,180,255),.64)
            heatmap,matched=draw_eit(metadata,eit,eit_times,t);matched_eit+=int(matched)
            canvas[544:1024,1300:1900]=heatmap
            canvas[1080:1280,:960]=imu_plot(imu,t,1,'Watch acceleration (m/s^2)',limits[0])
            canvas[1080:1280,960:]=imu_plot(imu,t,4,'Watch angular velocity (rad/s)',limits[1])
            encoder.stdin.write(canvas.tobytes())
            if f in {int(x*fps) for x in (5,25,50,100,150,250,300)}:
                cv2.imwrite(str(out/f'multimodal_preview_{round(t):03d}s.jpg'),canvas)
            if f%200==0:print(f'Multimodal video {f}/{count}',flush=True)
    except BaseException:
        encoder.stdin.close();encoder.wait();pending.unlink(missing_ok=True);raise
    finally:renderer.close()
    encoder.stdin.close()
    if encoder.wait():raise RuntimeError('Multimodal video encoding failed')
    pending.replace(output)
    report=dict(session=session.name,frames=count,fps=fps,durationSeconds=count/fps,resolution=[1920,1280],
        poseSha256=sha256(pose),framesWithMatchedPose=matched_pose,framesWithEit=matched_eit,
        eitClockStatus=metadata['clockStatus'],eitOffsetMs=metadata['offsetMs'],eitGaps=metadata['gaps'],
        imuSamplesInVideo=int(inside.sum()),imuSamplesTotal=len(imu),missingSamplesInterpolated=False,
        poseMethod=method,unvalidated3d=diagnostic)
    (out/'multimodal_video_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path)
    p.add_argument('--method',choices=['weighted','wilor'],default='weighted')
    p.add_argument('--diagnostic',action='store_true');a=p.parse_args();run(a.session,a.method,a.diagnostic)
