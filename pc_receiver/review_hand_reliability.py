"""Create fixed-time raw/overlay audit sheets and a synchronized before/after page."""
import argparse,hashlib,html,json,shutil
from pathlib import Path
import cv2
import numpy as np
from visualize_multicamera import Reader,rows,skeleton,label
from regularize_handpose import describe

STYLE='body{background:#10151e;color:#e5edf8;font:15px system-ui;max-width:1500px;margin:24px auto;padding:16px}a{color:#78d8e6}p{line-height:1.6}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}img,video{width:100%;display:block}section{background:#192331;padding:14px;border-radius:9px}button,input,select{margin:5px;padding:8px;background:#263446;color:white;border:1px solid #455a73}input[type=range]{width:60%}table{font-size:13px}td,th{padding:4px 10px}@media(max-width:850px){.grid{grid-template-columns:1fr}}'


def run(session):
    session=Path(session).resolve();root=session/'multicamera';src=root/'weighted_handpose';out=session/'visualization/weighted';out.mkdir(exist_ok=True,parents=True)
    obs=np.load(src/'observations.npz');p=np.load(src/'pose_3d.npz');pairs=rows(root/'frame_sets.jsonl');meta=json.loads((session/'timestamp.json').read_text())
    origin=meta['startedAtUnixMs'];times=(p['unixTimeMs']-origin)/1000
    targets=[t for t in (5,25,50,80) if times[0]<=t<=times[-1]]
    if not targets:targets=[int(round(float(times[len(times)//2])))]
    selection='Fixed times within the recording: '+', '.join(map(str,targets))+' seconds; all four cameras; index fingertip; not selected using weights'
    readers=[Reader(root) for _ in range(4)];samples=[];cards=[]
    annotations_path=out/'review_annotations.json'
    annotations=json.loads(annotations_path.read_text()) if annotations_path.exists() else {}
    try:
        for target in targets:
            f=int(np.argmin(np.abs(times-target)));sheet=np.full((4*336,1100,3),20,np.uint8)
            for ci in range(4):
                image=readers[ci].read(pairs[f]['frames'][f'cam{ci+1:02d}'])
                xy=obs['augment_xy'][ci,f,0];box=obs['boxes'][ci,f]
                if np.isfinite(box).all():center=(box[:2]+box[2:])/2;side=max(300,np.max(box[2:]-box[:2])*1.5)
                else:center=np.array([960,600]);side=1920
                s=500/side;affine=np.array([[s,0,250-s*center[0]],[0,s,150-s*center[1]]])
                crop=cv2.warpAffine(image,affine,(500,300));overlay=crop.copy();uv=cv2.transform(xy[None],affine)[0];valid=np.isfinite(xy).all(1)
                skeleton(overlay,uv,valid,(100,100,100),2)
                for j in np.flatnonzero(valid):
                    if np.abs(uv[j]).max()>10000:continue
                    w=p['weights'][ci,f,j];cv2.circle(overlay,tuple(uv[j].astype(int)),4,(60,int(70+175*w),int(245-170*w)),-1,cv2.LINE_AA)
                    label(overlay,str(j),tuple((uv[j]+[3,-3]).astype(int)),scale=.32)
                row=ci*336;sheet[row+30:row+330,:500]=crop;sheet[row+30:row+330,510:1010]=overlay
                label(sheet,f'{times[f]:.2f}s cam{ci+1:02d} | raw / joint-weight overlay | index tip (8): {p["weights"][ci,f,8]:.3f}',(8,row+22),scale=.57)
                name=f'review_{target:03d}s_cam{ci+1:02d}.jpg';cv2.imwrite(str(out/name),np.hstack([crop,overlay]))
                sample=dict(id=f'{target:03d}s_cam{ci+1:02d}',frame=f,timeSeconds=float(times[f]),camera=ci,joint=8,
                    jointName='index_tip',weight=float(p['weights'][ci,f,8]),imageWeight=float(obs['weight_image'][ci,f,8]),
                    stabilityPixels=float(obs['stability_pixels'][ci,f,8]),meshSurfaceVisibility=float(obs['visibility'][ci,f,8]),
                    geometryWeight=float(p['geometryWeights'][ci,f,8]),image=name,
                    annotation=dict(visibility='unreviewed',accuracy='unreviewed',reviewer=None),
                    cropToFullImage=dict(scale=s,center=center.tolist()))
                # JSON validity for missed detections.
                for key,value in sample.items():
                    if isinstance(value,float) and not np.isfinite(value):sample[key]=None
                if sample['id'] in annotations:sample['annotation']=annotations[sample['id']]
                samples.append(sample)
                cells=' · '.join(f'{k}: {sample[k]:.2f}' if sample[k] is not None else f'{k}: missing' for k in ('weight','stabilityPixels','meshSurfaceVisibility','geometryWeight'))
                cards.append(f'<section><h3>{sample["id"]} · Index fingertip (8)</h3><img src="{name}"><p>{cells}</p><p>Left: raw image. Right: joint weights. Red = low, green = high; numbers are joint IDs.</p><select data-id="{sample["id"]}"><option value="unreviewed">Unreviewed</option><option value="visible">Clearly visible</option><option value="occluded">Occluded</option><option value="unclear">Unclear / out of frame</option><option value="wrong_hand">Wrong hand selected</option></select></section>')
            cv2.imwrite(str(out/f'audit_sheet_{target:03d}s.jpg'),sheet)
    finally:
        for r in readers:r.close()
    (out/'review_samples.json').write_text(json.dumps(dict(selection=selection,samples=samples),indent=2,allow_nan=False))
    script='''const labels={};document.querySelectorAll('select[data-id]').forEach(s=>s.onchange=()=>labels[s.dataset.id]=s.value);document.getElementById('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({reviewer:'user',labels},null,2)],{type:'application/json'}));a.download='visibility_labels.json';a.click();URL.revokeObjectURL(a.href);};'''
    review=f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint Weight Review</title><style>{STYLE}</style><h1>Per-Joint Weight Review</h1><p><a href="viewer.html">Weighted Replay</a> · <a href="comparison.html">Before / After Comparison</a> · <a href="review_samples.json">Review Sample Data</a></p><p>Fixed samples at 5, 25, 50, and 80 seconds from all four views, focusing on the index fingertip (8). Mesh self-occlusion and crop stability can both be misleading. Compare with the raw images and export your own visibility labels. Agreement between models is not ground-truth validation.</p><div id="agentReview"></div><button id="export">Export My Visibility Labels</button><div class="grid">'+''.join(cards)+f'</div><script>{script}</script></html>'
    if annotations:
        notes='<h2>Assistant Visual Review (Not Independent Human Ground Truth)</h2><table><tr><th>Sample</th><th>Assessment</th><th>Image Weight / Final Weight</th><th>Notes</th></tr>'
        for sample in samples:
            a=sample['annotation']
            if a.get('reviewer'):
                notes+=f'<tr><td>{sample["id"]}</td><td>{html.escape(a["visibility"])}</td><td>{sample["imageWeight"]:.2f} / {sample["weight"]:.2f}</td><td>{html.escape(a.get("noteEnglish",a.get("note","")))}</td></tr>'
        notes+='</table><p>This review identifies obvious failures and missed reliability issues. The small sample has no pixel-level ground truth, so no accuracy score is reported.</p>'
        review=review.replace('<div id="agentReview"></div>',notes)
    review=review.replace('Fixed samples at 5, 25, 50, and 80 seconds from all four views, focusing on the index fingertip (8).',html.escape(selection)+'.')
    (out/'review.html').write_text(review)
    report=json.loads((src/'weighted_report.json').read_text());shutil.copy2(src/'weighted_report.json',out/'weighted_report.json')
    # Geometry agreement is evaluated against an excluded view; strata use only
    # image-derived weights, excluding the geometric factor to avoid tautology.
    e=p['leaveOneOutErrorPixels'];w=obs['weight_image'];known=np.isfinite(e)
    proxy=dict(meaning='Leave-one-camera-out consistency, grouped ONLY by image weights; no fitted output used; not ground truth',
        referenceCoverage=float(known.mean()),low=describe(e[known&(w<.3)]),medium=describe(e[known&(w>=.3)&(w<.6)]),high=describe(e[known&(w>=.6)]))
    (out/'reliability_proxy_audit.json').write_text(json.dumps(proxy,indent=2,allow_nan=False))
    e=report['highWeightReprojectionPixels'];b=report['before'];a=report['after']
    stats=f'Median reprojection error for high-weight 2D predictions: {e["before"]["median"]:.2f} → {e["after"]["median"]:.2f} px; acceleration P95: {b["accelerationMetresPerSecondSquared"]["p95"]:.2f} → {a["accelerationMetresPerSecondSquared"]["p95"]:.2f} m/s². Both are diagnostic metrics, not ground-truth accuracy.'
    revision=hashlib.sha256((src/'pose_3d.npz').read_bytes()).hexdigest()[:12]
    script='''const a=document.getElementById('old'),b=document.getElementById('new'),seek=document.getElementById('seek');let duration=102.8;
function jump(t){a.currentTime=t;b.currentTime=t;seek.value=t;}a.onloadedmetadata=()=>{duration=a.duration;seek.max=duration;};document.getElementById('play').onclick=async()=>{if(a.paused){b.currentTime=a.currentTime;await Promise.all([a.play(),b.play()]);}else{a.pause();b.pause();}};seek.oninput=()=>jump(+seek.value);document.getElementById('speed').onchange=e=>a.playbackRate=b.playbackRate=+e.target.value;document.querySelectorAll('[data-time]').forEach(x=>x.onclick=()=>jump(+x.dataset.time));a.onended=()=>b.pause();function sync(){if(!a.paused&&Math.abs(a.currentTime-b.currentTime)>.075)b.currentTime=a.currentTime;seek.value=a.currentTime;document.getElementById('time').textContent=a.currentTime.toFixed(2)+' s';requestAnimationFrame(sync);}sync();'''
    comparison=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>WiLoR Weighted Fit | Before / After</title><style>{STYLE}</style><h1>WiLoR | Before / After Weighted 2D Reprojection Optimization</h1><p><a href="../viewer.html">Baseline 3D Replay</a> · <a href="viewer.html">Weighted 3D / IMU / Per-Joint Weights</a> · <a href="review.html">Joint Weight Review</a></p><p>{stats}</p><p>Left: baseline bone-length constraints and temporal smoothing. Right: additional per-joint image reliability, cross-view consistency, and weighted 2D fitting. Both use the same source-video timeline; crops may differ when hand selection changes. On the right, 2D points range from red (low weight) to green (high weight). Cyan shows 3D reprojection; yellow circles indicate fewer than two supporting views.</p><button id="play">Play / Pause Both</button><input id="seek" type="range" min="0" max="102.8" step=".01"><span id="time"></span><select id="speed"><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1×</option></select><button data-time="5">5 s</button><button data-time="25">25 s</button><button data-time="50">50 s</button><button data-time="80">80 s</button><div class="grid"><section><h2>Baseline: 3D Constraints</h2><video id="old" muted playsinline preload="metadata" src="../rgb_pose_review.mp4" poster="../preview_050s.jpg"></video></section><section><h2>Updated: Per-Joint Weighted 2D Fit</h2><video id="new" muted playsinline preload="metadata" src="rgb_pose_review.mp4?v={revision}" poster="preview_050s.jpg?v={revision}"></video></section></div><script>{script}</script></html>'''
    duration=(meta['stoppedAtUnixMs']-origin)/1000
    comparison=comparison.replace('102.8',str(duration))
    for target in (5,25,50,80):
        if target>duration:comparison=comparison.replace(f'<button data-time="{target}">{target} s</button>','')
    if report.get('fitMode') in ('joint','trajectory'):
        comparison=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint Multiview Reconstruction</title><style>{STYLE}</style><h1>Direct Joint Multiview Reconstruction</h1><p><a href="viewer.html">3D Replay and IMU</a> · <a href="review.html">Joint Weight Review</a></p><p>{stats}</p><p>The before metrics refer to numerical camera-pair seeds. The fit directly combines weighted 2D observations, bone lengths and robust motion, with no baseline 3D anchor. Camera calibration is fixed. This page shows the final result, not a baseline comparison.</p><video controls muted playsinline preload="metadata" src="rgb_pose_review.mp4?v={revision}"></video></html>'''
    if report.get('fitMode')=='trajectory':
        comparison=comparison.replace('Direct Joint Multiview Reconstruction','Asynchronous 3D Trajectory Reconstruction').replace('with no baseline 3D anchor.', 'at each camera recorded time, with no baseline 3D anchor. Short interior gaps may be estimated and are flagged in the data.')
    (out/'comparison.html').write_text(comparison)
    if json.loads((session/'capture.json').read_text()).get('diagnosticCandidate'):
        notice='<div id="diagnosticNotice" style="padding:14px;background:#562b15;color:#ffe6ab"><strong>DIAGNOSTIC CANDIDATE: camera calibration has not passed all validation checks.</strong> Both compared results use the same unvalidated candidate calibration. Formal calibration remains unchanged.</div>'
        for name in ('comparison.html','review.html'):
            path=out/name;page=path.read_text();path.write_text(page.replace('</h1>','</h1>'+notice,1))
    print(json.dumps(proxy,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('session');run(ap.parse_args().session)
