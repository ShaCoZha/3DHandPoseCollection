"""Prepare recorded EIT measurement matrices for timestamp-based visualization.

EIT Unix seconds are not assumed to share the iPhone clock. The default mapping
is explicitly unverified. Missing intervals and invalid cells are preserved.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def prepare(session, offset_ms=0., clock_status='unverified'):
    session=Path(session).resolve();out=session/'visualization/eit';out.mkdir(parents=True,exist_ok=True)
    capture=json.loads((session/'timestamp.json').read_text())
    origin=capture['startedAtUnixMs'];duration=(capture['stoppedAtUnixMs']-origin)/1000
    times=[];ids=[];missing=[];magnitudes=[];perf=[];first=None;digest=hashlib.sha256()
    with (session/'eit_frames.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line);r=json.loads(line)
            if first is None:first=r
            if r['shape']!=first['shape'] or r['freqs']!=first['freqs']:
                raise ValueError('EIT shape/frequencies change within the recording')
            if r['shape']!=[16,1,16]:raise ValueError('Expected recorded EIT shape [16,1,16]')
            real=np.asarray(r['re'],dtype=np.float32).reshape(r['shape'])[:,0,:]
            imag=np.asarray(r['im'],dtype=np.float32).reshape(r['shape'])[:,0,:]
            magnitudes.append(np.hypot(real,imag));times.append(r['t_unix']*1000)
            perf.append(r['t_perf']);ids.append(r['frame_idx']);missing.append(r['n_missing'])
    times=np.asarray(times);perf=np.asarray(perf);values=np.asarray(magnitudes,dtype='<f4')
    if not len(times) or np.any(np.diff(times)<=0):raise ValueError('EIT timestamps must increase')
    relative=(times+offset_ms-origin)/1000
    intervals=np.diff(times)/1000
    tolerance=min(.1,max(.025,float(np.median(intervals))*1.5))
    finite=values[np.isfinite(values)]
    lo,hi=np.percentile(finite,[1,99]);hi=max(float(hi),float(lo)+1e-6)
    gaps=[dict(startSeconds=float(relative[i]),endSeconds=float(relative[i+1]),
               durationSeconds=float(intervals[i]),perfGapSeconds=float(perf[i+1]-perf[i]))
          for i in np.flatnonzero(intervals>max(.1,5*np.median(intervals)))]
    baseline_ids=np.flatnonzero((relative>=0)&(relative<1)&(np.array(missing)==0))
    baseline=np.nanmedian(values[baseline_ids],axis=0) if len(baseline_ids) else np.full((16,16),np.nan)
    values.tofile(out/'magnitude.f32')
    metadata=dict(session=session.name,shape=[16,16],frames=len(times),frequencyHz=first['freqs'][0],
        injectionPairs=first['inj_sequence'],measurementMode=first['mea_mode'],
        rawRelativeSeconds=((times-origin)/1000).tolist(),frameIndices=ids,missingCells=missing,
        offsetMs=offset_ms,clockStatus=clock_status,matchToleranceSeconds=tolerance,
        magnitudeRange=[float(lo),float(hi)],baselineMagnitude=np.where(np.isfinite(baseline),baseline,None).tolist(),
        binaryFile='magnitude.f32',binaryFormat='little-endian float32; frame, injection, measurement; NaN means invalid',
        meaning='Magnitude of recorded complex measurements, not a reconstructed conductivity image',
        units='recorded units; physical units not established by this file',
        sourceSha256=digest.hexdigest(),recordingDurationSeconds=duration,gaps=gaps,
        rawFrameCoverageSeconds=[float(relative[0]),float(relative[-1])],
        framesInsideVideo=int(((relative>=0)&(relative<=duration)).sum()),
        incompleteFrames=int((np.array(missing)>0).sum()),invalidCells=int((~np.isfinite(values)).sum()))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    shutil.copy2(Path(__file__).with_name('eit_viewer.js'),out/'eit_viewer.js')
    print(json.dumps({k:metadata[k] for k in ('frames','framesInsideVideo','incompleteFrames','invalidCells','gaps','clockStatus')},indent=2))
    return metadata


def attach(session, method='weighted'):
    session=Path(session).resolve();out=session/'visualization'
    if method!='wilor':out=out/method
    viewer=out/'viewer.html';text=viewer.read_text()
    prefix='../eit' if method!='wilor' else 'eit'
    if 'id="eitSection"' not in text:
        section='''<section class="card" id="eitSection" style="margin-top:18px">
<h2>EIT Measurement Frame</h2><p id="eitClock" class="muted"></p>
<div class="controls"><label>Display <select id="eitMode"><option value="magnitude">Magnitude</option><option value="relative">Change from stored baseline (%)</option></select></label>
<label>EIT time offset (ms) <input id="eitOffset" type="number" step="10" value="0" style="width:130px"></label></div>
<p class="muted">Positive offset moves EIT frames later on the video timeline. Changing this control is a visual adjustment, not clock calibration.</p>
<div class="plots"><div><canvas id="eitFrame" style="height:390px"></canvas><p id="eitInfo"></p></div><div><canvas id="eitTrace" style="height:200px"></canvas><p class="muted">Mean measurement magnitude over time. Gaps are not interpolated. Gray heatmap cells are invalid. Rows are injection pairs; columns are measurement indices.</p><p class="muted">Recorded complex-measurement magnitude, not a reconstructed conductivity image. Values use recorded units.</p></div></div></section>'''
        text=text.replace('<p class="footer">',section+'\n<p class="footer">',1)
        text=text.replace('</body>',f'<script src="{prefix}/eit_viewer.js" data-eit-root="{prefix}"></script></body>',1)
    (out/'viewer.pending.html').write_text(text);(out/'viewer.pending.html').replace(viewer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session',type=Path)
    p.add_argument('--offset-ms',type=float,default=0.)
    p.add_argument('--clock-status',choices=['unverified','phone_aligned','user_offset'],default='unverified')
    p.add_argument('--attach',choices=['wilor','weighted','poem']);a=p.parse_args()
    prepare(a.session,a.offset_ms,a.clock_status)
    if a.attach:attach(a.session,a.attach)
