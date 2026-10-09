"""Combine an existing four-camera review video with timestamp-matched 3D views.

Uses the existing RGB/2D/reprojection video without rerunning inference. Missing
poses remain empty; the 3D display uses a fixed center, orientation and scale.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import toml

from visualize_multicamera import CHAINS, Reader, label, nearest, rows, skeleton
from hand_reliability import cameras, project


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pose_panel(xyz, support, center, radius, yaw, pitch, title, clean=False, inferred=None):
    panel = np.full((432, 600, 3), (35, 28, 23), np.uint8)
    if not clean:
        label(panel, title, (18, 28), scale=.62)
        label(panel, 'Fixed world view | XYZ axes: 50 mm', (18, 52), scale=.47)
    if xyz is None or not np.isfinite(xyz).all(1).any():
        if not clean:
            label(panel, 'NO MATCHED 3D POSE', (125, 225), (90, 130, 255), .67)
        return panel
    valid = np.isfinite(xyz).all(1)

    def project(points):
        x, y, z = ((np.asarray(points) - center) / radius).T
        a = x * math.cos(yaw) + z * math.sin(yaw)
        b = -x * math.sin(yaw) + z * math.cos(yaw)
        c = y * math.cos(pitch) - b * math.sin(pitch)
        return np.column_stack([300 + a * 150, 238 + c * 150])

    uv = project(xyz)
    # Clip drawing to the plot area so an outlier cannot cover titles or notes.
    plot = panel[65:402].copy()
    def point(v):
        return tuple(np.clip(v - [0, 65], -100000, 100000).astype(int))
    origin = project(center[None])[0]
    axes = () if clean else ((123,148,255), (160,213,119), (255,186,137))
    for k, color in enumerate(axes):
        end = center.copy(); end[k] += .05
        ep = project(end[None])[0]
        cv2.line(plot, point(origin), point(ep), color, 1, cv2.LINE_AA)
        cv2.putText(plot, 'XYZ'[k], point(ep), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    for chain in CHAINS:
        for a, b in zip(chain, chain[1:]):
            if valid[a] and valid[b]:
                cv2.line(plot, point(uv[a]), point(uv[b]), (230,217,80), 2, cv2.LINE_AA)
    for j in np.flatnonzero(valid):
        color = (64,218,255) if not clean and support[j] < 2 else (230,217,80)
        cv2.circle(plot, point(uv[j]), 4, color, -1, cv2.LINE_AA)
        if inferred is not None and inferred[j]:cv2.circle(plot,point(uv[j]),7,(64,218,255),1,cv2.LINE_AA)
        if not clean:
            cv2.putText(plot, str(j), point(uv[j] + [6,-5]), cv2.FONT_HERSHEY_SIMPLEX, .32, (225,225,225), 1, cv2.LINE_AA)
    panel[65:402] = plot
    if not clean:
        label(panel, 'Yellow joints: fewer than 2 supporting views', (18, 420), (64,218,255), .46)
    return panel


class CleanRenderer:
    """Render from original RGB: diagnostic text is already baked into the review."""
    def __init__(self, root, pairs, xyz, origin, method='weighted'):
        self.pairs, self.xyz, self.origin = pairs, xyz, origin
        self.readers = [Reader(root) for _ in range(4)]
        self.records = [rows(root / f'cam{i+1:02d}_frames_aligned.jsonl') for i in range(4)]
        self.times = [np.array([(r['unixTimeMs']-origin)/1000 for r in rr]) for rr in self.records]
        if method=='weighted':
            with np.load(root / 'weighted_handpose/observations.npz') as obs:
                self.xy = obs['augment_xy'][:,:,0].copy()
        else:
            with np.load(root / 'handpose/wilor_2d.npz') as obs:
                self.xy = obs['keypoints'][...,:2].copy()
        self.inferred = self.view_inferred = None
        view_xyz = None
        if method=='weighted':
            with np.load(root / 'weighted_handpose/pose_3d.npz') as saved:
                if 'perViewJoints' in saved:
                    view_xyz=saved['perViewJoints'];self.inferred=saved['temporalInferred']
                    self.view_inferred=saved['perViewTemporalInferred']
        self.projections = []
        for ci,camera in enumerate(cameras(toml.load(root / 'calibration.toml'))):
            positions=xyz if view_xyz is None else view_xyz[ci]
            valid=np.isfinite(positions).all(-1)
            uv = np.full((*xyz.shape[:-1],2),np.nan)
            uv[valid] = project(positions[valid], camera)
            self.projections.append(uv)

    def draw(self, t, pi, paired, center, radius, support):
        canvas = np.full((1080,1920,3), (30,21,16), np.uint8)
        label(canvas,'Orange: WiLoR',(24,38),(30,170,255),.8)
        label(canvas,'Cyan: projected 3D',(310,38),(255,245,0),.8)
        if self.inferred is not None:label(canvas,'Yellow rings: temporal estimates',(660,38),(64,218,255),.62)
        for ci, reader in enumerate(self.readers):
            record = self.pairs[pi]['frames'][f'cam{ci+1:02d}'] if paired else self.records[ci][nearest(self.times[ci],t)]
            image = reader.read(record)
            xy = self.xy[ci,pi] if paired else np.full((21,2),np.nan)
            good = np.isfinite(xy).all(1)
            if good.any():
                low, high = np.nanmin(xy,axis=0), np.nanmax(xy,axis=0)
                crop_center = (low+high)/2
                side = max(380,float((high-low).max())*1.9)
            else:
                crop_center, side = np.array([960,600]), 1920
            scale = 640/side
            affine = np.array([[scale,0,320-scale*crop_center[0]],[0,scale,240-scale*crop_center[1]]])
            tile = cv2.warpAffine(image,affine,(640,480))
            if paired:
                skeleton(tile,cv2.transform(xy[None],affine)[0],good,(30,170,255))
                skeleton(tile,cv2.transform(self.projections[ci][pi][None],affine)[0],np.isfinite(self.projections[ci][pi]).all(1),(255,245,0),5)
                if self.view_inferred is not None:
                    uv=cv2.transform(self.projections[ci][pi][None],affine)[0]
                    for j in np.flatnonzero(self.view_inferred[ci,pi]&np.isfinite(uv).all(1)):
                        if np.abs(uv[j]).max()<10000:cv2.circle(tile,tuple(uv[j].astype(int)),8,(64,218,255),1,cv2.LINE_AA)
            tile[372:472,472:632] = cv2.resize(image,(160,100))
            cv2.rectangle(tile,(471,371),(632,472),(200,200,200),1)
            x,y = (ci%2)*640,64+(ci//2)*480
            canvas[y:y+480,x:x+640] = tile
        for y,yaw in [(64,.2),(544,.2+math.pi/2)]:
            panel = pose_panel(self.xyz[pi] if paired else None,support,center,radius,yaw,-.35,'',clean=True,inferred=self.inferred[pi] if self.inferred is not None and paired else None)
            canvas[y:y+480,1300:1900] = cv2.copyMakeBorder(panel,24,24,0,0,cv2.BORDER_CONSTANT,value=(35,28,23))
        return canvas

    def close(self):
        for reader in self.readers:
            reader.close()


def export(session, clean=False):
    session = Path(session).resolve()
    root = session / 'multicamera'
    out = session / 'visualization/weighted'
    source = out / 'rgb_pose_review.mp4'
    pose_path = root / 'weighted_handpose/pose_3d.npz'
    report = json.loads((out / 'visualization_report.json').read_text())
    digest = sha256(pose_path)
    if not report['revision'].endswith(digest[:12]):
        raise ValueError('The review video report does not match the current 3D pose. Regenerate the review first.')
    with np.load(pose_path) as saved:
        xyz = saved['joints'].copy()
        stamps = saved['unixTimeMs'].copy()
        support = saved['camerasUsed'].copy()
    meta = json.loads((session / 'timestamp.json').read_text())
    origin = float(meta['startedAtUnixMs'])
    times = (stamps - origin) / 1000
    if np.any(np.diff(times) <= 0):
        raise ValueError('Pose timestamps must increase.')
    pairs = rows(root / 'frame_sets.jsonl')
    np.testing.assert_allclose(stamps, [p['unixTimeMs'] for p in pairs], rtol=0, atol=.001)
    points = xyz[np.isfinite(xyz).all(-1)]
    low, high = np.percentile(points, [.5,99.5], axis=0)
    center = (low + high) / 2
    radius = max(.12, float(((high-low)/2).max()) * 1.25)
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or abs(fps - report['fps']) > .001 or count != report['videoFrames']:
        raise ValueError('Source video metadata does not match its visualization report.')
    tolerance=report.get('poseMatchToleranceMs',1000*(.5/fps+.001))/1000
    stem = 'rgb_2d_3d_clean' if clean else 'rgb_2d_3d_combined'
    output = out / f'{stem}.mp4'
    pending = out / f'{stem}.pending.mp4'
    renderer = CleanRenderer(root,pairs,xyz,origin) if clean else None
    cv2.setNumThreads(2)
    command = ['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
               '-s','1920x1080','-r',str(fps),'-i','-','-an','-c:v','libx264',
               '-threads','4','-preset','fast','-crf','19','-pix_fmt','yuv420p',
               '-movflags','+faststart',str(pending)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    matched = 0
    try:
        for frame in range(count):
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f'Source video decode failed at frame {frame}')
            if image.shape[:2] != (856,1280):
                raise ValueError('Expected the existing 1280x856 four-view video.')
            t = frame / fps
            pi = nearest(times,t)
            paired = abs(times[pi]-t) <= tolerance
            matched += int(paired)
            if clean:
                canvas = renderer.draw(t,pi,paired,center,radius,support[pi])
                label(canvas, 'Weighted 3D joints', (1320,38), (230,217,80), .8)
            else:
                canvas = np.full((1080,1920,3), (30,21,16), np.uint8)
                label(canvas, 'WiLoR | Four-View RGB + 2D Skeletons + Weighted 3D Pose', (20,32), scale=.85)
                label(canvas, f'Session {session.name} | {t:.2f} / {count/fps:.2f} s', (20,59), scale=.52)
                canvas[76:932,:1280] = image
                for y, yaw, title in [(76,.2,'3D skeleton | View A'), (520,.2+math.pi/2,'3D skeleton | View B (90 deg rotated)')]:
                    canvas[y:y+432,1300:1900] = pose_panel(
                        xyz[pi] if paired else None, support[pi], center, radius, yaw, -.35, title)
                label(canvas, 'RGB overlays: orange = WiLoR 2D; cyan = projected weighted 3D.', (20,961), scale=.61)
                label(canvas, '2D dots: red = low weight, green = high weight. Full camera views appear in the insets.', (20,988), scale=.56)
                if paired:
                    label(canvas, f'Pose frame {pi} | Valid joints: {np.isfinite(xyz[pi]).all(1).sum()}/21 | Four-view span: {pairs[pi]["cameraSkewMs"]:.2f} ms', (20,1015), scale=.59)
                else:
                    label(canvas, f'No pose within {tolerance*1000:.1f} ms of this video time.', (20,1015), (90,130,255), .59)
                label(canvas, 'Cameras expose at different times. Weights are heuristic; overlays are not ground truth.', (20,1043), scale=.52)
                cv2.rectangle(canvas,(20,1060),(1900,1067),(70,60,50),-1)
                cv2.rectangle(canvas,(20,1060),(20+round(1880*(frame+1)/count),1067),(220,190,70),-1)
            encoder.stdin.write(canvas.tobytes())
            if frame in {round(s*fps) for s in (5,25,50,80)}:
                cv2.imwrite(str(out / f'{"clean" if clean else "combined"}_preview_{round(t):03d}s.jpg'),canvas)
            if frame % 200 == 0:
                print(f'Combined video {frame}/{count}',flush=True)
    except BaseException:
        encoder.stdin.close(); encoder.wait(); pending.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if renderer:
            renderer.close()
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError('Video encoder failed.')
    pending.replace(output)
    summary = dict(session=session.name,output=str(output),sourceVideo=str(source),
                   sourceVideoSha256=sha256(source),poseSha256=digest,
                   frames=count,fps=fps,durationSeconds=count/fps,resolution=[1920,1080],
                   framesWithMatchedPose=matched,poseMatchToleranceMs=tolerance*1000,
                   fixedDisplayCenterMetres=center.tolist(),fixedDisplayRadiusMetres=radius,
                   missingJointsFilled=0,labelsLanguage='en',
                   note='Reuses the existing RGB/2D/3D-reprojection video. Both independent 3D panels show the same timestamp-matched weighted pose from fixed orientations. No inference or interpolation.')
    if clean:
        summary.update(sourceVideo=str(root/'rgb'),sourceVideoSha256=None,
                       note='Rendered from original RGB and existing weighted 2D/3D predictions. Only orange WiLoR and cyan 3D skeletons; no diagnostic overlays, inference or interpolation.')
    with np.load(root/'weighted_handpose/pose_3d.npz') as saved:
        if 'trajectoryReconstruction' in saved:
            summary.update(trajectoryReconstruction=True,temporalInferredFrameJoints=int(saved['temporalInferred'].sum()),
                note='Native-camera-time 3D trajectory projections; independent panels show reference time. Yellow rings label short-gap estimates. No additional interpolation during export.')
    (out / ('clean_video_report.json' if clean else 'combined_video_report.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--clean',action='store_true',help='Render original RGB with only the WiLoR / projected 3D legend.')
    args = parser.parse_args()
    export(args.session,clean=args.clean)
