"""Offline timestamp pairing, WiLoR detection and Anipose RANSAC triangulation.

The driver uses only the standard library; each inference stage uses its own venv.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

JOINTS = ['base', 'thumb_CMC', 'thumb_MCP', 'thumb_DIP', 'thumb_tip',
          'index_MCP', 'index_PIP', 'index_DIP', 'index_tip',
          'middle_MCP', 'middle_PIP', 'middle_DIP', 'middle_tip',
          'ring_MCP', 'ring_PIP', 'ring_DIP', 'ring_tip',
          'pinky_MCP', 'pinky_PIP', 'pinky_DIP', 'pinky_tip']


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def detect(root, config):
    import cv2
    import numpy as np
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
    import wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline as pipeline_module
    from hand_reliability import fast_crop_gaussian
    # Same antialias kernel and boundary handling, already used by the weighted
    # pass; avoid repeatedly filtering full RGB frames through slow SciPy loops.
    pipeline_module.gaussian = fast_crop_gaussian
    pairs = rows(root/'frame_sets.jsonl')
    if not pairs:
        raise ValueError('No timestamp-matched four-camera frames')
    output = root/'handpose'
    output.mkdir(exist_ok=True)
    points = np.full((4, len(pairs), 21, 3), np.nan, dtype=np.float32)
    points[..., 2] = 0
    hands = np.full((4, len(pairs)), -1, dtype=np.int8)
    model = WiLorHandPose3dEstimationPipeline(device=torch.device('cuda'), dtype=torch.float16, verbose=False)
    readers = [None]*4
    files = [None]*4
    positions = [0]*4
    wanted = 1 if config['hand'] == 'right' else 0
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    try:
        for index, pair in enumerate(pairs):
            for ci, spec in enumerate(config['cameras']):
                record = pair['frames'][spec['name']]
                path = root/record['video']
                if files[ci] != path:
                    if readers[ci] is not None:
                        readers[ci].release()
                    readers[ci] = cv2.VideoCapture(str(path))
                    files[ci], positions[ci] = path, 0
                target = record['videoFrameIndex']
                if target < positions[ci]:
                    raise ValueError('Frame pairing reused or reordered video frames')
                image = None
                while positions[ci] <= target:
                    ok, image = readers[ci].read()
                    if not ok:
                        raise ValueError('Video missing frame {}: {}'.format(target, path))
                    positions[ci] += 1
                predictions = model.predict(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                candidates = [p for p in predictions if int(float(p['is_right']) > .5) == wanted]
                if not candidates:
                    continue
                hand = max(candidates, key=lambda p: np.prod(np.asarray(p['hand_bbox'])[2:]-np.asarray(p['hand_bbox'])[:2]))
                box = np.asarray(hand['hand_bbox'])
                xy = np.asarray(hand['wilor_preds']['pred_keypoints_2d'][0])
                points[ci, index, :, :2] = xy
                points[ci, index, :, 2] = ((xy >= box[:2]) & (xy <= box[2:])).all(axis=1)
                hands[ci, index] = wanted
            if index % 100 == 0:
                print('WiLoR {}/{}'.format(index+1, len(pairs)), flush=True)
    finally:
        for reader in readers:
            if reader is not None:
                reader.release()
    np.savez_compressed(output/'wilor_2d.npz', keypoints=points, handedness=hands,
                        unixTimeMs=[p['unixTimeMs'] for p in pairs], joint_names=JOINTS)
    (output/'wilor_report.json').write_text(json.dumps(dict(frames=len(pairs), hand=config['hand'],
        detectedPerCamera=(hands >= 0).sum(axis=1).tolist(),
        confidenceMeaning='inside detector bounding box, not calibrated keypoint probability',
        handSelection='Largest detection of configured handedness per view; use one hand in the capture volume'), indent=2))


def triangulate(root, config):
    import numpy as np
    from aniposelib.cameras import CameraGroup
    pairs = rows(root/'frame_sets.jsonl')
    output = root/'handpose'
    data = np.load(str(output/'wilor_2d.npz'))
    points = data['keypoints'].copy()
    xy = points[..., :2].copy()
    xy[points[..., 2] < .6] = np.nan
    n = len(pairs)
    group = CameraGroup.load(str(root/'calibration.toml'))
    if group.get_names() != [c['name'] for c in config['cameras']]:
        raise ValueError('Camera calibration order mismatch')
    # Chunk work to avoid a long unresponsive RANSAC call on a 15-minute session.
    xyz = np.full((n, 21, 3), np.nan)
    error = np.full((n, 21), np.nan)
    used = np.zeros((n, 21), dtype=int)
    for start in range(0, n, 100):
        end = min(n, start+100)
        values, picked, _, residual = group.triangulate_ransac(xy[:,start:end].reshape(4,-1,2), min_cams=2)
        xyz[start:end] = values.reshape(-1,21,3)
        error[start:end] = residual.reshape(-1,21)
        used[start:end] = picked.sum(axis=(0,2)).reshape(-1,21)
        print('Anipose {}/{}'.format(end,n), flush=True)
    # Reject high reprojection residuals rather than labelling all finite points valid.
    valid = np.isfinite(xyz).all(axis=2) & np.isfinite(error) & (error <= 5) & (used >= 2)
    xyz[~valid] = np.nan
    np.savez_compressed(str(output/'pose_3d.npz'), joints=xyz, valid=valid,
                        reprojectionErrorPixels=error, camerasUsed=used,
                        unixTimeMs=data['unixTimeMs'], joint_names=JOINTS, units='metres')
    with (root.parent/'multicamera_hand_pose_aligned.jsonl').open('w') as handle:
        for index, pair in enumerate(pairs):
            joints = [{'name': name, 'valid': bool(valid[index,j]),
                       'position': xyz[index,j].tolist() if valid[index,j] else None,
                       'reprojectionErrorPixels': float(error[index,j]) if np.isfinite(error[index,j]) else None,
                       'camerasUsed': int(used[index,j])} for j,name in enumerate(JOINTS)]
            record = dict(type='multicamera_hand_pose', sampleIndex=index,
                          unixTimeMs=pair['unixTimeMs'], hand=config['hand'], joints=joints,
                          coordinateSpace='calibration world', units='metres',
                          cameraSkewMs=pair['cameraSkewMs'],
                          clockUncertaintyEstimateMs=max(r['clockUncertaintyEstimateMs'] for r in pair['frames'].values()),
                          sourceFrames={k:dict(video=r['video'],videoFrameIndex=r['videoFrameIndex'],
                                              cameraFrameId=r['cameraFrameId'],unixTimeMs=r['unixTimeMs'])
                                        for k,r in pair['frames'].items()})
            handle.write(json.dumps(record, allow_nan=False)+'\n')
    (output/'pose_report.json').write_text(json.dumps(dict(frames=n,
        validJointFraction=float(valid.mean()), validFrameFraction=float(valid.all(axis=1).mean()),
        reprojectionThresholdPixels=5, algorithm='WiLoR 2D + aniposelib RANSAC; no temporal interpolation',
        hardwareExposureSynchronized=False), indent=2))
    # Refresh immutable inputs for constraints whenever triangulation is rerun.
    shutil.copy2(output/'pose_3d.npz', output/'pose_3d_unconstrained.npz')
    shutil.copy2(output/'pose_report.json', output/'pose_report_unconstrained.json')
    shutil.copy2(root.parent/'multicamera_hand_pose_aligned.jsonl', root.parent/'multicamera_hand_pose_unconstrained.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--stage', choices=('all','wilor','triangulate','regularize','weighted'), default='all')
    args = parser.parse_args()
    session = args.session.resolve()
    root = session/'multicamera'
    config = json.loads((root/'config.json').read_text())
    if args.stage == 'wilor':
        detect(root, config)
    elif args.stage == 'triangulate':
        triangulate(root, config)
    elif args.stage == 'regularize':
        from regularize_handpose import main as regularize
        regularize(session)
    elif args.stage == 'weighted':
        subprocess.run([sys.executable,str(Path(__file__).with_name('process_weighted_handpose.py')),str(session)],check=True)
    else:
        from align_multicamera import align_session
        report = align_session(session)
        if not report['pairedFrames']:
            raise SystemExit('No valid frame pairs; inspect multicamera/alignment_report.json')
        output = root/'handpose'
        output.mkdir(exist_ok=True)
        env = dict(os.environ, HF_HUB_OFFLINE='1')
        state_path = output/'processing_status.json'
        state_path.write_text(json.dumps(dict(status='running')))
        try:
            stages=[('wilor', config['wilorPython']), ('triangulate', config['aniposePython'])]
            if config.get('regularizeHandpose',False) or config.get('weightedHandpose',False):
                stages.append(('regularize',config.get('regularizationPython',config['wilorPython'])))
            if config.get('weightedHandpose',False):
                stages.append(('weighted',sys.executable))
            for stage, python in stages:
                with (output/(stage+'.log')).open('w') as log:
                    subprocess.run([python, '-u', str(Path(__file__).resolve()), str(session), '--stage', stage],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
            state_path.write_text(json.dumps(dict(status='complete')))
        except Exception as exc:
            state_path.write_text(json.dumps(dict(status='failed', error=str(exc))))
            raise
        print('Hand pose: {}'.format(session/'multicamera_hand_pose_aligned.jsonl'))


if __name__ == '__main__':
    main()
