#!/usr/bin/env python3
"""Launch the existing pipeline with the appropriate configured Python."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pc_receiver'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/multicamera.local.json')
    ap.add_argument('--dry-run', action='store_true', help='Print the command without running it')
    ap.add_argument('task', choices=['collect', 'process', 'weighted', 'record-calibration',
        'detect-board', 'solve-calibration', 'viz', 'eit', 'export', 'export-clean', 'serve', 'test', 'check'])
    ap.add_argument('arguments', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    from multicamera_capture import CameraConfig
    config = CameraConfig.load(args.config)
    data = config.data
    env_file = ROOT/'configs/environments.local.json'
    env = json.loads(env_file.read_text()) if env_file.exists() else {}
    receiver = str(Path(env.get('receiverPython', sys.executable)).expanduser())
    analysis = data.get('reliabilityPython', data.get('regularizationPython', data['wilorPython']))
    script, python = {
        'collect': ('server.py', receiver), 'process': ('process_multicamera.py', receiver),
        'weighted': ('process_weighted_handpose.py', receiver),
        'record-calibration': ('record_calibration.py', receiver),
        'detect-board': ('calibrate_multicamera_recording.py', analysis),
        'solve-calibration': ('solve_multicamera_calibration.py', analysis),
        'viz': ('visualize_multicamera.py', analysis), 'eit': ('prepare_eit_visualization.py', analysis),
        'export': ('export_multimodal_video.py', analysis),
        'export-clean': ('export_pose_presentation.py', analysis),
        'serve': ('serve_visualization.py', receiver), 'test': ('', analysis), 'check': ('', receiver),
    }[args.task]
    extra = args.arguments
    if args.task == 'process' and '--stage' in extra:
        stage = extra[extra.index('--stage')+1]
        python = {'wilor': data['wilorPython'], 'triangulate': data['aniposePython'],
                  'regularize': analysis}.get(stage, receiver)
    if args.task == 'collect':
        extra = ['--host', '0.0.0.0', '--port', '8765', '--pose-source', 'multicamera',
                 '--camera-config', str(config.path), '--dataset-dir', str(ROOT/'data/dataset_multicamera'), *extra]
    if args.task == 'record-calibration':
        extra = ['--config', str(config.path), '--output-root', str(ROOT/'data/calibration_captures'), *extra]
    if args.task == 'test':
        command = [python, '-m', 'unittest', 'discover', '-s', str(ROOT/'tests'), *extra]
    elif args.task == 'check':
        print('All configured executables, detector adapter and calibration file exist.')
        for key in ('python', 'wilorPython', 'aniposePython', 'regularizationPython', 'reliabilityPython', 'calibration'):
            if key in data: print(f'{key}: {data[key]}')
        print('ffmpeg:', shutil.which('ffmpeg') or 'MISSING')
        print('This is a path check; camera connectivity, model weights and calibration quality are not checked.')
        return
    else:
        command = [python, '-u', str(ROOT/'pc_receiver'/script), *extra]
    if args.dry_run:
        print(shlex.join(command))
        return
    os.environ['PYTHONPATH'] = str(ROOT/'pc_receiver') + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
