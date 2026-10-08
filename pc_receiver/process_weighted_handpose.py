"""Run resumeable joint reliability inference, 2D fitting, visualization and audit."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('session',type=Path);args=ap.parse_args()
    session=args.session.resolve();root=session/'multicamera';out=root/'weighted_handpose';out.mkdir(exist_ok=True)
    config=json.loads((root/'config.json').read_text())
    python=config.get('reliabilityPython',config.get('regularizationPython'))
    if not python:raise SystemExit('Set reliabilityPython or regularizationPython to an environment with WiLoR, torch, scipy, cv2 and toml')
    for file in ('handpose/pose_3d.npz','handpose/wilor_2d.npz','calibration.toml','frame_sets.jsonl'):
        if not (root/file).exists():raise SystemExit(f'Missing input: {root/file}; complete the normal WiLoR/Anipose pipeline first')
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    steps=[('inference','infer_hand_reliability.py',[]),('fit','fit_weighted_handpose.py',[]),
        ('visualization','visualize_multicamera.py',['--method','weighted']),('review','review_hand_reliability.py',[])]
    env=dict(os.environ,HF_HUB_OFFLINE='1');state=out/'processing_status.json';start=time.time()
    try:
        for stage,script,extra in steps:
            state.write_text(json.dumps(dict(status='running',stage=stage)))
            print(f'{stage}: {session.name}',flush=True)
            with (out/(stage+'.log')).open('a') as log:
                subprocess.run([python,'-u',str(Path(__file__).with_name(script)),str(session),*extra],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.write_text(json.dumps(dict(status='complete',frames=json.loads((out/'weighted_report.json').read_text())['frames'],
            elapsedSeconds=time.time()-start,viewer=str(session/'visualization/weighted/viewer.html'))))
    except Exception as exc:
        state.write_text(json.dumps(dict(status='failed',error=str(exc))));raise
    print(session/'visualization/weighted/comparison.html')

if __name__=='__main__':main()
