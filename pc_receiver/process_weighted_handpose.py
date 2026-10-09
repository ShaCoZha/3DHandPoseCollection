"""Run resumeable joint reliability inference, 2D fitting, visualization and audit."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('session',type=Path)
    ap.add_argument('--inference-workers',type=int,default=1)
    ap.add_argument('--fit-mode',choices=('legacy','joint','trajectory'),default='legacy',
                    help='joint fits 2D observations directly; trajectory also uses native camera times and bounded short-gap recovery')
    args=ap.parse_args()
    if not 1<=args.inference_workers<=4:ap.error('--inference-workers must be between 1 and 4')
    session=args.session.resolve();root=session/'multicamera';out=root/'weighted_handpose';out.mkdir(exist_ok=True)
    config=json.loads((root/'config.json').read_text())
    python=config.get('reliabilityPython',config.get('regularizationPython'))
    if not python:raise SystemExit('Set reliabilityPython or regularizationPython to an environment with WiLoR, torch, scipy, cv2 and toml')
    required=['handpose/wilor_2d.npz','calibration.toml','frame_sets.jsonl']
    if args.fit_mode=='legacy':required.append('handpose/pose_3d.npz')
    for file in required:
        if not (root/file).exists():raise SystemExit(f'Missing input: {root/file}')
    lock=(out/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    fit_script='fit_joint_handpose.py' if args.fit_mode in ('joint','trajectory') else 'fit_weighted_handpose.py'
    steps=[('inference','infer_hand_reliability.py',[]),('fit',fit_script,['--mode','trajectory'] if args.fit_mode=='trajectory' else []),
        ('visualization','visualize_multicamera.py',['--method','weighted']),('review','review_hand_reliability.py',[])]
    env=dict(os.environ,HF_HUB_OFFLINE='1');state=out/'processing_status.json';start=time.time()
    try:
        for stage,script,extra in steps:
            state.write_text(json.dumps(dict(status='running',stage=stage)))
            print(f'{stage}: {session.name}',flush=True)
            if stage=='inference' and args.inference_workers>1:
                workers=[];handles=[]
                try:
                    for index in range(args.inference_workers):
                        log=(out/f'inference_worker_{index}.log').open('a');handles.append(log)
                        workers.append(subprocess.Popen([python,'-u',str(Path(__file__).with_name(script)),str(session),
                            '--shard-index',str(index),'--shard-count',str(args.inference_workers)],env=env,stdout=log,stderr=subprocess.STDOUT))
                    while any(p.poll() is None for p in workers):
                        if any(p.poll() not in (None,0) for p in workers):raise RuntimeError('A reliability inference worker failed; see worker logs')
                        time.sleep(1)
                    if any(p.returncode for p in workers):raise RuntimeError('A reliability inference worker failed; see worker logs')
                finally:
                    for p in workers:
                        if p.poll() is None:p.terminate()
                    for p in workers:
                        try:p.wait(timeout=10)
                        except subprocess.TimeoutExpired:p.kill();p.wait()
                    for log in handles:log.close()
                # The normal pass checks provenance, skips completed chunks,
                # and assembles observations using the unchanged algorithm.
            with (out/(stage+'.log')).open('a') as log:
                subprocess.run([python,'-u',str(Path(__file__).with_name(script)),str(session),*extra],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.write_text(json.dumps(dict(status='complete',frames=json.loads((out/'weighted_report.json').read_text())['frames'],
            elapsedSeconds=time.time()-start,viewer=str(session/'visualization/weighted/viewer.html'))))
    except Exception as exc:
        state.write_text(json.dumps(dict(status='failed',error=str(exc))));raise
    print(session/'visualization/weighted/comparison.html')

if __name__=='__main__':main()
