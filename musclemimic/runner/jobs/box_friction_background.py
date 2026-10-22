from musclemimic.runner.jobs.resources import gpu_for, cpu_set_for
from musclemimic.utils.paths import workspace_root, config_root
"""Detached, gated box friction curriculum. Launch in tmux box_friction_v2."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=workspace_root()
OUT=ROOT/'outputs/payload_box_friction_v2'
AUDIT=OUT/'audit'
STATE={}
MODULE='musclemimic.runner.workflows.box_friction'
SMOKE='smoke_platform_final'


def environment_signature():
    digest=hashlib.sha256()
    for path in [OUT/'bend/reference.npz',OUT/'bend/base_config.yaml',
                 ROOT/'musclemimic/environments/humanoids/myofullbody_box_friction.py']:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def status(phase,**kw):
    STATE.update(time=time.strftime('%Y-%m-%d %H:%M:%S %z'),phase=phase,**kw)
    p=AUDIT/'background_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(STATE,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
    print(STATE,flush=True)


def run(args,label,gpu=False):
    env=os.environ.copy();env.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if gpu:
        env.update(CUDA_VISIBLE_DEVICES=gpu_for('payload_box', '3'),JAX_PLATFORMS='cuda,cpu')
        env.pop('JAX_SKIP_CUDA_CONSTRAINTS_CHECK',None)
        python,cpus='.venv/bin/python',cpu_set_for('payload_box')
    else:
        env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_SKIP_CUDA_CONSTRAINTS_CHECK='1')
        python,cpus='.venv-stage2/bin/python',cpu_set_for('lift_evaluation')
    with (AUDIT/(label+'.log')).open('a') as f:
        p=subprocess.Popen(['taskset','-c',cpus,python,'-u','-m',*args],cwd=ROOT,
            env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status('running',step=label,pid=p.pid,gpu=3 if gpu else None)
        code=p.wait()
        if code:raise RuntimeError(f'{label}: exit={code}; inspect {label}.log')


def cp(label,updates):
    return OUT/'bend/training'/label/'checkpoints'/label/f'checkpoint_{updates}'


def train(label,mass,envs,updates,source=None,start_mode='beginning'):
    folder=OUT/'bend/training'/label
    if (folder/'completed.json').exists():return cp(label,updates)
    args=[MODULE,'train','--mass',str(mass),'--envs',str(envs),'--updates',str(updates),'--label',label]
    if source:args+=['--checkpoint',str(source)]
    args+=['--start-mode',start_mode]
    # Restart the same run to use its latest saved checkpoint, with a bounded retry count.
    for attempt in range(3):
        if (AUDIT/'STOP_BACKGROUND').exists():raise RuntimeError('Stop requested; no further stages launched')
        try:
            run(args,label,gpu=True);return cp(label,updates)
        except RuntimeError:
            if attempt==2:raise
            time.sleep(30)


def evaluate(path,label,mass):
    saved=OUT/'bend/evaluations'/label/'results.json'
    if saved.exists():
        data=json.loads(saved.read_text())
        if (data.get('environment_signature')==environment_signature()
            and data.get('checkpoint')==str(path) and data.get('mass_kg')==mass):return data
    run([MODULE,'evaluate','--checkpoint',str(path),'--label',label,'--mass',str(mass)],'eval_'+label)
    return json.loads(saved.read_text())


def report():
    previous=STATE.copy()
    run(['musclemimic.runner.workflows.box_friction_report'],'report')
    STATE.clear();STATE.update(previous)
    status(previous.get('phase','checking'))


def main():
    os.chdir(ROOT);AUDIT.mkdir(parents=True,exist_ok=True)
    lock=(AUDIT/'background.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('checking',supervisor_pid=os.getpid(),session='box_friction_v2')
    sha=hashlib.sha256((OUT/'bend/reference.npz').read_bytes()).hexdigest()
    for action,path,label in [('geometry',AUDIT/'full_geometry.json','geometry_final'),
                              ('validate',OUT/'bend/physics/acceptance.json','physics_final')]:
        data=json.loads(path.read_text()) if path.exists() else {}
        if not data.get('passed') or data.get('environment_signature')!=environment_signature():run([MODULE,action],label)
    initial=ROOT/'outputs/payload_box_friction_v2/bend/training/smoke_5kg/initialization/checkpoint_0'
    small=train(SMOKE,5,32,4,initial)
    evaluation=evaluate(small,SMOKE,5)
    metrics=[json.loads(line) for line in (OUT/'bend/training'/SMOKE/'metrics.jsonl').read_text().splitlines()]
    finite=bool(metrics) and all(math.isfinite(row['PPO/total_loss']) for row in metrics)
    gate=dict(passed=bool(finite and evaluation['numerical_contact_pass']),envs=32,updates=4,
        finite_ppo_loss=finite,numerical_contact_pass=evaluation['numerical_contact_pass'],
        physics='bend/physics/acceptance.json',scope='Simulation/training readiness; learned lifting success is evaluated separately.',
        reference_sha256=hashlib.sha256((OUT/'bend/reference.npz').read_bytes()).hexdigest(),
        environment_signature=environment_signature(),smoke_label=SMOKE,
        max_allowed_hand_box_penetration_m=.003)
    (AUDIT/'scaling_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    if not gate['passed']:raise RuntimeError('Small-environment gate failed; large training blocked')
    status('large_training_ready',scaling_gate=gate)
    report()
    # Start from original unladen weights, not the short smoke optimization.
    source=initial
    for mass,updates in [(1,200),(3,200),(5,400)]:
        passed=False
        for attempt in range(2):
            label=f'curriculum_{mass}kg_v2_{attempt+1}'
            source=train(label,mass,4096,updates,source,start_mode='beginning' if attempt==0 else 'random')
            result=evaluate(source,label,mass)
            report()
            if result['all_quality_pass']:
                passed=True;break
        if not passed:
            status('needs_policy_review',mass_kg=mass,checkpoint=str(source),
                   reason='Current load did not pass full trajectory, bilateral contact, friction support and penetration gates after two stages.')
            report();return
    status('complete',checkpoint=str(source),mass_kg=5)
    report()


if __name__=='__main__':
    try:main()
    except Exception:
        status('error',error=traceback.format_exc());raise
