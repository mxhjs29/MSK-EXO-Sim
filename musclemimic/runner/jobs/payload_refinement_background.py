from musclemimic.runner.jobs.resources import gpu_for, cpu_set_for
from musclemimic.utils.paths import workspace_root, config_root
"""Detached refinement pipeline. One session per assigned GPU/task."""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=workspace_root();OUT=ROOT/'outputs/payload_refinement_v3'
MODULE='musclemimic.runner.workflows.payload_refinement'
GPUS={'lift':gpu_for('payload_lift', '2'),'bend':gpu_for('payload_box', '3')}
STATE={}

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def status(task,phase,**kw):
    STATE.update(time=datetime.datetime.now().astimezone().isoformat(),phase=phase,**kw)
    save(OUT/task/'background_status.json',STATE);print(STATE,flush=True)

def run(task,args,label,gpu=False):
    env=os.environ.copy();env.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if gpu:
        env.update(CUDA_VISIBLE_DEVICES=GPUS[task],JAX_PLATFORMS='cuda,cpu');env.pop('JAX_SKIP_CUDA_CONSTRAINTS_CHECK',None)
        py='.venv/bin/python';cpus=cpu_set_for('payload_lift' if task=='lift' else 'payload_box')
    else:
        env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_SKIP_CUDA_CONSTRAINTS_CHECK='1')
        py='.venv-stage2/bin/python';cpus=cpu_set_for('lift_evaluation' if task=='lift' else 'box_evaluation')
    with (OUT/'audit'/f'{task}_{label}.log').open('a') as log:
        p=subprocess.Popen(['taskset','-c',cpus,py,'-u','-m',*args],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status(task,'running',step=label,pid=p.pid,gpu=(2 if task=='lift' else 3) if gpu else None)
        code=p.wait()
    if code:raise RuntimeError(f'{label} failed ({code}); see audit/{task}_{label}.log')

def train(task,label,mass,envs,updates,source=None):
    folder=OUT/task/'training'/label
    cp=folder/'checkpoints'/label/f'checkpoint_{updates}'
    if (folder/'completed.json').exists():return cp
    args=[MODULE,'train','--task',task,'--label',label,'--mass',str(mass),'--envs',str(envs),'--updates',str(updates)]
    if source:args+=['--checkpoint',str(source)]
    for attempt in range(3):
        if (OUT/task/'STOP_BACKGROUND').exists():raise RuntimeError('Stop marker present')
        try:run(task,args,label,gpu=True);return cp
        except RuntimeError:
            if attempt==2:raise
            time.sleep(30)

def evaluate(task,label,cp,mass):
    path=OUT/task/'evaluations'/label/'results.json'
    if not path.exists():
        run(task,[MODULE,'evaluate','--task',task,'--label',label,'--checkpoint',str(cp),'--mass',str(mass)],'eval_'+label)
    return json.loads(path.read_text())

def refresh(task):
    # Independent report producer uses a file lock; other GPU continues to train.
    run(task,['musclemimic.runner.workflows.payload_refinement_report'],'report')

def main(task):
    os.chdir(ROOT);folder=OUT/task
    lock=(folder/'background.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status(task,'checking',supervisor_pid=os.getpid(),session='payload_refine_'+task)
    assert json.loads((folder/'readiness.json').read_text())['passed']
    mass=2 if task=='lift' else 1
    small=train(task,'smoke',mass,32,4)
    result=evaluate(task,'smoke',small,mass)
    rows=[json.loads(x) for x in (folder/'training/smoke/metrics.jsonl').read_text().splitlines()]
    finite=bool(rows) and all(math.isfinite(r['PPO/total_loss']) for r in rows)
    stable=all(r['peak_root_error_m']<.25 for r in result['records'])
    if task=='bend':stable=stable and max(r['peak_penetration_m'] for r in result['records'])<.003
    save(folder/'smoke_gate.json',dict(passed=finite and stable,finite_loss=finite,stable_rollouts=stable,envs=32,updates=4))
    assert finite and stable,'Small environment gate failed'
    source=train(task,'pilot_20',mass,4096,20)
    result=evaluate(task,'pilot_20',source,mass)
    save(folder/'latest_evaluation.json',dict(label='pilot_20',checkpoint=str(source),mass_kg=mass))
    refresh(task)
    masses=[2] if task=='lift' else [1,3,5]
    for weight in masses:
        passed=result['all_quality_pass'] and weight==mass
        if not passed:
            for round_no in [1,2]:
                updates=200 if weight<5 else 400
                label=f'refine_{weight}kg_{round_no}'
                source=train(task,label,weight,4096,updates,source)
                result=evaluate(task,label,source,weight)
                save(folder/'latest_evaluation.json',dict(label=label,checkpoint=str(source),mass_kg=weight))
                refresh(task)
                if result['all_quality_pass']:passed=True;break
        if not passed:
            status(task,'needs_policy_review',checkpoint=str(source),mass_kg=weight,
                reason='Unchanged physical/tracking gates still fail after two targeted stages');return
        mass=weight
    status(task,'complete',checkpoint=str(source),mass_kg=mass)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['lift','bend']);a=p.parse_args()
    try:main(a.task)
    except Exception:
        status(a.task,'error',error=traceback.format_exc());raise
