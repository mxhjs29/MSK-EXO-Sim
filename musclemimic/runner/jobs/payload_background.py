from musclemimic.runner.jobs.resources import gpu_for, cpu_set_for
from musclemimic.utils.paths import workspace_root, config_root
"""Detach payload training recovery, evaluation and reporting from the chat.

Run in tmux from the repository root. Only uses the two explicitly assigned GPUs.
Existing trainers are observed; they are never duplicated or killed by this script.
"""
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT = workspace_root()
OUT = ROOT / 'outputs/payload_finetune_v1'
AUDIT = OUT / 'audit'
STATE = {}
MUTEX = threading.Lock()
GPUS = {2: gpu_for('payload_lift', '2'),
        3: gpu_for('payload_box', '3')}
WARM = OUT / 'lift/training/curriculum_05kg/checkpoints/curriculum_05kg/checkpoint_20'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def status(key, **value):
    with MUTEX:
        STATE[key] = dict(time=datetime.datetime.now().astimezone().isoformat(), **value)
        save(AUDIT / 'background_status.json', STATE)
    print(key, value, flush=True)


def existing_trainers(label):
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if (b'musclemimic.runner.workflows.payload_finetune' in args and
                    b'train' in args and label.encode() in args):
                found.append(int(p.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def command(args, name, gpu=None):
    env = os.environ.copy()
    env.update(MUJOCO_GL='osmesa', OPENBLAS_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if gpu is None:
        env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu',
                   JAX_SKIP_CUDA_CONSTRAINTS_CHECK='1')
        python, cpus = '.venv-stage2/bin/python', cpu_set_for('evaluation')
    else:
        env.update(CUDA_VISIBLE_DEVICES=GPUS[gpu], JAX_PLATFORMS='cuda,cpu')
        env.pop('JAX_SKIP_CUDA_CONSTRAINTS_CHECK', None)
        python, cpus = '.venv/bin/python', cpu_set_for('payload_lift' if gpu == 2 else 'payload_box')
    with (AUDIT / (name + '.log')).open('a') as log:
        p = subprocess.Popen(['taskset', '-c', cpus, python, '-u', '-m', *args],
                             cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
        status(name, phase='running', pid=p.pid, gpu=gpu)
        code = p.wait()
        status(name, phase='finished' if code == 0 else 'error', exit_code=code)
        if code:
            raise RuntimeError(f'{name} exited {code}; see audit/{name}.log')


def train(label, mass, gpu, updates=20, source=WARM, lr=None):
    folder = OUT / 'lift/training' / label
    for attempt in range(3):
        while (pids := existing_trainers(label)):
            status(label, phase='observing_existing_trainer', pids=pids, gpu=gpu)
            time.sleep(20)
        if (folder / 'completed.json').exists():
            status(label, phase='training_complete')
            return
        if (AUDIT / 'STOP_BACKGROUND').exists():
            raise RuntimeError('STOP_BACKGROUND requested; no new training launched')
        args = ['musclemimic.runner.workflows.payload_finetune', 'train', '--task', 'lift',
                '--mass', str(mass), '--envs', '4096', '--updates', str(updates),
                '--label', label, '--start-mode', 'beginning', '--checkpoint', str(source)]
        if lr is not None:
            args += ['--lr', str(lr)]
        try:
            command(args, label + '_background', gpu)
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(30)
    if not (folder / 'completed.json').exists():
        raise RuntimeError(f'{label} did not complete')


def checkpoint(label, update):
    return OUT / 'lift/training' / label / 'checkpoints' / label / f'checkpoint_{update}'


def evaluate(cp, label):
    path = OUT / 'lift/evaluations' / label / 'results.json'
    if not path.exists():
        command(['musclemimic.runner.workflows.payload_finetune', 'evaluate', '--task', 'lift',
                 '--mass', '2', '--checkpoint', str(cp), '--label', label], 'eval_' + label)
    return json.loads(path.read_text())


def rank(result):
    rows = result['records']
    return (sum(r['quality_pass'] for r in rows),
            sum(r['termination'] == 'reference_end' for r in rows),
            -sum(r['joint_mae_deg'] for r in rows),
            -sum(r['peak_root_error_m'] for r in rows))


def main():
    os.chdir(ROOT)
    AUDIT.mkdir(parents=True, exist_ok=True)
    lock = (AUDIT / 'background.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('supervisor', phase='running', pid=os.getpid(), session='payload_v1',
           recovery='Resume latest saved checkpoint if the original trainer exits')
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(train, 'curriculum_1kg_fullstart_v2', 1, 2),
                pool.submit(train, 'target_2kg_fullstart', 2, 3)]
        for job in jobs:
            job.result()
    candidates = []
    for stage in ['target_2kg_fullstart', 'curriculum_1kg_fullstart_v2']:
        for update in [10, 20]:
            cp = checkpoint(stage, update)
            if cp.exists():
                candidates.append(evaluate(cp, f'{stage}_cp{update}_at2kg'))
    best = max(candidates, key=rank)
    # A bounded continuation if neither initial branch meets the physical quality gate.
    for round_number in range(1, 4):
        if best['all_quality_pass']:
            break
        stage = f'target_2kg_refine_{round_number}'
        train(stage, 2, 2, updates=20, source=best['checkpoint'], lr=1e-5)
        for update in [10, 20]:
            candidates.append(evaluate(checkpoint(stage, update), stage + f'_cp{update}'))
        best = max(candidates, key=rank)
    final = evaluate(best['checkpoint'], 'final_2kg')
    save(OUT / 'final_selection.json', dict(
        lift=dict(checkpoint=best['checkpoint'], all_quality_pass=final['all_quality_pass'],
                  selection='4 deterministic starts; quality count, completion, joint MAE, root error'),
        bend=dict(checkpoint=str(OUT / 'bend/training/finetune_5kg/checkpoints/finetune_5kg/checkpoint_20'),
                  all_quality_pass=json.loads((OUT / 'bend/evaluations/final_5kg/results.json').read_text())['all_quality_pass'])))
    for task, labels in [('lift', ['original_2kg', 'final_2kg']), ('bend', ['original_5kg', 'final_5kg'])]:
        for label in labels:
            if not (OUT / task / 'evaluations' / label / 'rollout.mp4').exists():
                command(['musclemimic.runner.workflows.payload_report', '--task', task, '--label', label],
                        f'video_{task}_{label}')
    command(['musclemimic.runner.workflows.payload_report'], 'build_report')
    status('supervisor', phase='complete' if final['all_quality_pass'] else 'needs_quality_review',
           report=str(OUT / 'report.html'))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        status('supervisor', phase='error', traceback=traceback.format_exc())
        raise
