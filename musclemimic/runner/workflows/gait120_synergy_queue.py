from musclemimic.runner.jobs.resources import gpu_for, cpu_set_for
"""Durable, fail-closed queue after BOTH payload-refinement pipelines terminate.

Only the standard library is imported while waiting. GPU smoke tests and full
training use a frozen source directory after a two-minute quiet period.
"""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

GPUS = {'ppo': gpu_for('gait_ppo', '2'),
        'synergy_reward': gpu_for('gait_synergy', '3')}
TERMINAL = ('complete', 'needs_policy_review')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def stamp(**values):
    return dict(time=datetime.datetime.now().astimezone().isoformat(), **values)


def live_command(pid):
    try:
        return Path(f'/proc/{int(pid)}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (OSError, ValueError, TypeError):
        return ''


def payload_processes():
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        command = live_command(entry.name)
        # Match the program/module token, not text in a parent shell command.
        if is_payload_python(command):
            matches.append(dict(pid=int(entry.name), command=command))
    return matches


def is_payload_python(command):
    tokens = command.split()
    # tmux servers retain their original launch command after the task exits.
    # A substring match would wait forever, including on our own tmux session.
    if not tokens or not Path(tokens[0]).name.startswith('python'):
        return False
    return any(token.endswith('payload_refinement_background.py') or
               token in ('musclemimic.experiments.payload_refinement',
                         'musclemimic.experiments.payload_refinement_report',
                         'musclemimic.runner.workflows.payload_refinement',
                         'musclemimic.runner.workflows.payload_refinement_report') for token in tokens[1:])


def payload_gate(statuses, live):
    reasons = []
    for task in ('lift', 'bend'):
        state = statuses.get(task, {})
        if state.get('phase') not in TERMINAL:
            reasons.append(task+': '+str(state.get('phase', 'missing status')))
    if live:
        reasons.append('payload processes still alive: '+','.join(str(p['pid']) for p in live))
    return not reasons, reasons


def read_payload(project):
    statuses = {}
    for task in ('lift', 'bend'):
        path = project/'outputs/payload_refinement_v3'/task/'background_status.json'
        try:
            statuses[task] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            statuses[task] = dict(phase='unreadable', error=str(exc))
    live = payload_processes()
    ready, reasons = payload_gate(statuses, live)
    return dict(ready=ready, reasons=reasons, statuses=statuses, processes=live)


def require_release(root):
    release = json.loads((root/'release.json').read_text())
    assert release['authorized_after_payload'] and release['quiet_seconds'] >= 120
    evidence = read_payload(Path(release['project']))
    if not evidence['ready']:
        raise RuntimeError('Payload gate closed again: '+str(evidence['reasons']))
    if (root/'STOP_QUEUE').exists():
        raise RuntimeError('STOP_QUEUE present; no further jobs will start')


def gpu_available(uuid):
    """Query only the selected healthy GPU; never enumerate the stuck GPU4."""
    result = subprocess.run(['nvidia-smi', '-i', uuid, '--query-compute-apps=pid,gpu_uuid',
                             '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        return False, result.stderr.strip()
    lines = [line for line in result.stdout.splitlines() if uuid in line]
    return not lines, lines


def verify_snapshot(root):
    record = json.loads((root/'source_snapshot.json').read_text())
    for name, sha in record['files'].items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Frozen source/input changed: '+name)


def runtime_environment(root, arm, gpu):
    env = os.environ.copy()
    source = root/'source'
    env.update(PYTHONPATH=os.pathsep.join(map(str, [source, source/'models/myo_sim', source/'models/assist_sim'])),
               MUJOCO_GL='osmesa', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LP_NUM_THREADS='1',
               XLA_PYTHON_CLIENT_PREALLOCATE='false', CUDA_VISIBLE_DEVICES=GPUS[arm] if gpu else '',
               JAX_PLATFORMS='cuda,cpu' if gpu else 'cpu')
    return env


def run(root, project, arm, module, arguments, label, gpu):
    require_release(root)
    verify_snapshot(root)
    cpus = cpu_set_for(('gait_ppo' if arm == 'ppo' else 'gait_synergy') + ('' if gpu else '_evaluation'))
    command = ['taskset', '-c', cpus, str(project/'.venv/bin/python'), '-u', '-m', module,
               *arguments, '--root', str(root)]
    with (root/'logs'/f'{arm}_{label}.log').open('a') as stream:
        process = subprocess.Popen(command, cwd=root/'source', env=runtime_environment(root, arm, gpu),
                                   stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        save(root/arm/'queue_status.json', stamp(phase='running', stage=label, pid=process.pid,
             gpu_uuid=GPUS[arm] if gpu else None, command=command))
        code = process.wait()
    if code:
        raise RuntimeError(f'{arm}/{label} exited with {code}; see logs/{arm}_{label}.log')


def worker(root, project, arm):
    try:
        module = 'musclemimic.experiments.gait120_synergy_train'
        evaluator = 'musclemimic.experiments.gait120_synergy_evaluate'
        run(root, project, arm, module, ['train', '--arm', arm, '--label', 'smoke', '--envs', '32', '--updates', '2'], 'smoke', True)
        smoke = json.loads((root/arm/'smoke/completed.json').read_text())
        run(root, project, arm, evaluator, ['--arm', arm, '--checkpoint', smoke['checkpoint'], '--label', 'smoke', '--only-training-clip'], 'smoke_evaluation', False)
        rows = [json.loads(line) for line in (root/arm/'smoke/metrics.jsonl').read_text().splitlines()]
        import math
        finite = len(rows) >= 2 and all(math.isfinite(float(row['PPO/total_loss'])) for row in rows)
        result = json.loads((root/arm/'evaluations/smoke/summary.json').read_text())
        usable = result['clips'][0]['finite_rollout'] and result['clips'][0]['steps'] > 0
        save(root/arm/'smoke_gate.json', stamp(passed=finite and usable, finite_ppo_losses=finite,
             finite_cpu_rollout=usable, walking_quality_required_at_smoke=False))
        if not finite or not usable:
            raise RuntimeError('GPU smoke / CPU rollout gate failed; main training not started')
        run(root, project, arm, module, ['train', '--arm', arm, '--label', 'main', '--envs', '4096', '--updates', '10000'], 'main', True)
        result = json.loads((root/arm/'main/completed.json').read_text())
        run(root, project, arm, evaluator, ['--arm', arm, '--checkpoint', result['checkpoint'], '--label', 'final'], 'final_evaluation', False)
        save(root/arm/'queue_status.json', stamp(phase='complete', checkpoint=result['checkpoint']))
        return dict(arm=arm, complete=True)
    except Exception:
        error = traceback.format_exc()
        save(root/arm/'queue_status.json', stamp(phase='error', error=error))
        return dict(arm=arm, complete=False, error=error)


def compare_completed(root):
    """Keep constants and motion failures visible; compare like-for-like groups."""
    arms = {}
    for arm in GPUS:
        path = root/arm/'evaluations/final/summary.json'
        if path.exists():
            arms[arm] = {row['clip']: row for row in json.loads(path.read_text())['clips']}
    rows = []
    if len(arms) == 2:
        for clip in sorted(set(arms['ppo']) & set(arms['synergy_reward'])):
            pair = {arm: data[clip] for arm, data in arms.items()}
            if any(row['primary'] is None for row in pair.values()):
                rows.append(dict(clip=clip, winner_on_common_groups_if_both_quality_pass=None,
                                 reason='At least one run has no usable activation samples'))
                continue
            common = [name for name in pair['ppo']['primary']['groups']
                      if all(p['primary']['groups'][name]['pearson_r'] is not None for p in pair.values())]
            means = {arm: sum(row['primary']['groups'][name]['pearson_r'] for name in common)/len(common)
                     if common else None for arm, row in pair.items()}
            eligible = all(row['quality']['passed'] for row in pair.values()) and bool(common)
            winner = max(means, key=means.get) if eligible else None
            rows.append(dict(clip=clip, common_valid_groups=common, common_group_mean_r=means,
                quality_pass={arm: row['quality']['passed'] for arm, row in pair.items()},
                winner_on_common_groups_if_both_quality_pass=winner,
                all_six_mean_r={arm: row['primary']['all_six_mean_r'] for arm, row in pair.items()}))
    save(root/'comparison.json', stamp(available_arms=list(arms), clips=rows,
        note='No EMG ranking for motion failures; left symmetry is not an observation; Trial05 is descriptive.'))


def main(root, project):
    root, project = root.resolve(), project.resolve()
    (root/'logs').mkdir(exist_ok=True)
    lock = (root/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((root/'preflight.json').read_text())['passed']
    verify_snapshot(root)
    quiet_since = None
    while True:
        if (root/'STOP_QUEUE').exists():
            save(root/'queue_status.json', stamp(phase='stopped_before_training', pid=os.getpid()))
            return
        evidence = read_payload(project)
        if evidence['ready']:
            try:
                gpu_checks = {arm: gpu_available(uuid) for arm, uuid in GPUS.items()}
                ready = all(result[0] for result in gpu_checks.values())
            except (OSError, subprocess.TimeoutExpired) as exc:
                gpu_checks, ready = dict(error=str(exc)), False
        else:
            gpu_checks, ready = {}, False
        if ready:
            quiet_since = quiet_since or time.monotonic()
        else:
            quiet_since = None
        quiet = 0 if quiet_since is None else time.monotonic()-quiet_since
        save(root/'queue_status.json', stamp(phase='waiting_for_payload', pid=os.getpid(),
             predecessor_thread='添加物体交互', predecessor=evidence, gpu_checks=gpu_checks,
             quiet_seconds=quiet, training_started=False))
        if quiet >= 120:
            break
        time.sleep(30)
    save(root/'release.json', stamp(authorized_after_payload=True, project=str(project), quiet_seconds=quiet,
         predecessor=evidence, gpu_checks=gpu_checks))
    save(root/'queue_status.json', stamp(phase='running_experiments', pid=os.getpid(), training_started=True))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, root, project, arm) for arm in GPUS]
        results = [future.result() for future in futures]
    compare_completed(root)
    save(root/'queue_status.json', stamp(phase='complete' if all(r['complete'] for r in results) else 'needs_review',
                                       pid=os.getpid(), results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root, args.project)
    except Exception:
        save(args.root/'queue_status.json', stamp(phase='error', pid=os.getpid(), error=traceback.format_exc()))
        raise
