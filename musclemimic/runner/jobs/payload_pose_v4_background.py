from musclemimic.runner.jobs.resources import gpu_for, cpu_set_for
from musclemimic.utils.paths import workspace_root, config_root
"""Third, version-isolated 1 kg box pass with explicit box-pose tracking."""
import datetime, fcntl, json, math, os, shutil, subprocess, time, traceback
from pathlib import Path

ROOT=workspace_root()
OUT=ROOT/'outputs/payload_refinement_v4'
os.environ['PAYLOAD_REFINEMENT_OUT']=str(OUT)
import numpy as np
from omegaconf import OmegaConf, open_dict
from musclemimic.runner.workflows import payload_refinement as pr

SOURCE=ROOT/'outputs/payload_refinement_v3/bend/training/refine_1kg_2/checkpoints/refine_1kg_2/checkpoint_200'
GPU=gpu_for('box_pose', '1')
STATE={}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def status(phase,**values):
    STATE.update(time=datetime.datetime.now().astimezone().isoformat(),phase=phase,**values)
    save(OUT/'bend/background_status.json',STATE);print(STATE,flush=True)


def make_cpu(config):
    cfg=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
    cfg.experiment.env_params.reward_type='BoxPoseRefinementReward'
    return pr.make_cpu_original(cfg)


def prepare():
    from musclemimic.runner.workflows.payload_refinement import save_json
    folder=OUT/'bend';folder.mkdir(parents=True,exist_ok=True)
    source=ROOT/'outputs/payload_refinement_v3/bend'
    shutil.copy2(source/'reference.npz',folder/'reference.npz')
    shutil.copy2(source/'readiness.json',folder/'readiness.json')
    cfg=OmegaConf.load(source/'base_config.yaml')
    ep=cfg.experiment.env_params
    ep.reward_type='BoxPoseRefinementReward';ep.payload_mass_kg=1.
    weights=np.zeros(318);weights[:316]+=.3/316;weights[:65]+=.9/65;weights[110:160]+=.8/50
    ep.th_params=dict(random_start=True,fixed_start_conf=None,start_from_random_step=True,start_step_weights=weights.tolist())
    cfg.experiment.task_factory.params.custom_dataset_conf.traj_path=str((folder/'reference.npz').resolve())
    cfg.experiment.validation.th_params=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
    OmegaConf.save(cfg,folder/'base_config.yaml')
    save(folder/'changes.json',dict(source=str(SOURCE),reference_sha256=__import__('hashlib').sha256((folder/'reference.npz').read_bytes()).hexdigest(),
        mass_kg=1.,observation_dim=2838,action_dim=416,
        reward='previous grip/force/position objective plus 0.8/(1+(box quaternion angle / 0.40 rad)^2)',
        sampling='0.3 uniform + 0.9 frames 0-64 + 0.8 frames 110-159; normalized',
        geometry_unchanged=True,physics_unchanged=True))
    before=source/'evaluations/refine_1kg_2'
    baseline=folder/'evaluations/baseline';baseline.mkdir(parents=True,exist_ok=True)
    result=json.loads((before/'results.json').read_text());result['label']='baseline';result['checkpoint']=str(SOURCE)
    save(baseline/'results.json',result)
    for f in before.glob('start_*_rollout.npz'):shutil.copy2(f,baseline/f.name)
    # Backfill the new orientation readout from the already saved identical rollouts.
    for episode in result['records']:
        z=np.load(baseline/f"start_{episode['start']}_rollout.npz")
        q=z['qpos'][:,132:136];r=z['qref'][:,132:136]
        q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
        r=r/np.maximum(np.linalg.norm(r,axis=1,keepdims=True),1e-12)
        angle=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(q*r,axis=1)),0,1)))
        for row,value in zip(episode['trajectory'],angle):row['object_orientation_error_deg']=float(value)
        episode['mean_object_orientation_error_deg']=float(np.mean(angle))
        episode['peak_object_orientation_error_deg']=float(np.max(angle))
    save(baseline/'results.json',result)
    save(folder/'latest_evaluation.json',dict(label='baseline',checkpoint=str(SOURCE),mass_kg=1.))
    pr.OUT=OUT;pr.make_cpu_original=pr.make_cpu;pr.make_cpu=make_cpu


def run(args,label,gpu=False):
    env=os.environ.copy();env.update(MUJOCO_GL='osmesa',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if gpu:
        env.update(CUDA_VISIBLE_DEVICES=GPU,JAX_PLATFORMS='cuda,cpu');env.pop('JAX_SKIP_CUDA_CONSTRAINT_CHECK',None)
        py='.venv/bin/python';cpus=cpu_set_for('box_pose')
    else:
        env.update(CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',JAX_SKIP_CUDA_CONSTRAINT_CHECK='1')
        py='.venv-stage2/bin/python';cpus=cpu_set_for('box_evaluation')
    logpath=OUT/'audit'/f'{label}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
    with logpath.open('a') as log:
        proc=subprocess.Popen(['taskset','-c',cpus,py,'-u','-c',args],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status('running',step=label,pid=proc.pid,gpu=1 if gpu else None)
        rc=proc.wait()
    if rc:raise RuntimeError(f'{label} exited with {rc}; see {logpath}')


def assigned_gpu_busy():
    # Respect a live queue reservation if this GPU is assigned elsewhere later.
    for p in [ROOT/'outputs/gait120_myolegs26_training_v1/synergy_reward/queue_status.json',
              ROOT/'outputs/gait120_myolegs26_training_v1/ppo/queue_status.json']:
        if not p.exists():continue
        try:q=json.loads(p.read_text())
        except (OSError,json.JSONDecodeError):continue
        if q.get('gpu_uuid')!=GPU or q.get('phase')!='running':continue
        try:os.kill(int(q['pid']),0)
        except (OSError,KeyError,TypeError,ValueError):continue
        return p,q
    return None,None


def code(label,mass,envs,updates,source=None):
    return ('from musclemimic.runner.workflows import payload_refinement as p; '
            f'p.OUT=__import__("pathlib").Path("{OUT}"); '
            'p.make_cpu_original=p.make_cpu; '
            'p.make_cpu=lambda c: (__import__("fullbody.payload_pose_v4_background",fromlist=["make_cpu"]).make_cpu(c)); '
            f'p.train("bend","{label}",{envs},{updates},{mass},"{source or SOURCE}")')


def evaluate(label,checkpoint):
    from musclemimic.runner.workflows import payload_refinement as p
    p.OUT=OUT
    return p.evaluate('bend',label,checkpoint,1.)


def main():
    os.chdir(ROOT);folder=OUT/'bend'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'background.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status('checking',supervisor_pid=os.getpid(),session='payload_pose_v4')
    assert SOURCE.exists()
    prepare()
    reserved,q=assigned_gpu_busy()
    while reserved is not None:
        status('waiting_for_gpu',gpu=1,waiting_for=str(reserved),other_pid=q['pid'],other_stage=q.get('stage'))
        time.sleep(60)
        reserved,q=assigned_gpu_busy()
    # Four-update CPU/GPU smoke catches observation, reward and checkpoint-transfer issues.
    source=folder/'training/pose_smoke/checkpoints/pose_smoke/checkpoint_4'
    if not (folder/'training/pose_smoke/completed.json').exists():
        run(code('pose_smoke',1.,32,4), 'pose_smoke',gpu=True)
    else:
        status('eval_pose_smoke',gpu=1,step='pose_smoke')
    eval_file=folder/'evaluations/pose_smoke/results.json'
    if eval_file.exists() and json.loads(eval_file.read_text()).get('checkpoint')==str(source):
        result=json.loads(eval_file.read_text())
    else:
        result=evaluate('pose_smoke',source)
    rows=[json.loads(x) for x in (folder/'training/pose_smoke/metrics.jsonl').read_text().splitlines()]
    finite=bool(rows) and all(math.isfinite(x['PPO/total_loss']) for x in rows)
    stable=all(x['peak_root_error_m']<.25 and x['peak_penetration_m']<.01 for x in result['records'])
    save(folder/'smoke_gate.json',dict(passed=finite and stable,finite_loss=finite,stable_rollouts=stable,
        max_peak_penetration_m=max(x['peak_penetration_m'] for x in result['records']),
        task_failure_count=sum(x['termination']=='task_failure' for x in result['records']),
        envs=32,updates=4,penetration_limit_m=.01))
    assert finite and stable,'Smoke gate failed; do not start large run'
    # Load final 1 kg policy, keep model/optimizer update flow consistent with previous stages.
    source=folder/'training/pose_1kg_1/checkpoints/pose_1kg_1/checkpoint_200'
    if not (folder/'training/pose_1kg_1/completed.json').exists():
        run(code('pose_1kg_1',1.,4096,200,SOURCE),'pose_1kg_1',gpu=True)
    result=evaluate('pose_1kg_1',source)
    save(folder/'latest_evaluation.json',dict(label='pose_1kg_1',checkpoint=str(source),mass_kg=1.))
    run('from musclemimic.runner.workflows import payload_refinement_report as r; r.main()','report')
    status('complete' if result['all_quality_pass'] else 'needs_policy_review',checkpoint=str(source),mass_kg=1.,
        complete_count=result['complete_count'],quality_count=result['quality_count'],
        reason=None if result['all_quality_pass'] else 'Pose-targeted stage did not clear every existing 1 kg gate')


if __name__=='__main__':
    try:main()
    except Exception:
        status('error',error=traceback.format_exc());raise
