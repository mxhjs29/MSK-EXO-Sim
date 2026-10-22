"""Prepare, inspect, and fine-tune ideal-grip payload tasks from existing PPOs."""
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
from omegaconf import OmegaConf,open_dict

OUT=Path('outputs/payload_finetune_v1')
SOURCE=dict(lift='outputs/fullbody_lift_dumbbell/checkpoints/fullbody_lift_dumbbell/checkpoint_7400',
            bend='outputs/fullbody_bend/checkpoints/fullbody_bend/checkpoint_10000')


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def base_runtime(task):
    import yaml
    from musclemimic.runner.frozen_policy import Runtime
    from musclemimic.utils.paths import config_root
    cfg=yaml.safe_load((config_root()/'evaluation/frozen_policy.yaml').read_text())
    return Runtime(cfg['tasks'][task])


def make_cpu(config):
    from loco_mujoco.task_factories import TaskFactory
    cfg=OmegaConf.create(OmegaConf.to_container(config,resolve=True))
    ep=cfg.experiment.env_params
    ep.env_name='MyoFullBodyPayloadEnv';ep.mjx_backend='jax';ep.num_envs=1
    ep.th_params=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
    return TaskFactory.get_factory_cls('ImitationFactory').make(**ep,**cfg.experiment.task_factory.params)


def render(env,folder,frame_indices,policy=None):
    from PIL import Image,ImageDraw
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    m,d=env._model,env._data
    m.vis.global_.offwidth=1280;m.vis.global_.offheight=900
    renderer=mujoco.Renderer(m,height=720,width=960)
    opt=mujoco.MjvOption();opt.sitegroup[:]=0;opt.tendongroup[:]=0
    snapshots=[]
    for f in frame_indices:
        if policy is None:
            d.qpos[:]=env.th.traj.data.get_qpos(0,f,np);d.qvel[:]=0;mujoco.mj_forward(m,d)
        for view in ['full','grip']:
            cam=mujoco.MjvCamera();cam.azimuth=135 if view=='full' else 110;cam.elevation=-12
            cam.lookat[:]=d.xpos[env.payload_body_id] if view=='grip' else d.xpos[m.body('pelvis').id]+[0,0,.1]
            cam.distance=.48 if view=='grip' else 2.4
            renderer.update_scene(d,camera=cam,scene_option=opt)
            pic=Image.fromarray(renderer.render());draw=ImageDraw.Draw(pic)
            grip_mode='friction contact' if hasattr(env,'hand_friction') else 'ideal grip'
            draw.rectangle((0,0,960,30),fill='white');draw.text((10,8),f'{env.payload_kind} {env.payload_mass_kg:.2f} kg | frame {f} | {view} | {grip_mode}',fill='black')
            path=folder/f'frame_{f:04d}_{view}.png';pic.save(path);snapshots.append(str(path))
    renderer.close();return snapshots


def prepare(task):
    import jax
    from musclemimic.environments.humanoids.myofullbody_payload import MyoFullBodyPayloadEnv
    from loco_mujoco.trajectory import Trajectory,TrajectoryInfo,TrajectoryData,TrajectoryModel
    from loco_mujoco.trajectory.handler import materialize_trajectory
    rt=base_runtime(task);rt.reset(2026)
    folder=OUT/task;folder.mkdir(parents=True,exist_ok=True)
    config=OmegaConf.create(OmegaConf.to_container(rt.config,resolve=True))
    ep=OmegaConf.to_container(config.experiment.env_params,resolve=True)
    ep.pop('env_name',None);ep.update(mjx_backend='jax',num_envs=1,payload_kind='dumbbell' if task=='lift' else 'box',payload_mass_kg=2. if task=='lift' else 5.)
    if task=='lift':
        grip=json.loads((OUT/'audit/grip_fit.json').read_text())
        ep.update(payload_mount_offset=grip['mount_offset_m'],payload_finger_posture=grip['joint_angles_rad'])
    env=MyoFullBodyPayloadEnv(**ep)
    m,d=env._model,env._data
    source_count=int(rt.env.th.len_trajectory(0))
    start,end=(0,source_count) if task=='lift' else (140,481)
    original=np.asarray(rt.env.th.traj.data.qpos)[start:end].copy()
    qpos=np.broadcast_to(m.qpos0,(len(original),m.nq)).copy()
    qpos[:,:rt.model.nq]=original
    for name,value in env.payload_finger_posture.items():qpos[:,m.jnt_qposadr[m.joint(name).id]]=value
    if task=='bend':
        from scipy.optimize import least_squares
        hand_ids=[m.site('S_grasp_'+s).id for s in ['r','l']]
        dependent={int(m.eq_obj1id[e]) for e in range(m.neq) if m.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT}
        arm_names=[stem+'_'+side for side in ['r','l'] for stem in ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion'] if m.joint(stem+'_'+side).id not in dependent]
        arm_ids=np.array([m.jnt_qposadr[m.joint(n).id] for n in arm_names],int)
        bounds=np.array([m.jnt_range[m.joint(n).id] for n in arm_names])
        coupled=[e for e in range(m.neq) if m.eq_type[e]==mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[e]>=0 and m.jnt_qposadr[m.eq_obj2id[e]] in arm_ids]
        dep=m.jnt_qposadr[m.eq_obj1id[coupled]];drivers=m.jnt_qposadr[m.eq_obj2id[coupled]]
        coefficients=m.eq_data[coupled,:5].copy()
        def project_couplings():
            delta=d.qpos[drivers]-m.qpos0[drivers]
            d.qpos[dep]=m.qpos0[dep]+np.sum(coefficients*delta[:,None]**np.arange(5),axis=1)
        errors=[];max_changes=[]
        previous_correction=np.zeros(len(arm_ids))
        for i in range(len(qpos)):
            d.qpos[:]=qpos[i]
            project_couplings();mujoco.mj_kinematics(m,d)
            right,left=d.site_xpos[hand_ids].copy();middle=(right+left)/2
            orig=qpos[i,arm_ids].copy()
            initial=np.clip(orig+previous_correction,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
            def residual(v):
                d.qpos[arm_ids]=v;project_couplings();mujoco.mj_kinematics(m,d)
                r,l=d.site_xpos[hand_ids]
                return np.r_[(np.linalg.norm(l-r)-.21)*1000,((l+r)/2-middle)*10,(v-orig)*.02,(v-initial)*.15]
            fit=least_squares(residual,initial,bounds=(bounds[:,0],bounds[:,1]),max_nfev=150)
            previous_correction=fit.x-orig
            d.qpos[arm_ids]=fit.x;project_couplings();mujoco.mj_kinematics(m,d)
            right,left=d.site_xpos[hand_ids].copy();middle=(right+left)/2
            x=(left-right)/np.linalg.norm(left-right);z=np.array([0.,0.,1.]);z-=x*np.dot(z,x);z/=np.linalg.norm(z);y=np.cross(z,x)
            R=np.column_stack([x,y,z]);quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.ravel())
            targets=np.array([middle-x*.105,middle+x*.105])
            qpos[i]=d.qpos.copy();qpos[i,-7:]=np.r_[middle-z*.055,quat]
            d.qpos[:]=qpos[i];mujoco.mj_kinematics(m,d)
            errors.append(float(np.max(np.linalg.norm(d.site_xpos[hand_ids]-targets,axis=1))))
            max_changes.append(float(np.rad2deg(abs(fit.x-orig)).max()))
        max_jump=float(np.rad2deg(abs(np.diff(qpos[:,arm_ids],axis=0))).max())
        save_json(folder/'reference_adjustment.json',dict(start=start,end_exclusive=end,arm_joints=arm_names,max_grip_error_m=max(errors),max_joint_change_deg=max(max_changes),max_frame_joint_change_deg=max_jump,frame_errors_m=errors))
        assert max(errors)<.003,('IK grip alignment',max(errors),arm_names)
        assert max_jump<4.,('IK temporal smoothness',max_jump)
    info=TrajectoryInfo([m.joint(i).name for i in range(m.njnt)],TrajectoryModel(m.njnt,np.asarray(m.jnt_type)),100.,metadata=dict(task=task,source_start_frame=start,source_frame_count=source_count,ideal_grip=True))
    traj=Trajectory(info=info,data=TrajectoryData(qpos=qpos,qvel=np.zeros((len(qpos),m.nv)),split_points=np.array([0,len(qpos)])))
    traj=materialize_trajectory(traj,model=m,control_dt=.01,cache_type='sparse',site_names=env.sites_for_mimic)
    traj.save(str(folder/'reference.npz'))
    env.load_trajectory(traj=traj,cache_type='sparse',site_names=env.sites_for_mimic,warn=False)
    peak=int(np.argmax(qpos[:,m.jnt_qposadr[m.joint('elbow_flexion_r').id]])) if task=='lift' else len(qpos)//3
    pictures=render(env,folder/'reference_images',[0,peak,2*len(qpos)//3,len(qpos)-1])
    with open_dict(config):
        config.experiment.env_params.env_name='MjxMyoFullBodyPayloadEnv'
        config.experiment.env_params.payload_kind=ep['payload_kind']
        config.experiment.env_params.payload_mass_kg=ep['payload_mass_kg']
        config.experiment.env_params.payload_mount_offset=env.payload_mount_offset.tolist()
        config.experiment.env_params.payload_finger_posture=env.payload_finger_posture
        config.experiment.env_params.goal_type='PayloadGoalTrajMimic'
        config.experiment.env_params.goal_params.source_start_frame=start
        config.experiment.env_params.goal_params.source_frame_count=source_count
        config.experiment.env_params.reward_type='PayloadMimicReward'
        config.experiment.task_factory.params=dict(custom_dataset_conf=dict(traj_path=str((folder/'reference.npz').resolve()),trajectory_cache_type='sparse'))
        config.experiment.validation.amass_dataset_conf=None
        config.experiment.validation.c3d_dataset_conf=None
        config.experiment.validation.custom_dataset_conf=None
        config.experiment.validation.num_steps=len(qpos)-1
        config.experiment.validation.video_length=len(qpos)-1
        config.experiment.validation.video_frequency=0
        config.experiment.validation.num=4
        config.experiment.validation.num_envs=4
        config.experiment.validation.interval=20
        config.experiment.validation.joints_to_ignore=['payload_free'] if task=='bend' else []
        config.experiment.env_params.th_params=dict(random_start=True,fixed_start_conf=None,start_from_random_step=True)
        config.experiment.lr=3e-5
        config.experiment.anneal_lr=False
        config.experiment.total_timesteps_is_absolute=True
        config.experiment.max_checkpoints_to_keep=5
        config.experiment.auto_resume=True
        config.experiment.reset_lr_schedule_on_resume=False
        config.experiment.reset_logging_timestep=True
        config.wandb.mode='disabled'
    OmegaConf.save(config,folder/'base_config.yaml')
    assert env.info.observation_space.shape==rt.env.info.observation_space.shape
    assert env.msk_action_spec==rt.env.msk_action_spec
    save_json(folder/'preparation.json',dict(task=task,source_checkpoint=SOURCE[task],source_start=start,source_end_exclusive=end,
        target_mass_kg=ep['payload_mass_kg'],physical_added_mass_kg=float(m.body_mass.sum()-rt.model.body_mass.sum()),
        payload_body_mass_kg=float(m.body_mass[env.payload_body_id]),observation_shape=list(env.info.observation_space.shape),
        action_shape=list(env.info.action_space.shape),same_muscle_action_order=True,images=pictures,
        grip='rigid hand child' if task=='lift' else 'two soft connect constraints, no world support'))
    env.stop();rt.close();print('PREPARED',task,flush=True)


def evaluate(task,checkpoint=None,mass=None,label='baseline'):
    import jax,jax.numpy as jnp
    from musclemimic.runner.eval_utils import load_checkpoint
    from musclemimic.algorithms import PPOJax
    checkpoint=checkpoint or SOURCE[task]
    config=OmegaConf.load(OUT/task/'base_config.yaml')
    if mass is not None:config.experiment.env_params.payload_mass_kg=mass
    _,agent,_=load_checkpoint(checkpoint)
    env=make_cpu(config)
    conf=PPOJax.init_agent_conf(env,config)
    policy=jax.jit(lambda obs:conf.network.apply(dict(params=agent.train_state.params,run_stats=agent.train_state.run_stats),jnp.atleast_2d(obs))[0].mode())
    folder=OUT/task/'evaluations'/label;folder.mkdir(parents=True,exist_ok=True)
    records=[];n=int(env.th.len_trajectory(0));starts=[0,n//5,2*n//5,3*n//5]
    m=env._model
    joint='elbow_flexion_r' if task=='lift' else 'flex_extension'
    ja=int(m.jnt_qposadr[m.joint(joint).id]);jd=int(m.jnt_dofadr[m.joint(joint).id])
    lift8=['BIClong','BICshort','BRA','BRD','TRIlong','TRIlat','TRImed','ANC']
    muscles=[i for i in range(m.nu) if m.actuator(i).name in lift8+[v+'_r' for v in lift8]]
    if task=='lift':assert len(muscles)==8,[m.actuator(i).name for i in muscles]
    for start in starts:
        env.th.fixed_start_conf=[0,start];env.th.random_start=False
        obs=np.asarray(env.reset(jax.random.PRNGKey(2026)))
        reset_q=env._data.qpos.copy();reset_errors=[]
        if task=='bend':
            for s in ['r','l']:reset_errors.append(float(np.linalg.norm(env._data.site_xpos[env._model.site('S_grasp_'+s).id]-env._data.site_xpos[env._model.site('payload_grip_'+s).id])))
        steps=[];qposes=[];termination='reference_end';images=[]
        for step in range(n-start-1):
            action=np.asarray(policy(obs)).reshape(-1)
            obs,reward,absorbing,done,info=env.step(action)
            if not np.isfinite(env._data.qpos).all():raise FloatingPointError('nonfinite payload rollout')
            row=dict(t=step*.01,reward=float(reward),activation=float(np.sqrt(np.mean(env._data.act**2))),
                object_pos=env._data.xpos[env.payload_body_id].tolist(),root_pos=env._data.qpos[:3].tolist())
            d=env._data;qposes.append(d.qpos.copy())
            ref=env.th.get_current_traj_data(env._additional_carry,np).qpos
            origin=env.th.get_init_traj_data(env._additional_carry,np).qpos[:2]
            refroot=ref[:3].copy();refroot[:2]-=origin
            row.update(joint_angle_rad=float(d.qpos[ja]),reference_angle_rad=float(ref[ja]),
                joint_error_deg=float(np.rad2deg(abs(d.qpos[ja]-ref[ja]))),muscle_torque_nm=float(d.qfrc_actuator[jd]),
                root_error_m=float(np.linalg.norm(d.qpos[:3]-refroot)))
            if task=='lift':row['elbow8_activation_rms']=float(np.sqrt(np.mean(d.act[muscles]**2)))
            row['ground_force_n']=0.;row['payload_non_hand_contacts']=0
            for ci in range(d.ncon):
                ct=d.contact[ci]
                if m.geom('terrain').id in ct.geom:
                    cf=np.zeros(6);mujoco.mj_contactForce(m,d,ci,cf)
                    row['ground_force_n']+=float(abs((ct.frame.reshape(3,3).T@cf[:3])[2]))
                if any(int(g) in env.payload_geom_ids for g in ct.geom):row['payload_non_hand_contacts']+=1
            if task=='bend':
                row['grip_error_m']=max(float(np.linalg.norm(d.site_xpos[m.site('S_grasp_'+s).id]-d.site_xpos[m.site('payload_grip_'+s).id])) for s in ['r','l'])
                a=int(m.jnt_qposadr[m.joint('payload_free').id]);target=ref[a:a+3].copy();target[:2]-=origin
                row['object_error_m']=float(np.linalg.norm(d.qpos[a:a+3]-target))
                row['hand_support_world_n']=[]
                for s in ['r','l']:
                    eid=m.equality('payload_hold_'+s).id
                    rows=np.flatnonzero((d.efc_type==mujoco.mjtConstraint.mjCNSTR_EQUALITY)&(d.efc_id==eid))
                    assert len(rows)==3
                    row['hand_support_world_n'].append((-d.efc_force[rows]).tolist())
            steps.append(row)
            if start==0 and step in [0,(n-1)//3,2*(n-1)//3,n-2]:images+=render(env,folder/'images',[step],policy=True)
            if absorbing:termination='task_failure';break
            if done and step<n-start-2:termination='early_done';break
        records.append(dict(start=start,steps=len(steps),expected_steps=n-start-1,termination=termination,
            mean_reward=float(np.mean([v['reward'] for v in steps])),reset_grip_errors_m=reset_errors,
            peak_grip_error_m=max((v.get('grip_error_m',0) for v in steps),default=0),trajectory=steps,images=images,
            joint_mae_deg=float(np.mean([v['joint_error_deg'] for v in steps])),
            peak_root_error_m=max(v['root_error_m'] for v in steps),
            quality_pass=bool(termination=='reference_end' and np.mean([v['joint_error_deg'] for v in steps])<=20 and max(v['root_error_m'] for v in steps)<=.25 and max(v.get('grip_error_m',0) for v in steps)<=.03)))
        np.savez_compressed(folder/f'start_{start}_reset.npz',qpos=reset_q)
        np.savez_compressed(folder/f'start_{start}_rollout.npz',qpos=np.asarray(qposes),dt=.01)
        print('EVAL',task,label,start,termination,len(steps),flush=True)
    summary=dict(task=task,checkpoint=str(checkpoint),mass_kg=env.payload_mass_kg,all_complete=all(r['termination']=='reference_end' for r in records),
        all_quality_pass=all(r['quality_pass'] for r in records),quality_thresholds=dict(joint_mae_deg=20,peak_root_error_m=.25,peak_grip_error_m=.03),records=records)
    save_json(folder/'results.json',summary);env.stop()


def train(task,mass,envs,updates,label,source=None,lr=None,start_mode='random'):
    import jax
    jax.config.update('jax_default_matmul_precision','high')
    from musclemimic.runner.eval_utils import load_checkpoint
    from musclemimic.runner.engine import run_experiment
    from musclemimic.runner.logging import UnifiedHooks
    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.dataclasses import TrainState
    from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager,CheckpointMetadata
    cfg=OmegaConf.load(OUT/task/'base_config.yaml');folder=OUT/task/'training'/label
    with open_dict(cfg):
        exp=cfg.experiment;exp.env_params.payload_mass_kg=mass;exp.env_params.num_envs=envs;exp.num_envs=envs
        if lr is not None:exp.lr=lr
        if start_mode=='beginning':
            exp.env_params.th_params=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
        elif start_mode=='random':
            weights=exp.env_params.th_params.get('start_step_weights')
            exp.env_params.th_params=dict(random_start=True,fixed_start_conf=None,start_from_random_step=True)
            if weights is not None:exp.env_params.th_params.start_step_weights=weights
        if task=='lift':
            exp.env_params.reward_params.payload_joint_weight=.3
            exp.env_params.reward_params.root_pos_w_sum=.3
            exp.env_params.terminal_state_params.root_deviation_threshold=.25
            exp.validation.terminal_state_type='MeanRelativeSiteDeviationWithRootTerminalStateHandler'
            exp.validation.terminal_state_params=dict(mean_site_deviation_threshold=.5,root_deviation_threshold=.25,root_orientation_threshold=.7,enable_site_check=True)
        steps=16 if envs<128 else 64;mb=max(1,envs*steps//2048)
        exp.ppo_config.num_steps=steps;exp.ppo_config.num_minibatches=mb
        exp.num_steps=steps;exp.num_minibatches=mb
        exp.total_timesteps=envs*steps*updates
        exp.run_id=label;exp.checkpoint_root=str((folder/'checkpoints').resolve())
        exp.checkpoint_dir=str((folder/'checkpoints'/label).resolve())
        exp.validation.interval=min(10,updates)
        exp.validation.num_envs=min(envs,4)
        exp.validation.num=4
        exp.validation.video_frequency=0
        exp.validation.joints_to_ignore=['payload_free'] if task=='bend' else []
    warm=folder/'initialization';cp=warm/'checkpoint_0'
    if not cp.exists():
        source=source or SOURCE[task];_,old,_=load_checkpoint(source)
        cpu=make_cpu(cfg);conf=PPOJax.init_agent_conf(cpu,cfg)
        state=TrainState(apply_fn=conf.network.apply,params=old.train_state.params,tx=conf.tx,
            opt_state=conf.tx.init(old.train_state.params),step=0,run_stats=old.train_state.run_stats)
        manager=UnifiedCheckpointManager(str(warm.resolve()),max_to_keep=2)
        meta=CheckpointMetadata(step=0,update_number=0,global_timestep=0,target_global_timestep=envs*steps*updates,
            learning_rate=float(cfg.experiment.lr),num_envs=envs,num_steps=steps,num_minibatches=mb,update_epochs=1,env_name='MjxMyoFullBodyPayloadEnv')
        manager.save_checkpoint(0,conf,SimpleNamespace(train_state=state),meta);manager.close();cpu.stop()
        save_json(folder/'transfer.json',dict(source=str(source),weights_and_observation_statistics_preserved=True,optimizer_reinitialized=True,step_reset=True,mass_kg=mass))
    cfg.experiment.resume_from=str(cp.resolve())
    folder.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,folder/'config.yaml')
    # The shared runner normally receives this context from @hydra.main.
    from hydra.conf import HydraConf
    from hydra.core.hydra_config import HydraConfig
    hydra_cfg=OmegaConf.structured(HydraConf())
    hydra_cfg.runtime.output_dir=str(folder.resolve());hydra_cfg.runtime.cwd=str(Path.cwd())
    HydraConfig.instance().set_config(OmegaConf.create(dict(hydra=hydra_cfg)))
    class PayloadHooks(UnifiedHooks):
        def build_video_recorder(self,**kwargs):
            return None  # Record independent CPU evaluations after each mass stage.
        def enrich_log(self,log_dict,metrics_dict,env):
            def scalar(x):
                a=np.asarray(x)
                return float(a) if a.size==1 else a.tolist()
            with (folder/'metrics.jsonl').open('a') as f:
                f.write(json.dumps({k:scalar(v) for k,v in log_dict.items()},allow_nan=True)+'\n')
    print('TRAIN_START',task,mass,envs,updates,label,flush=True)
    run_experiment(cfg,hooks=PayloadHooks())
    logged_updates=len((folder/'metrics.jsonl').read_text().splitlines())
    save_json(folder/'completed.json',dict(task=task,label=label,mass_kg=mass,envs=envs,updates=updates,logged_updates=logged_updates,complete=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate','train']);p.add_argument('--task',choices=['lift','bend'],required=True)
    p.add_argument('--mass',type=float);p.add_argument('--checkpoint');p.add_argument('--envs',type=int,default=32);p.add_argument('--updates',type=int,default=2);p.add_argument('--label',default='smoke');p.add_argument('--lr',type=float);p.add_argument('--start-mode',choices=['random','beginning'],default='random')
    a=p.parse_args()
    if a.action=='prepare':prepare(a.task)
    elif a.action=='evaluate':evaluate(a.task,a.checkpoint,a.mass,a.label)
    else:train(a.task,a.mass,a.envs,a.updates,a.label,a.checkpoint,a.lr,a.start_mode)
