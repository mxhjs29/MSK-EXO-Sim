"""Versioned loaded-PPO refinement, measured contacts and unchanged quality gates."""
from musclemimic.runner.workflows import payload_finetune as legacy, box_friction as box
from musclemimic.runner.workflows.payload_finetune import *
from musclemimic.environments.humanoids.myofullbody_payload_refinement import ContactReadout
import hashlib
import shutil

OUT=Path(os.environ.get('PAYLOAD_REFINEMENT_OUT','outputs/payload_refinement_v3'))
OLD_LIFT=Path('outputs/payload_finetune_v1')
OLD_BOX=Path('outputs/payload_box_friction_v2')
SOURCES={
    'lift':OLD_LIFT/'lift/training/target_2kg_random_300/checkpoints/target_2kg_random_300/checkpoint_300',
    'bend':OLD_BOX/'bend/training/curriculum_1kg_v2_2/checkpoints/curriculum_1kg_v2_2/checkpoint_200'}


def make_cpu(config):
    from loco_mujoco.task_factories import TaskFactory
    cfg=OmegaConf.create(OmegaConf.to_container(config,resolve=True));ep=cfg.experiment.env_params
    ep.env_name='MyoFullBodyBoxRefinementEnv' if ep.payload_kind=='box' else 'MyoFullBodyPayloadEnv'
    ep.mjx_backend='jax';ep.num_envs=1
    ep.th_params=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
    return TaskFactory.get_factory_cls('ImitationFactory').make(**ep,**cfg.experiment.task_factory.params)


def prepare():
    for task,root,n in [('lift',OLD_LIFT,324),('bend',OLD_BOX,318)]:
        folder=OUT/task;folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/task/'reference.npz',folder/'reference.npz')
        cfg=OmegaConf.load(root/task/'base_config.yaml')
        with open_dict(cfg):
            ep=cfg.experiment.env_params
            ep.reward_type='LiftRefinementReward' if task=='lift' else 'BoxRefinementReward'
            ep.reward_params.root_pos_w_sum=.3
            ep.terminal_state_params.root_deviation_threshold=.25
            # Explicit mixture prevents forgetting early phases while targeting failure regions.
            weights=np.zeros(n)
            if task=='lift':
                weights[:n-2]+=.4/(n-2);weights[160:260]+=.6/100
                ep.reward_params.payload_joint_weight=.3
                ep.payload_mass_kg=2.
            else:
                weights[:n-2]+=.3/(n-2);weights[:70]+=.4/70;weights[110:160]+=.3/50
                ep.env_name='MjxMyoFullBodyBoxRefinementEnv';ep.goal_type='BoxRefinementGoal'
                ep.terminal_state_type='BoxRefinementTerminal';ep.reward_params.payload_grip_weight=.8
                ep.payload_mass_kg=1.
            ep.th_params=dict(random_start=True,fixed_start_conf=None,start_from_random_step=True,start_step_weights=weights.tolist())
            cfg.experiment.task_factory.params.custom_dataset_conf.traj_path=str((folder/'reference.npz').resolve())
            cfg.experiment.validation.terminal_state_type=ep.terminal_state_type
            cfg.experiment.validation.terminal_state_params=ep.terminal_state_params
            cfg.experiment.validation.th_params=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
        OmegaConf.save(cfg,folder/'base_config.yaml')
        save_json(folder/'changes.json',dict(source=str(SOURCES[task]),reference_unchanged_sha256=hashlib.sha256((folder/'reference.npz').read_bytes()).hexdigest(),
            sampling_weights=weights.tolist(),geometry_and_mass_unchanged=True,
            rewards='broad elbow tracking + late emphasis + velocity' if task=='lift' else 'bilateral force + dense box position/velocity + nonhand contact penalty + force observations',
            no_box_assistance=True))


def warm_start(task,label,envs,updates,mass,source):
    import jax,jax.numpy as jnp
    from musclemimic.runner.eval_utils import load_checkpoint
    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.dataclasses import TrainState
    from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager,CheckpointMetadata
    folder=OUT/task/'training'/label;warm=folder/'initialization'
    if (warm/'checkpoint_0').exists():return
    cfg=OmegaConf.load(OUT/task/'base_config.yaml');cfg.experiment.env_params.payload_mass_kg=mass
    e=make_cpu(cfg);conf=PPOJax.init_agent_conf(e,cfg)
    _,old,_=load_checkpoint(str(source));old_dim=int(old.train_state.run_stats['RunningMeanStd_0']['mean'].shape[0])
    new_dim=int(e.info.observation_space.shape[0])
    def expand(x,variance=False):
        if hasattr(x,'shape') and x.ndim and x.shape[0]==old_dim and new_dim>old_dim:
            shape=(new_dim-old_dim,*x.shape[1:])
            return jnp.concatenate((x,jnp.ones(shape,dtype=x.dtype) if variance else jnp.zeros(shape,dtype=x.dtype)),axis=0)
        return x
    params=jax.tree.map(expand,old.train_state.params)
    stats=jax.tree_util.tree_map_with_path(lambda p,x:expand(x,any(getattr(k,'key',None)=='var' for k in p)),old.train_state.run_stats)
    obs=np.asarray(e.reset(jax.random.PRNGKey(11)))
    oldconf=conf
    if old_dim!=new_dim:
        old_env=box.make_cpu(OmegaConf.load(OLD_BOX/'bend/base_config.yaml'))
        oldconf=PPOJax.init_agent_conf(old_env,cfg);old_env.stop()
    before=oldconf.network.apply(dict(params=old.train_state.params,run_stats=old.train_state.run_stats),jnp.atleast_2d(obs[:old_dim]))[0].mode()
    after=conf.network.apply(dict(params=params,run_stats=stats),jnp.atleast_2d(obs))[0].mode()
    error=float(np.max(abs(np.asarray(before)-np.asarray(after))));assert error<1e-5,error
    state=TrainState(apply_fn=conf.network.apply,params=params,tx=conf.tx,opt_state=conf.tx.init(params),step=0,run_stats=stats)
    steps=16 if envs<128 else 64
    meta=CheckpointMetadata(step=0,update_number=0,global_timestep=0,target_global_timestep=envs*steps*updates,
        learning_rate=float(cfg.experiment.lr),num_envs=envs,num_steps=steps,num_minibatches=max(1,envs*steps//2048),
        update_epochs=1,env_name=cfg.experiment.env_params.env_name)
    manager=UnifiedCheckpointManager(str(warm.resolve()),max_to_keep=2)
    manager.save_checkpoint(0,conf,SimpleNamespace(train_state=state),meta);manager.close();e.stop()
    save_json(folder/'transfer.json',dict(source=str(source),old_observation_dim=old_dim,new_observation_dim=new_dim,
        zero_initial_new_input_weights=True,max_policy_output_change=error,optimizer_reset=True))


def train(task,label,envs,updates,mass,source=None):
    source=source or SOURCES[task]
    assert json.loads((OUT/task/'readiness.json').read_text())['passed']
    if envs>=128:
        assert json.loads((OUT/task/'smoke_gate.json').read_text())['passed']
    warm_start(task,label,envs,updates,mass,source)
    legacy.OUT=OUT;legacy.make_cpu=make_cpu
    legacy.train(task,mass,envs,updates,label,str(source),lr=1e-5 if task=='lift' else 2e-5,start_mode='random')


def nonhand_contacts(e):
    m,d=e._model,e._data;hand=set(sum(e.box_hand_bodies,[]));out={}
    for i in range(d.ncon):
        c=d.contact[i];bodies=m.geom_bodyid[c.geom]
        if e.payload_body_id not in bodies:continue
        other=int(bodies[0] if bodies[1]==e.payload_body_id else bodies[1])
        if other==0 or other in hand:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        sign=1 if bodies[1]==e.payload_body_id else -1
        world=sign*c.frame.reshape(3,3).T@f[:3]
        name=m.body(other).name
        out[name]=out.get(name,0.)+float(world[2])
    return out


def evaluate(task,label,checkpoint=None,mass=None):
    import jax,jax.numpy as jnp
    from musclemimic.runner.eval_utils import load_checkpoint
    from musclemimic.algorithms import PPOJax
    checkpoint=checkpoint or SOURCES[task];mass=mass or (2. if task=='lift' else 1.)
    cfg=OmegaConf.load(OUT/task/'base_config.yaml');cfg.experiment.env_params.payload_mass_kg=mass
    e=make_cpu(cfg);m,d=e._model,e._data
    _,agent,_=load_checkpoint(str(checkpoint));conf=PPOJax.init_agent_conf(e,cfg)
    dim=int(agent.train_state.run_stats['RunningMeanStd_0']['mean'].shape[0])
    if dim!=e.info.observation_space.shape[0]:
        old_env=box.make_cpu(OmegaConf.load(OLD_BOX/'bend/base_config.yaml'))
        conf=PPOJax.init_agent_conf(old_env,cfg);old_env.stop()
    policy=jax.jit(lambda obs:conf.network.apply(dict(params=agent.train_state.params,run_stats=agent.train_state.run_stats),jnp.atleast_2d(obs[:dim]))[0].mode())
    folder=OUT/task/'evaluations'/label;folder.mkdir(parents=True,exist_ok=True)
    n=int(e.th.len_trajectory(0));records=[];sensor_error=0.;max_contacts=0
    if task=='bend':readout=ContactReadout(m);qa=int(m.jnt_qposadr[m.joint('payload_free').id])
    else:
        ja=int(m.jnt_qposadr[m.joint('elbow_flexion_r').id]);jd=int(m.jnt_dofadr[m.joint('elbow_flexion_r').id])
        names=['BIClong','BICshort','BRA','BRD','TRIlong','TRIlat','TRImed','ANC']
        muscles=[i for i in range(m.nu) if m.actuator(i).name in names+[v+'_r' for v in names]]
        assert len(muscles)==8
    for start in [0,n//5,2*n//5,3*n//5]:
        e.th.fixed_start_conf=[0,start];e.th.random_start=False
        obs=e.reset(jax.random.PRNGKey(2026));qposes=[];qrefs=[];rows=[];termination='reference_end'
        reset=d.qpos.copy()
        for step in range(n-start-1):
            obs,reward,absorbing,done,info=e.step(np.asarray(policy(obs)).reshape(-1))
            assert np.isfinite(d.qpos).all() and np.isfinite(obs).all()
            ref=e.th.get_current_traj_data(e._additional_carry,np).qpos.copy()
            origin=e.th.get_init_traj_data(e._additional_carry,np).qpos[:2]
            ref[:2]-=origin
            row=dict(t=(step+1)*.01,phase_t=(start+step+1)*.01,reward=float(reward),
                root_error_m=float(np.linalg.norm(d.qpos[:3]-ref[:3])),activation_rms=float(np.sqrt(np.mean(d.act**2))))
            if task=='lift':
                row.update(joint_angle_rad=float(d.qpos[ja]),reference_angle_rad=float(ref[ja]),
                    joint_error_deg=float(np.rad2deg(abs(d.qpos[ja]-ref[ja]))),muscle_torque_nm=float(d.qfrc_actuator[jd]),
                    elbow8_activation_rms=float(np.sqrt(np.mean(d.act[muscles]**2))))
            else:
                ref[qa:qa+2]-=origin;metrics=box.contact_metrics(e);row.update(metrics)
                f=readout(d,np);max_contacts=max(max_contacts,int(max(f['count'])))
                components=[float(np.max(abs(f['normal']-metrics['normal_n']))),
                    float(np.max(abs(f['friction_z']-metrics['friction_vertical_n']))),
                    abs(float(f['other'][2])-metrics['other_vertical_n'])]
                if max(components)>max(sensor_error,1e-5):
                    print('SENSOR_PARITY_DETAIL',start,step,components,'sensor',f,'raw',metrics,flush=True)
                sensor_error=max(sensor_error,*components)
                row.update(height_m=float(d.qpos[qa+2]),reference_height_m=float(ref[qa+2]),
                    object_error_m=float(np.linalg.norm(d.qpos[qa:qa+3]-ref[qa:qa+3])),
                    object_orientation_error_deg=float(np.rad2deg(2.*np.arccos(np.clip(abs(np.dot(d.qpos[qa+3:qa+7],ref[qa+3:qa+7])/
                        max(np.linalg.norm(d.qpos[qa+3:qa+7])*np.linalg.norm(ref[qa+3:qa+7]),1e-12)),0.,1.)))),
                    nonhand_by_body_n=nonhand_contacts(e),nonhand_force_norm_n=float(np.linalg.norm(f['other'])),
                    payload_pos_m=d.qpos[qa:qa+3].tolist(),reference_pos_m=ref[qa:qa+3].tolist())
            rows.append(row);qposes.append(d.qpos.copy());qrefs.append(ref)
            if absorbing:termination='task_failure';break
            if done and step<n-start-2:termination='early_done';break
        summary=dict(start=start,steps=len(rows),expected_steps=n-start-1,termination=termination,
            peak_root_error_m=max(r['root_error_m'] for r in rows),trajectory=rows)
        if task=='lift':
            summary['joint_mae_deg']=float(np.mean([r['joint_error_deg'] for r in rows]))
            summary['quality_pass']=termination=='reference_end' and summary['joint_mae_deg']<=20 and summary['peak_root_error_m']<=.25
        else:
            surface=e.box_support_pos[2]+e.box_support_half_size[2]+e.box_half_size[2]
            air=[r for r in rows if r['height_m']>surface+.04]
            bilateral=float(np.mean([min(r['normal_n'])>1. for r in air])) if air else 0.
            friction=float(np.mean([sum(r['friction_vertical_n']) for r in air])) if air else 0.
            other=float(np.mean([abs(r['other_vertical_n']) for r in air])) if air else 0.
            peak=max(r['max_penetration_m'] for r in rows);err=max(r['object_error_m'] for r in rows)
            summary.update(peak_penetration_m=peak,peak_object_error_m=err,bilateral_contact_fraction=bilateral,
                airborne_mean_friction_vertical_n=friction,airborne_other_body_vertical_n=other,
                mean_object_orientation_error_deg=float(np.mean([r['object_orientation_error_deg'] for r in rows])),
                peak_object_orientation_error_deg=max(r['object_orientation_error_deg'] for r in rows),
                quality_pass=termination=='reference_end' and peak<.003 and err<.10 and bilateral>.85 and friction>.7*mass*9.81 and other<.1*mass*9.81)
        records.append(summary)
        np.savez_compressed(folder/f'start_{start}_rollout.npz',qpos=qposes,qref=qrefs,reset=reset,dt=.01)
        print('EVAL',task,label,start,termination,len(rows),summary['quality_pass'],flush=True)
    result=dict(task=task,label=label,checkpoint=str(checkpoint),mass_kg=mass,all_quality_pass=all(r['quality_pass'] for r in records),
        complete_count=sum(r['termination']=='reference_end' for r in records),quality_count=sum(r['quality_pass'] for r in records),
        sensor_force_max_error_n=sensor_error,max_hand_contacts=max_contacts,records=records)
    if task=='bend':assert sensor_error<1e-5 and max_contacts<=24,(sensor_error,max_contacts)
    save_json(folder/'results.json',result);e.stop();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate','train']);p.add_argument('--task',choices=['lift','bend'])
    p.add_argument('--label',default='baseline');p.add_argument('--checkpoint');p.add_argument('--mass',type=float)
    p.add_argument('--envs',type=int,default=32);p.add_argument('--updates',type=int,default=4);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='evaluate':evaluate(a.task,a.label,a.checkpoint,a.mass)
    else:train(a.task,a.label,a.envs,a.updates,a.mass or (2. if a.task=='lift' else 1.),a.checkpoint)
