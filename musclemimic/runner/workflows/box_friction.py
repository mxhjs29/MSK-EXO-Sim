"""Prepare, audit and train the physical hand/box friction task."""
from musclemimic.runner.workflows import payload_finetune as legacy
from musclemimic.runner.workflows.payload_finetune import *
from musclemimic.environments.humanoids.myofullbody_box_friction import MyoFullBodyBoxFrictionEnv
import hashlib

OUT = Path('outputs/payload_box_friction_v2')
OLD = Path('outputs/payload_finetune_v1')


def make_cpu(config):
    from loco_mujoco.task_factories import TaskFactory
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    cfg.experiment.env_params.env_name = 'MyoFullBodyBoxFrictionEnv'
    cfg.experiment.env_params.mjx_backend = 'jax'
    cfg.experiment.env_params.num_envs = 1
    return TaskFactory.get_factory_cls('ImitationFactory').make(
        **cfg.experiment.env_params, **cfg.experiment.task_factory.params)


def hand_geoms(env):
    m = env._model
    return [[g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies and
             (m.geom_contype[g] or m.geom_conaffinity[g]) and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
            for bodies in env.box_hand_bodies]


def reference_hash():
    return hashlib.sha256((OUT/'bend/reference.npz').read_bytes()).hexdigest()


def environment_signature():
    digest=hashlib.sha256()
    for path in [OUT/'bend/reference.npz',OUT/'bend/base_config.yaml',
                 Path('musclemimic/environments/humanoids/myofullbody_box_friction.py')]:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def geometry():
    """Check every collidable body against the box at every reference frame."""
    e=make_cpu(OmegaConf.load(OUT/'bend/base_config.yaml'));m,d=e._model,e._data
    original=legacy.make_cpu(OmegaConf.load(OLD/'bend/base_config.yaml'))
    om=original._model;inertia_error=0.;mass_error=0.
    for b in range(1,m.nbody):
        name=m.body(b).name
        if name=='payload':continue
        ob=om.body(name).id
        mass_error=max(mass_error,float(abs(m.body_mass[b]-om.body_mass[ob])))
        inertia_error=max(inertia_error,float(np.max(abs(m.body_inertia[b]-om.body_inertia[ob]))))
    original.stop()
    q=np.asarray(e.th.traj.data.qpos).copy();origin=q[0,:2].copy()
    q[:,:2]-=origin;q[:,-7:-5]-=origin
    box=m.geom('payload_box').id
    others=[g for g in range(m.ngeom) if g!=box and m.geom_bodyid[g]>0 and
        ((m.geom_contype[g]&m.geom_conaffinity[box]) or (m.geom_contype[box]&m.geom_conaffinity[g]))]
    maximum=0.;worst=None;per_frame=[];sensor_error=0.;platform_depth=0.;platform_worst=None
    support=m.geom('box_support').id
    for frame,pose in enumerate(q):
        d.qpos[:]=pose;d.qvel[:]=0;mujoco.mj_forward(m,d)
        distances=[mujoco.mj_geomDistance(m,d,g,box,.1,None) for g in others]
        k=int(np.argmin(distances));depth=max(0.,-float(distances[k]));per_frame.append(depth)
        if depth>maximum:maximum=depth;worst=dict(frame=frame,geom=m.geom(others[k]).name)
        for g in others:
            dep=max(0.,-float(mujoco.mj_geomDistance(m,d,g,support,.1,None)))
            if dep>platform_depth:platform_depth=dep;platform_worst=dict(frame=frame,geom=m.geom(g).name)
        sensed=max(0.,max(-float(d.sensordata[m.sensor_adr[m.sensor('box_contact_depth_'+s).id]]) for s in ['r','l']))
        sensor_error=max(sensor_error,abs(sensed-contact_metrics(e)['max_penetration_m']))
    result=dict(passed=maximum<.001 and platform_depth<.001 and sensor_error<1e-8 and mass_error<1e-10 and inertia_error<1e-10,frames=len(q),
        max_human_platform_penetration_m=platform_depth,platform_worst=platform_worst,environment_signature=environment_signature(),
        max_human_body_mass_change_kg=mass_error,max_human_body_inertia_change=inertia_error,
        max_body_box_penetration_m=maximum,worst=worst,per_frame_penetration_m=per_frame,
        reference_sha256=reference_hash(),sensor_raw_max_difference_m=sensor_error,
        colliders=[m.geom(g).name for g in others],
        forearm_collision='Overlapping 30/24/17 mm radius capsules taper to the wrist; original body mass/inertia retained')
    save_json(OUT/'audit/full_geometry.json',result);e.stop()
    assert result['passed'],result
    print('FULL_GEOMETRY_PASSED',maximum,flush=True)


def prepare():
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from loco_mujoco.trajectory import Trajectory, TrajectoryInfo, TrajectoryData, TrajectoryModel
    from loco_mujoco.trajectory.handler import materialize_trajectory
    cfg = OmegaConf.load(OLD/'bend/base_config.yaml')
    old = legacy.make_cpu(cfg)
    original = np.asarray(old.th.traj.data.qpos).copy()
    first = int(np.argmin(original[:, -5]))
    original = original[first:]
    ep = OmegaConf.to_container(cfg.experiment.env_params, resolve=True)
    ep.pop('env_name')
    ep.update(goal_type='BoxFrictionGoal', reward_type='BoxFrictionReward',
              terminal_state_type='BoxFrictionTerminal', mjx_backend='jax', num_envs=1,
              payload_finger_posture={})
    # Low support: the original bend recording reaches knee height, not the floor.
    center0 = original[0, -7:-4].copy()
    support_height = float(center0[2] - .12)
    rootxy = original[0, :2]
    ep.update(box_support_pos=[*(center0[:2]-rootxy).astype(float).tolist(), support_height/2],
              box_support_half_size=[.07, .06, support_height/2])
    e = MyoFullBodyBoxFrictionEnv(**ep);m,d=e._model,e._data
    qpos = original.copy();n=len(qpos)
    dependent = {int(m.eq_obj1id[i]) for i in range(m.neq) if m.eq_type[i]==mujoco.mjtEq.mjEQ_JOINT}
    names = [stem+'_'+s for s in ['r','l'] for stem in
             ['elv_angle','shoulder_elv','shoulder_rot','elbow_flexion','pro_sup','deviation','flexion']
             if m.joint(stem+'_'+s).id not in dependent]
    ids = np.array([m.jnt_qposadr[m.joint(x).id] for x in names])
    bounds = np.array([m.jnt_range[m.joint(x).id] for x in names])
    coupled = [i for i in range(m.neq) if m.eq_type[i]==mujoco.mjtEq.mjEQ_JOINT and
               m.eq_obj2id[i]>=0 and m.jnt_qposadr[m.eq_obj2id[i]] in ids]
    dep=m.jnt_qposadr[m.eq_obj1id[coupled]];drivers=m.jnt_qposadr[m.eq_obj2id[coupled]]
    coefficients=m.eq_data[coupled,:5]
    palms=np.array([m.site('box_palm_'+s).id for s in ['r','l']])
    roots=[m.body('lunate_'+s).id for s in ['r','l']]
    geoms=hand_geoms(e);box=m.geom('payload_box').id
    pads=[m.geom('box_palm_pad_'+s).id for s in ['r','l']]
    hand_bodies=set(sum(e.box_hand_bodies,[]))
    avoid=[g for g in range(m.ngeom) if m.geom_bodyid[g]>0 and m.geom_bodyid[g] not in hand_bodies
           and g!=box and ((m.geom_contype[g]&m.geom_conaffinity[box]) or (m.geom_contype[box]&m.geom_conaffinity[g]))]
    finger_ids=[]
    for j in range(m.njnt):
        if m.jnt_bodyid[j] in sum(e.box_hand_bodies,[]) and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE and m.jnt_qposadr[j] not in ids and j not in dependent:
            finger_ids.append(j)
    finger_q=m.jnt_qposadr[finger_ids];finger_bounds=m.jnt_range[finger_ids]
    finger_pose=np.clip(np.zeros(len(finger_ids)),finger_bounds[:,0]+1e-5,finger_bounds[:,1]-1e-5)
    previous=None;errors=[];penetrations=[];angular=[]
    for i in range(n):
        d.qpos[:]=qpos[i];d.qpos[finger_q]=finger_pose
        center=qpos[i,-7:-4].astype(float).copy()
        # Preserve the recorded box path, with level sides for bilateral palm friction.
        oldR=Rotation.from_quat(np.roll(qpos[i,-4:],-1)).as_matrix()
        x=oldR[:,0].copy();x[2]=0;x/=np.linalg.norm(x)
        z=np.array([0.,0.,1.]);y=np.cross(z,x);R=np.column_stack((x,y,z))
        center += .08*z + .06*y
        quat=np.roll(Rotation.from_matrix(R).as_quat(),1)
        d.qpos[-7:]=np.r_[center,quat]
        targets=np.array([center-x*.12002+z*.02,center+x*.12002+z*.02])
        orig=qpos[i,ids].copy();initial=orig if previous is None else previous
        initial=np.clip(initial,bounds[:,0]+1e-5,bounds[:,1]-1e-5)
        initial=np.r_[initial, center]
        def forward(v):
            d.qpos[ids]=v[:len(ids)]
            d.qpos[-7:-4]=v[-3:]
            delta=d.qpos[drivers]-m.qpos0[drivers]
            d.qpos[dep]=m.qpos0[dep]+np.sum(coefficients*delta[:,None]**np.arange(5),axis=1)
            mujoco.mj_kinematics(m,d)
        def residual(v):
            forward(v)
            handR=d.xmat[roots].reshape(2,3,3)
            targets=np.array([v[-3:]-x*.12002+z*.02,v[-3:]+x*.12002+z*.02])
            clearance=np.array([mujoco.mj_geomDistance(m,d,g,box,.05,None) for g in avoid])
            palm_clearance=np.array([mujoco.mj_geomDistance(m,d,g,box,.05,None) for g in pads])
            # Full palm orientation: fingers down, palm normal toward the side wall.
            return np.r_[(d.site_xpos[palms]-targets).reshape(-1)*1000,
                         (handR[:,:,2]+x).reshape(-1)*50,
                         (v[:len(ids)]-orig)*.015,(v-initial)*2.,
                         (v[-3:]-center)*10,np.minimum(clearance-.003,0)*1500,
                         np.minimum(palm_clearance-.00002,0)*2000]
        fit=least_squares(residual,initial,bounds=(np.r_[bounds[:,0],center-.20],np.r_[bounds[:,1],center+.20]),max_nfev=220, diff_step=1e-4)
        forward(fit.x);previous=fit.x[:len(ids)]
        center=fit.x[-3:];targets=np.array([center-x*.12002+z*.02,center+x*.12002+z*.02])
        if i==0:print('FIT0',fit.message,fit.nfev,fit.cost, 'targets',targets,'palms',d.site_xpos[palms],flush=True)
        if i==0:
            # Move fingers clear of the wall; no finger is allowed inside the box.
            def fingers(v):
                d.qpos[finger_q]=v;mujoco.mj_kinematics(m,d)
                dist=np.array([mujoco.mj_geomDistance(m,d,g,box,.3,None) for side in geoms for g in side if 'pad' not in m.geom(g).name])
                return np.r_[np.minimum(dist-.001,0)*1000,v*.025]
            ffit=least_squares(fingers,finger_pose,bounds=(finger_bounds[:,0],finger_bounds[:,1]),max_nfev=200)
            finger_pose=ffit.x;fingers(finger_pose)
        distances=[mujoco.mj_geomDistance(m,d,g,box,.3,None) for side in geoms for g in side]
        qpos[i]=d.qpos.copy()
        errors.append(float(np.max(np.linalg.norm(d.site_xpos[palms]-targets,axis=1))))
        penetrations.append(max(0.,-min(distances)))
        angular.append(float(np.rad2deg(np.arccos(np.clip(-d.xmat[roots].reshape(2,3,3)[:,:,2]@x,-1,1))).max()))
        if i%50==0:print('IK',i,'pos',errors[-1],'penetration',penetrations[-1], 'angle',angular[-1],flush=True)
    folder=OUT/'bend';folder.mkdir(parents=True,exist_ok=True)
    info=TrajectoryInfo([m.joint(i).name for i in range(m.njnt)],TrajectoryModel(m.njnt,np.asarray(m.jnt_type)),100.,metadata=dict(
        task='bend',source_start_frame=140+first,source_frame_count=660,friction_only=True))
    traj=Trajectory(info=info,data=TrajectoryData(qpos=qpos,qvel=np.zeros((n,m.nv)),split_points=np.array([0,n])))
    traj=materialize_trajectory(traj,model=m,control_dt=.01,cache_type='sparse',site_names=e.sites_for_mimic)
    traj.save(str(folder/'reference.npz'));e.load_trajectory(traj=traj,cache_type='sparse',site_names=e.sites_for_mimic,warn=False)
    support=m.geom('box_support').id
    support_height=float(qpos[0,-5]-.12)
    ep['box_support_pos']=[*(qpos[0,-7:-5]-rootxy).astype(float).tolist(),support_height/2]
    ep['box_support_half_size']=[.07,.06,support_height/2]
    m.geom_pos[support]=ep['box_support_pos'];m.geom_size[support]=ep['box_support_half_size']
    m.geom_pos[support,:2]+=rootxy
    pictures=render(e,folder/'reference_images',[0,n//3,2*n//3,n-1])
    m.geom_pos[support,:2]-=rootxy
    with open_dict(cfg):
        ep['env_name']='MjxMyoFullBodyBoxFrictionEnv';ep['mjx_backend']='warp';ep['num_envs']=4096
        ep['goal_params']['source_start_frame']=140+first
        ep['reward_params']['payload_grip_weight']=.5
        ep['terminal_state_params']['root_deviation_threshold']=.25
        ep['payload_finger_posture']=dict(zip([m.joint(j).name for j in finger_ids],finger_pose.tolist()))
        ep['th_params']=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
        cfg.experiment.env_params=ep
        cfg.experiment.task_factory.params=dict(custom_dataset_conf=dict(traj_path=str((folder/'reference.npz').resolve()),trajectory_cache_type='sparse'))
        cfg.experiment.validation.terminal_state_type='BoxFrictionTerminal'
        cfg.experiment.validation.terminal_state_params=ep['terminal_state_params']
        cfg.experiment.validation.num_steps=n-1;cfg.experiment.validation.video_length=n-1
        cfg.experiment.validation.th_params=ep['th_params']
    OmegaConf.save(cfg,folder/'base_config.yaml')
    save_json(folder/'reference_audit.json',dict(frames=n,source_start=140+first,
        max_palm_position_error_m=max(errors),max_hand_box_penetration_m=max(penetrations),
        max_palm_normal_error_deg=max(angular),max_arm_frame_change_deg=float(np.rad2deg(abs(np.diff(qpos[:,ids],axis=0))).max()),
        support_height_m=support_height,friction=.8,hand_box_equalities=0,images=pictures,
        arm_joints=names,per_frame_penetration_m=penetrations))
    e.stop();old.stop()
    assert max(penetrations)<.001,('reference hand-box penetration',max(penetrations))
    assert max(errors)<.003,('palm alignment',max(errors))
    print('PREPARED_FRICTION',flush=True)


def contact_metrics(e):
    m,d=e._model,e._data;b=e.payload_body_id
    out=dict(normal_n=[0.,0.],friction_vertical_n=[0.,0.],support_vertical_n=[0.,0.],
             contacts=[0,0],max_penetration_m=0.,platform_vertical_n=0.,other_vertical_n=0.)
    for i in range(d.ncon):
        c=d.contact[i];bodies=m.geom_bodyid[c.geom]
        if b not in bodies:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f)
        sign=1 if bodies[1]==b else -1
        world=sign*c.frame.reshape(3,3).T@f[:3]
        tangent=sign*c.frame.reshape(3,3).T@np.r_[0.,f[1:3]]
        other=int(bodies[0] if bodies[1]==b else bodies[1])
        sides=[s for s,ids in enumerate(e.box_hand_bodies) if other in ids]
        if sides:
            s=sides[0];out['normal_n'][s]+=float(f[0]);out['contacts'][s]+=1
            out['support_vertical_n'][s]+=float(world[2]);out['friction_vertical_n'][s]+=float(tangent[2])
            out['max_penetration_m']=max(out['max_penetration_m'],float(-c.dist))
        elif m.geom('box_support').id in c.geom:out['platform_vertical_n']+=float(world[2])
        else:out['other_vertical_n']+=float(world[2])
    return out


def validate():
    import jax
    e=make_cpu(OmegaConf.load(OUT/'bend/base_config.yaml'));m,d=e._model,e._data
    audit=json.loads((OUT/'bend/reference_audit.json').read_text())
    assert audit['max_hand_box_penetration_m']<.001,audit['max_hand_box_penetration_m']
    assert not any('payload' in m.equality(i).name for i in range(m.neq))
    geoms=hand_geoms(e);box=m.geom('payload_box').id
    resets=[]
    support=m.geom('box_support').id
    body_geoms=[g for g in range(m.ngeom) if g!=box and m.geom_bodyid[g]>0 and
        ((m.geom_contype[g]&m.geom_conaffinity[box]) or (m.geom_contype[box]&m.geom_conaffinity[g]))]
    for start in np.linspace(0,len(e.th.traj.data.qpos)-2,9,dtype=int):
        e.th.fixed_start_conf=[0,int(start)];e.th.random_start=False
        obs=e.reset(jax.random.PRNGKey(71));q=d.qpos.copy()
        penetration=max(0.,-min(mujoco.mj_geomDistance(m,d,g,box,.3,None) for group in geoms for g in group))
        platform_pen=max(0.,-min(mujoco.mj_geomDistance(m,d,g,support,.3,None) for g in body_geoms))
        body_pen=max(0.,-min(mujoco.mj_geomDistance(m,d,g,box,.3,None) for g in body_geoms))
        assert penetration<.001,(start,penetration)
        assert platform_pen<.001 and body_pen<.001,('reset collision',start,platform_pen,body_pen)
        assert np.isfinite(obs).all()
        e.reset(jax.random.PRNGKey(71));assert np.array_equal(d.qpos,q)
        resets.append(dict(start=int(start),max_penetration_m=penetration,max_body_box_penetration_m=body_pen,
            max_human_platform_penetration_m=platform_pen,repeat_reset_identical=True))
    e.th.fixed_start_conf=[0,0];e.reset(jax.random.PRNGKey(71))
    # Exact per-hand engine sensors must agree with raw contacts and detect a deliberately invalid pose.
    initial_q=d.qpos.copy();a=int(m.jnt_qposadr[m.joint('payload_free').id])
    axis=d.xmat[e.payload_body_id].reshape(3,3)[:,0].copy()
    d.qpos[a:a+3]+=axis*.015;mujoco.mj_forward(m,d)
    sensor_depth=max(0.,max(-float(d.sensordata[m.sensor_adr[m.sensor('box_contact_depth_'+s).id]]) for s in ['r','l']))
    raw_depth=contact_metrics(e)['max_penetration_m']
    assert sensor_depth>.01 and abs(sensor_depth-raw_depth)<1e-8,(sensor_depth,raw_depth)
    terminal,_=e._terminal_state_handler.is_absorbing(e,None,{},d,e._additional_carry)
    assert terminal,'Penetrating state must terminate'
    d.qpos[:]=initial_q;d.qvel[:]=0;mujoco.mj_forward(m,d)
    fmt=lambda x:' '.join(str(float(v)) for v in np.asarray(x).reshape(-1))
    types={2:'sphere',3:'capsule',4:'ellipsoid',5:'cylinder',6:'box'}
    parts=['<mujoco><option timestep=".001" cone="elliptic" iterations="80"/><worldbody>']
    parts.append('<geom type="plane" size="2 2 .1" contype="1" conaffinity="4"/>')
    support=m.geom('box_support').id
    parts.append(f'<geom name="support" type="box" pos="{fmt(m.geom_pos[support])}" size="{fmt(m.geom_size[support])}" contype="1" conaffinity="4"/>')
    box_axis=d.xmat[e.payload_body_id].reshape(3,3)[:,0].copy()
    parts.append('<body name="carriage"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><joint name="lift" type="slide" axis="0 0 1" damping="1"/>')
    for side,gs in zip(['r','l'],geoms):
        b=m.body('lunate_'+side).id;R=d.xmat[b].reshape(3,3);pos=d.xpos[b].copy()
        inward=R.T@(box_axis*(1 if side=='r' else -1))
        parts.append(f'<body name="hand_{side}" pos="{fmt(pos)}" quat="{fmt(d.xquat[b])}"><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/><joint name="squeeze_{side}" type="slide" axis="{fmt(inward)}" limited="true" range="-.1 .03" damping="10"/>')
        for g in gs:
            local=R.T@(d.geom_xpos[g]-pos);rot=R.T@d.geom_xmat[g].reshape(3,3)
            quat=np.empty(4);mujoco.mju_mat2Quat(quat,rot.reshape(-1))
            parts.append(f'<geom name="{m.geom(g).name}" type="{types[int(m.geom_type[g])]}" size="{fmt(m.geom_size[g])}" pos="{fmt(local)}" quat="{fmt(quat)}" mass="0" contype="1" conaffinity="4" condim="3" friction=".8 0 0" priority="2" solref=".01 1" solimp=".95 .99 .001 .5 2"/>')
        parts.append('</body>')
    parts.append('</body>')
    b=e.payload_body_id
    parts.append(f'<body name="box" pos="{fmt(d.xpos[b])}" quat="{fmt(d.xquat[b])}"><freejoint name="box_free"/><geom name="box_geom" type="box" size="{fmt(m.geom_size[box])}" mass="5" contype="4" conaffinity="1" condim="3" priority="2" friction=".8 0 0" solref=".01 1" solimp=".95 .99 .001 .5 2"/></body></worldbody><actuator><position joint="lift" kp="20000" kv="300"/><motor joint="squeeze_r"/><motor joint="squeeze_l"/></actuator></mujoco>')
    folder=OUT/'bend/physics';folder.mkdir(parents=True,exist_ok=True)
    (folder/'friction_fixture.xml').write_text(''.join(parts))
    outcomes={}
    for label,mu in [('nominal',.8),('zero_friction',0.)]:
        fm=mujoco.MjModel.from_xml_string(''.join(parts));fd=mujoco.MjData(fm)
        fm.geom_friction[:]=[mu,0,0]
        mujoco.mj_forward(fm,fd)
        qbox=int(fm.jnt_qposadr[fm.joint('box_free').id]);z0=float(fd.qpos[qbox+2]);rows=[]
        for i in range(3200):
            t=i*.001
            rise=.20*(.5-.5*np.cos(np.pi*np.clip((t-.3)/.8,0,1)))
            fd.ctrl[:]=[rise,60*min(t/.2,1.) if t<2.3 else -30,60*min(t/.2,1.) if t<2.3 else -30]
            mujoco.mj_step(fm,fd)
            normal=0.;friction_z=0.;support_z=0.;pen=0.
            for ci in range(fd.ncon):
                c=fd.contact[ci]
                if fm.geom('box_geom').id not in c.geom:continue
                other=int(c.geom[0] if c.geom[1]==fm.geom('box_geom').id else c.geom[1])
                if fm.geom_bodyid[other] not in [fm.body('hand_r').id,fm.body('hand_l').id]:continue
                f=np.zeros(6);mujoco.mj_contactForce(fm,fd,ci,f)
                sign=1 if c.geom[1]==fm.geom('box_geom').id else -1
                normal+=f[0];friction_z+=(sign*c.frame.reshape(3,3).T@np.r_[0.,f[1:3]])[2]
                support_z+=(sign*c.frame.reshape(3,3).T@f[:3])[2];pen=max(pen,-c.dist)
            rows.append(dict(t=t,height_m=float(fd.qpos[qbox+2]),normal_n=float(normal),friction_vertical_n=float(friction_z),hand_vertical_n=float(support_z),penetration_m=float(pen)))
        held=[r for r in rows if 1.3<r['t']<2.2]
        outcome=dict(initial_height_m=z0,held_height_m=float(np.mean([r['height_m'] for r in held])),
            lift_m=float(np.mean([r['height_m'] for r in held])-z0),released_height_m=rows[-1]['height_m'],
            mean_friction_vertical_n=float(np.mean([r['friction_vertical_n'] for r in held])),
            mean_hand_vertical_n=float(np.mean([r['hand_vertical_n'] for r in held])),
            mean_total_normal_n=float(np.mean([r['normal_n'] for r in held])),
            peak_penetration_m=max(r['penetration_m'] for r in rows))
        outcomes[label]=outcome;save_json(folder/(label+'.json'),dict(summary=outcome,trajectory=rows))
        print('FIXTURE',label,outcome,flush=True)
    nominal=outcomes['nominal'];zero=outcomes['zero_friction']
    passed=bool(nominal['lift_m']>.15 and abs(nominal['mean_friction_vertical_n']-49.05)<2 and
                nominal['peak_penetration_m']<.003 and zero['lift_m']<.03 and
                nominal['held_height_m']-nominal['released_height_m']>.12)
    save_json(folder/'acceptance.json',dict(passed=passed,mass_kg=5.,hand_box_equalities=0,
        adhesion_actuators=0,reset_checks=resets,fixtures=outcomes,
        penetration_guard_test=dict(deliberate_depth_m=sensor_depth,raw_contact_depth_m=raw_depth,terminated=True),
        reference_sha256=hashlib.sha256((OUT/'bend/reference.npz').read_bytes()).hexdigest(),
        environment_signature=environment_signature(),
        scope='Actual compiled hand collision geometry driven by finite-force actuated slides in an isolated fixture; validates friction physics, not learned PPO success. 60 N squeeze per hand. No box position/velocity writes after reset.'))
    e.stop();assert passed,outcomes


def initialize_transfer(label, envs, updates, mass, source=None):
    import jax,jax.numpy as jnp
    from musclemimic.runner.eval_utils import load_checkpoint
    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.dataclasses import TrainState
    from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager,CheckpointMetadata
    folder=OUT/'bend/training'/label;warm=folder/'initialization'
    if (warm/'checkpoint_0').exists():return
    cfg=OmegaConf.load(OUT/'bend/base_config.yaml');cfg.experiment.env_params.payload_mass_kg=mass
    cpu=make_cpu(cfg);conf=PPOJax.init_agent_conf(cpu,cfg)
    source=source or SOURCE['bend'];oldcfg,old,_=load_checkpoint(source)
    old_dim=int(old.train_state.run_stats['RunningMeanStd_0']['mean'].shape[0])
    new_dim=cpu.info.observation_space.shape[0]
    def expand(x,variance=False):
        if hasattr(x,'shape') and x.ndim and x.shape[0]==old_dim and new_dim!=old_dim:
            pad=jnp.ones((new_dim-old_dim,*x.shape[1:]),dtype=x.dtype) if variance else jnp.zeros((new_dim-old_dim,*x.shape[1:]),dtype=x.dtype)
            return jnp.concatenate((x,pad),axis=0)
        return x
    params=jax.tree.map(expand,old.train_state.params)
    def stats(path,x):return expand(x,any(getattr(p,'key',None)=='var' for p in path))
    run_stats=jax.tree_util.tree_map_with_path(stats,old.train_state.run_stats)
    state=TrainState(apply_fn=conf.network.apply,params=params,tx=conf.tx,opt_state=conf.tx.init(params),step=0,run_stats=run_stats)
    # New object inputs have zero initial weights; transferred policy must be exactly preserved.
    if old_dim!=new_dim:
        oldconf=PPOJax.init_agent_conf(legacy.make_cpu(OmegaConf.load(OLD/'bend/base_config.yaml')),oldcfg)
        obs=np.asarray(cpu.reset(jax.random.PRNGKey(1)))
        a=oldconf.network.apply(dict(params=old.train_state.params,run_stats=old.train_state.run_stats),jnp.atleast_2d(obs[:old_dim]))[0].mode()
        b=conf.network.apply(dict(params=params,run_stats=run_stats),jnp.atleast_2d(obs))[0].mode()
        error=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert error<1e-5,error
    else:error=0.
    steps=16 if envs<128 else 64;mb=max(1,envs*steps//2048)
    manager=UnifiedCheckpointManager(str(warm.resolve()),max_to_keep=2)
    meta=CheckpointMetadata(step=0,update_number=0,global_timestep=0,target_global_timestep=envs*steps*updates,
        learning_rate=float(cfg.experiment.lr),num_envs=envs,num_steps=steps,num_minibatches=mb,update_epochs=1,env_name='MjxMyoFullBodyBoxFrictionEnv')
    manager.save_checkpoint(0,conf,SimpleNamespace(train_state=state),meta);manager.close();cpu.stop()
    save_json(folder/'transfer.json',dict(source=str(source),old_observation_dim=old_dim,new_observation_dim=new_dim,
        zero_initialized_object_input_weights=True,policy_preservation_max_error=error,optimizer_reinitialized=True,mass_kg=mass))


def train(mass,envs,updates,label,source=None,start_mode='beginning'):
    acceptance=json.loads((OUT/'bend/physics/acceptance.json').read_text())
    assert acceptance['passed'],'Physical gates have not passed'
    assert acceptance.get('environment_signature')==environment_signature(),'Physical gate is stale'
    if 'reference_sha256' in acceptance:
        assert acceptance['reference_sha256']==hashlib.sha256((OUT/'bend/reference.npz').read_bytes()).hexdigest()
    if envs>=128:
        for name in ['scaling_gate','full_geometry']:
            gate=json.loads((OUT/f'audit/{name}.json').read_text())
            assert gate['passed'] and gate['reference_sha256']==reference_hash() and gate.get('environment_signature')==environment_signature(),f'{name} is missing, failed, or stale'
        assert json.loads((OUT/'audit/warp_contact_sensor.json').read_text())['passed'],'Warp per-world collision guard has not passed'
    initialize_transfer(label,envs,updates,mass,source)
    save_json(OUT/'bend/training'/label/'provenance.json',dict(reference_sha256=reference_hash(),environment_signature=environment_signature(),
        env_source_sha256=hashlib.sha256(Path('musclemimic/environments/humanoids/myofullbody_box_friction.py').read_bytes()).hexdigest(),
        initial_source=str(source or SOURCE['bend']),mass_kg=mass,num_envs=envs,updates=updates,start_mode=start_mode))
    legacy.OUT=OUT;legacy.make_cpu=make_cpu
    legacy.train('bend',mass,envs,updates,label,source,start_mode=start_mode)


def evaluate(checkpoint=None,label='baseline',mass=5.):
    import jax,jax.numpy as jnp
    from musclemimic.runner.eval_utils import load_checkpoint
    from musclemimic.algorithms import PPOJax
    cfg=OmegaConf.load(OUT/'bend/base_config.yaml');cfg.experiment.env_params.payload_mass_kg=mass
    e=make_cpu(cfg);m,d=e._model,e._data
    checkpoint=checkpoint or OUT/'bend/training/smoke_5kg/initialization/checkpoint_0'
    _,agent,_=load_checkpoint(str(checkpoint));conf=PPOJax.init_agent_conf(e,cfg)
    policy=jax.jit(lambda obs:conf.network.apply(dict(params=agent.train_state.params,run_stats=agent.train_state.run_stats),jnp.atleast_2d(obs))[0].mode())
    folder=OUT/'bend/evaluations'/label;folder.mkdir(parents=True,exist_ok=True)
    episodes=[];n=int(e.th.len_trajectory(0));qa=int(m.jnt_qposadr[m.joint('payload_free').id])
    for start in [0,n//5,2*n//5,3*n//5]:
        e.th.fixed_start_conf=[0,start];e.th.random_start=False;obs=e.reset(jax.random.PRNGKey(2026))
        rows=[];qposes=[];reason='reference_end'
        for step in range(n-start-1):
            action=np.asarray(policy(obs)).reshape(-1)
            obs,reward,absorbing,done,info=e.step(action)
            assert np.isfinite(d.qpos).all() and np.isfinite(obs).all()
            ref=e.th.get_current_traj_data(e._additional_carry,np).qpos.copy()
            origin=e.th.get_init_traj_data(e._additional_carry,np).qpos[:2]
            ref[:2]-=origin;ref[qa:qa+2]-=origin
            row=contact_metrics(e);row.update(t=step*.01,reward=float(reward),height_m=float(d.qpos[qa+2]),
                reference_height_m=float(ref[qa+2]),object_error_m=float(np.linalg.norm(d.qpos[qa:qa+3]-ref[qa:qa+3])),
                root_error_m=float(np.linalg.norm(d.qpos[:3]-ref[:3])),activation_rms=float(np.sqrt(np.mean(d.act**2))))
            rows.append(row);qposes.append(d.qpos.copy())
            if absorbing:reason='task_failure';break
            if done and step<n-start-2:reason='early_done';break
        peak=max(r['max_penetration_m'] for r in rows)
        surface=sum(e.box_support_pos[2:])+e.box_support_half_size[2]+e.box_half_size[2]
        airborne=[r for r in rows if r['height_m']>surface+.04]
        bilateral=float(np.mean([all(f>1. for f in r['normal_n']) for r in airborne])) if airborne else 0.
        friction=float(np.mean([sum(r['friction_vertical_n']) for r in airborne])) if airborne else 0.
        other=float(np.mean([abs(r['other_vertical_n']) for r in airborne])) if airborne else 0.
        episodes.append(dict(start=start,steps=len(rows),termination=reason,peak_penetration_m=peak,
            bilateral_contact_fraction=bilateral,airborne_mean_friction_vertical_n=friction,
            airborne_other_body_vertical_n=other,
            quality_pass=reason=='reference_end' and peak<.003 and max(r['object_error_m'] for r in rows)<.10
                and bilateral>.85 and friction>.7*mass*9.81 and other<.1*mass*9.81,
            trajectory=rows))
        np.savez_compressed(folder/f'start_{start}_rollout.npz',qpos=qposes,dt=.01)
        print('BOX_EVAL',label,start,reason,len(rows),'penetration',peak,flush=True)
    result=dict(checkpoint=str(checkpoint),mass_kg=mass,reference_sha256=reference_hash(),environment_signature=environment_signature(),all_quality_pass=all(r['quality_pass'] for r in episodes),
                numerical_contact_pass=all(r['peak_penetration_m']<.003 for r in episodes),episodes=episodes)
    save_json(folder/'results.json',result);e.stop();return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','validate','geometry','train','evaluate'])
    p.add_argument('--mass',type=float,default=5.);p.add_argument('--envs',type=int,default=32)
    p.add_argument('--updates',type=int,default=4);p.add_argument('--label',default='smoke_5kg');p.add_argument('--checkpoint')
    p.add_argument('--start-mode',choices=['beginning','random'],default='beginning')
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='validate':validate()
    elif a.action=='geometry':geometry()
    elif a.action=='evaluate':evaluate(a.checkpoint,a.label,a.mass)
    else:train(a.mass,a.envs,a.updates,a.label,a.checkpoint,a.start_mode)
