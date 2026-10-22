"""Physical acceptance checks before scaling loaded imitation environments."""
from musclemimic.runner.workflows.payload_finetune import *


def validate(task):
    import jax
    cfg=OmegaConf.load(OUT/task/'base_config.yaml');e=make_cpu(cfg);m,d=e._model,e._data
    n=int(e.th.len_trajectory(0));checks=[]
    for start in np.linspace(0,n-2,9,dtype=int):
        e.th.fixed_start_conf=[0,int(start)];e.th.random_start=False
        obs=e.reset(jax.random.PRNGKey(71));q=d.qpos.copy()
        assert np.isfinite(obs).all() and np.isfinite(q).all()
        error=0.
        if task=='bend':
            error=max(float(np.linalg.norm(d.site_xpos[m.site('S_grasp_'+s).id]-d.site_xpos[m.site('payload_grip_'+s).id])) for s in ['r','l'])
            assert error<.003,(start,error)
        assert abs(float(obs[-1])-(int(start)+cfg.experiment.env_params.goal_params.source_start_frame)/cfg.experiment.env_params.goal_params.source_frame_count)<1e-6
        obs2=e.reset(jax.random.PRNGKey(71));assert np.array_equal(q,d.qpos) and np.allclose(obs,obs2)
        checks.append(dict(start=int(start),max_grip_error_m=error,phase=float(obs[-1]),repeat_reset_identical=True))
    b=e.payload_body_id
    mass=float(m.body_mass[b]);assert abs(mass-(2 if task=='lift' else 5))<1e-8
    assert np.all(m.body_inertia[b]>0)
    gravity={}
    if task=='lift':
        d.qpos[:]=e.th.traj.data.qpos[n//2];d.qvel[:]=0;mujoco.mj_forward(m,d)
        high=d.qfrc_bias.copy();q=d.qpos.copy()
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacBodyCom(m,d,jp,jr,b)
        oldmass=m.body_mass[b];oldinertia=m.body_inertia[b].copy()
        m.body_mass[b]=oldmass*1e-6;m.body_inertia[b]*=1e-6;mujoco.mj_setConst(m,d)
        d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
        delta=high-d.qfrc_bias;expected=jp[2]*(oldmass-m.body_mass[b])*9.81
        assert np.max(abs(delta-expected))<1e-6
        gravity=dict(root_added_weight_n=float(delta[2]),elbow_added_gravity_nm=float(delta[m.jnt_dofadr[m.joint('elbow_flexion_r').id]]),jacobian_load_max_error=float(np.max(abs(delta-expected))))
        m.body_mass[b]=oldmass;m.body_inertia[b]=oldinertia;mujoco.mj_setConst(m,d)
    # Copy the actual compiled payload inertial and collision geometry into a
    # fixture: no human, no muscles, no hidden support. Test hold and free drop.
    fmt=lambda x:' '.join(map(str,np.asarray(x).reshape(-1)))
    xml=['<mujoco><option timestep="0.002"/><worldbody><geom name="ground" type="plane" size="2 2 .1"/>']
    sites=[]
    if task=='bend':
        for s in ['r','l']:
            pos=m.site_pos[m.site('payload_grip_'+s).id]
            xml.append(f'<site name="anchor_{s}" pos="{fmt(pos+[0,0,1])}"/>')
            sites.append(f'<site name="grip_{s}" pos="{fmt(pos)}"/>')
    xml.append('<body name="load" pos="0 0 1"><freejoint/>')
    xml.append(f'<inertial mass="{mass}" diaginertia="{fmt(m.body_inertia[b])}" pos="{fmt(m.body_ipos[b])}" quat="{fmt(m.body_iquat[b])}"/>')
    types={3:'capsule',4:'ellipsoid',5:'cylinder',6:'box',2:'sphere'}
    for g in e.payload_geom_ids:
        xml.append(f'<geom type="{types[int(m.geom_type[g])]}" size="{fmt(m.geom_size[g])}" pos="{fmt(m.geom_pos[g])}" quat="{fmt(m.geom_quat[g])}" friction="{fmt(m.geom_friction[g])}" contype="{int(m.geom_contype[g])}" conaffinity="{int(m.geom_conaffinity[g])}"/>')
    xml.extend(sites);xml.append('</body></worldbody>')
    if task=='bend':xml.append('<equality>'+''.join(f'<connect site1="anchor_{s}" site2="grip_{s}" solref=".02 1"/>' for s in ['r','l'])+'</equality>')
    xml.append('</mujoco>')
    fixture=''.join(xml);p=OUT/task/'physics';p.mkdir(parents=True,exist_ok=True);(p/'fixture.xml').write_text(fixture)
    fm=mujoco.MjModel.from_xml_string(fixture);fd=mujoco.MjData(fm)
    hold={}
    if task=='bend':
        for i in range(500):mujoco.mj_step(fm,fd)
        hold=dict(vertical_constraint_force_n=float(fd.qfrc_constraint[2]),expected_weight_n=mass*9.81,
            height_m=float(fd.qpos[2]),equality_force_components=fd.efc_force[:6].tolist())
        assert abs(fd.qfrc_constraint[2]-mass*9.81)<.1
        fd.eq_active[:]=False
    forces=[]
    for i in range(1000):
        mujoco.mj_step(fm,fd)
        if i>900:
            force=0.
            for c in range(fd.ncon):
                cf=np.zeros(6);mujoco.mj_contactForce(fm,fd,c,cf);force+=cf[0]
            forces.append(force)
    contact=dict(final_height_m=float(fd.qpos[2]),contacts=int(fd.ncon),mean_ground_normal_n=float(np.mean(forces)),expected_weight_n=mass*9.81)
    assert fd.ncon>0 and abs(contact['mean_ground_normal_n']-mass*9.81)<.1
    save_json(p/'acceptance.json',dict(task=task,mass_kg=mass,inertia_kg_m2=m.body_inertia[b].tolist(),reset_checks=checks,
        gravity_transfer=gravity,two_hand_hold_fixture=hold,free_drop_fixture=contact,passed=True,
        scope='Ideal hand attachments replace gripping friction; ground/body collisions remain physical. Fixture uses compiled payload geometry/inertia.'))
    e.stop();print('PHYSICS_PASS',task,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['lift','bend'],required=True);validate(p.parse_args().task)
