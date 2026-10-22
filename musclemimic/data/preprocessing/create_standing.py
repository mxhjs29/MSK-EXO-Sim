"""Create a kinematic standing variant; does not alter training or source data."""
import os, sys, json
from pathlib import Path
from musclemimic.utils.paths import workspace_root
ROOT=workspace_root()
sys.path.insert(0,str(ROOT))
os.environ['JAX_PLATFORMS']='cpu'
os.environ['MUJOCO_GL']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']='0'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import fullbody.prepare_run as prep
os.environ['CONVERTED_AMASS_PATH']=str(ROOT/'outputs/kit3_dumbbell_visualization/amass_cache_root_aligned')
import numpy as np, mujoco, imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from loco_mujoco.task_factories import TaskFactory
from loco_mujoco.trajectory.dataclasses import Trajectory
OUT=ROOT/'outputs/fullbody_lift_dumbbell'
OUT.mkdir(parents=True,exist_ok=True)
motion='KIT/3/Experiment3a_legs_01_poses'
with initialize_config_dir(config_dir=str(ROOT/'fullbody'),version_base=None):
    cfg=compose(config_name='conf_fullbody_bend',overrides=[f'experiment.task_factory.params.amass_dataset_conf.rel_dataset_path=[{motion}]'])
p=OmegaConf.to_container(cfg.experiment.env_params,resolve=True)
p.update(env_name='MyoFullBodyAssistEnv',mjx_backend='jax',disable_fingers=False)
p.pop('num_envs',None)
p['th_params']=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False)
env=TaskFactory.get_factory_cls(cfg.experiment.task_factory.name).make(**p,**OmegaConf.to_container(cfg.experiment.task_factory.params,resolve=True))
m=env._model; d=mujoco.MjData(m)
print('DEFAULT ROOT',m.qpos0[:7],flush=True)
print('GEOMS',[(m.geom(i).name,m.body(int(m.geom_bodyid[i])).name,int(m.geom_type[i])) for i in range(m.ngeom) if m.geom_contype[i] or m.geom_conaffinity[i]],flush=True)
q=np.asarray(env.th.traj.data.qpos)[1811:2135].copy()
original=q.copy()
# Neutral model stance preserves anatomically coupled knee coordinates.
right_begin=int(m.jnt_qposadr[m.joint('sternoclavicular_r2_r').id])
left_begin=int(m.jnt_qposadr[m.joint('sternoclavicular_r2_l').id])
q[:]=m.qpos0
q[:,right_begin:left_begin]=original[:,right_begin:left_begin]
# Preserve source heading, remove root tilt; model default root supplies upright axes.
r0=Rotation.from_quat(m.qpos0[[4,5,6,3]])
rs=Rotation.from_quat(original[0,[4,5,6,3]])
yaw=(rs*r0.inv()).as_euler('xyz')[2]
quat=(Rotation.from_euler('z',yaw)*r0).as_quat()
q[:,3:7]=quat[[3,0,1,2]]
q[:,:2]=0
# Infer sole height from actual foot collision geometry in the neutral stance.
d.qpos[:]=q[0]; mujoco.mj_forward(m,d)
foot_geoms=[i for i in range(m.ngeom) if any(s in m.body(int(m.geom_bodyid[i])).name.lower() for s in ('calcn','talus','toes','foot')) and (m.geom_contype[i] or m.geom_conaffinity[i])]
assert foot_geoms,'No foot collision geometries found'
def bottom(i):
    typ=int(m.geom_type[i]); R=d.geom_xmat[i].reshape(3,3); size=m.geom_size[i]
    if typ==int(mujoco.mjtGeom.mjGEOM_SPHERE): extent=size[0]
    elif typ==int(mujoco.mjtGeom.mjGEOM_CAPSULE): extent=size[0]+abs(R[2,2])*size[1]
    elif typ==int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): extent=np.linalg.norm(R[2]*size)
    elif typ==int(mujoco.mjtGeom.mjGEOM_BOX): extent=abs(R[2])@size
    else: raise ValueError(f'Unhandled foot geom {typ}')
    return d.geom_xpos[i,2]-extent
low=min(bottom(i) for i in foot_geoms)
q[:,2]+=0.001-low
v=np.empty((len(q),m.nv))
for i in range(len(q)):
    a=max(0,i-1); b=min(len(q)-1,i+1)
    mujoco.mj_differentiatePos(m,v[i],(b-a)/100,q[a],q[b])
# Save in the same trajectory schema as the original reference. Empty derived arrays
# are intentional, as in the source cache; consumers recompute them from qpos/qvel.
src=next((ROOT/'outputs/kit3_dumbbell_visualization/amass_cache_root_aligned').rglob('Experiment3a_legs_01_poses.npz'))
with np.load(src,allow_pickle=True) as f: payload={k:f[k] for k in f.files}
payload.update(qpos=q,qvel=v,split_points=np.array([0,len(q)]))
meta=payload['metadata'].item().copy();meta.update(synthetic=True,source_motion=motion,source_interval_s=[18.14,21.38],edit='Neutral standing base and left arm; original right shoulder, arm and fingers retained; no object dynamics')
payload['metadata']=meta
np.savez_compressed(OUT/'standing_right_dumbbell.npz',**payload)
loaded=Trajectory.load(str(OUT/'standing_right_dumbbell.npz'),backend=np)
assert loaded.data.qpos.shape==q.shape and np.isfinite(v).all()
assert np.array_equal(q[:,right_begin:left_begin],original[:,right_begin:left_begin])
cam=mujoco.MjvCamera();cam.distance=3.2;cam.elevation=-10;cam.lookat[:]=[0,0,1.0]
renderer=mujoco.Renderer(m,height=480,width=480)
with imageio.get_writer(str(OUT/'standing_right_dumbbell.mp4'),fps=25,codec='libx264') as writer:
    for i in range(0,len(q),4):
        d.qpos[:]=q[i];mujoco.mj_forward(m,d)
        views=[]
        for angle in (90,150):
            cam.azimuth=angle;renderer.update_scene(d,camera=cam);views.append(renderer.render().copy())
        im=np.concatenate(views,axis=1);writer.append_data(im)
        if i==160: imageio.imwrite(OUT/'preview.jpg',im)
renderer.close()
d.qpos[:]=q[0];mujoco.mj_forward(m,d)
report=dict(source=str(src),source_interval_s=[18.14,21.38],fps=100,frames=len(q),duration_s=len(q)/100,right_arm_max_change_rad=float(abs(q[:,right_begin:left_begin]-original[:,right_begin:left_begin]).max()),foot_bottom_heights_m={m.geom(i).name:float(bottom(i)) for i in foot_geoms},root_height_m=float(q[0,2]),note='Synthetic kinematic reference, no dumbbell mass or dynamic balance validation.')
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
env.stop()
