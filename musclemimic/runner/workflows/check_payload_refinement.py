"""CPU/Warp contact readout parity on independent worlds, including nonhand support."""
from pathlib import Path
import json
import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from mujoco import mjx
from musclemimic.environments.humanoids.myofullbody_payload_refinement import add_contact_feedback,ContactReadout

s=mujoco.MjSpec.from_string('''<mujoco><option cone="elliptic" iterations="100" tolerance="1e-10"/>
<worldbody><body name="human">
<body name="right" pos="-.128 0 1"><geom type="sphere" size=".01" friction=".8 0 0"/></body>
<body name="left" pos=".128 0 1"><geom type="sphere" size=".01" friction=".8 0 0"/></body>
<body name="forearm" pos="0 0 .872"><geom type="sphere" size=".012" friction=".8 0 0"/></body>
</body><body name="box" pos="0 0 1"><freejoint/><geom type="box" size=".12 .1 .12" mass="1" friction=".8 0 0"/></body>
</worldbody></mujoco>''')
add_contact_feedback(s,('right','left'),'human','box');m=s.compile();read=ContactReadout(m)
positions=np.array([0.,.003,-.002],dtype=np.float32);cpu=[]
for x in positions:
    d=mujoco.MjData(m);d.qpos[0]=x;d.qvel[2]=-.2;mujoco.mj_forward(m,d)
    f=read(d,np);direct=np.zeros((2,3));normal=np.zeros(2);other=np.zeros(3)
    for ci in range(d.ncon):
        c=d.contact[ci];bodies=m.geom_bodyid[c.geom];box=m.body('box').id
        if box not in bodies:continue
        ff=np.zeros(6);mujoco.mj_contactForce(m,d,ci,ff);sign=1 if bodies[1]==box else -1
        w=sign*c.frame.reshape(3,3).T@ff[:3];b=int(bodies[0] if bodies[1]==box else bodies[1])
        if b in [m.body('right').id,m.body('left').id]:
            side=0 if b==m.body('right').id else 1;direct[side]+=w;normal[side]+=ff[0]
        else:other+=w
    assert np.allclose(f['hand'],direct,atol=1e-8)
    assert np.allclose(f['normal'],normal,atol=1e-8)
    assert np.allclose(f['other'],other,atol=1e-8)
    assert f['other'][2]>0
    cpu.append(np.r_[f['hand'].ravel(),f['normal'],f['other'],f['friction_z']])
mx=mjx.put_model(m,impl='warp')
@jax.jit
@jax.vmap
def run(x):
    d=mjx.make_data(m,impl='warp',naconmax=32,njmax=128)
    d=d.replace(qpos=d.qpos.at[0].set(x),qvel=d.qvel.at[2].set(-.2))
    f=read(mjx.forward(mx,d),jnp)
    return jnp.concatenate((f['hand'].reshape(-1),f['normal'],f['other'],f['friction_z']))
gpu=np.asarray(run(jnp.asarray(positions)));cpu=np.asarray(cpu)
error=float(np.max(abs(cpu-gpu)))
assert np.allclose(gpu,cpu,atol=.03,rtol=.01),(gpu,cpu,error)
assert not np.allclose(gpu[0],gpu[1])
out=Path('outputs/payload_refinement_v3/audit/warp_contact_feedback.json')
out.write_text(json.dumps(dict(passed=True,cpu=cpu.tolist(),warp=gpu.tolist(),max_error_n=error,
    independent_worlds=True,nonhand_support_detected=True,raw_cpu_force_parity=True),indent=2)+'\n')
print('CONTACT_FEEDBACK_PASSED',error,flush=True)
