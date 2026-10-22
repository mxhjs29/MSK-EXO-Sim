"""Verify independent per-world contact depth sensing on the actual Warp backend."""
import json
from pathlib import Path
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np

m=mujoco.MjModel.from_xml_string('''<mujoco><option cone="elliptic"/><worldbody>
<body name="hand" pos="0 0 1"><geom type="box" size=".01 .05 .05"/></body>
<body name="box" pos=".13 0 1"><freejoint/><geom type="box" size=".12 .1 .12" mass="5"/></body>
</worldbody><sensor><contact name="depth" subtree1="hand" body2="box" data="dist" reduce="mindist"/></sensor></mujoco>''')
mx=mjx.put_model(m,impl='warp')
@jax.jit
@jax.vmap
def run(x):
    d=mjx.make_data(m,impl='warp',naconmax=96,njmax=128)
    d=d.replace(qpos=d.qpos.at[0].set(x))
    return mjx.forward(mx,d).sensordata
positions=np.array([.131,.115,.125],np.float32)
gpu=np.asarray(run(jnp.array(positions))).reshape(-1)
cpu=[]
for x in positions:
    d=mujoco.MjData(m);d.qpos[0]=x;mujoco.mj_forward(m,d);cpu.append(float(d.sensordata[0]))
assert np.allclose(gpu,cpu,atol=2e-6),(gpu,cpu)
assert gpu[0]==0 and gpu[1]<-.014 and gpu[2]<-.004
result=dict(passed=True,cpu_depth_m=cpu,warp_depth_m=gpu.tolist(),independent_worlds=True)
Path('outputs/payload_box_friction_v2/audit/warp_contact_sensor.json').write_text(json.dumps(result,indent=2)+'\n')
print(result,flush=True)
