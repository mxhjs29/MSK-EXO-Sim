from types import SimpleNamespace
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from loco_mujoco.trajectory.handler import TrajectoryHandler,TrajState

@struct.dataclass
class Carry:
    key: object
    traj_state: object

def test_weighted_starts_reach_requested_region_deterministically_in_jit():
    h=object.__new__(TrajectoryHandler)
    h.traj=SimpleNamespace(data=SimpleNamespace(split_points=jnp.array([0,100])))
    h.random_start=True;h.use_fixed_start=False;h.start_from_random_step=True
    h.start_step_weights=np.r_[np.zeros(70),np.ones(20)/20,np.zeros(10)]
    @jax.jit
    @jax.vmap
    def sample(key):
        _,c=h.reset_state(None,None,None,Carry(key,TrajState(0,0,0)),jnp)
        return c.traj_state.subtraj_step_no
    keys=jax.random.split(jax.random.PRNGKey(7),1000)
    a=np.asarray(sample(keys));b=np.asarray(sample(keys))
    assert np.array_equal(a,b)
    assert a.min()==70 and a.max()==89
    assert len(np.unique(a))==20
