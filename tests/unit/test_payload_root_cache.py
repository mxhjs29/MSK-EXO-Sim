from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from musclemimic.core.terminal_state_handler.enhanced_fullbody import MeanRelativeSiteDeviationWithRootTerminalStateHandler


@pytest.mark.parametrize('backend',[np,jnp])
def test_root_check_uses_loaded_sparse_cache_order(backend):
    h=object.__new__(MeanRelativeSiteDeviationWithRootTerminalStateHandler)
    h.enable_site_check=True;h.root_site='pelvis_mimic';h.root_site_id=0
    h.root_traj_index=1  # stale constructor mapping points to the hand
    h._root_qpos_ids_xy=np.array([0,1]);h.root_deviation_threshold=.25
    ref=SimpleNamespace(site_xpos=backend.array([[4.,-2.,1.],[4.,-2.,1.6]]),qpos=backend.array([4.,-2.,1.,1.,0.,0.,0.]))
    th=SimpleNamespace(traj=SimpleNamespace(info=SimpleNamespace(site_names=['pelvis_mimic','hand_mimic'])),
        get_current_traj_data=lambda *a:ref,get_init_traj_data=lambda *a:ref)
    env=SimpleNamespace(th=th)
    def check(x):return h._check_root_deviation(env,SimpleNamespace(site_xpos=x),None,backend)
    if backend is jnp:check=jax.jit(check)
    assert not bool(check(backend.array([[0.,0.,1.]])))
    assert bool(check(backend.array([[.30,0.,1.]])))
