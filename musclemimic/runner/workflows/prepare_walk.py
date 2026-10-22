from musclemimic.utils.paths import workspace_root, config_root
"""Prepare the walking reference on CPU using the training model configuration."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys
sys.path.insert(0, str(workspace_root()))
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from loco_mujoco.task_factories import TaskFactory
import numpy as np

with initialize_config_dir(config_dir=str((config_root()/"experiments/fullbody")), version_base=None):
    config = compose(config_name='conf_fullbody_walk')
params = OmegaConf.to_container(config.experiment.env_params, resolve=True)
params['env_name'] = 'MyoFullBodyAssistEnv'
params['mjx_backend'] = 'jax'
params.pop('num_envs', None)
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
env = factory.make(**params, **config.experiment.task_factory.params)
traj = env.th.traj
assert traj.data.qpos.shape[-1] == env._model.nq
assert np.isfinite(traj.data.qpos).all()
print(f'READY: nq={env._model.nq}, nv={env._model.nv}, muscles={env._model.nu}, '
      f'frames={len(traj.data.qpos)}, fps={traj.info.frequency}', flush=True)
