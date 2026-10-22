"""Run a two-update PPO smoke test with a tiny in-memory trajectory."""

import os
from pathlib import Path

import hydra
import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from loco_mujoco.trajectory.dataclasses import (
    Trajectory,
    TrajectoryData,
    TrajectoryInfo,
    TrajectoryModel,
)
from musclemimic.environments.humanoids.myofullbody_assist import MjxMyoFullBodyAssistEnv
from musclemimic.runner.engine import run_experiment
from musclemimic.runner.logging import UnifiedHooks
import musclemimic.runner.engine as engine
from loco_mujoco.task_factories import TaskFactory


_TINY_TRAJ = None


def make_tiny_trajectory(env):
    model = env._model
    qpos = np.repeat(np.asarray(env._data.qpos, dtype=np.float32)[None, :], 2, axis=0)
    qvel = np.zeros((2, model.nv), dtype=np.float32)
    xpos = np.zeros((2, model.nbody, 3), dtype=np.float32)
    xquat = np.zeros((2, model.nbody, 4), dtype=np.float32)
    cvel = np.zeros((2, model.nbody, 6), dtype=np.float32)
    subtree_com = np.zeros((2, model.nbody, 3), dtype=np.float32)
    site_xpos = np.zeros((2, model.nsite, 3), dtype=np.float32)
    site_xmat = np.zeros((2, model.nsite, 9), dtype=np.float32)
    data = mujoco.MjData(model)
    for frame in range(2):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, data)
        xpos[frame] = data.xpos
        xquat[frame] = data.xquat
        cvel[frame] = data.cvel
        subtree_com[frame] = data.subtree_com
        site_xpos[frame] = data.site_xpos
        site_xmat[frame] = data.site_xmat
    info = TrajectoryInfo(
        joint_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
        model=TrajectoryModel(
            njnt=model.njnt,
            jnt_type=np.asarray(model.jnt_type),
            nbody=model.nbody,
            body_rootid=np.asarray(model.body_rootid),
            body_weldid=np.asarray(model.body_weldid),
            body_mocapid=np.asarray(model.body_mocapid),
            body_pos=np.asarray(model.body_pos),
            body_quat=np.asarray(model.body_quat),
            body_ipos=np.asarray(model.body_ipos),
            body_iquat=np.asarray(model.body_iquat),
            nsite=model.nsite,
            site_bodyid=np.asarray(model.site_bodyid),
            site_pos=np.asarray(model.site_pos),
            site_quat=np.asarray(model.site_quat),
        ),
        frequency=1.0 / env.dt,
        body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)],
        site_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)],
    )
    return Trajectory(
        info=info,
        data=TrajectoryData(
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
            cvel=cvel,
            subtree_com=subtree_com,
            site_xpos=site_xpos,
            site_xmat=site_xmat,
            split_points=np.asarray([0, 2]),
        ),
    )


@hydra.main(version_base=None, config_path="../fullbody", config_name="conf_fullbody_assist")
def main(config):
    global _TINY_TRAJ
    OmegaConf.set_struct(config, False)
    msk_key = os.environ.get("MUSCLEMIMIC_SMOKE_MSK_KEY", "myofullbody")
    device_key = os.environ.get("MUSCLEMIMIC_SMOKE_DEVICE_KEY", "HipExoskeleton_L1")
    terrain = {"terrain": os.environ.get("MUSCLEMIMIC_SMOKE_TERRAIN", "flat")}
    mjx_backend = os.environ.get("MUSCLEMIMIC_SMOKE_MJX_BACKEND", "warp")
    if mjx_backend not in {"warp", "jax"}:
        raise ValueError("MUSCLEMIMIC_SMOKE_MJX_BACKEND must be 'warp' or 'jax'")
    config.wandb.mode = "disabled"
    config.experiment.env_params.num_envs = 2
    config.experiment.env_params.init_state_type = "TrajInitialStateHandler"
    config.experiment.env_params.goal_type = "GoalTrajMimic"
    config.experiment.env_params.reward_type = "MimicReward"
    # The generic full-body terminal handler requires pelvis sites.  The
    # myoarms target view intentionally has no pelvis, so this tiny smoke test
    # uses the neutral handler for all target views.
    config.experiment.env_params.terminal_state_type = "NoTerminalStateHandler"
    config.experiment.env_params.goal_params = {}
    config.experiment.env_params.reward_params = {}
    config.experiment.env_params.th_params = {"random_start": False, "fixed_start_conf": (0, 0)}
    config.experiment.num_envs = 2
    config.experiment.total_timesteps = 8
    config.experiment.total_timesteps_is_absolute = True
    config.experiment.ppo_config.num_steps = 2
    config.experiment.ppo_config.num_minibatches = 1
    config.experiment.ppo_config.update_epochs = 1
    config.experiment.validation.active = False
    config.experiment.validation.num = 1
    config.experiment.save_checkpoints = False
    config.experiment.checkpoints_on_validation = False
    config.experiment.auto_resume = False
    config.experiment.actor_hidden_layers = [64]
    config.experiment.critic_hidden_layers = [64]
    config.experiment.n_seeds = 1
    config.experiment.task_factory.name = "ImitationFactory"
    config.experiment.task_factory.params = None
    config.experiment.env_params.msk_key = msk_key
    config.experiment.env_params.device_key = device_key
    config.experiment.env_params.terrain = terrain
    config.experiment.env_params.mjx_backend = mjx_backend
    config.experiment.env_params.device_control_mode = "zero"
    config.experiment.env_params.device_control_value = 0.0
    # Build the tiny reference from an MSK-only model.  Use flat terrain here
    # so the reference contains no warehouse free-body joints; ImitationFactory
    # adapts it by MSK joint name to the composed training model and initializes
    # all non-MSK coordinates from model.qpos0.
    trajectory_env = MjxMyoFullBodyAssistEnv(
        msk_key=msk_key,
        device_key=None,
        terrain={"terrain": "flat"},
        device_control_mode="zero",
        device_control_value=0.0,
        mjx_backend=mjx_backend,
        num_envs=1,
        nconmax=256,
        njmax=1024,
        disable_fingers=True,
        enable_muscle_length_observations=True,
        enable_muscle_velocity_observations=True,
        enable_muscle_force_observations=True,
        enable_muscle_excitation_observations=True,
        enable_muscle_activation_observations=True,
        enable_touch_sensor_observations=True,
        init_state_type="DefaultInitialStateHandler",
        goal_type="NoGoal",
        reward_type="NoReward",
        terminal_state_type="NoTerminalStateHandler",
    )
    tiny_traj = make_tiny_trajectory(trajectory_env)
    # Keep the Trajectory as a Python object; OmegaConf's structured wrapper
    # cannot represent its Union[Array, ndarray] fields.
    _TINY_TRAJ = tiny_traj

    def instantiate_smoke_env(cfg):
        factory_cls = TaskFactory.get_factory_cls(cfg.experiment.task_factory.name)
        env_params = OmegaConf.to_container(cfg.experiment.env_params, resolve=True)
        task_params = {
            "custom_dataset_conf": {"traj": _TINY_TRAJ, "trajectory_cache_type": "full"},
        }
        return factory_cls.make(**env_params, **task_params)

    engine.instantiate_env = instantiate_smoke_env
    run_experiment(config, hooks=UnifiedHooks())


if __name__ == "__main__":
    main()
