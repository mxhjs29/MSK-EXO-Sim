"""Small first-stage PPO smoke test for myolegs walking on flat terrain."""

import os

import hydra
from omegaconf import OmegaConf

from musclemimic.runner.engine import run_experiment
from musclemimic.runner.logging import UnifiedHooks


@hydra.main(version_base=None, config_path="../fullbody", config_name="conf_fullbody_assist")
def main(config):
    # The repository config is intentionally large; this test only checks one
    # rollout and one PPO update using the real KIT walking cache.
    OmegaConf.set_struct(config, False)
    config.wandb.mode = "disabled"
    msk_key = os.environ.get("MUSCLEMIMIC_SMOKE_MSK_KEY", "myolegs")
    config.experiment.env_params.msk_key = msk_key
    config.experiment.env_params.device_key = None
    config.experiment.env_params.terrain = {"terrain": "flat"}
    config.experiment.env_params.mjx_backend = "jax"
    num_envs = int(os.environ.get("MUSCLEMIMIC_SMOKE_NUM_ENVS", "2"))
    num_steps = int(os.environ.get("MUSCLEMIMIC_SMOKE_NUM_STEPS", "2"))
    total_timesteps = int(os.environ.get("MUSCLEMIMIC_SMOKE_TOTAL_TIMESTEPS", str(num_envs * num_steps * 2)))
    config.experiment.env_params.num_envs = num_envs
    config.experiment.env_params.goal_params = {}
    config.experiment.env_params.reward_params = {}
    config.experiment.env_params.terminal_state_type = "NoTerminalStateHandler"
    config.experiment.env_params.enable_touch_sensor_observations = msk_key != "myoarms"
    config.experiment.env_params.enable_muscle_length_observations = False
    config.experiment.env_params.enable_muscle_velocity_observations = False
    config.experiment.env_params.enable_muscle_force_observations = False
    config.experiment.env_params.enable_muscle_excitation_observations = False
    config.experiment.env_params.enable_muscle_activation_observations = False
    config.experiment.task_factory.params.amass_dataset_conf = {
        "rel_dataset_path": "KIT/11/WalkingStraightForwards01_poses",
        "retargeting_method": "gmr",
        "trajectory_cache_type": "sparse",
    }
    config.experiment.total_timesteps = total_timesteps
    config.experiment.total_timesteps_is_absolute = True
    config.experiment.ppo_config.num_steps = num_steps
    config.experiment.ppo_config.num_minibatches = 1
    config.experiment.ppo_config.update_epochs = 1
    config.experiment.actor_hidden_layers = [32]
    config.experiment.critic_hidden_layers = [32]
    config.experiment.use_layernorm = False
    config.experiment.validation.active = False
    config.experiment.validation.num = 1
    config.experiment.save_checkpoints = False
    config.experiment.checkpoints_on_validation = False
    config.experiment.auto_resume = False
    config.experiment.n_seeds = 1
    run_experiment(config, hooks=UnifiedHooks())


if __name__ == "__main__":
    main()
