"""Opt-in MyoLegs26 PPO comparison; never imported by existing training entrypoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf, open_dict

from musclemimic.rewards.common.imitation import MimicReward
from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv, MjxMyoFullBodyAssistEnv
from musclemimic.retargeting.gait120_myolegs26 import correct_dependent_ranges, STEMS

CLIP = 'S101_Trial01_Step01'
ARMS = ('ppo', 'synergy_reward')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


from musclemimic.rewards.tasks.gait120 import synergy_score


from musclemimic.rewards.tasks.gait120 import Gait120SynergyReward


from musclemimic.tasks.gait120.environment import MyoFullBodyGait120SynergyEnv


from musclemimic.tasks.gait120.environment import MjxMyoFullBodyGait120SynergyEnv


for cls in (Gait120SynergyReward, MyoFullBodyGait120SynergyEnv, MjxMyoFullBodyGait120SynergyEnv):
    cls.register()


def activation_filter(excitation, dynamics, sample_dt=.01, physics_dt=.002):
    """MuJoCo muscle dynamics, explicit substeps, steady periodic initialization.

    MVC-normalized EMG is assumed to be excitation, not calibrated activation.
    This transformation is a stated model prior, not an observed left/right activation.
    """
    excitation = np.clip(np.asarray(excitation), 0., 1.)
    count = int(round(sample_dt/physics_dt))
    if not np.isclose(count*physics_dt, sample_dt):
        raise ValueError('Sample interval must contain whole physics steps')
    act = excitation[0].copy()
    history = np.empty_like(excitation)
    for cycle in range(20):
        for i, row in enumerate(excitation):
            history[i] = act
            for substep in range(count):
                next_row = excitation[(i+1) % len(excitation)]
                u = row+(next_row-row)*(substep/count)
                derivative = np.array([mujoco.mju_muscleDynamics(float(c), float(a), np.ascontiguousarray(p[:3]))
                                       for c, a, p in zip(u, act, dynamics)])
                act = np.clip(act+physics_dt*derivative, 0., 1.)
    return history


def prepare(root, references, project):
    from hydra import compose, initialize_config_dir
    root.mkdir(parents=True, exist_ok=True)
    references, project = references.resolve(), project.resolve()
    audit = json.loads((references/'completion_audit.json').read_text())
    assert audit['complete'] and not audit['protected_files_changed']
    model = mujoco.MjModel.from_binary_path(str(references/'model/myolegs26_reference.mjb'))
    emg = np.load(references/'clips'/CLIP/'emg_reference.npz', allow_pickle=False)
    assert str(emg['split']) == 'train'
    names = emg['group_names']
    ids = np.array([model.actuator(str(n)).id for n in names])
    excitation = np.concatenate((emg['right_reconstructed'], emg['left_assumed']), axis=1)
    activation = activation_filter(excitation, model.actuator_dynprm[ids])
    scale = np.maximum(np.sqrt(np.mean(activation**2, axis=0)), .05)
    weights = np.r_[np.ones(6), np.full(6, .5)]
    np.savez_compressed(root/'training_targets.npz', activation_proxy=activation,
                        excitation_proxy=np.clip(excitation, 0., 1.), scale=scale, group_weights=weights,
                        group_names=names, source_time_s=emg['source_time_s'], clip_id=CLIP,
                        activation_dynamics_applied=True, left_measured=False)
    with initialize_config_dir(config_dir=str(project/'fullbody'), version_base=None):
        cfg = compose(config_name='conf_gait120_emg')
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        exp = cfg.experiment
        ep = exp.env_params
        ep.env_name = 'MjxMyoFullBodyGait120SynergyEnv'
        ep.msk_key = 'myolegs26'
        ep.knee_coupling_consistent = False  # Private subclass corrects knees AND hips.
        # This coupled model is unstable with the fullbody preset's explicit damping.
        # Retain MuJoCo's normal implicit Euler damping in BOTH comparison arms.
        ep.model_option_conf = dict(iterations=4, ls_iterations=8, disableflags=0)
        ep.reward_type = 'Gait120SynergyReward'
        ep.reward_params.synergy_targets = str((root/'training_targets.npz').resolve())
        ep.reward_params.synergy_weight = .2
        ep.reward_params.synergy_warmup_steps = 10
        ep.reward_params.activation_energy_coeff = 0.
        sites = ['pelvis_mimic', 'upper_body_mimic', 'head_mimic'] + [
            f'{side}_{joint}_mimic' for side in ('left', 'right') for joint in ('hip', 'knee', 'ankle', 'toes')]
        ep.goal_params.sites_for_mimic = sites
        ep.reward_params.sites_for_mimic = sites
        ep.terminal_state_params.mean_site_deviation_threshold = .25
        ep.terminal_state_params.root_deviation_threshold = .25
        ep.terminal_state_params.root_orientation_threshold = .7
        exp.task_factory.params.custom_dataset_conf.traj_path = str(references/'clips'/CLIP/'trajectory.npz')
        exp.validation.rel_site_names = sites
        exp.validation.terminal_state_type = ep.terminal_state_type
        exp.validation.terminal_state_params = OmegaConf.to_container(ep.terminal_state_params)
        exp.validation.num = 4
        exp.validation.num_steps = 95
        exp.validation.video_frequency = 0  # Queue performs separate CPU checkpoint evaluations.
        exp.seeds = [42]
        exp.n_seeds = 1
        exp.checkpoint_interval = 100
        exp.max_checkpoints_to_keep = 3
        exp.async_checkpointing = False
    OmegaConf.save(cfg, root/'base_config.yaml')
    save_json(root/'protocol.json', dict(arms=list(ARMS), reference_clip=CLIP, updates_per_arm=10000,
        num_envs=4096, steps_per_update=64, seed=42, action_dimension=26,
        coefficient_by_arm=dict(ppo=0., synergy_reward=.2), minimum_activation_penalty=0.,
        prior='Phase-conditioned W H reconstruction, filtered through MuJoCo muscle activation dynamics',
        activation_assumptions=dict(emg_as_excitation=True, mvc_gain=1., additional_delay_ms=0,
            periodic_filter_cycles=20, substep_s=.002, calibrated_to_subject=False),
        left_prior_weight=.5, missing_actuators_not_penalized=True, startup_reward_ramp_steps=10,
        primary_evaluation='Actual data.act versus original six measured right EMG groups, no lag, skip first 100 ms',
        secondary_evaluation='Peak normalized cross-correlation within +/- 100 ms, reported separately',
        quality_gates=dict(full_reference=True, root_max_m=.25, relative_site_mean_m=.075,
                           leg_joint_mae_deg=20, significant_joint_rom_ratio=[.7, 1.4]),
        shared_initialization='Same architecture, seed and fresh initialization; fullbody checkpoint dimensions differ',
        synergy_basis_fit='Trial01-03 only, K selected using Trial04; PPO motion reference is Trial01 only',
        heldout_caveat='Trial05 previously inspected; descriptive holdout only',
        numerical_settings=dict(timestep_s=.002, substeps=5, implicit_euler_damping=True,
                                iterations=4, ls_iterations=8, same_for_both_arms=True),
        references=str(references), started=False))


def config_for(root, arm, label, envs=4096, updates=10000, cpu=False):
    cfg = OmegaConf.load(root/'base_config.yaml')
    folder = (root/arm/label).resolve()
    with open_dict(cfg):
        exp = cfg.experiment
        exp.env_params.reward_params.synergy_weight = 0. if arm == 'ppo' else .2
        exp.env_params.num_envs = envs
        exp.num_envs = envs
        if cpu:
            exp.env_params.env_name = 'MyoFullBodyGait120SynergyEnv'
            exp.env_params.mjx_backend = 'jax'
            exp.env_params.model_option_conf = dict(iterations=4, ls_iterations=8, disableflags=0)
        steps = 16 if envs < 128 else 64
        exp.ppo_config.num_steps = exp.num_steps = steps
        exp.ppo_config.num_minibatches = exp.num_minibatches = max(1, envs*steps//2048)
        exp.total_timesteps = envs*steps*updates
        exp.run_id = arm+'_'+label
        exp.checkpoint_root = str(folder/'checkpoints')
        exp.checkpoint_dir = str(folder/'checkpoints'/exp.run_id)
        exp.resume_from = None
        exp.checkpoint_interval = min(100, updates)
        exp.validation.interval = min(100, updates)
        exp.validation.num_envs = min(envs, 8)
    return cfg, folder


def cpu_env(cfg, trajectory=None):
    from musclemimic.runner.engine import instantiate_env
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    ep = cfg.experiment.env_params
    ep.env_name = 'MyoFullBodyGait120SynergyEnv'
    ep.mjx_backend = 'jax'
    ep.num_envs = 1
    ep.th_params = dict(random_start=False, fixed_start_conf=[0, 0], start_from_random_step=False)
    ep.model_option_conf = dict(iterations=4, ls_iterations=8, disableflags=0)
    if trajectory:
        cfg.experiment.task_factory.params.custom_dataset_conf.traj_path = str(trajectory)
        ep.reward_params.synergy_weight = 0.  # Never reward a held-out clip using training H.
    return instantiate_env(cfg)


def preflight(root):
    import jax
    from musclemimic.algorithms import PPOJax
    protocol = json.loads((root/'protocol.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(Path(protocol['references'])/'model/myolegs26_reference.mjb'))
    rows = []
    for arm in ARMS:
        cfg, _ = config_for(root, arm, 'preflight', envs=1, updates=1, cpu=True)
        env = cpu_env(cfg)
        for field in ['jnt_range', 'eq_data', 'actuator_gainprm', 'actuator_dynprm', 'body_mass']:
            np.testing.assert_array_equal(getattr(env._model, field), getattr(model, field))
        obs = env.reset(jax.random.PRNGKey(42))
        conf = PPOJax.init_agent_conf(env, cfg)
        for step in range(5):
            obs, reward, absorbing, done, info = env.step(np.full(26, .05))
            assert np.isfinite(obs).all() and np.isfinite(reward)
            assert int(env._additional_carry.traj_state.subtraj_step_no) == step+1
        assert env.info.action_space.shape == (26,)
        rows.append(dict(arm=arm, observation_dim=len(obs), action_dim=26,
                         finite_cpu_steps=5, matching_exported_model=True))
        env.stop()
    save_json(root/'preflight.json', dict(passed=True, checks=rows, gpu_training_started=False,
        note='CPU structure/reward/physics checks only. GPU smoke runs are gated on payload completion.'))
    print('PREFLIGHT_PASS', rows, flush=True)


def train(root, arm, label, envs, updates):
    from musclemimic.runner.workflows.gait120_synergy_queue import require_release
    require_release(root)
    assert json.loads((root/'preflight.json').read_text())['passed']
    if envs >= 128:
        assert json.loads((root/arm/'smoke_gate.json').read_text())['passed']
    import jax
    from hydra.conf import HydraConf
    from hydra.core.hydra_config import HydraConfig
    from musclemimic.runner.engine import run_experiment
    from musclemimic.runner.logging import ExperimentHooks
    from datetime import datetime, timezone
    cfg, folder = config_for(root, arm, label, envs, updates)
    folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, folder/'config.yaml')
    hydra = OmegaConf.structured(HydraConf())
    hydra.runtime.output_dir = str(folder)
    hydra.runtime.cwd = str(Path.cwd())
    HydraConfig.instance().set_config(OmegaConf.create(dict(hydra=hydra)))
    class Hooks(ExperimentHooks):
        def build_video_recorder(self, **kwargs):
            return None

        def enrich_log(self, log_dict, metrics_dict, env):
            values = {k: np.asarray(v).tolist() for k, v in log_dict.items()}
            values['recorded_at'] = datetime.now(timezone.utc).isoformat()
            values['update'] = int(round(float(values['Current Timestep'])/(envs*cfg.experiment.num_steps)))
            with (folder/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(values, allow_nan=False)+'\n')
            save_json(folder/'progress.json', values)
    jax.config.update('jax_default_matmul_precision', 'high')
    run_experiment(cfg, Hooks())
    checkpoint = Path(cfg.experiment.checkpoint_dir)/f'checkpoint_{updates}'
    if not checkpoint.is_dir():
        raise RuntimeError('Training exited without the requested final checkpoint')
    progress = json.loads((folder/'progress.json').read_text())
    assert progress['update'] >= updates
    save_json(folder/'completed.json', dict(complete=True, updates=updates, arm=arm,
                                           checkpoint=str(checkpoint), envs=envs))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'train'])
    parser.add_argument('--root', type=Path, default=Path('outputs/gait120_myolegs26_training_v1'))
    parser.add_argument('--references', type=Path, default=Path('outputs/gait120_myolegs26_synergy_v1'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--arm', choices=ARMS, default='synergy_reward')
    parser.add_argument('--label', default='main')
    parser.add_argument('--envs', type=int, default=4096)
    parser.add_argument('--updates', type=int, default=10000)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root.resolve(), args.references, args.project)
    elif args.action == 'preflight':
        preflight(args.root.resolve())
    else:
        train(args.root.resolve(), args.arm, args.label, args.envs, args.updates)


if __name__ == '__main__':
    main()
