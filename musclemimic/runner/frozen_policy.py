"""Local CPU rollout adapter. No modifications to training environments or policies."""
import copy
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf
from musclemimic.data.fingerprints import digest
from musclemimic.data.fingerprints import model_fingerprint


class Runtime:
    def __init__(self, task, physics_hz=None):
        from musclemimic.runner.eval_utils import load_checkpoint
        from musclemimic.algorithms import PPOJax
        from loco_mujoco.task_factories import TaskFactory
        for key in ['AMASS_PATH','CONVERTED_AMASS_PATH','SMPL_MODEL_PATH','CONVERTED_C3D_PATH']:
            if key in task:
                os.environ[key] = str(Path(task[key]).resolve())
        saved, agent, _ = load_checkpoint(str(Path(task['checkpoint']).resolve()))
        if saved.experiment.get('n_seeds',1) != 1 or saved.experiment.get('len_obs_history',1) != 1:
            raise ValueError('First collector supports single-seed, single-frame policies')
        params = OmegaConf.to_container(saved.experiment.env_params,resolve=True)
        if params['env_name'] != 'MjxMyoFullBodyAssistEnv' or params.get('device_key') is not None:
            raise ValueError('Expected the unassisted Assist checkpoint; physical devices need a calibrated adapter')
        params.update(env_name='MyoFullBodyAssistEnv', mjx_backend='jax', headless=True,
                      th_params=dict(random_start=False,fixed_start_conf=[0,0],start_from_random_step=False))
        params.pop('num_envs',None)
        self.original_dt = float(params.get('timestep',.002))
        self.control_dt = self.original_dt * int(params.get('n_substeps',5))
        if not np.isclose(self.control_dt,.01):
            raise ValueError('This clock configuration requires a 100 Hz human policy')
        if physics_hz is not None:
            steps = physics_hz * self.control_dt
            if physics_hz % 60 or not np.isclose(steps,round(steps)):
                raise ValueError('Physics frequency must be divisible by 100 and 60')
            params.update(timestep=1/physics_hz,n_substeps=int(round(steps)))
        factory = TaskFactory.get_factory_cls(saved.experiment.task_factory.name)
        factory_params = OmegaConf.to_container(saved.experiment.task_factory.params,resolve=True)
        if factory_params.get('amass_dataset_conf') is not None:
            factory_params['amass_dataset_conf']['clear_cache'] = False
        self.env = factory.make(**params,**factory_params)
        self.model, self.data = self.env._model, self.env._data
        self.config = saved
        conf = PPOJax.init_agent_conf(self.env, saved)
        self.variables = {'params':agent.train_state.params,'run_stats':agent.train_state.run_stats}
        self.policy = jax.jit(lambda obs: conf.network.apply(self.variables,jnp.atleast_2d(obs))[0].mode())
        self.identity = dict(policy_sha256=digest(task['checkpoint']), model_sha256=model_fingerprint(self.model),
                             source_sha256=digest(task['source']), reference_source=task['source'],
                             checkpoint=str(Path(task['checkpoint']).resolve()), original_physics_dt_s=self.original_dt,
                             physics_dt_s=float(self.model.opt.timestep), human_control_dt_s=self.control_dt,
                             muscle_action_names=[self.model.actuator(int(i)).name for i in self.env._action_indices])
        self.max_steps = int(self.env.th.len_trajectory(0))-1

    def reset(self, seed, start_frame=0):
        if start_frame < 0 or start_frame >= self.max_steps:
            raise ValueError("Start frame outside reference")
        self.env.th.fixed_start_conf = [0, int(start_frame)]
        self.obs = np.asarray(self.env.reset(jax.random.PRNGKey(seed)))
        self.data = self.env._data
        if not np.isfinite(self.obs).all():
            raise FloatingPointError('Nonfinite initial observation')
        self.action()  # fail immediately on checkpoint observation/action mismatch
        return self.obs

    def action(self):
        action = np.asarray(self.policy(self.obs)).reshape(-1)
        if action.shape != (len(self.env._action_indices),) or not np.isfinite(action).all():
            raise ValueError('Policy output incompatible with muscle action schema')
        return action

    def snapshot(self):
        state = np.empty(mujoco.mj_stateSize(self.model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(self.model,self.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        return (state, copy.deepcopy(self.env._additional_carry), self.obs.copy(), copy.deepcopy(self.env._info))

    def restore(self, snapshot):
        state, carry, obs, info = snapshot
        mujoco.mj_setState(self.model,self.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        self.env._additional_carry = copy.deepcopy(carry)
        self.env._obs = obs.copy(); self.obs = obs.copy(); self.env._info = copy.deepcopy(info)
        mujoco.mj_forward(self.model,self.data)
        # Restore solver warm-start state after refreshing derived features.
        mujoco.mj_setState(self.model,self.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)

    def step(self, before_substep=None):
        action = self.action()
        if before_substep is None:
            result = self.env.step(action)
        else:
            result = step_with_observer(self.env, action, before_substep)
        self.obs = np.asarray(result[0]); self.data = self.env._data
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.obs).all():
            raise FloatingPointError('Nonfinite rollout')
        return result

    def close(self):
        self.env.stop()


def step_with_observer(env, action, callback):
    """Mirror Mujoco.step hooks exactly, opening only the physical substep loop.

    Regression-test against env.step whenever the upstream step implementation changes.
    callback(model,data) runs at the beginning of each actual integration interval.
    """
    info = env._info.copy()
    carry = env._additional_carry.replace(last_action=action)
    processed, carry = env._preprocess_action(action,env._model,env._data,carry)
    env._model,env._data,carry = env._simulation_pre_step(env._model,env._data,carry)
    for _ in range(env._n_intermediate_steps):
        ctrl,carry = env._compute_action(processed,env._model,env._data,carry)
        env._data.ctrl[env._action_indices] = ctrl
        for _ in range(env._n_substeps):
            callback(env._model,env._data)
            mujoco.mj_step(env._model,env._data)
    env._data,carry = env._simulation_post_step(env._model,env._data,carry)
    obs,carry = env._create_observation(env._model,env._data,carry)
    obs,env._data,info,carry = env._step_finalize(obs,env._model,env._data,info,carry)
    info = env._update_info_dictionary(info,obs,env._data,carry)
    absorbing,carry = env._is_absorbing(obs,info,env._data,carry)
    reward,carry = env._reward(env._obs,action,obs,absorbing,info,env._model,env._data,carry)
    done = env._is_done(obs,absorbing,info,env._data,carry)
    env._additional_carry = carry.replace(cur_step_in_episode=carry.cur_step_in_episode+1)
    env._obs = obs
    return np.asarray(obs),reward,absorbing,done,info
