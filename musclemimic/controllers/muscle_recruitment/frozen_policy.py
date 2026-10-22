"""Activation-aware zero-assistance recruitment for frozen Gait120 PPO.

Applies bounded least-squares recruitment to all muscle
actuators and the current model's control/substep duration. No EMG is read.
"""
import mujoco
import numpy as np
from scipy.optimize import lsq_linear
from musclemimic.controllers.muscle_recruitment.kinematics import independent_directions
from musclemimic.controllers.muscle_recruitment.dynamics import activation_step


def advance_activation(a, u, times, dt, steps):
    a = np.array(a, dtype=float, copy=True)
    for _ in range(steps):
        a = activation_step(a, u, times, dt)
    return a


def invert_activation(a, target, times, dt, steps):
    lo, hi = np.zeros_like(a), np.ones_like(a)
    for _ in range(28):
        mid = (lo + hi) / 2
        below = advance_activation(a, mid, times, dt, steps) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


class FrozenRecruitment:
    def __init__(self, env, strength=30., warmup_s=.1):
        self.env, self.model = env, env._model
        m = self.model
        self.ids = np.flatnonzero(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        self.adr = m.actuator_actadr[self.ids]
        self.times = m.actuator_dynprm[self.ids, :2]
        if np.any(m.actuator_dynprm[self.ids, 2] != 0):
            raise ValueError("This recruitment integrator requires unsmoothed muscle dynamics")
        self.steps = int(round(env.dt / m.opt.timestep))
        if not np.isclose(self.steps * m.opt.timestep, env.dt):
            raise ValueError("Non-integral control/substep ratio")
        self.strength, self.warmup_s = float(strength), float(warmup_s)
        self.nominal = env._data.act[self.adr].copy()
        self.nominal_before_last = self.nominal.copy()
        self.raw_nominal = env._data.ctrl[self.ids].copy()
        self.shadow = mujoco.MjData(m)
        self.dense = np.zeros((m.nu, m.nv))
        self.tick = 0
        self.records = []
        self.channels = {k: [] for k in ("act", "ctrl", "actuator_force")}
        actuator_lookup = {int(i): j for j, i in enumerate(self.ids)}
        activation_lookup = {int(i): j for j, i in enumerate(self.adr)}
        for observation in env.obs_container.values():
            kind = observation.data_type()
            if kind not in self.channels:
                continue
            lookup = activation_lookup if kind == "act" else actuator_lookup
            for obs_id, data_id in zip(observation.obs_ind, observation.data_type_ind):
                if int(data_id) in lookup:
                    self.channels[kind].append((int(obs_id), lookup[int(data_id)]))

    def actor_observation(self, observation):
        obs = np.array(observation, copy=True)
        if self.strength == 0 or self.tick * self.env.dt <= self.warmup_s:
            return obs
        for k, j in self.channels["act"]:
            obs[k] = self.nominal[j]
        for k, j in self.channels["ctrl"]:
            obs[k] = self.raw_nominal[j]
        m, d = self.model, self.env._data
        for k, j in self.channels["actuator_force"]:
            i = self.ids[j]
            gain = mujoco.mju_muscleGain(d.actuator_length[i], d.actuator_velocity[i],
                                       m.actuator_lengthrange[i], m.actuator_acc0[i], m.actuator_gainprm[i, :9])
            bias = mujoco.mju_muscleBias(d.actuator_length[i], m.actuator_lengthrange[i],
                                       m.actuator_acc0[i], m.actuator_biasprm[i, :9])
            obs[k] = gain * self.nominal_before_last[j] + bias
        return obs

    def transform(self, ctrl):
        m, d = self.model, self.env._data
        ctrl = np.asarray(ctrl).copy()
        self.raw_nominal = ctrl[self.ids].copy()
        self.nominal_before_last = advance_activation(self.nominal, self.raw_nominal, self.times,
                                                     m.opt.timestep, self.steps - 1)
        future = advance_activation(self.nominal, self.raw_nominal, self.times, m.opt.timestep, self.steps)
        if self.tick * self.env.dt >= self.warmup_s and self.strength > 0:
            mujoco.mj_copyState(m, d, self.shadow, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m, self.shadow)
            shadow = self.shadow
            mujoco.mju_sparse2dense(self.dense, shadow.actuator_moment, shadow.moment_rownnz,
                                   shadow.moment_rowadr, shadow.moment_colind)
            gain = np.array([mujoco.mju_muscleGain(shadow.actuator_length[i], shadow.actuator_velocity[i],
                            m.actuator_lengthrange[i], m.actuator_acc0[i], m.actuator_gainprm[i, :9]) for i in self.ids])
            directions = independent_directions(m, shadow)
            B = (self.dense[self.ids] @ directions).T * gain
            B = B[np.max(np.abs(B), axis=1) > 1e-8]
            target = B @ future
            current = d.act[self.adr].copy()
            lower = advance_activation(current, np.zeros_like(current), self.times, m.opt.timestep, self.steps)
            upper = advance_activation(current, np.ones_like(current), self.times, m.opt.timestep, self.steps)
            n = len(self.ids)
            A = np.vstack([B, np.eye(n) * np.sqrt((1 + self.strength) / n)])
            b = np.r_[target, future / np.sqrt(n * (1 + self.strength))]
            solution = lsq_linear(A, b, bounds=(lower + 1e-10, upper - 1e-10),
                                  method="bvls", tol=1e-7, max_iter=5 * n)
            solver = "bvls"
            if not solution.success:
                solution = lsq_linear(A, b, bounds=(lower + 1e-10, upper - 1e-10),
                                      method="trf", lsq_solver="exact", tol=1e-7, max_iter=500)
                solver = "trf_fallback"
            if not solution.success or not np.isfinite(solution.x).all():
                raise RuntimeError("Recruitment failed: " + solution.message)
            ctrl[self.ids] = invert_activation(current, solution.x, self.times, m.opt.timestep, self.steps)
            self.records.append(dict(time_s=self.tick * self.env.dt,
                                     torque_residual_norm_nm=float(np.linalg.norm(B @ solution.x - target)),
                                     nominal_rms=float(np.sqrt(np.mean(future ** 2))),
                                     target_rms=float(np.sqrt(np.mean(solution.x ** 2))),
                                     solver=solver, iterations=int(solution.nit),
                                     solver_optimality=float(solution.optimality)))
        self.nominal = future
        self.tick += 1
        return ctrl
