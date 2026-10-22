"""Physical torque interface for the two ideal Hippo joint actuators."""

import mujoco
import numpy as np


class HippoTorqueAdapter:
    names = ("Exo_R", "Exo_L")
    joints = ("hip_flexion_r", "hip_flexion_l")

    def __init__(self, model, limit=5.0, slew=100.0):
        if not np.isfinite([limit, slew]).all() or limit <= 0 or slew <= 0:
            raise ValueError("Torque limit and slew must be finite and positive")
        self.limit, self.slew = float(limit), float(slew)
        self.indices = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.names], dtype=np.int32
        )
        if np.any(self.indices < 0):
            raise ValueError("external_torque requires Hippo Exo_R and Exo_L")
        self.q_indices, self.v_indices, gains = [], [], []
        for aid, name in zip(self.indices, self.joints, strict=True):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if (
                jid < 0
                or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE
                or model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[aid, 0] != jid
                or model.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED
                or model.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE
                or model.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE
            ):
                raise ValueError("Expected ideal fixed-gain Hippo hinge actuators")
            gain = float(model.actuator_gainprm[aid, 0] * model.actuator_gear[aid, 0])
            if not np.isfinite(gain) or gain == 0:
                raise ValueError("Invalid Hippo transmission gain")
            lo, hi = sorted(model.actuator_ctrlrange[aid] * gain)
            if model.actuator_ctrllimited[aid] and not (lo <= -limit and hi >= limit):
                raise ValueError("Experiment torque exceeds actuator control range")
            if model.actuator_forcelimited[aid]:
                raise ValueError("Force-limited actuators need a separate calibrated adapter")
            self.q_indices.append(int(model.jnt_qposadr[jid]))
            self.v_indices.append(int(model.jnt_dofadr[jid]))
            gains.append(gain)
        self.gains = np.asarray(gains)

    def command(self, requested, previous, dt, xp=np):
        """Works on (..., 2) arrays, including JIT/vmap tracers."""
        target = xp.clip(xp.asarray(requested), -self.limit, self.limit)
        previous = xp.asarray(previous)
        applied = previous + xp.clip(target - previous, -self.slew * dt, self.slew * dt)
        return applied, applied / xp.asarray(self.gains)


def actuator_contribution(model, data, indices):
    """Sum only the selected actuator forces, supporting MuJoCo sparse moments."""
    moment = np.asarray(data.actuator_moment)
    if moment.ndim == 2:
        return np.asarray(data.actuator_force)[indices] @ moment[indices]
    dense = np.zeros((model.nu, model.nv))
    for i in indices:
        start = int(data.moment_rowadr[i])
        end = start + int(data.moment_rownnz[i])
        dense[i, np.asarray(data.moment_colind[start:end], dtype=int)] = moment[start:end]
    return np.asarray(data.actuator_force)[indices] @ dense[indices]
