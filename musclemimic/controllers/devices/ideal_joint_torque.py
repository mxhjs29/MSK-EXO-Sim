"""Left/right ideal joint torque, scheduled on exact 60 Hz boundaries."""
import numpy as np
import mujoco
from .channels import HIP_TORQUE_NAMES, TORQUE_NAMES, TASK_CHANNELS


class IdealJointTorque:
    def __init__(self, model, limit, slew, names=TORQUE_NAMES, active=None):
        if not np.isfinite([limit, slew]).all() or min(limit, slew) <= 0:
            raise ValueError('Positive finite torque limit/slew required')
        self.limit, self.slew = float(limit), float(slew)
        self.names = list(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError('Unique torque joint names required')
        joints = [model.joint(name).id for name in self.names]
        self.active = np.ones(len(joints), dtype=bool)
        if active is not None:
            if len(set(active)) != len(active) or any(i < 0 or i >= len(joints) for i in active):
                raise ValueError('Invalid active torque channels')
            self.active[:] = False
            self.active[active] = True
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError('Ideal torque channels must be hinges')
        self.indices = np.array([model.jnt_dofadr[j] for j in joints])
        self.previous = np.zeros(len(joints))
        self.contribution = np.zeros(len(joints))

    def reset(self):
        self.previous[:] = 0
        self.contribution[:] = 0

    def command(self, requested):
        requested = np.asarray(requested, dtype=float)
        if requested.shape != self.previous.shape or not np.isfinite(requested).all():
            raise ValueError('Requested torque must match configured channels in Nm')
        if np.any(requested[~self.active] != 0):
            raise ValueError('Nonzero request on inactive torque channel')
        self.previous += np.clip(np.clip(requested, -self.limit, self.limit) - self.previous,
                                 -self.slew / 60, self.slew / 60)
        return self.previous.copy()

    def apply(self, data):
        # Replace only this adapter's previous contribution, retaining other forces.
        data.qfrc_applied[self.indices] += self.previous - self.contribution
        self.contribution = self.previous.copy()


class IdealHipTorque(IdealJointTorque):
    """Compatibility interface for existing two-channel data/tests."""
    def __init__(self, model, limit, slew):
        super().__init__(model, limit, slew, names=HIP_TORQUE_NAMES)


def excitation(kind, frames, seed, limit, channels=2):
    if frames < 1 or limit <= 0:
        raise ValueError('Positive frames and limit required')
    rng = np.random.default_rng(seed)
    out = np.zeros((frames, channels))
    if kind == 'zero':
        return out
    if kind == 'hold':
        out[:] = rng.uniform(-limit, limit, channels)
    elif kind == 'random':
        knots = np.arange(0, frames + 30, 30)
        targets = rng.uniform(-limit, limit, (len(knots), channels))
        out = np.stack([np.interp(np.arange(frames), knots, targets[:, i]) for i in range(channels)], -1)
    elif kind == 'pulse':
        for start in range(30, frames, 60):
            side = (start // 60) % channels
            out[start:min(start + 12, frames), side] = limit * rng.choice([-1, 1])
    else:
        raise ValueError(f'Unknown excitation: {kind}')
    return out


def task_excitation(kind, frames, seed, limit, task, history_frames=30):
    """Open-loop waveform with an identical zero-torque prehistory for every branch."""
    active = TASK_CHANNELS[task]
    if history_frames < 29 or frames < 1:
        raise ValueError('Need a positive recording length and >=30 history states')
    out = np.zeros((frames, len(TORQUE_NAMES)))
    n = frames - history_frames
    if n <= 0:
        return out  # Keep short recordings even when they end during prehistory.
    if kind in ('full_positive', 'full_negative'):
        from .waveforms import full_horizon_torque
        return full_horizon_torque(task,frames,limit,1 if kind=='full_positive' else -1,history_frames)
    if kind in ('pulse_positive', 'pulse_negative'):
        wave = np.zeros((n, len(active)))
        sign = 1 if kind == 'pulse_positive' else -1
        width = min(12, n)
        wave[:width] = sign * limit * np.sin(np.pi * np.arange(1, width+1)/(width+1))[:, None]**2
    elif kind == 'multisine':
        rng = np.random.default_rng(seed)
        t = np.arange(n) / 60
        freqs = np.array([.5, 1., 2., 3., 5.])
        phase = rng.uniform(0, 2*np.pi, (len(active), len(freqs)))
        wave = np.sin(2*np.pi*t[:,None,None]*freqs + phase[None]).sum(-1)
        wave *= limit / np.maximum(np.max(np.abs(wave), axis=0), 1e-12)
    else:
        wave = excitation(kind, n, seed, limit, channels=len(active))
    out[history_frames:, active] = wave
    return out
