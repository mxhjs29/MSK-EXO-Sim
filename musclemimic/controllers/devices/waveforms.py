"""Deterministic torque schedules; every row drives one 60 Hz interval."""
import numpy as np
from .channels import TORQUE_NAMES, TASK_CHANNELS


def sustained_torque(task, frames, amplitude_nm, sign, onset_frame=30, hold_seconds=1., ramp_seconds=.1):
    if sign not in (-1, 1) or amplitude_nm < 0 or not np.isfinite(amplitude_nm):
        raise ValueError('Require a nonnegative finite magnitude and sign +/-1')
    hold = round(hold_seconds*60); ramp = round(ramp_seconds*60)
    if hold < 60 or ramp < 1 or not np.isclose(hold/60,hold_seconds) or not np.isclose(ramp/60,ramp_seconds):
        raise ValueError('Hold must be >=1 s; hold/ramp must align to 60 Hz')
    if onset_frame < 30 or frames <= onset_frame + 2*ramp + hold:
        raise ValueError('Trajectory must contain prehistory, the full stimulus and its ending')
    out = np.zeros((frames, len(TORQUE_NAMES)), dtype=float)
    ramp_up = np.arange(1,ramp+1)/ramp
    envelope = np.concatenate([ramp_up, np.ones(hold), 1-ramp_up])
    out[onset_frame:onset_frame+len(envelope),TASK_CHANNELS[task]] = sign*amplitude_nm*envelope[:,None]
    return out


def full_horizon_torque(task, frames, amplitude_nm, sign, onset_frame=30):
    """Smooth varying load until the last interval, including incomplete cycles.

    After shared prehistory: 0.1 s onset ramp and a 0.8 s repeating envelope
    ranging from 25% to 100% of the signed limit. No sustained peak plateau.
    Positive/negative branches test each loading direction separately.
    """
    if sign not in (-1, 1) or amplitude_nm < 0 or not np.isfinite(amplitude_nm):
        raise ValueError('Require a nonnegative finite magnitude and sign +/-1')
    if onset_frame < 0 or frames <= onset_frame:
        raise ValueError('Need at least one interval after onset')
    out = np.zeros((frames, len(TORQUE_NAMES)), dtype=float)
    t = np.arange(frames-onset_frame)/60
    envelope = (.625 + .375*np.cos(2*np.pi*t/.8))*np.minimum((np.arange(len(t))+1)/6, 1.)
    out[onset_frame:, TASK_CHANNELS[task]] = sign*amplitude_nm*envelope[:, None]
    return out
