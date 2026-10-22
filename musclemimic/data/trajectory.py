"""Explicit endpoint-preserving retiming of a single reference clip."""

from dataclasses import replace

import numpy as np


def retime_reference(traj, duration, dt):
    from loco_mujoco.trajectory.dataclasses import interpolate_trajectories

    if not np.isfinite([duration, dt]).all() or duration <= 0 or dt <= 0:
        raise ValueError("Reference duration and dt must be finite and positive")
    steps = round(duration / dt)
    if not np.isclose(steps * dt, duration, atol=1e-9):
        raise ValueError("Reference duration must be an integer number of control steps")
    points = np.asarray(traj.data.split_points)
    if len(points) != 2 or points[0] != 0 or points[-1] < 2:
        raise ValueError("Reference retiming requires one nonempty motion clip")
    frames = int(points[-1])
    target = steps + 1
    original_frequency = float(traj.info.frequency)
    metadata = dict(traj.info.metadata or {})
    metadata["reference_retiming"] = {
        "source_frames": frames,
        "source_frequency": original_frequency,
        "target_frames": target,
        "duration_seconds": float(duration),
    }
    # The existing resampler chooses round(N * new_frequency / old_frequency)
    # endpoint-inclusive samples. Supply its sampling ratio, then restore physical Hz.
    info = replace(traj.info, frequency=frames / target / dt)
    data, info = interpolate_trajectories(traj.data, info, 1 / dt, backend=np)
    assert len(data.qpos) == target
    return replace(traj, data=data, info=replace(info, metadata=metadata))
