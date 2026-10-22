"""Map an MSK-only trajectory into a composed Assist model by joint name."""

from __future__ import annotations

import mujoco
import numpy as np

from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel

from .msk_manifest import manifest_from_model


def adapt_msk_trajectory_to_composed_model(trajectory: Trajectory, *, model, msk_component) -> Trajectory:
    runtime = manifest_from_model("runtime", model)
    owned = set(msk_component.joints) & set(runtime.joint_names)
    source_names = set(trajectory.info.joint_names)
    foreign = source_names - owned
    missing = owned - source_names
    if foreign:
        raise ValueError(f"MSK trajectory contains non-MSK joints: {sorted(foreign)}")
    if missing:
        raise ValueError(f"MSK trajectory is missing runtime MSK joints: {sorted(missing)}")

    count = np.asarray(trajectory.data.qpos).shape[0]
    qpos = np.broadcast_to(np.asarray(model.qpos0), (count, model.nq)).copy()
    qvel = np.zeros((count, model.nv), dtype=np.asarray(trajectory.data.qvel).dtype)
    for name in trajectory.info.joint_names:
        src_q = trajectory.info.joint_name2ind_qpos[name]
        src_v = trajectory.info.joint_name2ind_qvel[name]
        qpos[:, runtime.qpos_indices[name]] = np.asarray(trajectory.data.qpos)[:, src_q]
        qvel[:, runtime.qvel_indices[name]] = np.asarray(trajectory.data.qvel)[:, src_v]

    names = tuple(model.joint(i).name for i in range(model.njnt))
    info = TrajectoryInfo(
        list(names),
        TrajectoryModel(model.njnt, np.asarray(model.jnt_type)),
        trajectory.info.frequency,
        metadata=dict(trajectory.info.metadata or {}),
    )
    data = TrajectoryData(qpos=qpos, qvel=qvel, split_points=np.asarray(trajectory.data.split_points))
    return Trajectory(info=info, data=data)
