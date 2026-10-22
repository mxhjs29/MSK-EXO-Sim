"""Public AMASS-to-Assist-MSK retargeting entry point."""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import myo_sim
import numpy as np
from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory
from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel

from .msk_manifest import build_msk_manifest
from .msk_profiles import load_fullbody_solver_config, load_msk_retarget_profile


def retarget_dataset_to_assist_msk(
    dataset_paths: str | Sequence[str],
    *,
    msk_key: str,
    method: str = "gmr",
    gmr_config: dict | None = None,
    clear_cache: bool = False,
    target_manifest=None,
):
    manifest = target_manifest or build_msk_manifest(msk_key)
    profile = load_msk_retarget_profile(msk_key, manifest)
    namespace = f"msk/{msk_key}/{manifest.fingerprint}/{profile.version}"
    metadata = {
        "msk_key": msk_key,
        "myo_sim_model_fingerprint": manifest.fingerprint,
        "retarget_profile_version": profile.version,
    }
    paths = list(dataset_paths) if not isinstance(dataset_paths, str) else [dataset_paths]
    if msk_key == "myofullbody":
        from musclemimic.environments.humanoids.myofullbody import MyoFullBody

        solver_config = load_fullbody_solver_config()
        finger_names = {f"{stem}_{side}" for stem in MyoFullBody._FINGER_JOINT_STEMS
                        for side in ("l", "r")}
        keep_fingers = bool(finger_names.intersection(manifest.joint_names))
        solver_config.env_params.disable_fingers = not keep_fingers
        if keep_fingers:
            # Older caches used the default finger-disabled solver even when
            # their namespace described a full model. Do not reuse those files.
            namespace += "/fingers-enabled-v1"
            metadata["solver_disable_fingers"] = False
        trajectory = load_retargeted_amass_trajectory(
            "MyoFullBodyAssistEnv", paths if len(paths) > 1 else paths[0],
            robot_conf=solver_config, retargeting_method=method,
            gmr_config=gmr_config, clear_cache=clear_cache,
            cache_namespace=namespace, cache_metadata=metadata,
        )
    else:
        # Stage A: reuse the already solved Assist fullbody cache.  The target
        # models share canonical joint names, so no fixed column slicing is used.
        from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
        from .msk_manifest import manifest_from_model

        source_env = MyoFullBodyAssistEnv(msk_key="myofullbody", headless=True)
        source = retarget_dataset_to_assist_msk(
            paths, msk_key="myofullbody", method=method, gmr_config=gmr_config,
            clear_cache=clear_cache, target_manifest=manifest_from_model("myofullbody", source_env._model),
        )
        # Build the target through the Assist environment so its canonical
        # mimic sites are present in the trajectory model.  The standalone
        # myo_sim asset contains thousands of muscle sites, whereas the
        # imitation handler expects the compact semantic site list exposed by
        # ``env.sites_for_mimic``.
        target_env = MyoFullBodyAssistEnv(
            msk_key=msk_key,
            device_key=None,
            terrain={"terrain": "flat"},
            headless=True,
        )
        target_model = target_env._model
        source_q = source.info.joint_name2ind_qpos
        source_v = source.info.joint_name2ind_qvel
        qpos = np.broadcast_to(target_model.qpos0, (len(source.data.qpos), target_model.nq)).copy()
        qvel = np.zeros((len(source.data.qvel), target_model.nv), dtype=np.float32)
        for jid in range(target_model.njnt):
            name = target_model.joint(jid).name
            if name not in source_q:
                if msk_key == "myoarms" and _is_optional_finger_joint(name):
                    continue
                raise ValueError(f"{msk_key} target joint {name!r} is absent from fullbody source")
            q0, q1 = int(target_model.jnt_qposadr[jid]), int(target_model.jnt_qposadr[jid]) + _joint_qpos_width(target_model.jnt_type[jid])
            v0, v1 = int(target_model.jnt_dofadr[jid]), int(target_model.jnt_dofadr[jid]) + _joint_qvel_width(target_model.jnt_type[jid])
            qpos[:, q0:q1] = np.asarray(source.data.qpos)[:, source_q[name]]
            qvel[:, v0:v1] = np.asarray(source.data.qvel)[:, source_v[name]]
        trajectory = _make_target_trajectory(target_model, qpos, qvel, source.info.frequency, source.data.split_points, metadata)
    trajectory.info.metadata = {**(trajectory.info.metadata or {}), **metadata}
    return trajectory


def _joint_qpos_width(joint_type):
    return {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4, mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[joint_type]


def _joint_qvel_width(joint_type):
    return {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3, mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[joint_type]


def _is_optional_finger_joint(name: str) -> bool:
    return any(token in name for token in ("cmc_", "mp_", "ip_", "mcp", "pm", "md", "deviation_"))


def _make_target_trajectory(model, qpos, qvel, frequency, split_points, metadata):
    data = mujoco.MjData(model)
    qvel = np.zeros((len(qpos), model.nv), dtype=np.float32)
    dt = 1.0 / frequency
    for start, end in zip(np.asarray(split_points[:-1]), np.asarray(split_points[1:])):
        if end - start <= 1:
            continue
        differences = np.empty((end - start - 1, model.nv), dtype=np.float64)
        for offset in range(end - start - 1):
            mujoco.mj_differentiatePos(model, differences[offset], dt, qpos[start + offset], qpos[start + offset + 1])
        qvel[start] = differences[0]
        qvel[start + 1:end - 1] = 0.5 * (differences[:-1] + differences[1:])
        qvel[end - 1] = differences[-1]
    site_xpos = np.empty((len(qpos), model.nsite, 3), dtype=np.float32)
    site_xmat = np.empty((len(qpos), model.nsite, 9), dtype=np.float32)
    for i, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        site_xpos[i] = data.site_xpos
        site_xmat[i] = data.site_xmat
    model_info = TrajectoryModel(
        model.njnt, np.asarray(model.jnt_type), nbody=model.nbody - 1,
        body_rootid=np.asarray(model.body_rootid[1:]), body_weldid=np.asarray(model.body_weldid[1:]),
        body_mocapid=np.asarray(model.body_mocapid[1:]), body_pos=np.asarray(model.body_pos[1:]),
        body_quat=np.asarray(model.body_quat[1:]), body_ipos=np.asarray(model.body_ipos[1:]),
        body_iquat=np.asarray(model.body_iquat[1:]), nsite=model.nsite,
        site_bodyid=np.asarray(model.site_bodyid), site_pos=np.asarray(model.site_pos), site_quat=np.asarray(model.site_quat),
    )
    info = TrajectoryInfo(
        [model.joint(i).name for i in range(model.njnt)], model_info, frequency,
        # Assist runtime caches are sparse qpos/qvel + tracking sites.  Do not
        # advertise body arrays when they are not materialized in
        # ``TrajectoryData``; the generic handler would otherwise try to
        # extend a one-dimensional placeholder body array.
        body_names=None,
        site_names=[model.site(i).name for i in range(model.nsite)], metadata=metadata,
    )
    traj_data = TrajectoryData(qpos=qpos, qvel=qvel, site_xpos=site_xpos, site_xmat=site_xmat, split_points=np.asarray(split_points))
    return Trajectory(info=info, data=traj_data)
