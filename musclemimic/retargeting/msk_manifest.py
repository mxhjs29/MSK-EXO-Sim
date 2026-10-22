"""Authoritative manifests for MSK models supplied by :mod:`myo_sim`."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import mujoco
import myo_sim


_QPOS_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}
_QVEL_WIDTH = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}


@dataclass(frozen=True)
class MskManifest:
    msk_key: str
    fingerprint: str
    joint_names: tuple[str, ...]
    joint_types: tuple[int, ...]
    qpos_indices: Mapping[str, tuple[int, ...]]
    qvel_indices: Mapping[str, tuple[int, ...]]
    body_names: tuple[str, ...]
    site_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    root_joint_name: str | None


def manifest_from_model(msk_key: str, model: mujoco.MjModel) -> MskManifest:
    joint_names = tuple(model.joint(i).name for i in range(model.njnt))
    joint_types = tuple(int(value) for value in model.jnt_type)
    qpos = {}
    qvel = {}
    for jid, (name, joint_type) in enumerate(zip(joint_names, model.jnt_type)):
        q0 = int(model.jnt_qposadr[jid])
        v0 = int(model.jnt_dofadr[jid])
        qpos[name] = tuple(range(q0, q0 + _QPOS_WIDTH[joint_type]))
        qvel[name] = tuple(range(v0, v0 + _QVEL_WIDTH[joint_type]))
    bodies = tuple(model.body(i).name for i in range(1, model.nbody))
    sites = tuple(model.site(i).name for i in range(model.nsite))
    actuators = tuple(model.actuator(i).name for i in range(model.nu))
    payload = json.dumps(
        {
            "joints": list(zip(joint_names, joint_types, qpos.values(), qvel.values())),
            "bodies": bodies,
            "sites": [(model.site(i).name, model.body(int(model.site_bodyid[i])).name) for i in range(model.nsite)],
        },
        sort_keys=True,
    ).encode()
    root = next((name for name, kind in zip(joint_names, model.jnt_type) if kind == mujoco.mjtJoint.mjJNT_FREE), None)
    return MskManifest(
        msk_key=msk_key,
        fingerprint=hashlib.sha256(payload).hexdigest()[:16],
        joint_names=joint_names,
        joint_types=joint_types,
        qpos_indices=MappingProxyType(qpos),
        qvel_indices=MappingProxyType(qvel),
        body_names=bodies,
        site_names=sites,
        actuator_names=actuators,
        root_joint_name=root,
    )


def build_msk_manifest(msk_key: str) -> MskManifest:
    """Compile the uncomposed authoritative model and describe its layout."""
    return manifest_from_model(msk_key, myo_sim.load_spec(msk_key).compile())
