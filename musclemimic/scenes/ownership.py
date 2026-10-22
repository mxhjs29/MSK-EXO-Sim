"""Resolve named components into compiled MuJoCo addresses."""
from dataclasses import dataclass
import mujoco


@dataclass(frozen=True)
class ComponentOwnership:
    entity_id: str
    body_ids: tuple[int, ...]
    joint_ids: tuple[int, ...]
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    actuator_indices: tuple[int, ...]
    activation_indices: tuple[int, ...]
    sensor_indices: tuple[int, ...]


def resolve_ownership(model, entity_id, names):
    joints = tuple(model.joint(n).id for n in names.joints)
    actuators = tuple(model.actuator(n).id for n in names.actuators)
    qpos, qvel = [], []
    for j in joints:
        kind = model.jnt_type[j]
        nq, nv = (7, 6) if kind == mujoco.mjtJoint.mjJNT_FREE else ((4, 3) if kind == mujoco.mjtJoint.mjJNT_BALL else (1, 1))
        qpos.extend(range(int(model.jnt_qposadr[j]), int(model.jnt_qposadr[j]) + nq))
        qvel.extend(range(int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + nv))
    activation = tuple(a for i in actuators for a in range(int(model.actuator_actadr[i]), int(model.actuator_actadr[i] + model.actuator_actnum[i])) if a >= 0)
    sensors = tuple(a for n in names.sensors for a in range(int(model.sensor(n).adr[0]), int(model.sensor(n).adr[0] + model.sensor(n).dim[0])))
    return ComponentOwnership(entity_id, tuple(model.body(n).id for n in names.bodies), joints,
                              tuple(qpos), tuple(qvel), actuators, activation, sensors)
