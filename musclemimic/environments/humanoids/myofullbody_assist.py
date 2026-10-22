"""Single-agent MSK training in an MSK + optional device + terrain world."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import mujoco
import numpy as np
from loco_mujoco.core import ObservationType

from musclemimic.models import ComponentNames, build_msk_device_terrain_spec
from musclemimic.retargeting.msk_manifest import manifest_from_model

from .myofullbody import MyoFullBody, MjxMyoFullBody


class MyoFullBodyAssistEnv(MyoFullBody):
    """Train only the MSK policy while an optional robot shares the physics."""

    def __init__(
        self,
        msk_key: str = "myofullbody",
        device_key: str | None = None,
        terrain: dict | None = None,
        include_robot_observations: bool = False,
        device_control_mode: str = "zero",
        device_control_value: float | Sequence[float] = 0.0,
        reference_duration: float | None = None,
        device_torque_limit: float = 5.0,
        device_torque_slew: float = 100.0,
        knee_coupling_consistent: bool = False,
        **kwargs,
    ) -> None:
        if "spec" in kwargs:
            raise ValueError("MyoFullBodyAssistEnv builds its spec; do not pass spec")
        if device_control_mode not in {"zero", "constant", "external_torque"}:
            raise ValueError("device_control_mode must be zero, constant, or external_torque")

        composition = build_msk_device_terrain_spec(msk_key, device_key, terrain)
        self.composition = composition
        self.msk_key = msk_key
        self.robot_manifest = composition.components.get("robot", ComponentNames())
        self.device_key = device_key or None
        self.terrain_config = {"terrain": "flat"} if terrain is None else dict(terrain)
        self.include_robot_observations = include_robot_observations
        self.device_control_mode = device_control_mode
        self.device_control_value = device_control_value
        self.reference_duration = reference_duration
        self.knee_coupling_consistent = knee_coupling_consistent
        self.knee_range_corrections = []

        msk = composition.components["msk"]
        robot = composition.components.get("robot", ComponentNames())
        self._msk_joint_names = frozenset(msk.joints)
        self._msk_actuator_names = frozenset(msk.actuators)
        self._msk_body_names = frozenset(msk.bodies)
        self._robot_joint_names = tuple(robot.joints)
        self._robot_actuator_names = tuple(robot.actuators)
        self.msk_observation_spec = ()
        self.robot_observation_spec = ()
        self.msk_action_spec = ()
        self.robot_action_spec = ()

        super().__init__(spec=composition.spec, **kwargs)
        self.msk_manifest = manifest_from_model(msk_key, self._model)

        self.msk_action_spec = tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(index))
            for index in self._action_indices
        )
        self.robot_action_spec = tuple(
            name for name in self._robot_actuator_names
            if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0
        )
        self._robot_action_indices = np.asarray(
            [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in self.robot_action_spec
            ],
            dtype=np.int32,
        )
        robot_joint_ids = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self._robot_joint_names
        ]
        robot_joint_ids = [joint_id for joint_id in robot_joint_ids if joint_id >= 0]
        qpos_width = {
            mujoco.mjtJoint.mjJNT_FREE: 7,
            mujoco.mjtJoint.mjJNT_BALL: 4,
            mujoco.mjtJoint.mjJNT_SLIDE: 1,
            mujoco.mjtJoint.mjJNT_HINGE: 1,
        }
        qvel_width = {
            mujoco.mjtJoint.mjJNT_FREE: 6,
            mujoco.mjtJoint.mjJNT_BALL: 3,
            mujoco.mjtJoint.mjJNT_SLIDE: 1,
            mujoco.mjtJoint.mjJNT_HINGE: 1,
        }
        self._robot_qpos_indices = np.asarray(
            [
                index
                for joint_id in robot_joint_ids
                for index in range(
                    int(self._model.jnt_qposadr[joint_id]),
                    int(self._model.jnt_qposadr[joint_id]) + qpos_width[self._model.jnt_type[joint_id]],
                )
            ],
            dtype=np.int32,
        )
        self._robot_qvel_indices = np.asarray(
            [
                index
                for joint_id in robot_joint_ids
                for index in range(
                    int(self._model.jnt_dofadr[joint_id]),
                    int(self._model.jnt_dofadr[joint_id]) + qvel_width[self._model.jnt_type[joint_id]],
                )
            ],
            dtype=np.int32,
        )
        self._robot_ctrl = self._make_robot_ctrl()
        self.torque_adapter = None
        if device_control_mode == "external_torque":
            from musclemimic.controllers.devices.torque_adapter import HippoTorqueAdapter
            self.torque_adapter = HippoTorqueAdapter(self._model, device_torque_limit, device_torque_slew)

    def _apply_spec_changes(self, spec):
        spec = super()._apply_spec_changes(spec)
        if self.knee_coupling_consistent:
            from musclemimic.models.joint_coupling import correct_knee_coupling_ranges
            self.knee_range_corrections = correct_knee_coupling_ranges(spec)
        return spec

    def load_trajectory(self, traj=None, traj_path=None, warn=True, cache_type="full", site_names=None):
        if self.reference_duration is not None:
            from loco_mujoco.trajectory import Trajectory
            from musclemimic.data.trajectory import retime_reference

            if (traj is None) == (traj_path is None):
                raise ValueError("Provide exactly one of traj and traj_path")
            source = Trajectory.load(traj_path) if traj is None else traj
            traj = retime_reference(source, self.reference_duration, self.dt)
            traj_path = None
        return super().load_trajectory(traj=traj, traj_path=traj_path, warn=warn,
                                       cache_type=cache_type, site_names=site_names)

    @property
    def body2sites_for_mimic(self):
        """Return only tracking bodies present in the selected MSK target.

        ``MyoFullBody`` historically assumes the full 17-point skeleton.
        Assist target views are deliberately smaller: ``myolegs`` has no arm
        bodies and ``myoarms`` has no leg bodies.  Filtering the canonical map
        here keeps one environment entry point while preserving the stable
        site names used by the retargeting profiles.
        """
        canonical = super().body2sites_for_mimic
        if self.msk_key == "myofullbody":
            return canonical
        available = getattr(self, "_msk_body_names", frozenset())
        return {body: site for body, site in canonical.items() if body in available}

    def _get_observation_specification(self, spec):
        msk_spec = super()._get_observation_specification(spec)
        available = {joint.name for joint in spec.joints}
        robot_joints = [name for name in self._robot_joint_names if name in available]
        robot_spec = []
        if robot_joints:
            robot_spec = [
                ObservationType.JointPosArray("robot_qpos", robot_joints),
                ObservationType.JointVelArray("robot_qvel", robot_joints),
            ]
        self.msk_observation_spec = tuple(msk_spec)
        self.robot_observation_spec = tuple(robot_spec)
        return msk_spec + robot_spec if self.include_robot_observations else msk_spec

    def _make_robot_ctrl(self) -> np.ndarray:
        count = len(self._robot_action_indices)
        if self.device_control_mode in {"zero", "external_torque"}:
            return np.zeros(count, dtype=np.float64)
        values = np.asarray(self.device_control_value, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(count, float(values), dtype=np.float64)
        if values.shape != (count,):
            raise ValueError(f"device_control_value must contain {count} values, got {values.shape}")
        if count:
            ranges = self._model.actuator_ctrlrange[self._robot_action_indices]
            limited = self._model.actuator_ctrllimited[self._robot_action_indices].astype(bool)
            if np.any(limited & ((values < ranges[:, 0]) | (values > ranges[:, 1]))):
                raise ValueError("device_control_value is outside a robot actuator ctrlrange")
        return values

    def step_with_torque(self, muscle_action, requested_torque):
        """CPU: preserve the policy action space; hold device ctrl through substeps."""
        if self.torque_adapter is None:
            raise ValueError("Use device_control_mode=external_torque")
        request = np.asarray(requested_torque)
        if request.shape != (2,) or not np.isfinite(request).all():
            raise ValueError("requested_torque must be finite [right, left] N m")
        adapter = self.torque_adapter
        previous = self._data.ctrl[adapter.indices] * adapter.gains
        _, ctrl = adapter.command(request, previous, self.dt)
        self._data.ctrl[adapter.indices] = ctrl
        return self.step(muscle_action)

    def mjx_step_with_torque(self, state, muscle_action, requested_torque):
        """JIT/vmap API: time-varying commands travel in data, not Python attributes."""
        if self.torque_adapter is None:
            raise ValueError("Use device_control_mode=external_torque")
        adapter = self.torque_adapter
        indices = jnp.asarray(adapter.indices)
        previous = state.data.ctrl[indices] * jnp.asarray(adapter.gains)
        _, ctrl = adapter.command(requested_torque, previous, self.dt, jnp)
        state = state.replace(data=state.data.replace(ctrl=state.data.ctrl.at[indices].set(ctrl)))
        return self.mjx_step(state, muscle_action)

    def get_robot_observation(self, data=None):
        """Return robot joint qpos/qvel independently of the policy input."""
        data = self._data if data is None else data
        qpos = data.qpos[self._robot_qpos_indices]
        qvel = data.qvel[self._robot_qvel_indices]
        if isinstance(qpos, np.ndarray):
            return np.concatenate((qpos, qvel))
        return jnp.concatenate((qpos, qvel))

    def _simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._simulation_pre_step(model, data, carry)
        if len(self._robot_action_indices) and self.device_control_mode != "external_torque":
            data.ctrl[self._robot_action_indices] = self._robot_ctrl
        return model, data, carry

    def _reset_carry(self, model, data, carry):
        data, carry = super()._reset_carry(model, data, carry)
        if len(self._robot_qpos_indices):
            data.qpos[self._robot_qpos_indices] = model.qpos0[self._robot_qpos_indices]
            data.qvel[self._robot_qvel_indices] = 0.0
        if len(self._robot_action_indices):
            data.ctrl[self._robot_action_indices] = self._robot_ctrl
        return data, carry

    def _mjx_simulation_pre_step(self, model, data, carry):
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        if len(self._robot_action_indices) and self.device_control_mode != "external_torque":
            data = data.replace(
                ctrl=data.ctrl.at[jnp.asarray(self._robot_action_indices)].set(jnp.asarray(self._robot_ctrl))
            )
        return model, data, carry

    def _mjx_reset_carry(self, model, data, carry):
        data, carry = super()._mjx_reset_carry(model, data, carry)
        if len(self._robot_qpos_indices):
            data = data.replace(
                qpos=data.qpos.at[jnp.asarray(self._robot_qpos_indices)].set(
                    model.qpos0[jnp.asarray(self._robot_qpos_indices)]
                ),
                qvel=data.qvel.at[jnp.asarray(self._robot_qvel_indices)].set(0.0),
            )
        if len(self._robot_action_indices):
            data = data.replace(
                ctrl=data.ctrl.at[jnp.asarray(self._robot_action_indices)].set(jnp.asarray(self._robot_ctrl))
            )
        return data, carry


class MjxMyoFullBodyAssistEnv(MyoFullBodyAssistEnv, MjxMyoFullBody):
    """MJX-enabled first-stage assist environment."""

    mjx_enabled = True
