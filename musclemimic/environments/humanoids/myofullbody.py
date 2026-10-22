"""
MyoFullBody environment - A full-body muscle-actuated humanoid model.
"""

import mujoco
import myo_sim
from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from musclemimic.environments import LocoEnv
from musclemimic.utils.logging import setup_logger
from mujoco import MjSpec

logger = setup_logger(__name__, identifier="[MyoFullBody]")


class MyoFullBody(LocoEnv):
    """
    Description
    ------------

    MuJoCo environment of a full-body humanoid model with muscle actuation.
    This model combines the full skeletal structure with comprehensive muscle control
    throughout the entire body, including legs, torso, arms, and optional fingers.

    The model uses muscle actuators (MuJoCo type 4) with Hill-type muscle dynamics,
    providing biomechanically realistic force generation and movement patterns.

    .. note:: Control range for all muscles is modified from default [0,1] to [-1,1]
              for normalized muscle policy actions.

    Default Observation Space
    -----------------

    The observation space consists of joint positions and velocities for the full body,
    plus optional muscle state observations (length, velocity, force, activation).

    Default Action Space
    -------------------

    Control function type: **DefaultControl**

    All muscle actuators use control range [-1.0, 1.0] where:
    - Negative values (-1.0 to 0.0) represent muscle relaxation to baseline activation
    - Positive values (0.0 to 1.0) represent increasing muscle activation levels

    Methods
    ------------

    """

    mjx_enabled = False

    _FINGER_JOINT_STEMS = (
        "cmc_flexion", "cmc_abduction", "mp_flexion", "ip_flexion",
        "mcp2_flexion", "mcp2_abduction", "mcp3_flexion", "mcp3_abduction",
        "mcp4_flexion", "mcp4_abduction", "mcp5_flexion", "mcp5_abduction",
        "md2_flexion", "md3_flexion", "md4_flexion", "md5_flexion",
        "pm2_flexion", "pm3_flexion", "pm4_flexion", "pm5_flexion",
    )
    _FINGER_MUSCLE_STEMS = (
        "FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5",
        "EDC2", "EDC3", "EDC4", "EDC5", "EDM", "EIP", "EPL", "EPB",
        "FPL", "APL", "OP", "RI2", "RI3", "RI4", "RI5", "LU_RB2",
        "LU_RB3", "LU_RB4", "LU_RB5", "UI_UB2", "UI_UB3", "UI_UB4", "UI_UB5",
    )

    def __init__(
        self,
        timestep: float = 0.002,
        n_substeps: int = 5,
        disable_fingers: bool = True,
        enable_joint_pos_observations: bool = True,
        enable_joint_vel_observations: bool = True,
        enable_muscle_length_observations: bool = False,
        enable_muscle_velocity_observations: bool = False,
        enable_muscle_force_observations: bool = False,
        enable_muscle_excitation_observations: bool = False,
        enable_muscle_activation_observations: bool = False,
        enable_touch_sensor_observations: bool = True,
        spec: str | MjSpec = None,
        observation_spec: list[ObservationType] = None,
        actuation_spec: list[str] = None,
        mjx_backend: str = "jax",
        **kwargs,
    ) -> None:
        """
        Constructor for MyoFullBody environment.

        Args:
            timestep (float): Simulation timestep in seconds. Default 0.002 (500Hz physics).
            n_substeps (int): Number of physics substeps per control step. Default 5 (100Hz control).
            disable_fingers (bool): If True (default), finger joints and muscles are disabled
            enable_joint_pos_observations (bool): If True (default), include joint positions in observations
            enable_joint_vel_observations (bool): If True (default), include joint velocities in observations
            enable_muscle_length_observations (bool): If True, include muscle length in observations
            enable_muscle_velocity_observations (bool): If True, include muscle velocity in observations
            enable_muscle_force_observations (bool): If True, include muscle force in observations
            enable_muscle_excitation_observations (bool): If True, include muscle excitation (neural drive from data.ctrl) in observations
            enable_muscle_activation_observations (bool): If True, include muscle activation (actual state from data.act) in observations
            spec (Union[str, MjSpec]): Path to XML file or MjSpec object. If None, uses default.
            observation_spec (List[ObservationType]): Custom observation specification.
            actuation_spec (List[str]): Custom action specification.
            **kwargs: Additional arguments passed to parent class.
        """

        self._disable_fingers = disable_fingers
        self._enable_joint_pos_observations = enable_joint_pos_observations
        self._enable_joint_vel_observations = enable_joint_vel_observations
        self._enable_muscle_length_observations = enable_muscle_length_observations
        self._enable_muscle_velocity_observations = enable_muscle_velocity_observations
        self._enable_muscle_force_observations = enable_muscle_force_observations
        self._enable_muscle_excitation_observations = enable_muscle_excitation_observations
        self._enable_muscle_activation_observations = enable_muscle_activation_observations
        self._enable_touch_sensor_observations = enable_touch_sensor_observations

        # Store mjx_backend for use in _modify_spec_for_mjx
        self.mjx_backend = mjx_backend

        # Store nconmax and num_envs from kwargs for automatic scaling
        self.nconmax = kwargs.get("nconmax", None)
        self.njmax = kwargs.get("njmax", None)  # Also handle njmax
        self.num_envs = kwargs.get("num_envs", 1)

        if spec is None:
            spec = myo_sim.load_spec("myofullbody")

        # Load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # Apply changes to the MjSpec
        spec = self._apply_spec_changes(spec)

        # Get observation and action specifications
        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)

        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # Modify spec for MJX if enabled
        if self.mjx_enabled:
            spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            mjx_backend=mjx_backend,
            **kwargs,
        )

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        """
        Apply changes to the MjSpec including:
        1. Disabling fingers if requested
        2. Adding mimic sites for trajectory tracking
        3. Modifying muscle control ranges from [0, 1] to [-1, 1] for normalized muscle policy actions

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            MjSpec: Modified specification
        """

        # Naming always comes from myo_sim.  disable_fingers only controls
        # removal and never selects an alternate legacy naming convention.
        if self._disable_fingers:
            # Device composition rebuilds full-length keyframes.  Deleting hand
            # DOFs changes nq/nv/nu, so stale key arrays must not survive.  The
            # training reset path restores trajectory state by observation name;
            # an empty key falls back to the model's qpos0 for non-trajectory use.
            for key in spec.keys:
                key.qpos = []
                key.qvel = []
                key.ctrl = []
            finger_joints = {
                f"{stem}_{side}" for stem in self._FINGER_JOINT_STEMS for side in ("r", "l")
            }
            finger_actuators = {
                name
                for stem in self._FINGER_MUSCLE_STEMS
                for name in (stem, f"{stem}_l")
            }
            finger_tendons = {
                name
                for stem in self._FINGER_MUSCLE_STEMS
                for name in (f"{stem}_tendon", f"{stem}_tendon_l")
            }
            for actuator in list(spec.actuators):
                if actuator.name in finger_actuators:
                    spec.delete(actuator)
            for tendon in list(spec.tendons):
                if tendon.name in finger_tendons:
                    spec.delete(tendon)
            for joint in list(spec.joints):
                if joint.name in finger_joints:
                    spec.delete(joint)

        # Add mimic sites for trajectory tracking
        for body_name, site_name in self.body2sites_for_mimic.items():
            b = spec.body(body_name)
            if b is None:
                raise ValueError(f"Required MSK body {body_name!r} is missing from the model")
            if spec.site(site_name) is not None:
                continue
            pos = [0.0, 0.0, 0.0]
            b.add_site(
                name=site_name,
                group=4,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.075, 0.05, 0.025],
                rgba=[1.0, 0.0, 0.0, 0.5],
                pos=pos,
            )

        
        # Modify control range to -1 to 1
        for actuator in spec.actuators:
            # Only modify muscle actuators (type 4 in MuJoCo), not motor actuators
            if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
                actuator.ctrlrange = [-1.0, 1.0]
                actuator.ctrllimited = True
        return spec

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        """
        Get observation specification including joint positions/velocities
        and optional muscle observations.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            List[ObservationType]: List of observations
        """
        obs_spec = []

        available_joints = {joint.name for joint in spec.joints}
        owned_joints = getattr(self, "_msk_joint_names", available_joints)
        j_names = [
            joint.name for joint in spec.joints
            if joint.name in owned_joints and joint.name != self.root_free_joint_xml_name
        ]
        has_free_root = spec.joint(self.root_free_joint_xml_name) is not None

        # Add joint position observations if enabled
        if self._enable_joint_pos_observations:
            # Add free joint observation (position without x,y)
            if has_free_root:
                obs_spec.append(ObservationType.FreeJointPosNoXY("q_free_joint", self.root_free_joint_xml_name))
            # Add all joint positions
            obs_spec.append(ObservationType.JointPosArray("q_all_pos", j_names))

        # Add joint velocity observations if enabled
        if self._enable_joint_vel_observations:
            # Add free joint velocities
            if has_free_root:
                obs_spec.append(ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name))
            # Add all joint velocities
            obs_spec.append(ObservationType.JointVelArray("dq_all_vel", j_names))

        # Add muscle observations if enabled
        owned_actuators = getattr(self, "_msk_actuator_names", None)
        for actuator in spec.actuators:
            if actuator.dyntype != mujoco.mjtDyn.mjDYN_MUSCLE:
                continue
            if owned_actuators is not None and actuator.name not in owned_actuators:
                continue
            actuator_name = actuator.name

            # Add muscle length observations
            if self._enable_muscle_length_observations:
                obs_name = f"muscle_length_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorLength(obs_name, xml_name=actuator_name))

            # Add muscle velocity observations
            if self._enable_muscle_velocity_observations:
                obs_name = f"muscle_velocity_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorVelocity(obs_name, xml_name=actuator_name))

            # Add muscle force observations
            if self._enable_muscle_force_observations:
                obs_name = f"muscle_force_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorForce(obs_name, xml_name=actuator_name))

            # Add muscle excitation observations (neural drive from data.ctrl)
            if self._enable_muscle_excitation_observations:
                obs_name = f"muscle_excitation_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorExcitation(obs_name, xml_name=actuator_name))

            # Add muscle activation observations (actual state from data.act)
            if self._enable_muscle_activation_observations:
                obs_name = f"muscle_activation_{actuator_name.lower()}"
                obs_spec.append(ObservationType.ActuatorActivation(obs_name, xml_name=actuator_name))

        # Add touch sensor observations for foot contact feedback if enabled
        # These sensors are crucial for locomotion balance and gait control
        if self._enable_touch_sensor_observations:
            touch_sensors = ["r_foot", "r_toes", "l_foot", "l_toes"]
            for sensor_name in touch_sensors:
                if spec.sensor(sensor_name) is None:
                    # Standalone Assist targets (for example ``myoarms``) do
                    # not necessarily expose the locomotion contact sensors.
                    # Keep the common full-body defaults strict while making
                    # the target view composable.
                    if getattr(self, "msk_key", "myofullbody") != "myofullbody":
                        continue
                    raise ValueError(f"Required MSK touch sensor {sensor_name!r} is missing")
                obs_name = f"touch_{sensor_name}"
                obs_spec.append(ObservationType.TouchSensor(obs_name, xml_name=sensor_name))

        return obs_spec

    def _get_action_specification(self, spec: MjSpec) -> list[str]:
        """
        Get the MSK-only muscle action specification.

        Args:
            spec (MjSpec): The MuJoCo model specification

        Returns:
            List[str]: List of actuator names
        """
        owned_actuators = getattr(self, "_msk_actuator_names", None)
        return [
            actuator.name
            for actuator in spec.actuators
            if actuator.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
            and (owned_actuators is None or actuator.name in owned_actuators)
        ]

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """
        Modify the model specification for MJX backend compatibility.
        Uses the same contact-simplification idea as the older full-body setup,
        adapted for the maintained MyoFullBody asset.
        """
        if hasattr(self, "mjx_backend") and self.mjx_backend == "warp":
            # MuJoCo 3.3.7+: nconmax is per-env, naconmax is total across all envs
            per_env_contacts = 96  # Contacts per environment
            num_envs = getattr(self, "num_envs", 1)

            if not hasattr(self, "nconmax") or self.nconmax is None:
                self.nconmax = per_env_contacts  # Per-env contacts
            if not hasattr(self, "naconmax") or getattr(self, "naconmax", None) is None:
                self.naconmax = self.nconmax * num_envs  # Total across all envs
            if not hasattr(self, "njmax") or self.njmax is None:
                self.njmax = 768  # Per-world constraints

            # Apply the limits to the spec (spec uses per-env values)
            spec.nconmax = self.nconmax
            spec.njmax = self.njmax

            logger.info(
                "nconmax=%s (per-env), naconmax=%s (total), njmax=%s",
                self.nconmax,
                self.naconmax,
                self.njmax,
            )
            logger.info("Keeping all contacts enabled for Warp backend")
        else:
            for g in spec.geoms:
                # Keep essential ground contacts but disable others
                if g.name and ("floor" in g.name.lower() or "ground" in g.name.lower()):
                    continue  # Keep floor/ground contacts
                g.contype = 0
                g.conaffinity = 0

        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default path to the xml file of the environment.
        """
        raise RuntimeError("MyoFullBody uses the authoritative in-memory myo_sim specification")

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the root free joint in the Mujoco xml file.
        """
        return "root"

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body in the Mujoco xml file.
        """
        return "pelvis"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body in the Mujoco xml file.
        """
        return "torso"

    @info_property
    def root_height_healthy_range(self) -> tuple[float, float]:
        """
        Returns the healthy range of the root height.
        """
        return (0.6, 1.5)

    @info_property
    def body2sites_for_mimic(self) -> dict[str, str]:
        """
        Returns a dictionary mapping body names to their corresponding mimic site names.
        Tailored for the maintained full-body muscle model.

        Focus on essential tracking points: ankle and toe for proper foot orientation.
        Let forward kinematics handle heel (calcn) positioning.
        """
        body2sitemimic = {
            "pelvis": "pelvis_mimic",
            "lumbar1": "upper_body_mimic",
            "head": "head_mimic",
            # Left arm (note: left uses capital L suffix)
            "humerus_l": "left_shoulder_mimic",
            "ulna_l": "left_elbow_mimic",
            "lunate_l": "left_hand_mimic",
            # Right arm (note: right has no suffix)
            "humerus_r": "right_shoulder_mimic",
            "ulna_r": "right_elbow_mimic",
            "lunate_r": "right_hand_mimic",
            "femur_l": "left_hip_mimic",
            "tibia_l": "left_knee_mimic",
            "talus_l": "left_ankle_mimic",
            "toes_l": "left_toes_mimic",
            "femur_r": "right_hip_mimic",
            "tibia_r": "right_knee_mimic",
            "talus_r": "right_ankle_mimic",
            "toes_r": "right_toes_mimic",
        }
        return body2sitemimic

    @info_property
    def sites_for_mimic(self) -> list[str]:
        """
        Returns a list of all mimic sites.
        """
        return list(self.body2sites_for_mimic.values())

    @info_property
    def goal_visualization_arrow_offset(self) -> list[float]:
        """
        Returns the offset for the goal visualization arrow.
        """
        return [0, 0, 0.4]


class MjxMyoFullBody(MyoFullBody):
    """
    MJX version of MyoFullBody with support for JAX and warp backends.
    """

    mjx_enabled = True

    def __init__(self, timestep: float = 0.002, n_substeps: int = 5, mjx_backend: str = "jax", **kwargs):
        """
        Constructor for MJX version of MyoFullBody.

        Args:
            timestep (float): Timestep of the simulation.
            n_substeps (int): Number of substeps.
            mjx_backend (str): MJX backend to use ('jax' or 'warp'). Default: 'jax'.
            **kwargs: Additional arguments.
        """
        # Extract goal-related parameters to prevent them from being passed to viewer
        goal_related_params = [
            "visualize_goal",
            "enable_enhanced_visualization",
            "target_geom_rgba",
            "n_step_lookahead",
            "goal_type",
            "goal_params",
        ]

        extracted_goal_params = {}
        for param in goal_related_params:
            if param in kwargs:
                extracted_goal_params[param] = kwargs.pop(param)

        if "model_option_conf" not in kwargs.keys():
            model_option_conf = dict(iterations=4, ls_iterations=8, disableflags=mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
        else:
            model_option_conf = kwargs["model_option_conf"]
            del kwargs["model_option_conf"]

        # Pass goal-related parameters back through kwargs
        kwargs.update(extracted_goal_params)

        # Store mjx_backend for use in parent class
        self.mjx_backend = mjx_backend

        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            mjx_backend=mjx_backend,
            **kwargs,
        )
