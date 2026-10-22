"""Acceptance coverage for first-stage MSK-only assist training."""

import numpy as np
import pytest

from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
from musclemimic.models import build_msk_device_terrain_spec


@pytest.mark.integration
@pytest.mark.parametrize(
    ("device_key", "expected_nu", "expected_robot_actions"),
    [(None, 416, 0), ("HipExoskeleton_L1", 418, 2)],
)
def test_builder_splits_msk_and_robot_ownership(device_key, expected_nu, expected_robot_actions):
    result = build_msk_device_terrain_spec("myofullbody", device_key, {"terrain": "flat"})
    model = result.spec.compile()

    assert model.nu == expected_nu
    assert len(result.components["msk"].actuators) == 416
    assert len(result.components["robot"].actuators) == expected_robot_actions
    assert set(result.components["msk"].actuators).isdisjoint(result.components["robot"].actuators)


@pytest.mark.integration
@pytest.mark.parametrize("disable_fingers", [False, True])
def test_hand_names_follow_myo_sim_independent_of_disable_fingers(disable_fingers):
    env = MyoFullBodyAssistEnv(
        device_key=None,
        terrain={"terrain": "flat"},
        disable_fingers=disable_fingers,
        enable_touch_sensor_observations=True,
        horizon=10,
    )

    actuator_names = [
        env._model.actuator(index).name for index in range(env._model.nu)
    ]
    assert not any("_left" in name for name in actuator_names)
    assert ("FDS5_l" in actuator_names) is (not disable_fingers)
    assert np.isfinite(env.reset()).all()


@pytest.mark.integration
def test_robot_observation_is_optional_and_robot_action_is_not_policy_action():
    common = dict(
        device_key="HipExoskeleton_L1",
        terrain={"terrain": "flat"},
        disable_fingers=False,
        device_control_mode="constant",
        device_control_value=[0.1, -0.1],
        horizon=10,
    )
    msk_only = MyoFullBodyAssistEnv(include_robot_observations=False, **common)
    with_robot = MyoFullBodyAssistEnv(include_robot_observations=True, **common)

    assert msk_only.info.action_space.shape == (416,)
    assert with_robot.info.action_space.shape == (416,)
    assert len(with_robot.robot_action_spec) == 2
    assert with_robot.info.observation_space.shape[0] - msk_only.info.observation_space.shape[0] == 4

    with_robot.reset()
    assert with_robot.get_robot_observation().shape == (4,)
    observation, *_ = with_robot.step(np.zeros(416))
    assert np.isfinite(observation).all()
    np.testing.assert_allclose(with_robot._data.ctrl[with_robot._robot_action_indices], [0.1, -0.1])
