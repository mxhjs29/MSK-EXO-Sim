import mujoco
import numpy as np
import pytest

from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel
from musclemimic.retargeting import (
    adapt_msk_trajectory_to_composed_model,
    build_msk_manifest,
    load_msk_retarget_profile,
)


def test_manifest_uses_mujoco_layout_and_is_stable():
    first = build_msk_manifest("myofullbody")
    second = build_msk_manifest("myofullbody")
    assert first.fingerprint == second.fingerprint
    assert first.root_joint_name == "root"
    assert len(first.qpos_indices["root"]) == 7
    assert len(first.qvel_indices["root"]) == 6
    assert first.joint_names and first.body_names and first.site_names


def test_profiles_register_component_stages():
    assert load_msk_retarget_profile("myoarms").target_msk_key == "myoarms"
    assert load_msk_retarget_profile("myolegs").target_msk_key == "myolegs"


def test_adapter_initializes_robot_from_qpos0():
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body><joint name='human'/><geom size='.1'/></body>
        <body><joint name='robot'/><geom size='.1'/></body></worldbody>
        <keyframe><key qpos='0.25 0.75'/></keyframe></mujoco>"""
    )
    source = Trajectory(
        TrajectoryInfo(["human"], TrajectoryModel(1, np.array([mujoco.mjtJoint.mjJNT_HINGE])), 30.0),
        TrajectoryData(qpos=np.array([[1.0], [2.0]]), qvel=np.array([[3.0], [4.0]]), split_points=np.array([0, 2])),
    )
    component = type("Names", (), {"joints": ("human",)})()
    result = adapt_msk_trajectory_to_composed_model(source, model=model, msk_component=component)
    np.testing.assert_allclose(result.data.qpos[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(result.data.qpos[:, 1], model.qpos0[1])
    np.testing.assert_allclose(result.data.qvel[:, 1], 0.0)


def test_fullbody_solver_keeps_runtime_fingers(monkeypatch):
    from types import SimpleNamespace

    from musclemimic.retargeting import msk_retarget

    manifest = build_msk_manifest("myofullbody")
    captured = {}

    def load(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(info=SimpleNamespace(metadata={}))

    monkeypatch.setattr(msk_retarget, "load_retargeted_amass_trajectory", load)
    msk_retarget.retarget_dataset_to_assist_msk(
        "KIT/11/WalkingStraightForwards01_poses", msk_key="myofullbody", target_manifest=manifest
    )
    assert captured["robot_conf"].env_params.disable_fingers is False
    assert captured["cache_namespace"].endswith("/fingers-enabled-v1")
    assert captured["cache_metadata"]["myo_sim_model_fingerprint"] == manifest.fingerprint


@pytest.mark.parametrize("msk_key", ["myoarms", "myolegs"])
def test_component_retarget_preserves_joint_motion_and_tracking_sites(monkeypatch, msk_key):
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    from musclemimic.retargeting import msk_retarget

    source_env = MyoFullBodyAssistEnv(msk_key="myofullbody", headless=True, disable_fingers=False)
    model = source_env._model
    joint_name = "elbow_flexion_r" if msk_key == "myoarms" else "hip_flexion_r"
    qpos = np.repeat(model.qpos0[None, :], 3, axis=0)
    qpos[:, model.joint(joint_name).qposadr[0]] += [0.0, 0.001, 0.002]
    source = msk_retarget._make_target_trajectory(
        model,
        qpos,
        np.zeros((3, model.nv)),
        100.0,
        np.array([0, 3]),
        {"msk_key": "myofullbody"},
    )
    monkeypatch.setattr(msk_retarget, "load_retargeted_amass_trajectory", lambda *args, **kwargs: source)
    target = msk_retarget.retarget_dataset_to_assist_msk("synthetic/reaching", msk_key=msk_key)

    np.testing.assert_array_equal(
        target.data.qpos[:, target.info.joint_name2ind_qpos[joint_name]],
        source.data.qpos[:, source.info.joint_name2ind_qpos[joint_name]],
    )
    np.testing.assert_allclose(target.data.qvel[:, target.info.joint_name2ind_qvel[joint_name]], 0.1)
    assert target.info.metadata["msk_key"] == msk_key
    tracking_site = "right_hand_mimic" if msk_key == "myoarms" else "right_ankle_mimic"
    assert tracking_site in target.info.site_names
    assert np.isfinite(target.data.site_xpos).all()
