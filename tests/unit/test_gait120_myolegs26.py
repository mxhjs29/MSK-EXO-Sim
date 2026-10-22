import mujoco
import numpy as np

from musclemimic.retargeting.gait120_myolegs26 import (
    BODIES, JOINTS, RetargetIK, bilateral_basis, build_target,
    group_six, symmetric_coefficients, concatenate_isolated, continuous_root_quaternions,
)
from musclemimic.retargeting.gait120_retarget import project_joint_couplings


def test_six_group_mapping_preserves_solid_channel_mean():
    # The 11-channel loader already averages original SM=8 and SL=9.
    source = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.5, 10., 11.])[:, None]
    np.testing.assert_allclose(group_six(source)[:, 0], [1., 3., 8.5, 1., 4.5, 6.5])


def test_bilateral_weights_keep_independent_coefficients():
    w = np.arange(12).reshape(6, 2)/12
    right, left = np.array([.2, .8]), np.array([.9, .1])
    out = bilateral_basis(w)@np.r_[right, left]
    np.testing.assert_allclose(out[:6], w@right)
    np.testing.assert_allclose(out[6:], w@left)


def test_half_cycle_shift_preserves_measured_right_and_swaps_phase():
    phase = np.linspace(0, 1, 101)
    right = (1 + np.sin(2*np.pi*phase))[None, :]
    original = right.copy()
    left = symmetric_coefficients(right, phase)
    np.testing.assert_allclose(left[0], 1-np.sin(2*np.pi*phase), atol=1e-12)
    np.testing.assert_array_equal(right, original)
    discontinuous = np.array([[0., 1., 2., 3., 4.]])
    shifted = symmetric_coefficients(discontinuous, np.linspace(0, 1, 5))
    assert shifted[0, 2] == 2  # endpoint mean closes the periodic proxy seam
    np.testing.assert_array_equal(discontinuous, [[0., 1., 2., 3., 4.]])


def test_private_target_allows_walking_knee_and_ik_recovers_known_pose():
    env, changes = build_target()
    model = env._model
    assert changes and model.nu == 26 and model.nq == 47
    q = model.qpos0.copy()
    for side in ['r', 'l']:
        q[model.joint('knee_angle_'+side).qposadr[0]] = -1.1
        q[model.joint('hip_flexion_'+side).qposadr[0]] = .3
    q = project_joint_couplings(model, q)
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            assert model.jnt_range[j, 0]-1e-8 <= q[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]+1e-8
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    eq = data.efc_pos[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    np.testing.assert_allclose(eq, 0, atol=1e-10)
    ids = [model.body(n).id for n in BODIES]
    x = np.r_[q[:3], np.zeros(3), [q[model.joint(n).qposadr[0]] for n in JOINTS]]
    solver = RetargetIK(model)
    _, recovered = solver.solve(data.xpos[ids].copy(), data.xmat[ids].reshape(-1, 3, 3).copy(), x, None)
    np.testing.assert_allclose(recovered, q, atol=2e-6)


def test_separate_split_exports_do_not_alias_or_modify_source_metadata():
    from musclemimic.retargeting.msk_retarget import _make_target_trajectory
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint name="hinge"/><geom size=".1"/></body></worldbody></mujoco>')
    a = _make_target_trajectory(model, np.array([[0.], [.1]]), None, 100., [0, 2], {'clip': 'a'})
    b = _make_target_trajectory(model, np.array([[.2], [.3]]), None, 100., [0, 2], {'clip': 'b'})
    full = concatenate_isolated([a, b], {'split': 'all'})
    train = concatenate_isolated([a, b], {'split': 'train'})
    assert a.info.metadata == {'clip': 'a'} and b.info.metadata == {'clip': 'b'}
    assert full.info.metadata == {'split': 'all'}
    assert train.info.metadata == {'split': 'train'}
    np.testing.assert_array_equal(full.data.split_points, [0, 2, 4])


def test_quaternion_hemisphere_is_continuous_without_changing_rotations():
    from scipy.spatial.transform import Rotation
    q = np.zeros((3, 9))
    q[:, 3:7] = [[1, 0, 0, 0], [-1, 0, 0, 0], [-.99995, -.009999875, 0, 0]]
    original = q.copy()
    fixed = continuous_root_quaternions(q)
    assert np.all(np.sum(fixed[:-1, 3:7]*fixed[1:, 3:7], axis=1) > 0)
    np.testing.assert_array_equal(q, original)
    np.testing.assert_allclose(Rotation.from_quat(q[:, [4, 5, 6, 3]]).as_matrix(),
                               Rotation.from_quat(fixed[:, [4, 5, 6, 3]]).as_matrix(), atol=1e-15)
