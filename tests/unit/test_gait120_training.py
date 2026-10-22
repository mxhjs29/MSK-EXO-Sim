"""Regression checks for the Gait120 experiment's data and control contracts."""
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from loco_mujoco.trajectory.dataclasses import TrajectoryData
from musclemimic.experiments.gait120_recruitment import advance_activation, invert_activation
from musclemimic.experiments.gait120_evaluate import correlation_metrics, pearson


def test_missing_vector_cache_preserves_vmap_axes_without_becoming_present():
    data = TrajectoryData(qpos=jnp.zeros((6, 3)), qvel=jnp.ones((6, 3)), split_points=jnp.array([0, 6]))
    selected = data.get(jnp.zeros((2, 3), dtype=int), jnp.array([[0, 1, 2], [3, 4, 5]]))
    assert selected.subtree_com_root.shape == (2, 3, 0)
    result = jax.jit(jax.vmap(jax.vmap(lambda d: d.qvel.sum() if d.subtree_com_root.size == 0 else -1)))(selected)
    np.testing.assert_array_equal(result, np.full((2, 3), 3))
    sliced = TrajectoryData.dynamic_slice_in_dim(data, 0, 1, 3)
    assert sliced.subtree_com_root.shape == (3, 0) and sliced.subtree_com_root.size == 0


def test_activation_prediction_matches_mujoco_dynamics_and_inverse():
    a = np.array([.1, .4, .8])
    u = np.array([.7, .05, .9])
    times = np.array([[.01, .04], [.015, .05], [.02, .06]])
    expected = a.copy()
    for _ in range(5):
        expected += .002 * np.array([mujoco.mju_muscleDynamics(float(ui), float(ai), np.r_[ts, 0.])
                                     for ai, ui, ts in zip(expected, u, times)])
    actual = advance_activation(a, u, times, .002, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    reconstructed = invert_activation(a, actual, times, .002, 5)
    np.testing.assert_allclose(reconstructed, u, atol=1e-8)


def test_constant_emg_is_undefined_and_negative_correlation_is_retained():
    x = np.linspace(0, 1, 100)
    assert pearson(x, np.ones(100)) is None
    assert pearson(x, -x) == pytest.approx(-1)
    a = np.tile(x[:, None], (1, 12))
    emg = {"source_time_s":x, "normalized":a.copy()}
    metrics, _ = correlation_metrics(a, x, emg)
    assert metrics["valid_groups"] == 11 and metrics["mean_r"] == pytest.approx(1)
    emg["normalized"][:, 3] = 1
    metrics, _ = correlation_metrics(a, x, emg)
    assert metrics["mean_r"] is None and metrics["constant_or_invalid_groups"] == ["TA"]


def test_constructor_variant_matches_exported_reference_and_does_not_modify_default():
    from musclemimic.environments.humanoids.myofullbody_assist import MyoFullBodyAssistEnv
    from musclemimic.retargeting.gait120_retarget import make_reference_env
    original = MyoFullBodyAssistEnv(headless=True)
    before = original._model.jnt_range.copy()
    exported, _ = make_reference_env()
    training = MyoFullBodyAssistEnv(headless=True, knee_coupling_consistent=True)
    np.testing.assert_array_equal(training._model.jnt_range, exported._model.jnt_range)
    np.testing.assert_array_equal(original._model.jnt_range, before)
    assert len(training.knee_range_corrections) == 11
    np.testing.assert_array_equal(training._model.actuator_gainprm, original._model.actuator_gainprm)


def test_full_published_emg_tables_and_mvc_identity_when_available():
    from musclemimic.retargeting.gait120_emg import Gait120Mat
    path = Path("outputs/gait120_emg_comparison_v1/data/ProcessedData.mat")
    if not path.exists():
        pytest.skip("Downloaded Gait120 data not installed")
    mat = Gait120Mat(path)
    trial = mat.data["LevelWalking"]["Trial01"]
    envelope, names = mat.table(trial["Step01"]["EMGs_env"])
    normalized, other_names = mat.table(trial["Step01"]["EMGs_norm"])
    assert envelope.shape == normalized.shape == (1960, 12)
    assert names == other_names and names[3] == "TibialisAnterior"
    np.testing.assert_allclose(envelope / trial["MVCs"], normalized, rtol=1e-12)


def test_ranking_waits_for_all_final_checkpoints_and_excludes_bad_motion(tmp_path):
    import json
    from musclemimic.experiments.gait120_evaluate import summarize, ARMS
    results = [("ppo", .95, False), ("recruitment_reward", .4, True), ("frozen_recruitment", .6, True)]
    for arm, score, quality in results:
        folder = tmp_path / arm / "evaluation"
        folder.mkdir(parents=True)
        record = dict(update_number=10000, eligible_for_ranking=quality,
                      primary=dict(mean_r=score), quality=dict(passed=quality))
        (folder / "metrics.json").write_text(json.dumps(record))
        summarize(tmp_path)
        report = json.loads((tmp_path / "comparison.json").read_text())
        if arm != ARMS[-1]:
            assert report["winner_among_quality_eligible_methods"] is None
    assert report["winner_among_quality_eligible_methods"] == "frozen_recruitment"
    assert report["all_methods_quality_eligible"] is False


def test_shared_policy_ranking_rejects_mismatched_checkpoints(tmp_path):
    import json
    from musclemimic.experiments.gait120_evaluate import summarize, ARMS

    (tmp_path / "protocol.json").write_text(json.dumps({"arm3": {"source_arm": "ppo"}}))
    for arm in ARMS:
        folder = tmp_path / arm / "evaluation"
        folder.mkdir(parents=True)
        record = dict(update_number=10000, checkpoint=str(tmp_path / arm / "checkpoint_10000"),
                      eligible_for_ranking=True, primary=dict(mean_r=.5), quality=dict(passed=True))
        (folder / "metrics.json").write_text(json.dumps(record))
    summarize(tmp_path)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["shared_checkpoint_verified"] is False
    assert report["winner_among_quality_eligible_methods"] is None
    path = tmp_path / "frozen_recruitment/evaluation/metrics.json"
    record = json.loads(path.read_text())
    record["checkpoint"] = str(tmp_path / "ppo/checkpoint_10000")
    path.write_text(json.dumps(record))
    summarize(tmp_path)
    report = json.loads((tmp_path / "comparison.json").read_text())
    assert report["shared_checkpoint_verified"] is True
    assert report["status"] == "complete"
