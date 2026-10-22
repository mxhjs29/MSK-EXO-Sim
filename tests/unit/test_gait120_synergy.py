import numpy as np
import pytest

from musclemimic.experiments.gait120_synergy import (
    choose_rank, fit_nmf, project_nonnegative, reconstruction_metrics,
)


def test_fixed_basis_recovers_unseen_coefficients_without_changing_basis():
    rng = np.random.default_rng(1)
    basis = rng.random((11, 4))
    original = basis.copy()
    expected = rng.random((4, 50))
    actual = project_nonnegative(basis, basis @ expected)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_array_equal(basis, original)


def test_nmf_recovers_nonnegative_low_rank_signal():
    rng = np.random.default_rng(2)
    # Anchor rows/columns make the synthetic basis identifiable up to scale/permutation.
    w = np.vstack([np.eye(3), rng.random((8, 3))])
    h = np.column_stack([np.eye(3), rng.random((3, 70))])
    signal = w @ h
    basis, coeff, _ = fit_nmf(signal, 3, max_iter=4000)
    assert np.all(basis >= 0) and np.all(coeff >= 0)
    assert np.linalg.norm(signal - basis @ coeff) / np.linalg.norm(signal) < .005
    with pytest.raises(ValueError):
        fit_nmf(-signal, 3)


def test_rank_selection_ignores_test_and_requires_all_measured_muscles():
    good = {"global_uncentered_vaf": .95, "minimum_muscle_uncentered_vaf": .80}
    bad = {"global_uncentered_vaf": .99, "minimum_muscle_uncentered_vaf": .50}
    rows = [{"rank": 2, "train": good, "validation": bad, "test": good},
            {"rank": 3, "train": good, "validation": good, "test": bad}]
    assert choose_rank(rows) == 3
    assert choose_rank(rows[:1]) is None


def test_constant_reconstruction_does_not_get_fabricated_correlation():
    signal = np.array([[1., 2., 3.], [2., 4., 8.]])
    constant = np.broadcast_to(signal.mean(axis=1, keepdims=True), signal.shape)
    metrics = reconstruction_metrics(signal, constant)
    assert metrics["per_muscle_pearson_r"] == [None, None]
    assert metrics["mean_pearson_r"] is None
    np.testing.assert_allclose(metrics["per_muscle_centered_r2"], 0, atol=1e-14)
