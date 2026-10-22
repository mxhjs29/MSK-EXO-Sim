import numpy as np
import jax
import jax.numpy as jnp

from musclemimic.experiments.gait120_synergy_train import synergy_score, activation_filter
from musclemimic.experiments.gait120_synergy_queue import payload_gate, is_payload_python
from musclemimic.experiments.gait120_synergy_evaluate import correlations, cross_correlation


def test_gate_waits_for_whole_pipelines_and_live_supervisor():
    complete = dict(lift=dict(phase='complete'), bend=dict(phase='complete'))
    assert payload_gate(complete, [dict(pid=123)])[0] is False
    assert payload_gate(dict(lift=dict(phase='complete'), bend=dict(phase='running')), [])[0] is False
    assert payload_gate(dict(lift=dict(phase='complete'), bend=dict(phase='error')), [])[0] is False
    assert payload_gate({}, [])[0] is False
    assert payload_gate(complete, [])[0] is True
    # A bounded curriculum can finish normally without meeting walking/contact gates.
    assert payload_gate(dict(lift=dict(phase='complete'), bend=dict(phase='needs_policy_review')), [])[0] is True


def test_gate_ignores_tmux_server_retaining_finished_launch_command():
    assert is_payload_python('python3 -u fullbody/payload_refinement_background.py bend')
    assert is_payload_python('.venv/bin/python -m musclemimic.experiments.payload_refinement train')
    assert not is_payload_python('tmux new-session -d -s payload_refine_lift exec python3 fullbody/payload_refinement_background.py lift')
    assert not is_payload_python('bash -lc python3 fullbody/payload_refinement_background.py lift')
    assert not is_payload_python('python -m musclemimic.experiments.gait120_synergy_queue')


def test_synergy_target_beats_zero_and_has_finite_jax_gradient():
    target = np.linspace(.1, .8, 12, dtype=np.float32)
    scales, weights = np.full(12, .1), np.r_[np.ones(6), np.full(6, .5)]
    expected = synergy_score(target, target, scales, weights)
    assert expected == 1.
    assert synergy_score(np.zeros(12), target, scales, weights) < expected
    score = jax.jit(lambda act: synergy_score(act, jnp.asarray(target), scales, weights, jnp))
    np.testing.assert_allclose(score(jnp.zeros(12)), synergy_score(np.zeros(12), target, scales, weights), rtol=1e-6)
    gradient = jax.grad(score)(jnp.zeros(12))
    assert np.isfinite(gradient).all() and np.any(gradient > 0)


def test_activation_proxy_has_memory_and_remains_bounded():
    u = np.zeros((100, 2))
    u[20:40] = .8
    parameters = np.tile([.01, .04, 0.], (2, 1))
    a = activation_filter(u, parameters)
    assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= .8
    assert a[20, 0] < a[25, 0] and a[40, 0] > a[45, 0]
    assert a[41, 0] > 0  # Activation remains after excitation stops.
    np.testing.assert_allclose(activation_filter(np.full((20, 2), .3), parameters), .3)


def test_constant_group_does_not_discard_other_five_correlations():
    x = np.tile(np.linspace(0, 1, 60)[:, None], (1, 6))
    a = x.copy()
    a[:, 3] = 0
    result = correlations(a, x)
    assert result['valid_group_count'] == 5 and result['all_six_mean_r'] is None
    np.testing.assert_allclose(result['valid_only_mean_r'], 1.)
    assert result['invalid_groups'] == ['vasti']


def test_cross_correlation_reports_delayed_activation_sign():
    rng = np.random.default_rng(17)
    emg = rng.normal(size=80)
    act = np.r_[np.zeros(3), emg[:-3]]
    row = cross_correlation(act, emg)
    assert row['lag_ms'] == 30 and row['r'] > .999999
