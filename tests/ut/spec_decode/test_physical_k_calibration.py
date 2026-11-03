# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_ascend.dynamic_spec import _create_controller
from vllm_ascend.worker.v2.spec_decode.physical_k_calibration import (
    AcceptanceCalibrator,
    ProbeBudget,
    RuntimeCostCalibrator,
)
from vllm_ascend.worker.v2.spec_decode.physical_k_feedback import (
    Forecast,
    PhysicalKFeedback,
    _PendingFeedback,
)

_GREEDY = (0.0, 1.0, 100, 0.0)
_SAMPLED = (1.0, 1.0, 100, 0.0)


def observe(calibrator, raw, width=5, sampled=1, rejected=5, context=10, params=_GREEDY):
    rows = len(raw)
    calibrator.observe(
        raw,
        np.full(rows, width),
        [params] * rows,
        np.full(rows, context),
        np.full(rows, sampled),
        np.full(rows, rejected),
        np.full(rows, width),
    )


def test_unobserved_model_starts_from_raw_probability():
    raw = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5]])
    predicted = AcceptanceCalibrator(5).predict(raw, 5, [_GREEDY], np.asarray([10]))
    np.testing.assert_allclose(predicted.mean, raw)
    assert np.all(predicted.lower <= predicted.mean)
    assert np.all(predicted.upper >= predicted.mean)


def test_av_censored_tail_never_becomes_negative_labels():
    calibrator = AcceptanceCalibrator(5)
    raw = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5]])
    # Three admitted drafts, all accepted. Scheduled/physical K was five.
    for _ in range(64):
        observe(calibrator, raw, sampled=4, rejected=0)
    predicted = calibrator.predict(raw, 5, [_GREEDY], np.asarray([10]))
    assert np.all(predicted.mean[0, :3] > raw[0, :3])
    np.testing.assert_allclose(predicted.mean[0, 3:], raw[0, 3:])
    assert all(not histogram.count[3:].any() for histogram in calibrator.groups.values())


def test_calibration_isolated_by_actual_k_and_sampling_parameters():
    calibrator = AcceptanceCalibrator(5)
    raw = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5]])
    for _ in range(64):
        observe(calibrator, raw)
    assert calibrator.predict(raw, 5, [_GREEDY], np.asarray([10])).mean[0, 0] < 0.2
    np.testing.assert_allclose(calibrator.predict(raw, 3, [_GREEDY], np.asarray([10])).mean, raw[:, :3])
    np.testing.assert_allclose(calibrator.predict(raw, 5, [_SAMPLED], np.asarray([10])).mean, raw)


def test_large_batch_does_not_count_as_many_independent_steps():
    raw = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5]])
    small, large = AcceptanceCalibrator(5), AcceptanceCalibrator(5)
    observe(small, raw)
    observe(large, np.repeat(raw, 64, axis=0))
    a = small.predict(raw, 5, [_GREEDY], np.asarray([10]))
    b = large.predict(raw, 5, [_GREEDY], np.asarray([10]))
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.lower, b.lower)


def test_mature_context_statistics_override_broad_fallback():
    calibrator = AcceptanceCalibrator(5)
    raw = np.asarray([[0.9, 0.8, 0.7, 0.6, 0.5]])
    for _ in range(64):
        observe(calibrator, raw, context=10)
        observe(calibrator, raw, context=14, sampled=6, rejected=0)
    low = calibrator.predict(raw, 5, [_GREEDY], np.asarray([10])).mean
    high = calibrator.predict(raw, 5, [_GREEDY], np.asarray([14])).mean
    assert low[0, 0] < 0.25
    assert high[0, -1] > 0.8
    assert np.all(np.diff(high, axis=1) <= 0)


@pytest.mark.parametrize("sampled,rejected", [(0, 0), (7, 0), (2, -1), (2, 5)])
def test_invalid_or_prefill_counts_do_not_update(sampled, rejected):
    calibrator = AcceptanceCalibrator(5)
    observe(calibrator, np.full((1, 5), 0.8), sampled=sampled, rejected=rejected)
    assert not calibrator.groups


def test_bounded_state_and_old_data_decay():
    calibrator = AcceptanceCalibrator(5)
    raw = np.full((1, 5), 0.8)
    for context in range(200):
        observe(calibrator, raw, context=context)
    assert len(calibrator.groups) <= 128
    for _ in range(64):
        observe(calibrator, raw, context=10)
    before = calibrator.predict(raw, 5, [_GREEDY], np.asarray([10])).mean[0, 0]
    calibrator.clock += 2048
    after = calibrator.predict(raw, 5, [_GREEDY], np.asarray([10])).mean[0, 0]
    assert after > before


def test_cost_correction_tracks_actual_width_context_and_complete_time():
    costs = RuntimeCostCalibrator()
    key = (3, 64, 13)
    for _ in range(64):
        costs.observe(key, 10, 30)
    assert costs.estimate(key)[0] > 2.5
    assert costs.estimate((5, 64, 13)) == (1, 1, 1)
    assert costs.estimate((3, 64, 10)) == (1, 1, 1)
    costs.observe(key, 0, 30)
    assert len(costs.groups) == 1


def test_probe_budget_charges_time_and_cycles_candidates():
    budget = ProbeBudget(0.02)
    costs = {3: 10.0, 4: 15.0, 5: 20.0}
    for _ in range(32):
        budget.advance(10)
    assert budget.choose([3, 4, 5], costs, 5) is None
    for _ in range(268):
        budget.advance(10)
    assert budget.choose([3, 4, 5], costs, 5) == 3
    assert budget.credit == pytest.approx(0)
    for _ in range(300):
        budget.advance(10)
    assert budget.choose([3, 4, 5], costs, 5) == 4
    assert budget.full_width_interval(10, 20) == 300
    assert ProbeBudget(0).full_width_interval(10, 20) == 0


def test_probe_bursts_respect_estimated_time_budget_when_step_cost_changes():
    budget = ProbeBudget(0.02)
    costs = {3: 10.0, 4: 40.0, 5: 20.0}
    total_ms = spent_ms = 0.0
    seen = set()
    for step in range(5000):
        step_ms = (10.0, 100.0, 1.0)[step % 3]
        total_ms += step_ms
        budget.advance(step_ms)
        candidate = budget.choose([3, 4, 5], costs, 5)
        if candidate is not None:
            seen.add(candidate)
            spent_ms += 3 * max(costs[candidate], costs[5])
        assert spent_ms <= budget.fraction * total_ms + 1e-8
    assert seen == {3, 4}


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_probe_does_not_spend_credit_on_invalid_step_costs(invalid):
    budget = ProbeBudget(0.02)
    for _ in range(1000):
        budget.advance(100)
    credit = budget.credit
    assert budget.choose([3, 5], {3: 10.0, 5: invalid}, 5) is None
    assert budget.credit == credit


def make_feedback():
    return PhysicalKFeedback(5, {"calibration": "shadow", "min_gain": 0.02, "probe_budget": 0.02})


def test_feedback_never_waits_for_device_completion():
    feedback = make_feedback()
    pending = SimpleNamespace(ready=SimpleNamespace(query=lambda: False))
    feedback.pending.append(pending)
    feedback.poll()
    assert len(feedback.pending) == 1
    assert feedback.completed == 0


def test_feedback_uses_frozen_prediction_and_excludes_transition_cost():
    feedback = make_feedback()
    raw = np.full((1, 5), 0.8)
    forecast = Forecast(("original",), raw.copy(), np.asarray([5]), [_GREEDY], np.asarray([10]), np.asarray([5]), 1, 10)
    raw.fill(0.1)  # Later slot reuse must not replace this prediction.
    rejected = SimpleNamespace(numpy=lambda: np.asarray([1]))
    item = _PendingFeedback(
        forecast,
        SimpleNamespace(num_sampled_tokens_np=np.asarray([3])),
        rejected,
        SimpleNamespace(query=lambda: True),
        SimpleNamespace(elapsed_time=lambda end: 20),
        object(),
        3,
        object(),
    )
    feedback.pending.append(item)
    feedback.poll()
    assert feedback.completed == 1
    assert not feedback.cost.groups
    assert feedback.acceptance.groups
    histogram = feedback.acceptance.groups[(5, _GREEDY, 10)]
    assert histogram.count[0, 12] == 1
    assert histogram.count[3:].sum() == 0


@pytest.mark.parametrize("probe_k,stable_k", [(3, 5), (4, 5), (5, 3)])
def test_scheduler_probe_produces_a_stable_width_cost_sample(probe_k, stable_k):
    controller = _create_controller(
        SimpleNamespace(
            use_v2_model_runner=True,
            speculative_config=SimpleNamespace(num_speculative_tokens=5, enable_adaptive_verification=True),
            additional_config={
                "dynamic_spec_config": {"method": "dspark", "physical_k": {"min_k": 3, "calibration": "online"}}
            },
        )
    )
    controller.recommend(64, stable_k, probe_k=probe_k, probe_interval=0)
    for _ in range(3):
        controller.advance_profile(64)
    feedback = make_feedback()
    previous_k = stable_k
    for _ in range(3):
        current_k = controller.cap(5, 64)
        forecast = Forecast(
            tuple(str(index) for index in range(64)),
            np.ones((64, 5)),
            np.full(64, previous_k),
            [_GREEDY] * 64,
            np.full(64, 10),
            np.full(64, previous_k),
            64,
            10.0,
        )
        feedback.pending.append(
            _PendingFeedback(
                forecast,
                SimpleNamespace(num_sampled_tokens_np=np.ones(64, dtype=int)),
                SimpleNamespace(numpy=lambda: np.ones(64, dtype=int)),
                SimpleNamespace(query=lambda: True),
                SimpleNamespace(elapsed_time=lambda end: 20.0),
                object(),
                current_k,
                object(),
            )
        )
        feedback.poll()
        previous_k = current_k
    assert (probe_k, 64, 10) in feedback.cost.groups
    assert feedback.cost.groups[(probe_k, 64, 10)].count == 1
    assert controller._state(64).stable_k == stable_k


def test_feedback_drops_changed_request_order_before_copying_device_data():
    feedback = make_feedback()
    feedback.started = object()
    feedback.forecast = Forecast(
        ("old",), np.ones((1, 5)), np.asarray([5]), [_GREEDY], np.asarray([10]), np.asarray([5]), 1
    )
    runner = SimpleNamespace(speculator=SimpleNamespace(_vllm_ascend_last_runtime_k=5))
    output = SimpleNamespace(model_runner_output=SimpleNamespace(req_ids=["new"]))
    feedback.finish_step(runner, output, SimpleNamespace(has_structured_output_reqs=False))
    assert not feedback.pending


def test_forecast_follows_request_ids_when_worker_sorts_batch():
    raw = np.asarray([[0.9] * 5, [0.2] * 5])
    forecast = Forecast(
        ("first", "second"),
        raw,
        np.asarray([5, 3]),
        [_GREEDY, _SAMPLED],
        np.asarray([10, 14]),
        np.asarray([5, 3]),
        2,
        calibrated=raw.copy(),
    )
    aligned = forecast.align_to(("second", "first"))
    assert aligned is not None
    np.testing.assert_allclose(aligned.raw[0], raw[1])
    assert aligned.physical_ks.tolist() == [3, 5]
    assert aligned.sampling == [_SAMPLED, _GREEDY]
    assert forecast.align_to(("first", "unknown")) is None
    assert forecast.align_to(("first", "first")) is None


def test_shadow_error_uses_predictions_frozen_before_learning_feedback():
    feedback = make_feedback()
    forecast = Forecast(
        ("request",),
        np.full((1, 5), 0.8),
        np.asarray([5]),
        [_GREEDY],
        np.asarray([10]),
        np.asarray([5]),
        1,
        calibrated=np.full((1, 5), 0.2),
    )
    feedback.pending.append(
        _PendingFeedback(
            forecast,
            SimpleNamespace(num_sampled_tokens_np=np.asarray([2])),
            SimpleNamespace(numpy=lambda: np.asarray([0])),
            SimpleNamespace(query=lambda: True),
            object(),
            object(),
            5,
            object(),
        )
    )
    feedback.poll()
    assert feedback.labels == 1
    assert feedback.raw_squared_error == pytest.approx(0.04)
    assert feedback.calibrated_squared_error == pytest.approx(0.64)
