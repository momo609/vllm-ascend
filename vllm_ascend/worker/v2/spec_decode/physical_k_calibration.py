# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Bounded CPU statistics for physical-K utility prediction.

These probabilities price work only. They must never replace the probabilities
used by the rejection sampler. Each instance belongs to one deployed model pair.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import ceil

import numpy as np

from vllm_ascend.dynamic_spec import PHYSICAL_K_PROBE_COST_STEPS, PHYSICAL_K_PROBE_INTERVAL

_CONFIDENCE_BINS = 16
_PRIOR_STEPS = 8.0
_HALF_LIFE_STEPS = 256.0
_MAX_GROUPS = 128
_UNCERTAINTY_MULTIPLIER = 2.0

SamplingKey = tuple[float, float, int, float]


def context_bucket(length: int) -> int:
    return max(1, int(length)).bit_length() - 1


@dataclass
class ProbabilityEstimate:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class _Histogram:
    count: np.ndarray
    accepted: np.ndarray
    updated: int


class AcceptanceCalibrator:
    """Shrink per-position, per-confidence-bin observations to raw predictions.

    A whole feedback batch contributes at most one observation to each cell.
    This prevents a large, correlated batch from instantly overwhelming the
    prior. Error bars are conservative decision heuristics, not independent
    Bernoulli confidence guarantees. Missing/censored positions add no labels.
    """

    def __init__(self, max_k: int):
        self.max_k = max_k
        self.clock = 0
        self.groups: OrderedDict[tuple, _Histogram] = OrderedDict()

    def _histogram(self, key: tuple, *, create: bool) -> _Histogram | None:
        histogram = self.groups.get(key)
        if histogram is None:
            if not create:
                return None
            if len(self.groups) >= _MAX_GROUPS:
                self.groups.popitem(last=False)
            shape = (self.max_k, _CONFIDENCE_BINS)
            histogram = _Histogram(np.zeros(shape), np.zeros(shape), self.clock)
            self.groups[key] = histogram
        self.groups.move_to_end(key)
        decay = 0.5 ** ((self.clock - histogram.updated) / _HALF_LIFE_STEPS)
        histogram.count *= decay
        histogram.accepted *= decay
        histogram.updated = self.clock
        return histogram

    def observe(
        self,
        raw: np.ndarray,
        physical_ks: np.ndarray,
        sampling: list[SamplingKey],
        contexts: np.ndarray,
        num_sampled: np.ndarray,
        num_rejected: np.ndarray,
        scheduled: np.ndarray,
    ) -> None:
        """Consume sampler counts before EOS/output-limit truncation.

        V2 reports sampled = accepted drafts + one target token and rejected =
        admitted drafts - accepted drafts. Thus admitted, unlike scheduled,
        excludes AV's unverified tail. Invalid/prefill rows are discarded.
        """
        self.clock += 1
        updates: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        for row, (width, params, context) in enumerate(zip(physical_ks, sampling, contexts)):
            width = int(width)
            accepted = int(num_sampled[row]) - 1
            rejected = int(num_rejected[row])
            admitted = accepted + rejected
            if not (0 < width <= self.max_k and 0 <= accepted <= admitted <= min(width, scheduled[row])):
                continue
            values = raw[row, :admitted]
            if not np.all(np.isfinite(values)):
                continue
            bins = np.minimum((np.clip(values, 0, 1) * _CONFIDENCE_BINS).astype(int), _CONFIDENCE_BINS - 1)
            positions = np.arange(admitted)
            labels = positions < accepted
            # The broad fallback shares sampling settings and actual K, never
            # another K or model. Context-specific cells remain independent.
            for key in ((width, params, -1), (width, params, int(context))):
                if key not in updates:
                    shape = (self.max_k, _CONFIDENCE_BINS)
                    updates[key] = (np.zeros(shape), np.zeros(shape))
                count, successes = updates[key]
                np.add.at(count, (positions, bins), 1)
                np.add.at(successes, (positions, bins), labels)
        for key, (count, successes) in updates.items():
            histogram = self._histogram(key, create=True)
            assert histogram is not None
            observed = count > 0
            histogram.count[observed] += 1.0
            histogram.accepted[observed] += successes[observed] / count[observed]

    def predict(
        self, raw: np.ndarray, physical_k: int, sampling: list[SamplingKey], contexts: np.ndarray
    ) -> ProbabilityEstimate:
        values = np.clip(raw[:, :physical_k], 0.0, 1.0)
        mean = values.copy()
        count = np.zeros_like(values)
        for row, (params, context) in enumerate(zip(sampling, contexts)):
            broad = self._histogram((physical_k, params, -1), create=False)
            local = self._histogram((physical_k, params, int(context)), create=False)
            bins = np.minimum((values[row] * _CONFIDENCE_BINS).astype(int), _CONFIDENCE_BINS - 1)
            positions = np.arange(physical_k)
            for histogram in (broad, local):
                if histogram is None:
                    continue
                n = histogram.count[positions, bins]
                # Prefer a mature local cell; do not count it and its broad
                # fallback as independent evidence from the same observations.
                use = (n >= _PRIOR_STEPS) | ((count[row] == 0) & (n > 0))
                calibrated = (histogram.accepted[positions, bins] + _PRIOR_STEPS * values[row]) / (n + _PRIOR_STEPS)
                mean[row, use] = calibrated[use]
                count[row, use] = n[use]
        radius = _UNCERTAINTY_MULTIPLIER * np.sqrt(
            np.maximum(mean * (1.0 - mean), 1.0 / _CONFIDENCE_BINS) / (count + _PRIOR_STEPS + 1.0)
        )
        return ProbabilityEstimate(
            np.minimum.accumulate(mean, axis=1),
            np.minimum.accumulate(np.clip(mean - radius, 0, 1), axis=1),
            np.minimum.accumulate(np.clip(mean + radius, 0, 1), axis=1),
        )


@dataclass
class _CostMoments:
    count: float = 0.0
    ratio: float = 0.0
    squared_ratio: float = 0.0


class RuntimeCostCalibrator:
    """Correct startup step costs using completed, stable-width decode steps."""

    def __init__(self):
        self.groups: OrderedDict[tuple[int, int, int], _CostMoments] = OrderedDict()

    def observe(self, key: tuple[int, int, int], predicted_ms: float, elapsed_ms: float) -> None:
        if not (np.isfinite(predicted_ms) and np.isfinite(elapsed_ms) and predicted_ms > 0 and elapsed_ms > 0):
            return
        if key not in self.groups:
            if len(self.groups) >= _MAX_GROUPS:
                self.groups.popitem(last=False)
            self.groups[key] = _CostMoments()
        self.groups.move_to_end(key)
        moments = self.groups[key]
        decay = 0.5 ** (1.0 / _HALF_LIFE_STEPS)
        ratio = elapsed_ms / predicted_ms
        moments.count = decay * moments.count + 1.0
        moments.ratio = decay * moments.ratio + ratio
        moments.squared_ratio = decay * moments.squared_ratio + ratio * ratio

    def estimate(self, key: tuple[int, int, int]) -> tuple[float, float, float]:
        moments = self.groups.get(key)
        if moments is None:
            return 1.0, 1.0, 1.0
        weight = moments.count + _PRIOR_STEPS
        mean = (moments.ratio + _PRIOR_STEPS) / weight
        variance = max(0.0, (moments.squared_ratio + _PRIOR_STEPS) / weight - mean * mean)
        radius = _UNCERTAINTY_MULTIPLIER * np.sqrt(variance / weight)
        return mean, max(np.finfo(float).eps, mean - radius), mean + radius

    def effective_samples(self, key: tuple[int, int, int]) -> float:
        moments = self.groups.get(key)
        return moments.count if moments is not None else 0.0


class ProbeBudget:
    """Charge complete predicted probe time, rather than a fixed step fraction.

    Pricing uses the runtime-corrected step model. This is an estimated time
    budget, not a hard wall-clock bound under external load changes.
    """

    def __init__(self, fraction: float):
        self.fraction = fraction
        self.credit = 0.0
        self.steps = 0
        self.last_probe = 0
        self.next_candidate = 0
        self.max_probe_ms = 0.0

    def advance(self, step_ms: float) -> None:
        self.steps += 1
        self.credit = min(self.credit + self.fraction * step_ms, max(2.0 * step_ms, self.max_probe_ms))

    def choose(self, candidates: list[int], costs: dict[int, float], current_k: int) -> int | None:
        if self.fraction <= 0 or self.steps - self.last_probe < PHYSICAL_K_PROBE_INTERVAL:
            return None
        alternatives = [k for k in candidates if k != current_k]
        if not alternatives:
            return None
        if any(not np.isfinite(costs[k]) or costs[k] <= 0 for k in candidates):
            return None
        self.max_probe_ms = PHYSICAL_K_PROBE_COST_STEPS * max(costs[k] for k in candidates)
        # Cycle through candidates so a width cannot be starved by the score
        # that was itself fitted using the previous width's workload.
        candidate = alternatives[self.next_candidate % len(alternatives)]
        cost = PHYSICAL_K_PROBE_COST_STEPS * max(costs[candidate], costs[current_k])
        if not np.isfinite(cost) or cost <= 0 or cost > self.credit + np.finfo(float).eps * cost * self.steps:
            return None
        self.credit = max(0.0, self.credit - cost)
        self.last_probe = self.steps
        self.next_candidate += 1
        return candidate

    def full_width_interval(self, current_ms: float, full_ms: float) -> int:
        if self.fraction <= 0:
            return 0
        return max(
            PHYSICAL_K_PROBE_INTERVAL,
            ceil(
                PHYSICAL_K_PROBE_COST_STEPS
                * max(full_ms, current_ms)
                / (self.fraction * max(current_ms, np.finfo(float).eps))
            ),
        )
