# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Asynchronous verification feedback for physical-K performance decisions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm_ascend.dynamic_spec import PHYSICAL_K_RECOMMEND_INTERVAL
from vllm_ascend.worker.v2.spec_decode.physical_k_calibration import (
    AcceptanceCalibrator,
    ProbeBudget,
    RuntimeCostCalibrator,
    SamplingKey,
    context_bucket,
)

_MAX_PENDING_FEEDBACK = 4
_FULL_WIDTH_PROBE_SHARE = 0.5


@dataclass
class Forecast:
    req_ids: tuple[str, ...]
    raw: np.ndarray
    physical_ks: np.ndarray
    sampling: list[SamplingKey]
    contexts: np.ndarray
    scheduled: np.ndarray
    bucket: int
    predicted_ms: float = 0.0
    calibrated: np.ndarray | None = None

    def align_to(self, req_ids: tuple[str, ...]) -> Forecast | None:
        if req_ids == self.req_ids:
            return self
        if len(req_ids) != len(self.req_ids) or len(set(req_ids)) != len(req_ids) or set(req_ids) != set(self.req_ids):
            return None
        positions = {req_id: row for row, req_id in enumerate(self.req_ids)}
        order = np.asarray([positions[req_id] for req_id in req_ids])
        return Forecast(
            req_ids,
            self.raw[order],
            self.physical_ks[order],
            [self.sampling[row] for row in order],
            self.contexts[order],
            self.scheduled[order],
            self.bucket,
            self.predicted_ms,
            self.calibrated[order] if self.calibrated is not None else None,
        )


@dataclass
class _PendingFeedback:
    forecast: Forecast
    output: Any
    rejected_cpu: Any
    ready: Any
    start: Any
    end: Any
    draft_k: int
    # Retain the source until its asynchronous copy completes.
    rejected_gpu: Any


class PhysicalKFeedback:
    def __init__(self, max_k: int, params: dict[str, Any]):
        self.mode = params["calibration"]
        self.min_gain = params["min_gain"]
        self.probe_fraction = params["probe_budget"] * _FULL_WIDTH_PROBE_SHARE
        self.acceptance = AcceptanceCalibrator(max_k)
        self.cost = RuntimeCostCalibrator()
        self.probes: dict[int, ProbeBudget] = {}
        self.choices: dict[int, int] = {}
        self.draft_widths: dict[str, int] = {}
        self.pending: deque[_PendingFeedback] = deque()
        self.forecast: Forecast | None = None
        self.sampling_states: Any = None
        self.started: Any = None
        self.steps = 0
        self.capture = False
        self.previous_widths: set[int] = set()
        self.completed = 0
        self.labels = 0
        self.raw_squared_error = 0.0
        self.calibrated_squared_error = 0.0
        self.cost_factors: dict[tuple[int, int], float] = {}

    def probe_budget(self, bucket: int) -> ProbeBudget:
        if bucket not in self.probes:
            self.probes[bucket] = ProbeBudget(self.probe_fraction)
        return self.probes[bucket]

    def begin_step(self, runner: Any) -> None:
        """Poll completed samples without waiting on a device or output future."""
        self.poll()
        self.forecast = None
        self.started = None
        self.sampling_states = getattr(getattr(runner, "sampler", None), "sampling_states", None)
        active_ids = runner.req_states.req_id_to_index
        self.draft_widths = {key: value for key, value in self.draft_widths.items() if key in active_ids}
        widths = set(self.draft_widths.values())
        self.capture = self.steps % PHYSICAL_K_RECOMMEND_INTERVAL == 0 or widths != self.previous_widths
        self.previous_widths = widths
        self.steps += 1
        if self.capture and len(self.pending) < _MAX_PENDING_FEEDBACK:
            import torch

            self.started = torch.npu.Event(enable_timing=True)
            self.started.record()
        else:
            self.capture = False

    def features(self, manager: Any, req_ids: list[str]) -> tuple[list[SamplingKey], np.ndarray]:
        slots = [manager.req_states.req_id_to_index[req_id] for req_id in req_ids]
        contexts = np.asarray([context_bucket(manager.req_states.num_computed_tokens_np[slot]) for slot in slots])
        states = self.sampling_states
        if states is None:
            # No implicit greedy/non-greedy mixing when a runner cannot expose
            # the sampling parameters (e.g. a version-incompatible upstream).
            return [], contexts
        params = [
            (
                float(states.temperature.np[s]),
                float(states.top_p.np[s]),
                int(states.top_k.np[s]),
                float(states.min_p.np[s]),
            )
            for s in slots
        ]
        return params, contexts

    def capture_forecast(self, manager: Any, per_req: dict, drafts: dict, num_target_tokens: int) -> None:
        if not self.capture or manager.num_bonus_tokens != 1:
            return
        req_ids = list(per_req)
        if not req_ids or any(req_id not in self.draft_widths for req_id in req_ids):
            return
        scheduled = np.asarray([len(drafts.get(req_id, ())) for req_id in req_ids])
        slots = np.asarray([manager.req_states.req_id_to_index[req_id] for req_id in req_ids])
        non_draft = np.asarray(list(per_req.values())) - scheduled
        # Exclude prefills, prefill lookahead, and mixed batches. A decode step
        # has exactly one non-draft token per row for this sampler contract.
        if np.any(non_draft != 1) or np.any(
            manager.req_states.num_computed_tokens_np[slots] < manager.req_states.prefill_len.np[slots]
        ):
            return
        sampling, contexts = self.features(manager, req_ids)
        if not sampling:
            return
        confidence = manager._stale_confidences[manager._stale_idx].np[slots].astype(np.float64)
        raw = np.cumprod(confidence, axis=1)
        if not np.all(np.isfinite(raw)):
            return
        widths = np.asarray([self.draft_widths[req_id] for req_id in req_ids])
        bucket = 1 << (len(req_ids) - 1).bit_length()
        forecast = Forecast(tuple(req_ids), raw, widths, sampling, contexts, scheduled, bucket)
        forecast.calibrated = raw.copy()
        for width in np.unique(widths):
            rows = np.flatnonzero(widths == width)
            if not 0 < width <= raw.shape[1]:
                continue
            prediction = self.acceptance.predict(raw[rows], int(width), [sampling[row] for row in rows], contexts[rows])
            forecast.calibrated[rows, :width] = prediction.mean
        if np.all(widths == widths[0]):
            table = manager._physical_k_draft_costs.get(int(widths[0]), {})
            draft_ms = table.get(bucket)
            verify = manager.cost_tables[1]
            if draft_ms is not None and 0 <= num_target_tokens < len(verify):
                forecast.predicted_ms = float(draft_ms + verify[num_target_tokens])
        self.forecast = forecast

    def finish_step(self, runner: Any, output: Any, input_batch: Any) -> None:
        speculator = getattr(runner, "speculator", None)
        draft_k = getattr(speculator, "_vllm_ascend_last_runtime_k", None)
        container = getattr(output, "model_runner_output", output)
        if container is not None and draft_k is not None:
            for req_id in container.req_ids:
                self.draft_widths[req_id] = int(draft_k)
        forecast, self.forecast = self.forecast, None
        if forecast is None or self.started is None or draft_k is None:
            return
        # Structured-output masks can invalidate a proposal for reasons absent
        # from our features. Do not train on those labels.
        if input_batch is None or input_batch.has_structured_output_reqs:
            return
        if container is None:
            return
        forecast = forecast.align_to(tuple(container.req_ids))
        if forecast is None:
            return
        if not hasattr(output, "num_sampled_tokens_np") or not hasattr(output, "sampler_output"):
            return
        rejected = output.sampler_output.num_rejected
        if rejected is None:
            return
        import torch

        ended = torch.npu.Event(enable_timing=True)
        ended.record()
        rejected_cpu = torch.empty(rejected.shape, dtype=rejected.dtype, device="cpu", pin_memory=True)
        ready = torch.npu.Event()
        with torch.npu.stream(runner.output_copy_stream):
            runner.output_copy_stream.wait_stream(torch.npu.current_stream())
            rejected_cpu.copy_(rejected, non_blocking=True)
            ready.record(runner.output_copy_stream)
        self.pending.append(
            _PendingFeedback(forecast, output, rejected_cpu, ready, self.started, ended, int(draft_k), rejected)
        )

    def poll(self) -> None:
        while self.pending and self.pending[0].ready.query():
            feedback = self.pending.popleft()
            forecast = feedback.forecast
            sampled = np.asarray(feedback.output.num_sampled_tokens_np)
            rejected = feedback.rejected_cpu.numpy()
            if sampled.shape != forecast.physical_ks.shape or rejected.shape != sampled.shape:
                continue
            if forecast.calibrated is not None:
                accepted = sampled - 1
                admitted = accepted + rejected
                valid_rows = (
                    (accepted >= 0)
                    & (rejected >= 0)
                    & (admitted <= forecast.physical_ks)
                    & (admitted <= forecast.scheduled)
                )
                positions = np.arange(forecast.raw.shape[1])[None, :]
                observed = valid_rows[:, None] & (positions < admitted[:, None])
                labels = positions < accepted[:, None]
                self.labels += int(np.count_nonzero(observed))
                self.raw_squared_error += float(np.sum((forecast.raw[observed] - labels[observed]) ** 2))
                self.calibrated_squared_error += float(np.sum((forecast.calibrated[observed] - labels[observed]) ** 2))
            self.acceptance.observe(
                forecast.raw,
                forecast.physical_ks,
                forecast.sampling,
                forecast.contexts,
                sampled,
                rejected,
                forecast.scheduled,
            )
            # A transition mixes verification under the old K with drafting
            # under the new K; do not attribute its complete cost to either.
            if np.all(forecast.physical_ks == feedback.draft_k) and forecast.predicted_ms > 0:
                key = (feedback.draft_k, forecast.bucket, int(np.max(forecast.contexts)))
                self.cost.observe(key, forecast.predicted_ms, feedback.start.elapsed_time(feedback.end))
            self.completed += 1
