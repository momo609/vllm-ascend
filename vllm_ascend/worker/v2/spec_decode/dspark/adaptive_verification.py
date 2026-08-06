# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Hardware-aware verification for the Ascend DSpark MRV2 path.

The MVP deliberately uses one stale CPU allocation result as the source of
truth for every downstream metadata structure. This avoids a device-to-host
sync and, unlike a mixed CPU/device layout, cannot produce mismatched request
or logits boundaries.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

    from vllm_ascend.ascend_config import DSparkHardwareAwareVerificationConfig


def build_cost_tables_from_curves(
    draft_curve: list[tuple[int, float]],
    verify_curve: list[tuple[int, float]],
    max_num_reqs: int,
    max_batch_tokens: int,
    cudagraph_limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build request/token cost tables from measured NPU curves.

    At and below ``cudagraph_limit`` execution pads to a captured token
    bucket, so the table is a step function. Above the graph limit, adjacent
    measured points are interpolated. Curves are made monotonic to suppress
    normal profiling noise.
    """

    def build_table(limit: int, curve: list[tuple[int, float]]) -> np.ndarray:
        if not curve:
            raise ValueError("A non-empty hardware cost curve is required.")
        xs, ys = np.asarray(curve, dtype=np.float64).T
        ys = np.maximum.accumulate(ys)
        values = np.arange(limit + 1)
        bucket_indices = np.searchsorted(xs, values, side="left")
        result = ys[np.minimum(bucket_indices, len(xs) - 1)]

        if cudagraph_limit:
            interpolate_mask = values > cudagraph_limit
            curve_mask = xs > cudagraph_limit
            if interpolate_mask.any() and curve_mask.any():
                result[interpolate_mask] = np.interp(values[interpolate_mask], xs[curve_mask], ys[curve_mask])

        if len(xs) > 1:
            extrapolate_mask = values > xs[-1]
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            result[extrapolate_mask] = ys[-1] + slope * (values[extrapolate_mask] - xs[-1])
        return result

    draft_table = np.maximum(build_table(max_num_reqs, draft_curve), 0.0)
    verify_table = np.maximum(build_table(max_batch_tokens, verify_curve), np.finfo(np.float64).eps)
    return draft_table, verify_table


@dataclass(frozen=True)
class DSparkVerificationPlan:
    """One batch's stale-CPU verification allocation."""

    draft_budget: int
    full_draft_budget: int
    draft_lengths: dict[str, int]
    predicted_throughput: float
    fixed_throughput: float

    @property
    def trimmed_draft_tokens(self) -> int:
        return self.full_draft_budget - self.draft_budget


def select_verification_plan(
    req_ids: list[str],
    confidence_probs: np.ndarray,
    scheduled_drafts: np.ndarray,
    num_non_draft_tokens: np.ndarray,
    draft_cost_table: np.ndarray,
    verify_cost_table: np.ndarray,
    min_predicted_gain: float,
) -> DSparkVerificationPlan | None:
    """Select a global draft budget and assign it as continuous prefixes."""

    num_reqs = len(req_ids)
    if num_reqs == 0 or confidence_probs.shape[0] != num_reqs:
        return None
    if scheduled_drafts.shape != (num_reqs,):
        return None
    if num_non_draft_tokens.shape != (num_reqs,):
        return None
    if np.any(scheduled_drafts < 0) or np.any(num_non_draft_tokens <= 0):
        return None

    full_draft_budget = int(scheduled_drafts.sum())
    base_tokens = int(num_non_draft_tokens.sum())
    if full_draft_budget == 0:
        return None
    if num_reqs >= len(draft_cost_table):
        return None
    if base_tokens + full_draft_budget >= len(verify_cost_table):
        return None

    num_steps = confidence_probs.shape[1]
    if np.any(scheduled_drafts > num_steps):
        return None
    clipped_confidences = np.clip(confidence_probs.astype(np.float64, copy=False), 0.0, 1.0)
    survival = np.cumprod(clipped_confidences, axis=1)
    valid = np.arange(num_steps)[None, :] < scheduled_drafts[:, None]
    flat_survival = survival.reshape(-1)
    flat_valid = valid.reshape(-1)
    valid_indices = np.flatnonzero(flat_valid)
    order = valid_indices[np.argsort(-flat_survival[valid_indices], kind="stable")]
    sorted_scores = flat_survival[order]

    expected_tokens = np.concatenate(
        (
            np.asarray([float(num_reqs)]),
            float(num_reqs) + np.cumsum(sorted_scores),
        )
    )
    costs = draft_cost_table[num_reqs] + verify_cost_table[base_tokens : base_tokens + full_draft_budget + 1]
    throughput = expected_tokens / costs
    draft_budget = int(np.argmax(throughput))
    fixed_throughput = float(throughput[full_draft_budget])
    predicted_throughput = float(throughput[draft_budget])

    if predicted_throughput < fixed_throughput * (1.0 + min_predicted_gain):
        draft_budget = full_draft_budget
        predicted_throughput = fixed_throughput

    admitted = order[:draft_budget]
    per_request = np.bincount(admitted // num_steps, minlength=num_reqs).astype(np.int32)
    if np.any(per_request > scheduled_drafts):
        raise AssertionError("DSpark allocation exceeded a scheduled draft prefix.")

    return DSparkVerificationPlan(
        draft_budget=draft_budget,
        full_draft_budget=full_draft_budget,
        draft_lengths={req_id: int(length) for req_id, length in zip(req_ids, per_request)},
        predicted_throughput=predicted_throughput,
        fixed_throughput=fixed_throughput,
    )


class AsyncStaleConfidenceRing:
    """Double-buffered, non-blocking NPU-to-CPU confidence publication.

    A completed copy is promoted only when the next confidence block is
    recorded. Since planning happens before recording in a model step, the
    planner can never use the confidence belonging to the drafts it is about
    to verify.
    """

    def __init__(
        self,
        max_num_reqs: int,
        num_speculative_steps: int,
        device: torch.device,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self._device_confidences = torch.ones(
            (max_num_reqs, num_speculative_steps),
            dtype=torch.float32,
            device=device,
        )
        self._host_tensors = [
            torch.ones(
                (max_num_reqs, num_speculative_steps),
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self._events = [torch.npu.Event() for _ in range(2)]
        self._copy_stream = torch.npu.Stream()
        self._active_idx = 0
        self._pending_idx: int | None = None
        self._epochs = np.zeros(max_num_reqs, dtype=np.int64)
        self._valid = np.zeros(max_num_reqs, dtype=np.bool_)
        self._buffer_epochs = [self._epochs.copy() for _ in range(2)]
        self._buffer_valid = [self._valid.copy() for _ in range(2)]

    def reset_slot(self, req_idx: int) -> None:
        self._epochs[req_idx] += 1
        self._valid[req_idx] = False
        self._device_confidences[req_idx].fill_(1.0)

    def get_rows(self, slots: np.ndarray) -> np.ndarray | None:
        active_epochs = self._buffer_epochs[self._active_idx][slots]
        active_valid = self._buffer_valid[self._active_idx][slots]
        if not np.all(active_valid & (active_epochs == self._epochs[slots])):
            return None
        return self._host_tensors[self._active_idx].numpy()[slots]

    def record(
        self,
        confidence_probs: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        valid_rows: np.ndarray,
    ) -> None:
        if self._pending_idx is not None:
            if not self._events[self._pending_idx].query():
                return
            self._active_idx = self._pending_idx
            self._pending_idx = None

        active_rows = np.flatnonzero(valid_rows)
        if active_rows.size == 0:
            return
        active_rows_device = torch.from_numpy(active_rows).to(device=idx_mapping.device, dtype=torch.long)
        slots = idx_mapping[active_rows_device].to(torch.long)
        values = confidence_probs[active_rows_device]
        self._device_confidences.index_copy_(0, slots, values)
        self._valid[idx_mapping_np[active_rows]] = True

        write_idx = self._active_idx ^ 1
        self._buffer_epochs[write_idx] = self._epochs.copy()
        self._buffer_valid[write_idx] = self._valid.copy()
        current_stream = torch.npu.current_stream()
        self._copy_stream.wait_stream(current_stream)
        with torch.npu.stream(self._copy_stream):
            self._host_tensors[write_idx].copy_(self._device_confidences, non_blocking=True)
            self._events[write_idx].record()
        self._pending_idx = write_idx

    def set_stale_confidences_for_test(self, confidence_probs: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Inject a published snapshot without touching a device."""
        np.copyto(self._host_tensors[self._active_idx].numpy(), confidence_probs)
        if valid is None:
            valid = np.ones(self.max_num_reqs, dtype=np.bool_)
        self._valid[:] = valid
        self._buffer_valid[self._active_idx] = valid.copy()
        self._buffer_epochs[self._active_idx] = self._epochs.copy()


class AscendAdaptiveVerificationManager:
    """Plan and apply stale-CPU DSpark verification budgets."""

    def __init__(
        self,
        req_states: RequestState,
        num_bonus_tokens: int,
        config: DSparkHardwareAwareVerificationConfig,
    ) -> None:
        self.req_states = req_states
        self.num_bonus_tokens = num_bonus_tokens
        self.config = config
        self.cost_tables = build_cost_tables_from_curves(
            config.draft_cost_curve,
            config.verify_cost_curve,
            req_states.max_num_reqs,
            req_states.max_num_batched_tokens,
            config.cudagraph_limit,
        )
        self.confidences = AsyncStaleConfidenceRing(
            req_states.max_num_reqs,
            req_states.num_speculative_steps,
            req_states.device,
        )
        self.last_plan: DSparkVerificationPlan | None = None

    def reset_slot(self, req_idx: int) -> None:
        self.confidences.reset_slot(req_idx)

    def plan_scheduler_output(self, scheduler_output: SchedulerOutput) -> SchedulerOutput:
        self.last_plan = None
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            return scheduler_output

        req_ids = list(scheduler_output.num_scheduled_tokens)
        # The MVP handles decode-only batches. A mixed batch falls back to the
        # fixed layout rather than applying an approximate metadata rewrite.
        if not req_ids or any(req_id not in draft_tokens for req_id in req_ids):
            return scheduler_output

        scheduled_drafts = np.fromiter(
            (len(draft_tokens[req_id]) for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        num_non_draft_tokens = np.fromiter(
            (scheduler_output.num_scheduled_tokens[req_id] - len(draft_tokens[req_id]) for req_id in req_ids),
            dtype=np.int32,
            count=len(req_ids),
        )
        if np.any(num_non_draft_tokens != self.num_bonus_tokens):
            return scheduler_output

        try:
            slots = np.fromiter(
                (self.req_states.req_id_to_index[req_id] for req_id in req_ids),
                dtype=np.int32,
                count=len(req_ids),
            )
        except KeyError:
            return scheduler_output
        stale_confidences = self.confidences.get_rows(slots)
        if stale_confidences is None:
            return scheduler_output

        draft_cost_table, verify_cost_table = self.cost_tables
        plan = select_verification_plan(
            req_ids,
            stale_confidences,
            scheduled_drafts,
            num_non_draft_tokens,
            draft_cost_table,
            verify_cost_table,
            self.config.min_predicted_gain,
        )
        self.last_plan = plan
        if plan is None or plan.trimmed_draft_tokens == 0:
            return scheduler_output

        compacted = copy.copy(scheduler_output)
        compacted.scheduled_spec_decode_tokens = {
            req_id: draft_tokens[req_id][: plan.draft_lengths[req_id]]
            for req_id in req_ids
            if plan.draft_lengths[req_id] > 0
        }
        compacted.num_scheduled_tokens = {
            req_id: int(num_non_draft_tokens[i]) + plan.draft_lengths[req_id] for i, req_id in enumerate(req_ids)
        }
        compacted.total_num_scheduled_tokens = sum(compacted.num_scheduled_tokens.values())
        return compacted

    def record_confidences(self, input_batch: InputBatch, probs: torch.Tensor) -> None:
        valid_rows = ~input_batch.is_prefilling_np
        self.confidences.record(
            probs[: input_batch.num_reqs],
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            valid_rows,
        )
