# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Scheduler-side control for Ascend variable physical draft width.

The worker owns confidence processing and NPU cost modelling. It publishes a
compact recommendation; the scheduler applies it to a matching batch bucket.
The acceptance policy remains a safe fallback if AV profiling is unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

from vllm.logger import logger

from vllm_ascend import envs

_ACCEPTANCE_EMA_ALPHA = 0.2
_BUCKET_SWITCH_STEPS = 2
_MIN_K_DWELL_STEPS = 8
# Include every non-empty active-batch bucket in AV profiling and runtime K
# selection. This is important for RL, where a rollout can decay from a large
# batch (for example 128) through 64/32/16 down to 8 or fewer requests.
PHYSICAL_K_MIN_TUNED_BATCH_SIZE = 1
PHYSICAL_K_RECOMMEND_INTERVAL = 16
PHYSICAL_K_MAX_TP_SIZE = 8
_ACCEPTANCE_THRESHOLD = 0.6
_DOWNSHIFT_STEPS = 3
_UPSHIFT_STEPS = 2
PHYSICAL_K_PROBE_INTERVAL = 32
# One step changes draft width; the next can price matching verify/draft work.
PHYSICAL_K_PROBE_STEPS = 2
# Also reserve the step that returns from the probe to the selected width.
PHYSICAL_K_PROBE_COST_STEPS = PHYSICAL_K_PROBE_STEPS + 1
PHYSICAL_K_DEFAULT_MIN_GAIN = 0.02
PHYSICAL_K_DEFAULT_PROBE_BUDGET = 0.02


def _trace_physical_k(message: str, *args: Any) -> None:
    if envs.VLLM_ASCEND_PHYSICAL_K_TRACE:
        logger.info("ASCEND_AV_PHYSICAL_K_TRACE %s", message % args if args else message)


def batch_bucket(batch_size: int) -> int:
    return 1 << (max(int(batch_size), 1) - 1).bit_length()


def plan_draft_slots(spec: Any, max_k: int, lookup: list[int] | None, batch_size: int) -> tuple[int, int]:
    """Return the planned K and extra input slots for one scheduling iteration."""
    slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
    if spec is None or lookup is None:
        return max_k, slots
    if spec.use_dflash():
        planned_k = lookup[min(batch_size, len(lookup) - 1)]
        return planned_k, min(slots, planned_k)
    if spec.parallel_drafting and not spec.uses_draft_model():
        planned_k = lookup[min(batch_size, len(lookup) - 1)]
        return planned_k, min(slots, max(0, planned_k - 1))
    return max_k, slots


def resolve_physical_k(dynamic_config: dict[str, Any]) -> dict[str, Any] | None:
    physical = dynamic_config.get("physical_k")
    if physical is None:
        return None
    if not isinstance(physical, dict):
        raise ValueError("physical_k must be an object")
    if dynamic_config.get("method") not in ("dspark", "dflash"):
        raise ValueError("physical_k requires dspark or dflash")
    unknown = set(physical) - {"min_k", "auto_tune", "calibration", "min_gain", "probe_budget"}
    if unknown:
        raise ValueError(f"Unknown physical_k fields: {sorted(unknown)}")
    min_k = physical.get("min_k", 3)
    auto_tune = physical.get("auto_tune", True)
    if type(min_k) is not int or min_k < 1:
        raise ValueError(f"physical_k.min_k has invalid value {min_k!r}")
    if not isinstance(auto_tune, bool):
        raise ValueError(f"physical_k.auto_tune has invalid value {auto_tune!r}")
    calibration = physical.get("calibration", "shadow")
    if calibration not in ("off", "shadow", "online"):
        raise ValueError("physical_k.calibration must be off, shadow or online")
    min_gain = physical.get("min_gain", PHYSICAL_K_DEFAULT_MIN_GAIN)
    probe_budget = physical.get("probe_budget", PHYSICAL_K_DEFAULT_PROBE_BUDGET)
    for name, value in (("min_gain", min_gain), ("probe_budget", probe_budget)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"physical_k.{name} must be a finite number in [0, 1)")
    return {
        "min_k": min_k,
        "auto_tune": auto_tune,
        "calibration": calibration,
        "min_gain": float(min_gain),
        "probe_budget": float(probe_budget),
    }


def physical_k_runtime_enabled(vllm_config: Any) -> bool:
    """Enable physical K only when a narrower width can actually be used.

    TP8 is enabled for measurement, not assumed to outperform adaptive
    verification. Its extra collective and graph-management costs must be
    checked against an otherwise identical adaptive benchmark. A minimum K
    equal to the configured K has no physical-width choices and should use
    the ordinary adaptive path without its controller overhead.
    """

    dynamic = (getattr(vllm_config, "additional_config", None) or {}).get("dynamic_spec_config", {})
    if not isinstance(dynamic, dict) or (params := resolve_physical_k(dynamic)) is None:
        return False
    parallel_config = getattr(vllm_config, "parallel_config", None)
    tensor_parallel_size = int(getattr(parallel_config, "tensor_parallel_size", 1))
    speculative_config = getattr(vllm_config, "speculative_config", None)
    max_k = int(getattr(speculative_config, "num_speculative_tokens", 0))
    return tensor_parallel_size <= PHYSICAL_K_MAX_TP_SIZE and params["min_k"] < max_k


@dataclass
class _BucketState:
    stable_k: int
    profile_k: int | None
    empirical_k: int
    survival: list[float]
    survival_seen: list[bool]
    observations: int = 0
    down_steps: int = 0
    up_steps: int = 0
    dwell_steps: int = 0
    last_probe_observation: int = -1
    probe_interval: int = PHYSICAL_K_PROBE_INTERVAL
    pending_probe_k: int | None = None
    active_probe_k: int | None = None
    probe_steps_remaining: int = 0


@dataclass
class AdaptiveDraftKController:
    """Apply AV cost recommendations with an acceptance-only fallback.

    RL rollouts commonly begin with a large decode batch and shrink as
    requests encounter EOS.  Each power-of-two batch bucket therefore owns
    independent acceptance and K state.  Bucket changes and K changes are
    debounced so a short-lived rollout tail does not make graph width flap.
    """

    max_k: int
    min_k: int = 1
    auto_tune: bool = True
    require_profile_for_downshift: bool = False
    probe_steps: int = 1

    def __post_init__(self) -> None:
        self.max_k = max(int(self.max_k), 0)
        self.min_k = min(max(int(self.min_k), 1), self.max_k) if self.max_k else 0
        self._bucket_states: dict[int, _BucketState] = {}
        self._active_bucket: int | None = None
        self._pending_bucket: int | None = None
        self._pending_bucket_steps = 0
        self._single_request_id: str | None = None
        self._last_traced_cap: dict[int, int] = {}
        self._last_cap_bucket: int | None = None
        self.probe_steps = max(1, int(self.probe_steps))

    @staticmethod
    def _cancel_probe(state: _BucketState) -> None:
        state.pending_probe_k = None
        state.active_probe_k = None
        state.probe_steps_remaining = 0

    def _begin_probe(self, state: _BucketState, physical_k: int) -> int:
        state.active_probe_k = physical_k
        state.probe_steps_remaining = self.probe_steps - 1
        return physical_k

    def observe_single_request(self, request_id: str) -> None:
        """Do not carry a previous request's K decision into a new rollout.

        At concurrency one the bucket is reused for every request, although
        draft acceptance can differ sharply between prompts. Start a new
        request at the safe full width until its own confidence is priced.
        """

        if self._single_request_id is not None and self._single_request_id != request_id:
            self._bucket_states.pop(1, None)
        self._single_request_id = request_id

    _batch_bucket = staticmethod(batch_bucket)

    def _state(self, bucket: int) -> _BucketState:
        state = self._bucket_states.get(bucket)
        if state is None:
            state = _BucketState(
                stable_k=self.max_k,
                profile_k=None,
                empirical_k=self.max_k,
                survival=[1.0] * self.max_k,
                survival_seen=[False] * self.max_k,
            )
            self._bucket_states[bucket] = state
        return state

    def _settled_bucket(self, batch_size: int) -> int:
        """Require two consecutive steps before following a shrinking batch."""

        bucket = self._batch_bucket(batch_size)
        if self._active_bucket is None:
            self._active_bucket = bucket
        elif bucket == self._active_bucket:
            self._pending_bucket = None
            self._pending_bucket_steps = 0
        elif bucket == self._pending_bucket:
            self._pending_bucket_steps += 1
            if self._pending_bucket_steps >= _BUCKET_SWITCH_STEPS:
                self._active_bucket = bucket
                self._pending_bucket = None
                self._pending_bucket_steps = 0
        else:
            self._pending_bucket = bucket
            self._pending_bucket_steps = 1
        return self._active_bucket

    def _desired_k(self, state: _BucketState) -> int:
        if self.auto_tune and state.profile_k is not None:
            desired = state.profile_k
        elif self.auto_tune and self.require_profile_for_downshift:
            # AV can price each captured draft width. Keep collecting fresh
            # full-width confidences until it has made that comparison;
            # acceptance alone cannot account for an eager narrow graph.
            desired = self.max_k
        else:
            desired = state.empirical_k
        return max(self.min_k, min(desired, self.max_k))

    def _advance_state(self, state: _BucketState) -> None:
        desired = self._desired_k(state)
        if state.dwell_steps:
            state.dwell_steps -= 1
        if desired < state.stable_k:
            state.down_steps += 1
            state.up_steps = 0
            if state.down_steps >= _DOWNSHIFT_STEPS and state.dwell_steps == 0:
                state.stable_k = desired
                state.down_steps = 0
                state.dwell_steps = _MIN_K_DWELL_STEPS
        elif desired > state.stable_k:
            state.up_steps += 1
            state.down_steps = 0
            if state.up_steps >= _UPSHIFT_STEPS and state.dwell_steps == 0:
                state.stable_k = desired
                state.up_steps = 0
                state.dwell_steps = _MIN_K_DWELL_STEPS
        else:
            state.down_steps = 0
            state.up_steps = 0

    def recommend(
        self,
        batch_size: int,
        physical_k: int,
        cost_saving_k: int | None = None,
        *,
        probe_k: int | None = None,
        probe_interval: int = PHYSICAL_K_PROBE_INTERVAL,
    ) -> None:
        # Accept the legacy transport argument without allowing an independent
        # acceptance threshold to override the worker's complete utility score.
        del cost_saving_k
        physical_k = max(self.min_k, min(int(physical_k), self.max_k))
        state = self._state(self._batch_bucket(batch_size))
        state.profile_k = physical_k
        state.pending_probe_k = max(self.min_k, min(int(probe_k), self.max_k)) if probe_k is not None else None
        state.probe_interval = max(0, int(probe_interval))

    def advance_profile(self, batch_size: int) -> None:
        """Advance a profiled bucket without rebuilding per-request samples."""

        if batch_size <= 0:
            return
        state = self._state(self._batch_bucket(batch_size))
        if state.profile_k is None:
            return
        state.observations += 1
        self._advance_state(state)

    def cap(self, configured_k: int, batch_size: int | None = None) -> int:
        configured_k = max(min(int(configured_k), self.max_k), 0)
        next_bucket = self._batch_bucket(batch_size) if batch_size else None
        if next_bucket != self._last_cap_bucket:
            if (
                self._last_cap_bucket is not None
                and (previous := self._bucket_states.get(self._last_cap_bucket)) is not None
            ):
                self._cancel_probe(previous)
            self._last_cap_bucket = next_bucket
        if configured_k == 0:
            if next_bucket is not None and (state := self._bucket_states.get(next_bucket)) is not None:
                self._cancel_probe(state)
            return 0
        if batch_size is not None and batch_size < PHYSICAL_K_MIN_TUNED_BATCH_SIZE:
            return configured_k
        if batch_size:
            bucket = self._batch_bucket(batch_size)
            state = self._state(bucket)
            if state.probe_steps_remaining:
                if state.active_probe_k is not None and state.active_probe_k <= configured_k:
                    state.probe_steps_remaining -= 1
                    return state.active_probe_k
                self._cancel_probe(state)
            if state.pending_probe_k is not None:
                probe_k, state.pending_probe_k = state.pending_probe_k, None
                return self._begin_probe(state, min(probe_k, configured_k))
            if state.profile_k == state.stable_k == self.max_k:
                return self._cap_result(state, bucket, batch_size, configured_k)
            bucket = self._settled_bucket(batch_size)
            state = self._state(bucket)
            if (
                bucket == next_bucket
                and state.probe_interval
                and state.observations
                and state.observations % state.probe_interval == 0
                and state.last_probe_observation != state.observations
            ):
                state.last_probe_observation = state.observations
                return self._begin_probe(state, configured_k)
            return self._cap_result(state, bucket, batch_size, configured_k)
        return configured_k

    def _cap_result(self, state: _BucketState, bucket: int, batch_size: int, configured_k: int) -> int:
        result = min(state.stable_k, configured_k)
        if self._last_traced_cap.get(bucket) != result:
            _trace_physical_k(
                "cap bucket=%d batch=%d configured=%d profile=%s stable=%d result=%d",
                bucket,
                batch_size,
                configured_k,
                state.profile_k,
                state.stable_k,
                result,
            )
            self._last_traced_cap[bucket] = result
        return result

    def observe(
        self,
        scheduled_widths: Sequence[int],
        sampled_token_ids: Sequence[Sequence[int]],
    ) -> None:
        if len(scheduled_widths) != len(sampled_token_ids):
            return
        if self.auto_tune:
            active_batch_size = sum(int(width) > 0 for width in scheduled_widths)
            if not active_batch_size:
                return
            state = self._state(self._batch_bucket(active_batch_size))
            state.observations += 1
            if state.profile_k is not None:
                self._advance_state(state)
                return
        pairs = [(int(width), tokens) for width, tokens in zip(scheduled_widths, sampled_token_ids) if width > 0]
        if not pairs:
            return
        widths = [width for width, _ in pairs]
        if len(widths) < PHYSICAL_K_MIN_TUNED_BATCH_SIZE:
            return
        if not self.auto_tune:
            state = self._state(self._batch_bucket(len(widths)))
            state.observations += 1
        accepted = [min(width, max(len(tokens) - 1, 0)) for width, tokens in pairs]
        alpha = _ACCEPTANCE_EMA_ALPHA
        for position in range(1, self.max_k + 1):
            eligible = [index for index, width in enumerate(widths) if width >= position]
            if not eligible:
                continue
            observed = sum(accepted[index] >= position for index in eligible) / len(eligible)
            index = position - 1
            if state.survival_seen[index]:
                state.survival[index] = (1.0 - alpha) * state.survival[index] + alpha * observed
            else:
                state.survival[index] = observed
                state.survival_seen[index] = True

        # Acceptance survival is monotone by definition. Enforce that after
        # EMA updates where later positions may have fewer eligible samples.
        for index in range(1, self.max_k):
            state.survival[index] = min(state.survival[index], state.survival[index - 1])
        useful_positions = sum(
            probability >= _ACCEPTANCE_THRESHOLD
            for probability, seen in zip(state.survival, state.survival_seen)
            if seen
        )
        ordered = sorted(accepted)
        quantile_position = math.ceil(0.5 * (len(ordered) - 1))
        quantile_k = ordered[quantile_position]
        state.empirical_k = max(
            self.min_k,
            min(
                self.max_k,
                useful_positions,
                quantile_k,
            ),
        )
        self._advance_state(state)


def _create_controller(vllm_config: Any) -> AdaptiveDraftKController | None:
    if not getattr(vllm_config, "use_v2_model_runner", False):
        return None
    if not physical_k_runtime_enabled(vllm_config):
        return None
    dynamic = (getattr(vllm_config, "additional_config", None) or {}).get("dynamic_spec_config", {})
    if not isinstance(dynamic, dict):
        return None
    params = resolve_physical_k(dynamic)
    if params is None:
        return None
    max_k = int(getattr(getattr(vllm_config, "speculative_config", None), "num_speculative_tokens", 0))
    return AdaptiveDraftKController(
        max_k=max_k,
        min_k=params["min_k"],
        auto_tune=params["auto_tune"],
        require_profile_for_downshift=bool(
            getattr(vllm_config.speculative_config, "enable_adaptive_verification", False)
        ),
        probe_steps=PHYSICAL_K_PROBE_STEPS if params["auto_tune"] and params["calibration"] == "online" else 1,
    )


def _update_controller(controller, scheduler_output, model_runner_output) -> None:
    recommendation = getattr(model_runner_output, "physical_k_recommendation", None)
    scheduled = getattr(scheduler_output, "scheduled_spec_decode_tokens", None) or {}
    scheduler_batch_size = len(scheduled)
    req_ids = getattr(model_runner_output, "req_ids", ())
    if scheduler_batch_size == 1 and len(req_ids) == 1:
        controller.observe_single_request(req_ids[0])
    if recommendation is not None:
        recommended_batch_size, physical_k = recommendation
        _trace_physical_k(
            "recommend worker_batch=%d scheduler_batch=%d physical_k=%d",
            recommended_batch_size,
            scheduler_batch_size,
            physical_k,
        )
        if scheduler_batch_size and controller._batch_bucket(recommended_batch_size) != controller._batch_bucket(
            scheduler_batch_size
        ):
            # The AV worker may only have fresh draft rows for a subset of
            # scheduled requests. Never apply a K priced for that smaller
            # subset to a different scheduler batch bucket.
            controller.recommend(scheduler_batch_size, controller.max_k)
            _trace_physical_k(
                "recommend_rejected worker_bucket=%d scheduler_bucket=%d",
                controller._batch_bucket(recommended_batch_size),
                controller._batch_bucket(scheduler_batch_size),
            )
        else:
            controller.recommend(
                recommended_batch_size,
                physical_k,
                probe_k=getattr(model_runner_output, "physical_k_probe", None),
                probe_interval=getattr(model_runner_output, "physical_k_probe_interval", PHYSICAL_K_PROBE_INTERVAL),
            )
    if recommendation is None and scheduler_batch_size < PHYSICAL_K_MIN_TUNED_BATCH_SIZE:
        return
    if recommendation is None and controller.auto_tune and scheduler_batch_size:
        state = controller._state(controller._batch_bucket(scheduler_batch_size))
        if state.profile_k is not None:
            if state.profile_k != state.stable_k or state.stable_k < controller.max_k:
                controller.advance_profile(scheduler_batch_size)
            return
    sampled = getattr(model_runner_output, "sampled_token_ids", None)
    if sampled is not None and len(req_ids) == len(sampled):
        controller.observe(
            [len(scheduled.get(req_id, ())) for req_id in req_ids],
            sampled,
        )


def install_scheduler_policy() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    if not getattr(original_init, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            config = args[0] if args else kwargs.get("vllm_config", getattr(self, "vllm_config", None))
            self._ascend_physical_k_controller = _create_controller(config)

        patched_init._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler.__init__ = patched_init

    original_after = Scheduler._update_after_schedule
    if not getattr(original_after, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_after)
        def patched_after(self, scheduler_output):
            controller = getattr(self, "_ascend_physical_k_controller", None)
            if controller is not None:
                batch_size = len(scheduler_output.scheduled_spec_decode_tokens)
                scheduler_output.num_spec_tokens_to_schedule = controller.cap(
                    scheduler_output.num_spec_tokens_to_schedule, batch_size
                )
            return original_after(self, scheduler_output)

        patched_after._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler._update_after_schedule = patched_after

    original_output = Scheduler.update_from_output
    if not getattr(original_output, "_vllm_ascend_physical_k_patched", False):

        @wraps(original_output)
        def patched_output(self, scheduler_output, model_runner_output):
            outputs = original_output(self, scheduler_output, model_runner_output)
            if (controller := getattr(self, "_ascend_physical_k_controller", None)) is not None:
                _update_controller(controller, scheduler_output, model_runner_output)
            return outputs

        patched_output._vllm_ascend_physical_k_patched = True  # type: ignore[attr-defined]
        Scheduler.update_from_output = patched_output
