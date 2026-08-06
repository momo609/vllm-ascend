# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np

from vllm_ascend.worker.v2.spec_decode.dspark.adaptive_verification import (
    AscendAdaptiveVerificationManager,
    build_cost_tables_from_curves,
    select_verification_plan,
)


def test_cost_table_preserves_graph_steps_and_interpolates_tail() -> None:
    _, verify = build_cost_tables_from_curves(
        [(1, 0.5), (4, 1.0)],
        [(4, 1.0), (8, 2.0), (10, 3.0), (12, 4.0)],
        max_num_reqs=4,
        max_batch_tokens=12,
        cudagraph_limit=8,
    )

    assert np.all(verify[:5] == 1.0)
    assert np.all(verify[5:9] == 2.0)
    assert verify[9] == 3.0
    assert verify[10] == 3.0
    assert verify[11] == 3.5


def test_plan_uses_global_confidence_order_and_continuous_prefixes() -> None:
    plan = select_verification_plan(
        ["low", "high"],
        np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32),
        np.asarray([2, 2], dtype=np.int32),
        np.asarray([1, 1], dtype=np.int32),
        np.zeros(3, dtype=np.float64),
        np.asarray([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
        min_predicted_gain=0.0,
    )

    assert plan is not None
    assert plan.draft_budget == 1
    assert plan.draft_lengths == {"low": 0, "high": 1}
    assert plan.trimmed_draft_tokens == 3


def test_minimum_gain_keeps_fixed_verification() -> None:
    plan = select_verification_plan(
        ["a"],
        np.asarray([[0.9, 0.9]], dtype=np.float32),
        np.asarray([2], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.zeros(2, dtype=np.float64),
        np.ones(4, dtype=np.float64),
        min_predicted_gain=1.0,
    )

    assert plan is not None
    assert plan.draft_budget == plan.full_draft_budget == 2


class _PublishedConfidences:
    def __init__(self, values: np.ndarray):
        self.values = values

    def get_rows(self, slots: np.ndarray) -> np.ndarray:
        return self.values[slots]


def _make_manager() -> AscendAdaptiveVerificationManager:
    manager = AscendAdaptiveVerificationManager.__new__(AscendAdaptiveVerificationManager)
    manager.req_states = SimpleNamespace(req_id_to_index={"low": 0, "high": 1})
    manager.num_bonus_tokens = 1
    manager.config = SimpleNamespace(min_predicted_gain=0.0)
    manager.cost_tables = (
        np.zeros(3, dtype=np.float64),
        np.asarray([1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0]),
    )
    manager.confidences = _PublishedConfidences(np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=np.float32))
    manager.last_plan = None
    return manager


def test_compacted_scheduler_output_is_consistent_and_non_mutating() -> None:
    manager = _make_manager()
    original = SimpleNamespace(
        num_scheduled_tokens={"low": 3, "high": 3},
        total_num_scheduled_tokens=6,
        scheduled_spec_decode_tokens={
            "low": [10, 11],
            "high": [20, 21],
        },
    )

    compacted = manager.plan_scheduler_output(original)

    assert compacted is not original
    assert compacted.scheduled_spec_decode_tokens == {"high": [20]}
    assert compacted.num_scheduled_tokens == {"low": 1, "high": 2}
    assert compacted.total_num_scheduled_tokens == 3
    assert original.scheduled_spec_decode_tokens == {
        "low": [10, 11],
        "high": [20, 21],
    }
    assert original.total_num_scheduled_tokens == 6


def test_mixed_batch_falls_back_to_original_layout() -> None:
    manager = _make_manager()
    original = SimpleNamespace(
        num_scheduled_tokens={"low": 3, "prefill": 8},
        total_num_scheduled_tokens=11,
        scheduled_spec_decode_tokens={"low": [10, 11]},
    )

    assert manager.plan_scheduler_output(original) is original
