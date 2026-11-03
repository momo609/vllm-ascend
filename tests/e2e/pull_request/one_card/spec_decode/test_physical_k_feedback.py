# SPDX-License-Identifier: Apache-2.0

"""Exercise the real NPU event/copy path without loading model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.worker.v2.spec_decode.physical_k_feedback import PhysicalKFeedback


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
def test_physical_k_feedback_async_copy_and_poll():
    states = SimpleNamespace(
        req_id_to_index={"request": 0},
        num_computed_tokens_np=np.asarray([100]),
        prefill_len=SimpleNamespace(np=np.asarray([0])),
    )
    sampling = SimpleNamespace(
        temperature=SimpleNamespace(np=np.asarray([0.0])),
        top_p=SimpleNamespace(np=np.asarray([1.0])),
        top_k=SimpleNamespace(np=np.asarray([100])),
        min_p=SimpleNamespace(np=np.asarray([0.0])),
    )
    runner = SimpleNamespace(
        req_states=states,
        sampler=SimpleNamespace(sampling_states=sampling),
        speculator=SimpleNamespace(_vllm_ascend_last_runtime_k=5),
        output_copy_stream=torch.npu.Stream(),
    )
    manager = SimpleNamespace(
        req_states=states,
        num_bonus_tokens=1,
        _stale_idx=0,
        _stale_confidences=[SimpleNamespace(np=np.full((1, 5), 0.9))],
        _physical_k_draft_costs={5: {1: 1.0}},
        cost_tables=(np.ones(2), np.ones(16)),
    )
    feedback = PhysicalKFeedback(5, {"calibration": "online", "min_gain": 0.02, "probe_budget": 0.02})
    feedback.draft_widths["request"] = 5
    feedback.begin_step(runner)
    feedback.capture_forecast(manager, {"request": 6}, {"request": [1] * 5}, 4)
    output = SimpleNamespace(
        model_runner_output=SimpleNamespace(req_ids=["request"]),
        num_sampled_tokens_np=np.asarray([3]),
        sampler_output=SimpleNamespace(num_rejected=torch.tensor([1], device="npu")),
    )
    feedback.finish_step(runner, output, SimpleNamespace(has_structured_output_reqs=False))
    assert len(feedback.pending) == 1
    # Tests may wait; the production poll path must only query completion.
    feedback.pending[0].ready.synchronize()
    feedback.poll()
    assert feedback.completed == 1
    assert not feedback.pending
    assert feedback.cost.groups
    histogram = feedback.acceptance.groups[(5, (0.0, 1.0, 100, 0.0), 6)]
    assert np.all(histogram.count[:3].sum(axis=1) == 1)
    assert not histogram.count[3:].any()
