# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import numpy as np
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.model_runner import (
    NPUModelRunner,
    _ProposalSkippingSpeculator,
)


def test_zero_k_proxy_skips_proposer_and_preserves_delegate_state() -> None:
    class _Speculator:
        draft_tokens = np.arange(12).reshape(4, 3)
        draft_logits = "verifier-state"

        def propose(self, *args, **kwargs):
            raise AssertionError("The underlying proposer must not run for K=0.")

    proxy = _ProposalSkippingSpeculator(_Speculator())

    output = proxy.propose(SimpleNamespace(num_reqs=2))

    assert output.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert not proxy.supports_mm_inputs
    assert proxy.draft_logits == "verifier-state"


def test_full_graph_padding_uses_dynamic_speculative_query_len() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.decode_query_len = 6
    query_start_loc = np.full(10, -1, dtype=np.int32)
    query_start_loc[:4] = [0, 1, 2, 3]

    padded, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=3,
        num_reqs_padded=3,
        num_reqs=3,
        query_start_loc_np=query_start_loc,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=3,
        uniform_token_count=1,
    )

    assert num_reqs_padded == 3
    assert padded[:4].tolist() == [0, 1, 2, 3]


def test_full_graph_dynamic_k_uses_one_dummy_request_for_token_padding() -> None:
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.decode_query_len = 6
    query_start_loc = np.full(10, -1, dtype=np.int32)
    query_start_loc[:4] = [0, 1, 2, 3]

    padded, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=4,
        num_reqs_padded=3,
        num_reqs=3,
        query_start_loc_np=query_start_loc,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=3,
        uniform_token_count=1,
    )

    assert num_reqs_padded == 4
    assert padded[:5].tolist() == [0, 1, 2, 3, 4]
