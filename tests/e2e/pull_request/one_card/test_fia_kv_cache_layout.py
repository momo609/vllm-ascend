# SPDX-License-Identifier: Apache-2.0

"""Validate split K/V allocation against FIA and the original paged layout."""

import math

import pytest
import torch
import torch_npu

from vllm_ascend.worker.v2.attn_utils import _reshape_combined_attention_kv_cache


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
@pytest.mark.parametrize(
    "dtype,block_size,head_size",
    [(torch.float16, 128, 64), (torch.bfloat16, 256, 128)],
)
def test_split_fia_cache_matches_paged_cache_after_updates(dtype, block_size, head_size):
    generator = torch.Generator().manual_seed(37)
    shape = (4, block_size, 1, head_size)
    initial_key = torch.randn(shape, generator=generator).to(dtype).to("npu")
    initial_value = torch.randn(shape, generator=generator).to(dtype).to("npu")
    query = torch.randn((2, 2, head_size), generator=generator).to(dtype).to("npu")
    page_bytes = 2 * block_size * head_size * initial_key.element_size()
    raw = torch.zeros(4 * page_bytes, dtype=torch.uint8, device="npu")
    strided_key, strided_value = _reshape_combined_attention_kv_cache(raw, (2, *shape), dtype, page_bytes)
    dense_key, dense_value = torch.empty_like(initial_key), torch.empty_like(initial_value)
    strided_key.copy_(initial_key)
    strided_value.copy_(initial_value)
    dense_key.copy_(initial_key)
    dense_value.copy_(initial_value)
    assert not strided_key.is_contiguous()
    assert dense_key.is_contiguous() and dense_value.is_contiguous()

    block_ids = [[2, 0], [3, 1]]
    lengths = [block_size + 5, block_size + 13]
    block_table = torch.tensor(block_ids, dtype=torch.int32, device="npu")

    def attention(key, value):
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key.view(4, block_size, head_size),
            value.view(4, block_size, head_size),
            num_heads=2,
            num_key_value_heads=1,
            input_layout="TND",
            scale=1 / math.sqrt(head_size),
            block_table=block_table,
            block_size=block_size,
            actual_seq_lengths=[1, 2],
            actual_seq_lengths_kv=lengths,
            sparse_mode=0,
        )
        return output.cpu().float()

    before_update = attention(dense_key, dense_value)
    for updated in (False, True):
        if updated:
            strided_value[3].fill_(7)
            dense_value[3].fill_(7)
        actual = attention(dense_key, dense_value)
        original = attention(strided_key, strided_value)
        torch.testing.assert_close(actual, original, rtol=0.01, atol=0.01)
        key_cpu, value_cpu, query_cpu = dense_key.cpu().float(), dense_value.cpu().float(), query.cpu().float()
        expected = []
        for index, (ids, length) in enumerate(zip(block_ids, lengths)):
            key = key_cpu[ids].reshape(-1, head_size)[:length]
            value = value_cpu[ids].reshape(-1, head_size)[:length]
            probabilities = (query_cpu[index] @ key.T / math.sqrt(head_size)).softmax(dim=-1)
            expected.append(probabilities @ value)
        torch.testing.assert_close(actual, torch.stack(expected), rtol=0.02, atol=0.02)
        if updated:
            assert not torch.allclose(actual, before_update)
