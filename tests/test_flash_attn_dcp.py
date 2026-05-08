# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

try:
    from vllm_musa.v1.attention.backends import flash_attn as flash_attn_module
    from vllm_musa.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadata,
    )
except ModuleNotFoundError as exc:
    pytest.skip(f"requires full vllm-musa runtime: {exc}", allow_module_level=True)


class _FakeDCPGroup:
    world_size = 2
    rank_in_group = 0

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat((tensor, tensor + 1), dim=dim)


def test_forward_with_dcp_runs_context_suffix_and_merge(monkeypatch):
    calls = []

    def fake_merge_attn_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
    ):
        assert prefix_output.shape == suffix_output.shape == output.shape
        assert prefix_lse.shape == suffix_lse.shape == (2, 3)
        output.copy_(prefix_output + suffix_output)

    def fake_flash_attn_varlen_func(**kwargs):
        q = kwargs["q"]
        calls.append(kwargs)
        out = torch.ones(
            q.shape[0], q.shape[1], q.shape[2], dtype=q.dtype, device=q.device
        )
        lse = torch.ones(q.shape[1], q.shape[0], dtype=q.dtype, device=q.device)
        return out, lse

    def fake_dcp_combine(context_out, context_lse, group, return_lse=False):
        assert return_lse is True
        assert group.world_size == 2
        return context_out[:, :2, :].contiguous(), context_lse[:, :2].contiguous()

    monkeypatch.setattr(flash_attn_module, "get_dcp_group", lambda: _FakeDCPGroup())
    monkeypatch.setattr(
        flash_attn_module, "merge_attn_states", fake_merge_attn_states
    )
    monkeypatch.setattr(
        flash_attn_module,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )

    impl = FlashAttentionImpl(
        num_heads=2,
        head_size=4,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    impl.dcp_world_size = 2
    impl.dcp_combine = fake_dcp_combine

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([4, 5], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        decode_query_start_loc=None,
        decode_seq_lens=None,
        decode_block_table=None,
        num_prefills=2,
        num_prefill_tokens=3,
        prefill_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        prefill_max_seq_len=2,
        cu_seqlens_k=None,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=4,
        dcp_context_kv_lens=torch.tensor([3, 3], dtype=torch.int32),
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=0,
        causal=True,
    )

    query = torch.zeros(3, 2, 4)
    key = torch.zeros(3, 1, 4)
    value = torch.zeros(3, 1, 4)
    key_cache = torch.zeros(2, 16, 1, 4)
    value_cache = torch.zeros(2, 16, 1, 4)
    output = torch.empty(3, 2, 4)

    impl._forward_with_dcp(
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        metadata,
    )

    assert len(calls) == 2
    assert calls[0]["causal"] is False
    assert calls[0]["seqused_k"] is metadata.dcp_context_kv_lens
    assert calls[0]["max_seqlen_k"] == metadata.max_dcp_context_kv_len
    assert calls[1]["causal"] is True
    assert calls[1]["cu_seqlens_k"] is metadata.query_start_loc
    assert torch.all(output == 2)
