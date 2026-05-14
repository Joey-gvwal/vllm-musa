# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the vllm-musa sparse FlashMLA correctness provider."""

import torch


def _reference_sparse_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, q_dim = q.shape
    topk = indices.shape[-1]
    idx = indices[:, 0, :].to(torch.long)
    valid = (idx >= 0) & (idx < kv.shape[0])
    if topk_length is not None:
        valid &= torch.arange(topk).unsqueeze(0) < topk_length.to(torch.long).view(
            -1, 1
        )
    gathered = kv[:, 0, :].to(torch.float32).index_select(
        0, idx.masked_fill(~valid, 0).reshape(-1)
    )
    gathered = gathered.view(num_tokens, topk, kv.shape[-1])
    logits = torch.einsum("thd,tkd->thk", q.to(torch.float32), gathered) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(1), -float("inf"))
    no_key_mask = ~valid.any(dim=-1)
    lse = torch.logsumexp(logits, dim=-1).masked_fill(
        no_key_mask[:, None], -float("inf")
    )
    max_logits = torch.max(logits, dim=-1).values.masked_fill(
        no_key_mask[:, None], -float("inf")
    )
    lse_for_o = lse
    if attn_sink is not None:
        sink = attn_sink[:num_heads].to(torch.float32).view(1, num_heads)
        lse_for_o = torch.logaddexp(lse, sink.expand(num_tokens, -1))
    safe_lse = lse_for_o.masked_fill(torch.isneginf(lse_for_o), float("inf"))
    weights = torch.exp(
        logits - safe_lse.unsqueeze(-1)
    )
    weights = weights.masked_fill(~valid.unsqueeze(1), 0.0)
    out = torch.einsum("thk,tkd->thd", weights, gathered[:, :, :d_v])
    out = out.masked_fill(no_key_mask[:, None, None], 0.0).to(q.dtype)
    return out, max_logits, lse.masked_fill(no_key_mask[:, None], float("inf"))


def test_sparse_flashmla_provider_is_available_without_env(monkeypatch):
    monkeypatch.delenv(
        "VLLM_MUSA_ENABLE_TORCH_SPARSE_FLASHMLA_FALLBACK", raising=False
    )

    from vllm_musa.v1.attention.ops import flashmla

    monkeypatch.setattr(
        flashmla.current_platform, "get_device_capability", lambda: (3, 1)
    )

    assert (
        flashmla.flash_mla_sparse_fwd
        is not flashmla._raise_flashmla_sparse_unavailable
    )
    assert flashmla.is_flashmla_sparse_supported() == (True, None)


def test_musa_deepseek_v4_sparse_backend_uses_s5000_block_size():
    from vllm_musa.v1.attention.backends.mla.flashmla_sparse import (
        MUSADeepseekV4FlashMLASparseBackend,
    )

    assert MUSADeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes() == [
        256
    ]


def test_sparse_flashmla_prefill_provider_matches_reference_production_dims():
    from vllm_musa.v1.attention.ops import flashmla

    torch.manual_seed(0)
    q = torch.randn((2, 3, 576), dtype=torch.bfloat16)
    kv = torch.randn((5, 1, 576), dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 2, -1]], [[4, 99, 1]]], dtype=torch.int32)
    topk_length = torch.tensor([2, 1], dtype=torch.int32)
    attn_sink = torch.tensor([0.25, -0.5, 0.125], dtype=torch.float32)
    out = torch.empty((2, 3, 512), dtype=torch.bfloat16)

    actual, max_logits, lse = flashmla._torch_flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=0.125,
        d_v=512,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=out,
    )
    expected, expected_max_logits, expected_lse = _reference_sparse_mla(
        q,
        kv,
        indices,
        sm_scale=0.125,
        d_v=512,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )

    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(max_logits, expected_max_logits)
    torch.testing.assert_close(lse, expected_lse)


def test_sparse_flashmla_kvcache_provider_matches_reference_production_dims():
    from vllm_musa.v1.attention.ops import flashmla

    torch.manual_seed(1)
    q = torch.randn((1, 2, 3, 576), dtype=torch.bfloat16)
    k_cache = torch.randn((6, 1, 576), dtype=torch.bfloat16)
    indices = torch.tensor([[[0, 2, -1], [4, 99, 1]]], dtype=torch.int32)
    topk_length = torch.tensor([[2, 1]], dtype=torch.int32)
    attn_sink = torch.tensor([0.25, -0.5, 0.125], dtype=torch.float32)
    out = torch.empty((1, 2, 3, 512), dtype=torch.bfloat16)

    actual, lse = flashmla._torch_flash_mla_with_kvcache_sparse_fallback(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=torch.empty(0),
        softmax_scale=0.125,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=out,
    )
    expected, _, expected_lse = _reference_sparse_mla(
        q.reshape(2, 3, 576),
        k_cache,
        indices.reshape(2, 3).unsqueeze(1),
        sm_scale=0.125,
        d_v=512,
        topk_length=topk_length.reshape(2),
        attn_sink=attn_sink,
    )

    assert actual is out
    torch.testing.assert_close(
        actual,
        expected.reshape(1, 2, 3, 512),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(lse, expected_lse.reshape(1, 2, 3).permute(0, 2, 1))
