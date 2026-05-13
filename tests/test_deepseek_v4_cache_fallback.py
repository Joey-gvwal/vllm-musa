# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DeepSeek-V4 MUSA cache fallback patch helpers."""

import importlib.util
from pathlib import Path

import torch


def _load_musa_combine_topk_swa_indices_fallback():
    patch_path = (
        Path(__file__).parents[1]
        / "vllm_musa"
        / "patches"
        / "vllm__v1__attention__ops__deepseek_v4_ops__cache_utils.patch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_cache_utils_patch", patch_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    patch_source = module.PATCHES[0][1]
    start = patch_source.index("def _musa_combine_topk_swa_indices_fallback")
    namespace = {
        "torch": torch,
        "_SPARSE_PREFILL_TOPK_ALIGNMENT": 128,
    }
    exec(patch_source[start:], namespace)
    return namespace["_musa_combine_topk_swa_indices_fallback"]


def _reference_combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    combined_topk = (topk + window_size + 127) // 128 * 128
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    base = int(query_start_loc[0].item())
    for batch_idx in range(seq_lens.shape[0]):
        query_start = int(query_start_loc[batch_idx].item()) - base
        query_end = int(query_start_loc[batch_idx + 1].item()) - base
        query_len = query_end - query_start
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - query_len
        gather_start = seq_len - gather_len
        req_offset = int(M) * batch_idx
        for token_idx in range(query_start, query_end):
            pos = start_pos + token_idx - query_start
            topk_len = min((pos + 1) // int(compress_ratio), int(topk))
            swa_len = min(pos + 1, int(window_size))
            if topk_len > 0:
                combined_indices[token_idx, :topk_len] = (
                    topk_indices[token_idx, :topk_len].to(torch.int32) + req_offset
                )
            if swa_len > 0:
                combined_indices[token_idx, topk_len : topk_len + swa_len] = (
                    torch.arange(
                        swa_len, device=topk_indices.device, dtype=torch.int32
                    )
                    + req_offset
                    + int(N)
                    + pos
                    - swa_len
                    + 1
                    - gather_start
                )
            combined_lens[token_idx] = topk_len + swa_len
    return combined_indices, combined_lens


def test_musa_combine_topk_swa_indices_fallback_matches_upstream_slots():
    fallback = _load_musa_combine_topk_swa_indices_fallback()
    topk_indices = torch.tensor(
        [
            [2, -1, 5, 0],
            [-1, 4, 1, 3],
            [-1, 6, 0, 2],
            [7, -1, 3, 1],
            [5, 4, -1, 0],
        ],
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([3, 5, 8], dtype=torch.int32)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    gather_lens = torch.tensor([6, 8], dtype=torch.int32)

    actual_indices, actual_lens = fallback(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=3,
        compress_ratio=2,
        topk=4,
        M=20,
        N=7,
    )
    expected_indices, expected_lens = _reference_combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        window_size=3,
        compress_ratio=2,
        topk=4,
        M=20,
        N=7,
    )

    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_lens, expected_lens)
    assert actual_indices[2, :7].tolist() == [19, 26, 20, 22, 30, 31, 32]
    assert int(actual_lens[2].item()) == 7
