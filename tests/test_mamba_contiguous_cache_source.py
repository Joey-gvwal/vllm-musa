# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the v0.28 page-aligned Mamba cache fix."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0135-MUSA-fix-Qwen38-Mamba-page-aligned-state-layout.patch"
)
POOL_PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0130-MUSA-restore-contiguous-segregated-Mamba-cache-pools.patch"
)


def test_page_aligned_mamba_patch_covers_both_v028_cache_builders():
    source = PATCH.read_text()

    assert "vllm/v1/worker/gpu/attn_utils.py" in source
    assert "vllm/v1/worker/gpu_model_runner.py" in source
    assert "page_size_bytes = kv_cache_spec.page_size_bytes" in source
    assert ".view(num_blocks, 1, 1, page_size_bytes)" in source


def test_mamba_bind_accepts_segregated_state_pools():
    source = POOL_PATCH.read_text()

    assert "torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]" in source
    assert "if isinstance(kv_cache, (list, tuple))" in source
    assert "self.kv_cache = tuple(kv_cache)" in source
    assert "Unexpected number of Mamba state pools" in source


def test_patch_keeps_legacy_packed_page_fallback():
    source = POOL_PATCH.read_text()

    assert "pages = kv_cache.squeeze(dim=(1, 2))" in source
    assert "Packing conv/SSM bytes into every physical page" in source
