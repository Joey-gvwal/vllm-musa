# SPDX-License-Identifier: Apache-2.0
"""Source contracts for the v0.28 contiguous Mamba cache adaptation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0112-MUSA-restore-contiguous-segregated-Mamba-cache-pools.patch"
)


def test_contiguous_mamba_patch_keeps_v032_cache_binding() -> None:
    source = PATCH.read_text()
    allocator = (
        ROOT / "third_party/vllm/vllm/v1/worker/utils.py"
    ).read_text()

    assert "vllm/model_executor/layers/mamba/abstract.py" in source
    assert "vllm/v1/worker/gpu_model_runner.py" in source
    assert "if isinstance(kv_cache, (list, tuple))" in source
    assert "separate_musa_pools = kv_cache_config.musa_mamba_num_blocks is not None" in allocator
    assert "torch.zeros(tensor.size, dtype=torch.int8, device=device)" in allocator


def test_mamba_bind_accepts_segregated_state_pools():
    source = PATCH.read_text()

    assert "torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]" in source
    assert "if isinstance(kv_cache, (list, tuple))" in source
    assert "self.kv_cache = tuple(kv_cache)" in source
    assert "Unexpected number of Mamba state pools" in source


def test_patch_keeps_legacy_packed_page_fallback():
    source = PATCH.read_text()

    assert "pages = kv_cache.squeeze(dim=(1, 2))" in source
    assert "state = pages[:, offset : offset + nbytes].view(dtype)" in (
        ROOT / "third_party/vllm/vllm/model_executor/layers/mamba/abstract.py"
    ).read_text()
