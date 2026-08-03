# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001
"""Focused allocation contract for the MUSA Qwen GDN output buffer."""

from types import SimpleNamespace

# isort: off
import torchada  # noqa: F401
import torch

# isort: on

from vllm.config import CUDAGraphMode

from vllm_musa.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn


def _patch_forward_context(
    monkeypatch,
    *,
    runtime_mode: CUDAGraphMode,
    bucket_size: int | None,
    has_attention_metadata: bool = True,
) -> None:
    batch_descriptor = (
        SimpleNamespace(num_tokens=bucket_size) if bucket_size is not None else None
    )
    forward_context = SimpleNamespace(
        attn_metadata=object() if has_attention_metadata else None,
        batch_descriptor=batch_descriptor,
        cudagraph_runtime_mode=runtime_mode,
    )
    monkeypatch.setattr(gdn, "get_forward_context", lambda: forward_context)


def _patch_allocators(monkeypatch) -> list[str]:
    calls: list[str] = []

    def fake_empty(shape, *, dtype, device):
        calls.append("empty")
        return torch.full(shape, -7, dtype=dtype, device=device)

    def fake_zeros(shape, *, dtype, device):
        calls.append("zeros")
        return torch.full(shape, 0, dtype=dtype, device=device)

    monkeypatch.setattr(gdn.torch, "empty", fake_empty)
    monkeypatch.setattr(gdn.torch, "zeros", fake_zeros)
    return calls


def test_exact_cudagraph_bucket_skips_gdn_output_zeroing(monkeypatch) -> None:
    _patch_forward_context(
        monkeypatch,
        runtime_mode=CUDAGraphMode.FULL,
        bucket_size=25,
    )
    calls = _patch_allocators(monkeypatch)

    def fail_runtime_config_lookup():
        raise AssertionError("runtime config lookup")

    monkeypatch.setattr(gdn, "get_current_vllm_config", fail_runtime_config_lookup)

    output = gdn._allocate_gdn_output(
        (25, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        capture_sizes=tuple(range(1, 65)),
    )

    assert calls == ["empty"]
    assert torch.equal(output, torch.full_like(output, -7))


def test_padded_cudagraph_bucket_zeros_only_danger_zone(monkeypatch) -> None:
    _patch_forward_context(
        monkeypatch,
        runtime_mode=CUDAGraphMode.FULL,
        bucket_size=8,
    )
    calls = _patch_allocators(monkeypatch)

    output = gdn._allocate_gdn_output(
        (8, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        capture_sizes=(1, 4, 8),
    )

    assert calls == ["empty"]
    assert torch.equal(output[:4], torch.full_like(output[:4], -7))
    assert torch.equal(output[4:], torch.zeros_like(output[4:]))


def test_eager_gdn_output_keeps_zero_initialized_contract(monkeypatch) -> None:
    _patch_forward_context(
        monkeypatch,
        runtime_mode=CUDAGraphMode.NONE,
        bucket_size=None,
    )
    calls = _patch_allocators(monkeypatch)

    output = gdn._allocate_gdn_output(
        (25, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        capture_sizes=tuple(range(1, 65)),
    )

    assert calls == ["zeros"]
    assert torch.equal(output, torch.zeros_like(output))


def test_profile_warmup_keeps_zero_initialized_contract(monkeypatch) -> None:
    _patch_forward_context(
        monkeypatch,
        runtime_mode=CUDAGraphMode.FULL,
        bucket_size=25,
        has_attention_metadata=False,
    )
    calls = _patch_allocators(monkeypatch)

    output = gdn._allocate_gdn_output(
        (25, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        capture_sizes=tuple(range(1, 65)),
    )

    assert calls == ["zeros"]
    assert torch.equal(output, torch.zeros_like(output))
