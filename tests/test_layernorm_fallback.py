# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MUSA RMSNorm fallback behavior."""

import torch


def test_musa_rmsnorm_residual_uses_native_when_vllm_fused_op_missing(monkeypatch):
    from vllm_musa.model_executor.layers import layernorm

    norm = object.__new__(layernorm.MusaRMSNorm)
    torch.nn.Module.__init__(norm)
    norm.hidden_size = 2
    norm.variance_epsilon = 1e-6
    norm.variance_size_override = None
    norm.has_weight = True
    norm.weight = torch.nn.Parameter(torch.tensor([1.25, 0.75]))
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    residual = torch.tensor([[0.5, -1.0], [1.0, -0.5]], dtype=torch.float32)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fused_add_rms_norm should not be called")

    monkeypatch.setattr(
        layernorm, "_vllm_fused_add_rms_norm_available", lambda: False
    )
    monkeypatch.setattr(layernorm, "fused_add_rms_norm", fail_if_called)

    expected = norm.forward_native(x.clone(), residual.clone())
    actual = norm.forward_oot(x.clone(), residual.clone())

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
