# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def test_deepgemm_post_process_upcasts_e8m0_scales_when_disabled():
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable")

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    weight = torch.empty((2, 2), dtype=torch.uint8)
    scale_bytes = torch.tensor([126, 127, 128, 129], dtype=torch.uint8)
    scales = scale_bytes.view(e8m0_dtype).reshape(2, 2)

    out_weight, out_scales = deepgemm_post_process_fp8_weight_block(
        weight,
        scales,
        quant_block_shape=(128, 128),
        use_e8m0=False,
    )

    assert out_weight is weight
    expected = (scale_bytes.to(torch.int32) << 23).view(torch.float32).reshape(2, 2)
    assert out_scales.dtype == torch.float32
    torch.testing.assert_close(out_scales, expected)


def test_deepgemm_post_process_keeps_e8m0_scales_when_enabled():
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if e8m0_dtype is None:
        pytest.skip("torch.float8_e8m0fnu is unavailable")

    from vllm_musa.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )

    weight = torch.empty((2, 2), dtype=torch.uint8)
    scales = torch.tensor([126, 127, 128, 129], dtype=torch.uint8).view(
        e8m0_dtype
    )

    out_weight, out_scales = deepgemm_post_process_fp8_weight_block(
        weight,
        scales,
        quant_block_shape=(128, 128),
        use_e8m0=True,
    )

    assert out_weight is weight
    assert out_scales is scales
