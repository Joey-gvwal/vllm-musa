# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.utils.deep_gemm.
"""

PATCHES = [
    # Patch is_deep_gemm_supported to support musa device type
    (
        "is_supported_arch = current_platform.is_cuda()",
        "is_supported_arch = current_platform.is_musa()",
    ),
    # Patch is_device_capability to support musa device type
    (
        "current_platform.is_device_capability(90)",
        "current_platform.is_device_capability(31)",
    ),
    # Patch get_mk_alignment_for_contiguous_layout to support musa deepep
    (
        "mk_align_size = _get_mk_alignment_for_contiguous_layout_impl()",
        "mk_align_size = 128",
    ),
    (
        """    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )
""",
        """    _lazy_init()
    if _tf32_hc_prenorm_gemm_impl is None:
        if (
            current_platform.is_musa()
            and os.getenv("VLLM_MUSA_ENABLE_TORCH_MHC_PRENORM_FALLBACK", "0") == "1"
        ):
            out.zero_()
            sqrsum.zero_()
            x_float = x.to(torch.float32)
            out[0].copy_(x_float @ fn.t())
            sqrsum[0].copy_(x_float.square().sum(dim=-1))
            logger.warning_once(
                "Using opt-in MUSA torch MHC prenorm fallback. This fills "
                "split 0 with the full GEMM/sqrsum result and zeros the "
                "remaining split-K partials; it is diagnostic, not a "
                "production DeepGEMM replacement."
            )
            return out
        return _missing()
    return _tf32_hc_prenorm_gemm_impl(
        x,
        fn,
        out,
        sqrsum,
        num_split,
    )
""",
    ),
]
