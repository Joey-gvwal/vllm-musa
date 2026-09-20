# SPDX-License-Identifier: Apache-2.0
"""Source-level contract for the DeepSeek-V4 FlashMLA cache pack path."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_flashmla_cache_pack_uses_one_warp_per_fp8_group():
    source = (ROOT / "csrc/musa/attention/deepseek_v4_cache_store.mu").read_text()
    kernel = source.split(
        "__global__ void deepseek_v4_qnorm_rope_kv_pack_fused_kernel", 1
    )[1]
    kernel = kernel.split("}  // namespace", 1)[0]

    assert "const int warp = tid >> 5;" in kernel
    assert "if (warp < kNopeDim / kQuantBlockSize)" in kernel
    assert "warp_reduce_max(fmaxf(fabsf(x0), fabsf(x1)))" in kernel
    assert "reinterpret_cast<uint16_t*>(token_ptr + start)[lane]" in kernel
    assert "scale_ptr[warp]" in kernel
    assert "if (warp == 7)" in kernel


def test_flashmla_cache_pack_has_no_experiment_only_dispatch_gate():
    source = (ROOT / "csrc/musa/attention/deepseek_v4_cache_store.mu").read_text()

    assert "VLLM_MUSA_DEEPSEEK_V4_QNORM_ROPE_KV_PACK_IMPL" not in source
    assert "optimized_cache_pack_enabled" not in source
    assert "kOptimizedCachePack" not in source


def test_flashmla_q_pad_is_written_in_the_fused_kernel():
    source = (ROOT / "csrc/musa/attention/deepseek_v4_cache_store.mu").read_text()
    patch = (
        ROOT / "vllm_musa/patches/series/0014-MUSA-vllm.models.deepseek_v4.attention.patch"
    ).read_text()
    wrapper = (ROOT / "vllm_musa/_custom_ops.py").read_text()

    assert "int64_t in_heads, int64_t out_heads" in source
    assert "if (head >= in_heads)" in source
    assert "q_head_padded" in source
    assert "int q_head_padded=0) -> Tensor" in (
        ROOT / "csrc/musa/torch_bindings.cpp"
    ).read_text()
    assert "q_head_padded: int = 0" in wrapper
    assert "F.pad(q," not in patch
    assert "padded_heads," in patch
