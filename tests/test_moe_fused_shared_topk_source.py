# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unquantized_moe_uses_fused_shared_topk_extension() -> None:
    source = (
        REPO_ROOT
        / "vllm_musa/model_executor/layers/fused_moe/"
        "unquantized_fused_moe_method.py"
    ).read_text()

    assert "from vllm_musa.jit_kernel.extend_topk_shared import" in source
    assert "topk_weights, topk_ids = extend_topk_with_shared(" in source
    assert "VLLM_MUSA_MOE_FUSED_SHARED_TOPK" not in source
    assert "torch.cat([topk_weights, shared_weight]" not in source
    assert "topk_weights.shape[1] == routed_topk + 1" in source


def test_fused_shared_topk_preserves_bf16_sigmoid_rounding() -> None:
    source = (
        REPO_ROOT / "vllm_musa/jit_kernel/extend_topk_shared.py"
    ).read_text()

    assert "shared_logits_ptr.dtype.element_ty" in source


def test_fused_shared_topk_has_no_runtime_gate() -> None:
    source = (REPO_ROOT / "vllm_musa/utils/environ.py").read_text()

    assert "VLLM_MUSA_MOE_FUSED_SHARED_TOPK" not in source


def test_qwen_fold_combines_router_and_shared_gate_projection() -> None:
    patch = (
        REPO_ROOT
        / "vllm_musa/patches/series/"
        "0076-MUSA-model-fold-the-Qwen3.5-shared-expert-into-fused.patch"
    ).read_text()
    assert "num_experts=self.n_routed_experts" in patch
    assert "if self.shared_expert is None or self._musa_shared_fold" in patch
    assert "self.experts.router._musa_num_fused_shared_experts = 1" in patch


def test_musa_topk_consumes_combined_qwen_gate_output() -> None:
    router = (
        REPO_ROOT
        / "vllm_musa/model_executor/layers/fused_moe/router/"
        "grouped_topk_router.py"
    ).read_text()
    fallback = (
        REPO_ROOT
        / "vllm_musa/model_executor/layers/fused_moe/router/"
        "fused_topk_router.py"
    ).read_text()
    wrapper = (REPO_ROOT / "vllm_musa/jit_kernel/csrc/topk.py").read_text()
    kernel = (
        REPO_ROOT / "vllm_musa/jit_kernel/csrc/topk/topk_gating.mu"
    ).read_text()
    assert "num_fused_shared_experts=num_fused_shared_experts" in router
    assert "routed_experts = self.global_num_experts" in fallback
    assert "router_logits[:, :routed_experts].contiguous()" in fallback
    assert "router_logits[:, routed_experts:].contiguous()" in fallback
    assert "int(num_fused_shared_experts)" in wrapper
    assert "int num_fused_shared_experts" in kernel
    assert "num_experts == 257 && topk == 9" in kernel
    assert "topk_softmax_no_bias_renorm_warp_shared1_kernel_fixed_k" in kernel
