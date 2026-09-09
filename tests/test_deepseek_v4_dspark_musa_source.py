from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"


def _patch(name: str) -> str:
    return (SERIES / name).read_text()


def test_dspark_context_kv_uses_musa_custom_op() -> None:
    patch = _patch("0140-MUSA-vllm.models.deepseek_v4.nvidia.dspark.patch")
    assert "from vllm.platforms import current_platform" in patch
    assert "if current_platform.is_musa():" in patch
    assert "_musa_custom_ops.deepseek_v4_qnorm_rope_kv_insert(" in patch
    assert "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope" not in patch


def test_dspark_rejection_sampler_preserves_optional_feature_flags() -> None:
    patch = _patch(
        "0141-MUSA-vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils-dspark.patch"
    )
    assert "synthetic_mode = synthetic_conditional_rates is not None" in patch
    assert 'target_logits.device.type == "musa"' in patch
    assert "draft_logits = target_logits.new_empty((1, 1, 1))" in patch
    assert "SYNTHETIC_MODE=synthetic_mode" in patch
