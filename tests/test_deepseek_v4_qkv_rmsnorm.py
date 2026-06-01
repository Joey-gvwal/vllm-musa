from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def test_deepseek_v4_fused_qkv_rmsnorm_custom_op_is_registered():
    custom_ops = _read("vllm_musa/_custom_ops.py")
    bindings = _read("csrc/musa/torch_bindings.cpp")
    headers = _read("csrc/musa/musa_ops.h")
    setup = _read("setup.py")

    assert "def deepseek_v4_fused_q_kv_rmsnorm(" in custom_ops
    assert "torch.ops._C_musa_ops.deepseek_v4_fused_q_kv_rmsnorm" in custom_ops
    assert "deepseek_v4_fused_q_kv_rmsnorm(Tensor q" in bindings
    assert "&deepseek_v4_fused_q_kv_rmsnorm" in bindings
    assert "deepseek_v4_fused_q_kv_rmsnorm(" in headers
    assert "csrc/musa/attention/deepseek_v4_fused_qkv_rmsnorm.mu" in setup


def test_deepseek_v4_fused_qkv_rmsnorm_patch_uses_native_by_default():
    source = _read(
        "vllm_musa/patches/"
        "vllm__v1__attention__ops__deepseek_v4_ops__fused_qk_rmsnorm.patch.py"
    )

    assert "VLLM_MUSA_DEEPSEEK_V4_QKV_RMSNORM_FALLBACK" in source
    assert "return _musa_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)" in source
    assert "_musa_rmsnorm_fallback(qr, q_weight, eps)" in source
    assert "_OLD_HELPERS" in source
    assert "_OLD_FALLBACK_RETURN" in source
    assert source.index("return _musa_fused_q_kv_rmsnorm") < source.index(
        "_musa_rmsnorm_fallback(qr, q_weight, eps)"
    )
