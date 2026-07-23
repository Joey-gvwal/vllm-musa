from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0088-MUSA-DeepSeek-V4-enable-aux-overlap.patch"
)
MODEL = ROOT / "third_party/vllm/vllm/models/deepseek_v4/nvidia/model.py"


def test_deepseek_v4_aux_overlap_is_not_runtime_env_gated() -> None:
    patch = PATCH.read_text()
    source = MODEL.read_text()

    assert not any(
        line.startswith("+")
        and "VLLM_MUSA_DEEPSEEK_V4_DISABLE_AUX_OVERLAP" in line
        for line in patch.splitlines()
    )
    assert "VLLM_MUSA_DEEPSEEK_V4_DISABLE_AUX_OVERLAP" not in source
    assert "_musa_deepseek_v4_disable_aux_overlap" not in source
    assert "current_platform.is_rocm()" in source
    assert "current_platform.is_xpu()" in source
    assert "else [torch.cuda.Stream() for _ in range(3)]" in source
