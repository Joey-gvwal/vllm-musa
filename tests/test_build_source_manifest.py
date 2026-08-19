from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_regular_cuda_view_source_is_present_for_musa_binding() -> None:
    setup_source = (ROOT / "setup.py").read_text()
    cuda_view = ROOT / "third_party" / "vllm" / "csrc" / "cuda_view.cu"
    bindings = (
        ROOT / "third_party" / "vllm" / "csrc" / "torch_bindings.cpp"
    ).read_text()

    assert '"csrc/cuda_view.cu"' in setup_source
    assert cuda_view.is_file()
    assert "get_cuda_view_from_cpu_tensor" in cuda_view.read_text()
    assert "get_cuda_view_from_cpu_tensor" in bindings


def test_regular_moe_sources_use_resolvable_stable_helper_includes() -> None:
    moe_dir = ROOT / "third_party" / "vllm" / "csrc" / "libtorch_stable" / "moe"
    for source_name in (
        "topk_softmax_kernels.cu",
        "topk_softplus_sqrt_kernels.cu",
    ):
        source = (moe_dir / source_name).read_text()
        assert '#include "../cub_helpers.h"' in source
        assert '#include "../torch_utils.h"' in source
