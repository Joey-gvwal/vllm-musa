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
