import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_triton_gluon_is_optional_for_musa_triton_32() -> None:
    source = (
        ROOT / "third_party" / "vllm" / "vllm" / "triton_utils" / "__init__.py"
    ).read_text()
    tree = ast.parse(source)

    gluon_import_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "triton.experimental"
            for child in node.body
        )
        and any(
            isinstance(child, ast.ImportFrom)
            and child.module == "triton.language.core"
            and any(alias.name == "_aggregate" for alias in child.names)
            for child in node.body
        )
        and any(
            isinstance(handler.type, ast.Name)
            and handler.type.id == "ImportError"
            for handler in node.handlers
        )
    ]

    assert len(gluon_import_guards) == 1
    assert source.count("aggregate = TritonLanguagePlaceholder()") == 2


def test_moe_overrides_use_v028_routed_experts_api() -> None:
    fp8_source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "quantization"
        / "fp8.py"
    ).read_text()
    unquantized_source = (
        ROOT
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "fused_moe"
        / "unquantized_fused_moe_method.py"
    ).read_text()

    assert "from vllm.model_executor.layers.fused_moe import RoutedExperts" in (
        fp8_source
    )
    assert "layer: FusedMoE" not in fp8_source
    for source in (fp8_source, unquantized_source):
        assert (
            "from vllm.model_executor.layers.fused_moe.fused_moe import "
            "fused_experts"
        ) in source


def test_qwen_uniform_decode_selector_uses_v028_scheduled_tokens_array() -> None:
    source = (
        ROOT
        / "third_party"
        / "vllm"
        / "vllm"
        / "v1"
        / "worker"
        / "gpu"
        / "model_runner.py"
    ).read_text()

    assert "req_ids,\n                num_scheduled_tokens_np," in source
    assert "req_ids,\n                num_scheduled_tokens," not in source
    assert "num_scheduled_tokens_np,\n                batch_req_state.is_prefilling_np," in (
        source
    )
