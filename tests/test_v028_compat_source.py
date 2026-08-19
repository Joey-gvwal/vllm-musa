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
