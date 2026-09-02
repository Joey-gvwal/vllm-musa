from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEXER = (
    ROOT
    / "third_party"
    / "vllm"
    / "vllm"
    / "model_executor"
    / "layers"
    / "sparse_attn_indexer.py"
)
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0110-MUSA-preserve-DeepSeek-V4-MTP-verification-semantics.patch"
)


def test_dsv4_native_indexer_does_not_select_metadata_only_recent_path() -> None:
    """No-MTP DSV4 must retain learned-indexer semantics during graph capture."""
    source = INDEXER.read_text()
    patch = "\n".join(
        line[1:] if line.startswith("+") else line
        for line in PATCH.read_text().splitlines()
    )
    snippet = (
        "if self.use_musa_native_indexer:\n"
        "            return False\n"
        "        if _musa_sparse_indexer_mtp_requires_learned():\n"
        "            return False"
    )
    assert snippet in source
    assert snippet in patch
