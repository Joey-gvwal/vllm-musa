from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0138-MUSA-keep-DSV4-learned-indexer-on-graph-capture.patch"
)


def test_dsv4_native_indexer_does_not_select_metadata_only_recent_path() -> None:
    """No-MTP DSV4 must retain learned-indexer semantics during graph capture."""
    patch = PATCH.read_text()
    snippet = (
        "if self.use_musa_native_indexer:\n"
        "+            return False"
    )
    assert snippet in patch
    assert "The recent table is metadata-only" in patch
