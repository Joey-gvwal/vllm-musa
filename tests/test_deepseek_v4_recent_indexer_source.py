"""Source contracts for the DSV4 learned-only sparse indexer."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa/patches/series"
METADATA_PATCH = SERIES / "0145-MUSA-remove-DSV4-recent-indexer-metadata.patch"
DISPATCH_PATCH = SERIES / "0146-MUSA-keep-DSV4-learned-indexer-dispatch.patch"
GRAPH_PATCH = SERIES / "0147-MUSA-keep-DSV4-learned-indexer-on-graph-decode.patch"


def _additions(patch: Path) -> str:
    return "\n".join(
        line[1:] for line in patch.read_text().splitlines() if line.startswith("+")
    )


def test_recent_indexer_metadata_is_removed() -> None:
    patch = METADATA_PATCH.read_text()

    assert "-    recent_indices: torch.Tensor | None = None" in patch
    assert "-    recent_indices_ready: bool = False" in patch
    assert "-        self.recent_indices_buffer: torch.Tensor | None = None" in patch
    assert "-    def _build_recent_indices(" in patch
    additions = _additions(METADATA_PATCH)
    assert "recent_indices" not in additions
    assert "_build_recent_indices" not in additions


def test_attention_dispatch_keeps_learned_indexer_only() -> None:
    patch = DISPATCH_PATCH.read_text()
    additions = _additions(DISPATCH_PATCH)

    assert "skip_indexer_weights=use_musa_recent_indexer" in patch
    assert "lambda: indexer.forward_musa_recent(" in patch
    assert "def forward_musa_recent(" in patch
    assert "skip_indexer_weights" not in additions
    assert "forward_musa_recent" not in additions
    assert "uses_musa_recent_indices" not in additions
    assert "self._run_parallel_input_projections(hidden_states)" in additions
    assert "aux_fns[1] = indexer_weights_proj" in additions
    assert "lambda: indexer(" in additions


def test_graph_decode_uses_learned_indexer_not_recent_window() -> None:
    patch = GRAPH_PATCH.read_text()
    additions = _additions(GRAPH_PATCH)

    assert "_musa_fill_recent_sparse_indexer_indices" in patch
    assert "def uses_musa_recent_indices(self) -> bool:" in patch
    assert "def forward_musa_recent(self, hidden_states: torch.Tensor)" in patch
    assert "_musa_fill_recent_sparse_indexer_indices" not in additions
    assert "uses_musa_recent_indices" not in additions
    assert "forward_musa_recent" not in additions
    assert "return _musa_fill_exact_sparse_indexer_indices_capture(" in additions
    assert "def _musa_decode_request_max_seq_len(decode_metadata) -> int:" in additions
    assert "_musa_decode_request_max_seq_len(decode_metadata) > 4096" in additions
    assert "block_table.shape[1] * kv_cache.shape[1]" not in additions
    assert "does not support prefill" not in additions
    assert "MUSA learned-indexer capture supports the FP8 path only" in additions
