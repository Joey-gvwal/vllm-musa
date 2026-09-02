from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0116-MUSA-bound-DSV4-long-prefill-indexer-logits.patch"
)


def _text() -> str:
    return PATCH.read_text(encoding="utf-8")


def _changed_files(text: str) -> set[str]:
    return {
        line[len("+++ b/") :]
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }


def _diff_lines(text: str, prefix: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith(prefix) and not line.startswith(prefix * 3)
    )


def test_long_prefill_uses_bounded_materialized_logits() -> None:
    text = _text()
    added = _diff_lines(text, "+")
    removed = _diff_lines(text, "-")

    assert "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB" in added
    assert "rows_for_budget" in added
    assert "use_bounded_long_prefill" in added
    assert "_musa_custom_ops.sparse_indexer_topk(" in added
    assert "pages_per_block = block_size // 64" in added
    assert "paged_kv_cache = kv_cache.view(-1, 64" in added
    assert "if use_bounded_long_prefill and block_size != 64" in added
    assert "tp_partition_rows = use_bounded_long_prefill" in added
    assert "tp_group.all_gather(work_output, dim=0)" in added
    assert "Using bounded MUSA DeepSeek-V4 materialized indexer prefill" in added
    assert "or (is_deepseek_v4 and int(chunk.total_seq_lens) > 4096)" in removed


def test_long_prefill_routes_before_the_4k_native_kernel_gate() -> None:
    added = _diff_lines(_text(), "+")
    removed = _diff_lines(_text(), "-")
    route = added.index("allow_deepseek_v4=True")
    provider_return = added.index("return True", route)

    assert route < provider_return
    assert "and int(chunk.total_seq_lens) > 4096" in added[:route]
    assert "if _musa_try_fill_prefill_topk_from_materialized_logits(" not in removed


def test_page64_view_is_initialized_inside_the_prefill_helper() -> None:
    text = _text()
    start = text.index("def _musa_try_fill_prefill_topk_from_materialized_logits")
    end = text.index(
        "def _musa_try_fill_prefill_topk_from_indexer_cache_native", start
    )
    helper = text[start:end]

    initialized = helper.index("paged_block_size = block_size")
    used = helper.index("context_chunk, paged_block_size", initialized)
    assert initialized < used


def test_long_prefill_patch_only_changes_sparse_indexer() -> None:
    assert _changed_files(_text()) == {
        "vllm/model_executor/layers/sparse_attn_indexer.py"
    }
