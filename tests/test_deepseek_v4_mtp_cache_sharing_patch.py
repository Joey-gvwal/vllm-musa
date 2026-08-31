from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0112-MUSA-share-DSV4-MTP-indexer-cache-rows.patch"
)


def _text() -> str:
    return PATCH.read_text(encoding="utf-8")


def _changed_files(text: str) -> set[str]:
    return {
        line[len("+++ b/") :]
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }


def test_mtp_indexer_gathers_cache_once_per_request() -> None:
    text = _text()

    assert "batch_size = int(getattr(batch_descriptor, \"num_reqs\", 0) or 0)" in text
    assert "next_n = rows // batch_size" in text
    assert "request_block_table = raw_block_table[::next_n]" in text
    assert "+        batch_size,\n         max_pages,\n         total_dim," in text
    assert ".reshape(batch_size, next_n * 64, 128)" in text
    assert ".reshape(\n+        rows,\n+        64," in text


def test_mtp_indexer_cache_sharing_patch_is_scoped() -> None:
    assert _changed_files(_text()) == {
        "vllm/model_executor/layers/sparse_attn_indexer.py"
    }
