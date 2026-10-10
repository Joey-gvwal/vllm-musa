from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa"
    / "patches"
    / "series"
    / "0181-Bugfix-Mooncake-Report-failed-remote-KV-loads-to-the.patch"
)
UPSTREAM_COMMIT = "2c7ee87223f0c94f614a0b8600f6f193e6e4388f"


def _text() -> str:
    return PATCH.read_text(encoding="utf-8")


def _changed_files(text: str) -> set[str]:
    return {
        line[len("+++ b/") :].split("\t", 1)[0]
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }


def _diff_lines(text: str, prefix: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith(prefix) and not line.startswith(prefix * 3)
    )


def test_patch_only_touches_mooncake_connector() -> None:
    assert _changed_files(_text()) == {
        "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
    }


def test_patch_is_the_upstream_commit() -> None:
    assert f"(cherry picked from commit {UPSTREAM_COMMIT})" in _text()


def test_failed_pulls_are_reported_to_the_scheduler() -> None:
    text = _text()
    added = _diff_lines(text, "+")
    removed = _diff_lines(text, "-")

    # A failed pull finishes receiving with its blocks marked as load errors,
    # so the scheduler fails or recomputes the request and frees its blocks.
    assert "def _handle_failed_recv(" in added
    assert "self._invalid_block_ids.put(invalid)" in added
    assert "self.finished_recving_reqs.add(pull_meta.d_req_id)" in added
    assert "def get_block_ids_with_load_errors(self) -> set[int]:" in added
    assert "return self.connector_worker.get_block_ids_with_load_errors()" in added

    # Every receiver failure path reports instead of only logging.
    assert 'self._handle_failed_recv(pull_metas, req_ids, f"transfer failed: {e}")' in added
    assert 'response.err_msg or "transfer error"' in added
    assert 'pull_metas, response.err_reqs, response.err_msg or "unknown error"' in added
    assert "remote engine_id {remote_engine_id} not found from bootstrap" in added
    assert 'logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)' in removed


def test_success_after_a_failure_is_not_counted() -> None:
    added = _diff_lines(_text(), "+")

    assert "failed: bool = False" in added
    assert "if pull_meta.failed:" in added
