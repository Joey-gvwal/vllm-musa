# SPDX-License-Identifier: Apache-2.0
"""the musa_sync driver (report / verify plumbing).

Stdlib-only (musa_sync loads manifest.py + build_apply.py by file path) → runs
locally. The clone-dependent paths (`verify --target`, `rebase`, `regen`) are
exercised on the MUSA container against a real vLLM checkout; here we
cover `report`, the probe helper, and the verify row-builder against a synthetic
checkout.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location(
        "musa_sync_under_test", ROOT / "tools" / "musa_sync.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["musa_sync_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_report(ms, capsys):
    rc = ms.main(["report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "114 divergences" in out
    assert "'1': 53" in out and "'2': 24" in out and "'3': 1" in out
    assert "'4a': 2" in out and "'5': 26" in out and "'6': 8" in out


def test_report_doc(ms, capsys):
    rc = ms.main(["report", "--doc"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.lstrip().startswith("| id | cat |")
    assert "vllm__v1__spec_decode__eagle" in out


def test_probe_upstream(ms, tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "x.py").write_text("")
    assert ms._probe_upstream(tmp_path, "vllm/x.py")
    assert not ms._probe_upstream(tmp_path, "vllm/nope.py")
    assert ms._probe_upstream(tmp_path, None)  # no declared target → vacuously present


def test_read_pin(ms):
    assert ms.read_pin("VLLM_TAG") == "6d37570a1c6d8f96e9f2b2b72594ab9fd3ae0993"
    assert ms.read_pin("DOES_NOT_EXIST", "fallback") == "fallback"


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_verify_rows_synthetic(ms, tmp_path):
    # A synthetic checkout that has the cat-6 target files → those probe "present".
    clone = tmp_path / "vllm"
    clone.mkdir()
    for e in ms.manifest.object_entries():
        if e.upstream_path is None:  # torch.* cat-6 not probeable from the vLLM clone
            continue
        p = clone / e.upstream_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
    rows = ms._verify_rows(clone)
    assert len(rows) == len(ms.manifest.ENTRIES)
    cat6 = {did: status for did, cat, status, _ in rows if cat == "6"}
    assert cat6 and all(s == "present" for s in cat6.values()), cat6
    # a cat-6 entry whose target is absent must report missing-target
    missing = clone / "vllm" / "v1" / "spec_decode" / "eagle.py"
    missing.unlink()
    rows2 = ms._verify_rows(clone)
    statuses = {did: s for did, cat, s, _ in rows2 if cat == "6"}
    assert statuses["vllm__v1__spec_decode__eagle"] == "missing-target", statuses
