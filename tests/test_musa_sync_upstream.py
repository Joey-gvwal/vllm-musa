"""series/upstream/: upstream vLLM commits the pin predates, applied after the
MUSA entries. Covers regen's split, the series gate, the upgrade plan,
``backport`` and ``rebase --drop-upstreamed`` against small synthetic vLLM
histories."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location(
        "musa_sync_upstream_under_test", ROOT / "tools" / "musa_sync.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["musa_sync_upstream_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_EDITOR": "true"},
    ).stdout.strip()


def _new_repo(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    _git(path, "config", "user.email", "musa-sync@example.invalid")
    _git(path, "config", "user.name", "musa-sync test")
    _git(path, "checkout", "--quiet", "-b", "main")


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).write_text(text)
        _git(repo, "add", name)
    _git(repo, "commit", "--quiet", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _text(c: str, changes: dict[int, str] | None = None) -> str:
    lines = [f"{c}{i}" for i in range(1, 7)]
    for n, new in (changes or {}).items():
        lines[n - 1] = new
    return "".join(f"{line}\n" for line in lines)


BASE = {f"{c}.txt": _text(c) for c in "abcdefghi"}


def _line2(c: str, new: str) -> dict[str, str]:
    return {f"{c}.txt": _text(c, {2: new})}


def _edit(repo: Path, c: str, n: int, new: str) -> dict[str, str]:
    """``c.txt`` as it is in ``repo``, with line ``n`` replaced."""
    lines = (repo / f"{c}.txt").read_text().splitlines()
    lines[n - 1] = new
    return {f"{c}.txt": "".join(f"{line}\n" for line in lines)}


@pytest.fixture
def upstream(tmp_path: Path) -> SimpleNamespace:
    """pin -> main with fixes, reverts in both styles and a re-land; a release
    branch from the pin with re-committed fixes; a commit from elsewhere."""
    up = tmp_path / "upstream"
    _new_repo(up)
    pin = _commit(up, BASE, "v1.0")
    fix_a = _commit(up, _line2("a", "A2"), "[Bugfix] fix a (#101)")
    fix_b = _commit(up, _line2("b", "B2"), "[Core] fix b (#102)")
    feat = _commit(up, _line2("d", "D2"), "[Feature] d (#104)")
    # vLLM squash-merges a revert PR without "This reverts commit".
    _commit(up, {"d.txt": BASE["d.txt"]}, 'Revert "[Feature] d" (#105)')
    _commit(up, _line2("e", "E2"), "[Misc] tidy e")
    fix_f = _commit(up, _line2("f", "F2"), "[Bugfix] fix f (#106)")
    fix_g = _commit(up, _line2("g", "G2"), "[Bugfix] fix g (#107)")
    feat_h = _commit(up, _line2("h", "H2"), "[Feature] h (#109)")
    _commit(
        up,
        {"h.txt": BASE["h.txt"]},
        f'Revert "[Feature] h (#109)"\n\nThis reverts commit {feat_h}.',
    )
    feat_i = _commit(up, _line2("i", "I2"), "[Feature] i (#110)")
    _commit(up, {"i.txt": BASE["i.txt"]}, 'Revert "[Feature] i" (#111)')
    _commit(up, _line2("i", "I2"), 'Reapply "[Feature] i" (#112)')
    main = _git(up, "rev-parse", "HEAD")
    _git(up, "checkout", "--quiet", "-b", "release", pin)
    _git(up, "cherry-pick", fix_b)  # same subject, new sha
    release_b = _git(up, "rev-parse", "HEAD")
    _commit(up, _line2("f", "F2"), "Backport the f fix")  # same change, new subject
    _git(up, "cherry-pick", "-x", fix_g)  # names its main commit
    release = _git(up, "rev-parse", "HEAD")
    _git(up, "checkout", "--quiet", "main")
    # A commit this history never saw, making the same change as "tidy e".
    elsewhere = tmp_path / "elsewhere"
    _new_repo(elsewhere)
    _commit(elsewhere, BASE, "v1.0")
    tidy_e2 = _commit(elsewhere, _line2("e", "E2"), "[Misc] e2 (#200)")
    return SimpleNamespace(
        path=up,
        elsewhere=elsewhere,
        pin=pin,
        main=main,
        release=release,
        release_b=release_b,
        release_g=release,
        fix_a=fix_a,
        fix_b=fix_b,
        feat=feat,
        fix_f=fix_f,
        fix_g=fix_g,
        feat_h=feat_h,
        feat_i=feat_i,
        tidy_e2=tidy_e2,
    )


@pytest.fixture
def work(tmp_path: Path, upstream: SimpleNamespace, ms, monkeypatch) -> Path:
    """A third_party/vllm-like clone at the pin, with the series paths of a
    project in tmp_path."""
    clone = tmp_path / "work"
    subprocess.run(
        ["git", "clone", "--quiet", str(upstream.path), str(clone)], check=True
    )
    _git(clone, "config", "user.email", "musa-sync@example.invalid")
    _git(clone, "config", "user.name", "musa-sync test")
    _git(clone, "checkout", "--quiet", "--detach", upstream.pin)
    _git(clone, "fetch", "--quiet", str(upstream.elsewhere), upstream.tidy_e2)
    pins = tmp_path / "PINS"
    pins.write_text(f"VLLM_COMMIT={upstream.pin}\n")
    monkeypatch.setattr(ms, "WORKDIR", clone)
    monkeypatch.setattr(ms, "SERIES_DIR", tmp_path / "series")
    monkeypatch.setattr(ms, "PINS", pins)
    monkeypatch.setattr(
        ms.manifest, "series_apply_order", lambda phase=None: ms._series_dir_paths()[0]
    )
    return clone


def _pick(clone: Path, sha: str) -> None:
    _git(clone, "cherry-pick", "-x", sha)


def _regen(ms) -> None:
    assert ms.cmd_regen(argparse.Namespace(area=None)) == 0


def _entries(ms) -> list[str]:
    return [ms.build_apply.entry_name(p) for p in ms._series_dir_paths()[0]]


def _plan(ms, upstream, target: str) -> dict[str, tuple[str, str]]:
    rows, _ = ms._upgrade_plan_rows(upstream.path, target, upstream.pin)
    return {name: (verdict, reason) for name, _, verdict, reason in rows}


def _backport(ms, commit, **kw) -> int:
    args = dict(commit=commit, source=None, no_regen=False)
    args.update(kw)
    return ms.cmd_backport(argparse.Namespace(**args))


def _rebase(ms, tag, history=None, drop=False) -> int:
    args = argparse.Namespace(
        tag=tag, drop_upstreamed=drop, history=history, fetch=False
    )
    return ms.cmd_rebase(args)


def _set_pin(ms, sha: str) -> None:
    ms.PINS.write_text(f"VLLM_COMMIT={sha}\n")


SERIES = [
    "0001-MUSA-c.patch",
    "0002-MUSA-a5.patch",
    "upstream/0001-Bugfix-fix-a-101.patch",
    "upstream/0002-Core-fix-b-102.patch",
    "upstream/0003-Feature-d-104.patch",
    "upstream/0004-Bugfix-fix-f-106.patch",
    "upstream/0005-Misc-e2-200.patch",
]


@pytest.fixture
def series(ms, work: Path, upstream: SimpleNamespace) -> Path:
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _commit(work, _edit(work, "a", 5, "a5-musa"), "MUSA: a5")
    for sha in (
        upstream.fix_a,
        upstream.fix_b,
        upstream.feat,
        upstream.fix_f,
        upstream.tidy_e2,
    ):
        _pick(work, sha)
    _regen(ms)
    return ms.SERIES_DIR


def _snapshot(series: Path) -> dict[str, bytes]:
    return {
        p.relative_to(series).as_posix(): p.read_bytes()
        for p in series.rglob("*.patch")
    }


def _statuses(ms) -> dict[str, str]:
    return {name: status for name, status, _ in ms._series_format_rows()}


# ------------------------------------------------------------------ layout


def test_regen_splits_musa_patches_from_upstream_commits(ms, series, capsys):
    assert _entries(ms) == SERIES
    assert set(_statuses(ms).values()) == {"clean"}
    assert ms.main(["check-series"]) == 0
    out = capsys.readouterr().out
    assert (
        "--- numbering: 2 entries are unique and contiguous 0001..0002; "
        "upstream/ 5 entries are unique and contiguous 0001..0005 ---"
    ) in out


def test_index_lines_do_not_depend_on_the_object_count(ms, series):
    lines = [
        line
        for p in series.rglob("*.patch")
        for line in p.read_text().splitlines()
        if line.startswith("index ")
    ]
    assert lines
    assert all(re.match(r"index [0-9a-f]{10}\.\.[0-9a-f]{10} ", line) for line in lines)


def test_the_build_applies_musa_entries_first(ms, series, work, upstream, tmp_path):
    order = [ms.build_apply.entry_name(p) for p in ms.build_apply.series_files(series)]
    assert order == SERIES
    fresh = tmp_path / "fresh"
    subprocess.run(
        ["git", "clone", "--quiet", str(upstream.path), str(fresh)], check=True
    )
    _git(fresh, "checkout", "--quiet", "--detach", upstream.pin)
    results = ms.build_apply.apply_patch_series(fresh, series, strict=True)
    assert results == [(name, "applied") for name in SERIES]
    for name in BASE:
        assert (fresh / name).read_text() == (work / name).read_text(), name


def test_round_trip_and_replay(ms, series, work, upstream, tmp_path):
    rows = ms._series_format_rows(work, round_trip=True)
    assert {status for _, status, _ in rows} == {"clean"}
    at_pin = tmp_path / "at-pin"
    subprocess.run(
        ["git", "clone", "--quiet", str(upstream.path), str(at_pin)], check=True
    )
    _git(at_pin, "checkout", "--quiet", "--detach", upstream.pin)
    rows = ms._series_format_rows(at_pin, replay=True)
    assert {status for _, status, _ in rows} == {"clean"}


def test_rebase_keeps_upstream_subjects_so_regen_is_a_fixed_point(
    ms, series, work, upstream
):
    before = _snapshot(series)
    assert _rebase(ms, None) == 0
    subjects = _git(work, "log", "--format=%s", f"{upstream.pin}..HEAD").splitlines()
    assert "[Core] fix b (#102)" in subjects  # `git am` would drop "[Core]"
    _regen(ms)
    assert _snapshot(series) == before


def test_a_musa_commit_made_on_top_stays_a_musa_entry(ms, series, work, capsys):
    before = _snapshot(series)
    _commit(work, _edit(work, "g", 5, "g5-musa"), "MUSA: g")
    _regen(ms)
    assert "note:" not in capsys.readouterr().out
    assert _entries(ms) == SERIES[:2] + ["0003-MUSA-g.patch"] + SERIES[2:]
    after = _snapshot(series)
    assert {k: v for k, v in after.items() if k != "0003-MUSA-g.patch"} == before
    rows = ms._series_format_rows(work, round_trip=True)
    assert {status for _, status, _ in rows} == {"clean"}


def test_a_musa_fix_to_an_upstream_commit_stays_after_it(
    ms, series, work, upstream, capsys
):
    _commit(work, _edit(work, "a", 2, "A2-musa"), "MUSA: fix a")
    _regen(ms)
    out = capsys.readouterr().out
    assert "note: upstream/0006-MUSA-fix-a.patch changes a file" in out
    assert _entries(ms) == SERIES + ["upstream/0006-MUSA-fix-a.patch"]
    assert set(_statuses(ms).values()) == {"clean"}
    plan = _plan(ms, upstream, upstream.pin)
    assert plan["upstream/0006-MUSA-fix-a.patch"] == ("keep", "")
    verdict, reason = _plan(ms, upstream, upstream.main)[
        "upstream/0006-MUSA-fix-a.patch"
    ]
    assert verdict == "review" and "a MUSA change on top of upstream/0001" in reason


def test_a_musa_fix_is_reviewed_when_any_upstream_commit_before_it_goes(
    ms, work, upstream
):
    _pick(work, upstream.fix_a)
    _pick(work, upstream.fix_g)
    _commit(work, _edit(work, "a", 2, "A2-musa"), "MUSA: fix a")
    _regen(ms)
    plan = _plan(ms, upstream, upstream.pin)
    assert plan["upstream/0003-MUSA-fix-a.patch"] == ("keep", "")
    # The release carries fix g only. The fix is for fix a, but which upstream
    # commit it serves is not recorded, so it is reviewed all the same.
    plan = _plan(ms, upstream, upstream.release)
    assert plan["upstream/0001-Bugfix-fix-a-101.patch"][0] == "keep"
    verdict, reason = plan["upstream/0003-MUSA-fix-a.patch"]
    assert verdict == "review" and "upstream/0002-Bugfix-fix-g-107.patch" in reason


# -------------------------------------------------------------------- gate


def test_upstream_must_start_with_a_cherry_pick(ms, work):
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _regen(ms)
    (ms.SERIES_DIR / "upstream").mkdir()
    (ms.SERIES_DIR / "0001-MUSA-c.patch").rename(
        ms.SERIES_DIR / "upstream" / "0001-MUSA-c.patch"
    )
    assert _statuses(ms) == {"upstream/0001-MUSA-c.patch": "not-a-cherry-pick"}


def test_a_musa_entry_must_not_be_a_cherry_pick(ms, series):
    (series / "upstream" / "0005-Misc-e2-200.patch").rename(
        series / "0003-Misc-e2-200.patch"
    )
    assert _statuses(ms)["0003-Misc-e2-200.patch"] == "misplaced-cherry-pick"


def test_numbering_is_checked_per_directory(ms, series):
    up = series / "upstream"
    (up / "0005-Misc-e2-200.patch").rename(up / "0007-Misc-e2-200.patch")
    rows = ms._series_format_rows()
    numbering = ms._numbering_rows(ms._entry_file_rows(rows))
    assert [(name, status) for name, status, _ in numbering] == [
        ("series/upstream numbering", "non-contiguous")
    ]


def test_parse_upstream_pick(ms):
    mailbox = (
        "From 0000000000000000000000000000000000000000 Mon Sep 17 00:00:00 2001\n"
        "From: musa <musa@local>\n"
        "Subject: [PATCH] [Core] a long subject that git folds onto a\n"
        " second line (#102)\n\n"
        "Body.\n\n"
        f"(cherry picked from commit {'a' * 40})\r\n"
        f"(cherry picked from commit {'b' * 40})\n"
        f"(cherry picked from commit {'a' * 40})\n"
        "---\n x | 1 +\n"
    )
    pick = ms.manifest.parse_upstream_pick(mailbox.encode())
    assert pick.commits == ("a" * 40, "b" * 40)
    assert pick.pr == "102"
    garbage = ms.manifest.parse_upstream_pick(b"\xff\xfe garbage")
    assert garbage == ms.manifest.UpstreamPick()


# -------------------------------------------------------------------- plan


def test_history_signals(ms, upstream):
    history = ms._UpstreamHistory(upstream.path, upstream.main, upstream.pin)

    def state(sha):
        return history.ref_state(ms._Ref("commit", sha))[0]

    assert state(upstream.fix_a) == "in"
    assert state(upstream.feat) == "reverted"  # vLLM-style revert
    assert state(upstream.feat_h) == "reverted"  # git-style revert
    assert state(upstream.feat_i) == "in"  # reverted, then re-landed


def test_plan_against_main(ms, series, upstream):
    plan = _plan(ms, upstream, upstream.main)
    for name in (
        "upstream/0001-Bugfix-fix-a-101.patch",
        "upstream/0002-Core-fix-b-102.patch",
        "upstream/0004-Bugfix-fix-f-106.patch",
    ):
        verdict, reason = plan[name]
        assert verdict == "drop" and "is in the target history" in reason, name
    verdict, reason = plan["upstream/0003-Feature-d-104.patch"]
    assert verdict == "review" and "reverted by" in reason
    # Not in this history at all, but the target already has the change.
    assert plan["upstream/0005-Misc-e2-200.patch"] == (
        "drop",
        "the change is already in the target",
    )
    assert plan["0001-MUSA-c.patch"][0] == plan["0002-MUSA-a5.patch"][0] == "keep"


def test_plan_against_a_release_branch(ms, series, upstream):
    plan = _plan(ms, upstream, upstream.release)
    assert plan["upstream/0001-Bugfix-fix-a-101.patch"][0] == "keep"
    verdict, reason = plan["upstream/0002-Core-fix-b-102.patch"]
    assert verdict == "drop" and reason == f"#102 merged as {upstream.release_b[:12]}"
    assert plan["upstream/0003-Feature-d-104.patch"][0] == "keep"
    verdict, reason = plan["upstream/0004-Bugfix-fix-f-106.patch"]
    assert verdict == "drop" and "(patch-id)" in reason
    assert plan["upstream/0005-Misc-e2-200.patch"][0] == "keep"


def test_plan_on_the_current_pin_keeps_everything(ms, series, upstream):
    plan = _plan(ms, upstream, upstream.pin)
    assert {verdict for verdict, _ in plan.values()} == {"keep"}


def test_a_changed_backport_is_reviewed(ms, work, upstream):
    _pick(work, upstream.fix_a)
    (work / "a.txt").write_text(_text("a", {2: "A2", 5: "a5-local"}))
    _git(work, "commit", "--quiet", "--all", "--amend", "--no-edit")
    _regen(ms)
    ((verdict, reason),) = _plan(ms, upstream, upstream.main).values()
    assert verdict == "review" and "differs from its upstream commit" in reason


@pytest.mark.parametrize(
    "upstream_entry, state, content, verbatim, verdict",
    [
        (True, ("in", "ev"), None, True, "drop"),
        (True, ("in", "ev"), "conflict", True, "drop"),
        (True, ("in", "ev"), None, False, "review"),
        (True, ("in", "ev"), None, None, "review"),
        (True, ("in", "ev"), "applied", True, "review"),
        (True, ("out", ""), "already-applied", False, "drop"),
        (True, ("reverted", "ev"), "conflict", True, "review"),
        (True, ("unknown", "ev"), "conflict", None, "review"),
        (True, ("unknown", "ev"), "applied", None, "keep"),
        (True, ("out", ""), "applied", None, "keep"),
        (False, ("out", ""), "already-applied", None, "review"),
        (False, ("out", ""), "applied", None, "keep"),
    ],
)
def test_plan_verdicts(ms, upstream_entry, state, content, verbatim, verdict):
    assert ms._plan_verdict(upstream_entry, state, content, verbatim)[0] == verdict


def test_plan_refuses_an_entry_that_is_not_a_regular_file(ms, series, upstream):
    os.mkfifo(series / "0003-MUSA-fifo.patch")
    with pytest.raises(ms.RepoUnusable, match="not a regular file"):
        ms._upgrade_plan_rows(upstream.path, upstream.main, upstream.pin)


def test_rebase_drop_upstreamed_then_regen_on_the_new_pin(
    ms, series, work, upstream, capsys
):
    musa_c = (series / "0001-MUSA-c.patch").read_bytes()
    assert _rebase(ms, "main", history=str(upstream.path), drop=True) == 0
    out = capsys.readouterr().out
    assert "dropped 4 already upstream" in out
    assert "Set VLLM_COMMIT and VLLM_TAG in third_party/PINS" in out
    subjects = _git(work, "log", "--format=%s", f"{upstream.main}..HEAD").splitlines()
    assert subjects == ["[Feature] d (#104)", "MUSA: a5", "MUSA: c"]
    # PINS still names the old pin, so regen must not run.
    before = _snapshot(series)
    assert ms.cmd_regen(argparse.Namespace(area=None)) == 1
    assert "holds the series on" in capsys.readouterr().out
    assert _snapshot(series) == before
    _set_pin(ms, upstream.main)
    _regen(ms)
    assert _entries(ms) == [
        "0001-MUSA-c.patch",
        "0002-MUSA-a5.patch",
        "upstream/0001-Feature-d-104.patch",
    ]
    assert (series / "0001-MUSA-c.patch").read_bytes() == musa_c


# ---------------------------------------------------------------- backport


def test_backport_appends_without_touching_musa_entries(ms, work, upstream, capsys):
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _pick(work, upstream.fix_a)
    _regen(ms)
    musa_c = (ms.SERIES_DIR / "0001-MUSA-c.patch").read_bytes()
    assert _backport(ms, upstream.fix_b, source=str(upstream.path)) == 0
    assert (
        f"backported {upstream.fix_b[:12]} as upstream/0002" in capsys.readouterr().out
    )
    assert _entries(ms) == [
        "0001-MUSA-c.patch",
        "upstream/0001-Bugfix-fix-a-101.patch",
        "upstream/0002-Core-fix-b-102.patch",
    ]
    assert (ms.SERIES_DIR / "0001-MUSA-c.patch").read_bytes() == musa_c
    rows = ms._series_format_rows(work, round_trip=True)
    assert {status for _, status, _ in rows} == {"clean"}


def test_backport_after_a_musa_commit_made_on_top(ms, work, upstream):
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _pick(work, upstream.fix_a)
    _regen(ms)
    _commit(work, _edit(work, "g", 5, "g5-musa"), "MUSA: g")
    _regen(ms)
    before = _snapshot(ms.SERIES_DIR)
    # The new commit lands after every entry, not between MUSA: g and fix a.
    assert _backport(ms, upstream.fix_g, source=str(upstream.path)) == 0
    after = _snapshot(ms.SERIES_DIR)
    assert {k: v for k, v in after.items() if k in before} == before
    assert _entries(ms) == [
        "0001-MUSA-c.patch",
        "0002-MUSA-g.patch",
        "upstream/0001-Bugfix-fix-a-101.patch",
        "upstream/0002-Bugfix-fix-g-107.patch",
    ]
    assert (
        "+G2"
        in (ms.SERIES_DIR / "upstream" / "0002-Bugfix-fix-g-107.patch").read_text()
    )


def test_backport_refuses_a_commit_the_series_carries(ms, series, upstream, capsys):
    assert _backport(ms, upstream.fix_a) == 1
    assert "is already upstream/0001-Bugfix-fix-a-101.patch" in capsys.readouterr().out


def test_backport_of_a_release_cherry_pick_names_both_commits(ms, work, upstream):
    assert _backport(ms, upstream.release_g, source=str(upstream.path)) == 0
    (entry,) = ms._series_dir_paths()[0]
    commits = ms.manifest.upstream_pick(entry).commits
    assert commits == (upstream.fix_g, upstream.release_g)
    assert set(_statuses(ms).values()) == {"clean"}
    ((verdict, reason),) = _plan(ms, upstream, upstream.main).values()
    assert verdict == "drop" and upstream.fix_g[:12] in reason


def test_a_cherry_pick_conflict_resolves_then_regen(ms, work, upstream, capsys):
    _git(upstream.path, "checkout", "--quiet", "-b", "other", upstream.pin)
    other = _commit(upstream.path, _line2("a", "A2-other"), "[Bugfix] other a (#108)")
    _git(upstream.path, "checkout", "--quiet", "main")
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _pick(work, upstream.fix_a)
    _regen(ms)
    assert _backport(ms, other, source=str(upstream.path)) == 1
    assert "CONFLICT: cherry-pick" in capsys.readouterr().out
    (work / "a.txt").write_text(_text("a", {2: "A2-other"}))
    _git(work, "add", "a.txt")
    _git(work, "cherry-pick", "--continue")
    _regen(ms)
    assert _entries(ms) == [
        "0001-MUSA-c.patch",
        "upstream/0001-Bugfix-fix-a-101.patch",
        "upstream/0002-Bugfix-other-a-108.patch",
    ]
    assert set(_statuses(ms).values()) == {"clean"}


def test_an_empty_cherry_pick_leaves_the_clone_alone(ms, work, upstream, capsys):
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _pick(work, upstream.fix_b)
    _regen(ms)
    head = _git(work, "rev-parse", "HEAD")
    assert _backport(ms, upstream.release_b) == 1  # the same change again
    assert "failed" in capsys.readouterr().out
    assert _git(work, "rev-parse", "HEAD") == head
    assert ms._git_in_progress() is None


def test_backport_refuses_a_clone_that_does_not_hold_the_series(
    ms, work, upstream, capsys
):
    _commit(work, _edit(work, "c", 2, "c2-musa"), "MUSA: c")
    _regen(ms)
    entry = ms.SERIES_DIR / "0001-MUSA-c.patch"
    entry.write_text(entry.read_text().replace("+c2-musa", "+c2-musa-v2"))
    edited = entry.read_bytes()
    assert _backport(ms, upstream.fix_b) == 2
    assert "does not hold the series" in capsys.readouterr().out
    assert entry.read_bytes() == edited


def test_backport_refuses_a_clone_rebased_onto_another_pin(
    ms, series, upstream, capsys
):
    assert _rebase(ms, "main", history=str(upstream.path), drop=True) == 0
    capsys.readouterr()
    assert _backport(ms, upstream.fix_g) == 2
    assert "holds the series on" in capsys.readouterr().out


def test_backport_fetches_a_commit_that_is_no_branch_tip(
    ms, work, upstream, tmp_path, monkeypatch, capsys
):
    # Like a fresh third_party/vllm: no upstream commit past the pin.
    clone = tmp_path / "fresh"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--no-local",
            "--single-branch",
            "--branch",
            "release",
            str(upstream.path),
            str(clone),
        ],
        check=True,
    )
    _git(clone, "config", "user.email", "musa-sync@example.invalid")
    _git(clone, "config", "user.name", "musa-sync test")
    _git(clone, "checkout", "--quiet", "--detach", upstream.pin)
    missing = subprocess.run(
        ["git", "-C", str(clone), "cat-file", "-e", f"{upstream.fix_a}^{{commit}}"]
    )
    assert missing.returncode != 0
    monkeypatch.setattr(ms, "WORKDIR", clone)
    assert _backport(ms, upstream.fix_a, source=str(upstream.path)) == 0
    assert (
        f"backported {upstream.fix_a[:12]} as upstream/0001" in capsys.readouterr().out
    )
    assert _entries(ms) == ["upstream/0001-Bugfix-fix-a-101.patch"]


# ----------------------------------------------------------- shipped series


def test_shipped_upstream_entries():
    spec = importlib.util.spec_from_file_location(
        "musa_manifest_upstream_check", ROOT / "vllm_musa" / "patches" / "manifest.py"
    )
    manifest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manifest)
    series = [
        e for e in manifest.ENTRIES if e.category in manifest.BUILD_APPLIED_CATEGORIES
    ]
    upstream = [e for e in series if e.id.startswith("upstream/")]
    assert upstream and series[-len(upstream) :] == upstream  # applied last
    assert upstream[0].upstream_commits, "upstream/ starts with an upstream commit"
    for e in upstream:
        pick = manifest.upstream_pick(ROOT / e.path)
        assert pick.commits == e.upstream_commits, e.id
        assert bool(pick.commits) == bool(pick.pr), e.id
    for e in series[: -len(upstream)]:
        assert not manifest.upstream_pick(ROOT / e.path).commits, e.id
