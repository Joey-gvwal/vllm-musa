#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""the MDM driver — one tool for the whole divergence lifecycle.

The MUSA Divergence Manifest (MDM) keeps vllm-musa's divergence from upstream
vLLM in one declarative census (``vllm_musa/patches/manifest.py``) applied to a
pinned, cloned vLLM. This is the single CLI over that system.

Subcommands::

    apply  <repo> [--phase P] [--check-only] [--no-strict]
        Build-time: apply the build-applied diff series (categories 1/2/3/4b) to
        the cloned vLLM at <repo>, honoring apply_phase order. (setup.py calls this.)

    verify [--target TAG] [--repo PATH]
        OFFLINE pre-bump gate (no MUSA hardware). Fresh clone of vllm@TAG (or
        --repo for an existing checkout), then for EVERY divergence emit one
        status row: build-applied diffs are git-apply --check'd
        (clean/obsolete/conflict); cat-5/cat-6 existence-probe their upstream
        target. Exits non-zero on any conflict / missing / orphaned divergence —
        the bounded review surface a version bump needs.

    rebase <tag>
        Checkout the clone at <tag> and ``git am -3`` the series in order (true
        3-way; trivial upstream drift auto-merges, real conflicts halt).

    regen
        Regenerate the series from the clone's commits
        (``git format-patch --no-signature --no-numbered --zero-commit``,
        keeping the ``index`` blob lines for the next bump's 3-way).

    report [--doc]
        Census of the manifest: a status line per divergence, or (--doc) the
        Markdown census table.

Stdlib-only; loads manifest.py + build_apply.py BY FILE PATH so it never imports
the ``vllm_musa`` package (works before install, in plain CI).
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # tools/ -> repo root
PATCHES = ROOT / "vllm_musa" / "patches"
SERIES_DIR = PATCHES / "series"
MODULE_DRIFT_DIR = (
    PATCHES / "module-drift"
)  # cat-4a drift tripwires (never build-applied)
WORKDIR = ROOT / "third_party" / "vllm"
PINS = ROOT / "third_party" / "PINS"
VLLM_URL = "https://github.com/vllm-project/vllm.git"


def _load(name: str, path: Path):
    """Load a stdlib-only helper module by file path (no vllm_musa import)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (safe by-path load)
    spec.loader.exec_module(mod)
    return mod


manifest = _load("musa_mdm_manifest", PATCHES / "manifest.py")
build_apply = _load("musa_mdm_build_apply", PATCHES / "build_apply.py")


def read_pin(key: str, default: str | None = None) -> str | None:
    if not PINS.is_file():
        return default
    for line in PINS.read_text().splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line[len(key) + 1 :].split("#", 1)[0].strip()
    return default


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True
    )


def _probe_upstream(clone: Path, upstream_path: str | None) -> bool:
    """cat-5/cat-6 existence probe: does the declared upstream target exist in the
    clone? (File-level for now; symbol-level is a later enhancement.)"""
    if not upstream_path:
        return True
    return (Path(clone) / upstream_path).is_file()


def _module_tripwire(clone: Path, entry) -> "str | None":
    """cat-4a drift tripwire: the stable unified diff of the upstream file (in the
    clone) vs the MUSA shadow copy. Returns None if the upstream file is gone.
    difflib (no timestamps) so the stored tripwire is byte-stable across runs."""
    up = Path(clone) / (entry.upstream_path or "")
    shadow = ROOT / entry.path
    if not entry.upstream_path or not up.is_file() or not shadow.is_file():
        return None
    a = up.read_text(errors="replace").splitlines(keepends=True)
    b = shadow.read_text(errors="replace").splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, fromfile="upstream", tofile="musa", n=3))


def _tripwire_path(entry) -> Path:
    return MODULE_DRIFT_DIR / (entry.id + ".diff")


# --------------------------------------------------------------------------- apply
def cmd_apply(args) -> int:
    repo = Path(args.repo)
    order = manifest.series_apply_order(phase=args.phase)
    results = build_apply.apply_patch_series(
        repo, order=order, strict=not args.no_strict, check_only=args.check_only
    )
    for name, status in results:
        print(f"{status:16} {name}")
    n_conflict = sum(1 for _, s in results if s == "conflict")
    verb = "would-apply" if args.check_only else "applied"
    n_ok = sum(1 for _, s in results if s in ("applied", "would-apply"))
    n_skip = sum(1 for _, s in results if s == "already-applied")
    print(f"--- {n_ok} {verb}, {n_skip} already-applied, {n_conflict} conflict ---")
    return 1 if n_conflict else 0


# -------------------------------------------------------------------------- verify
# Map a build_apply check_only status (vs a pristine clone) to a verify verdict.
_VERIFY_STATUS = {
    "would-apply": "clean",  # applies cleanly to pristine upstream
    "already-applied": "obsolete",  # already in upstream -> candidate for removal
    "conflict": "conflict",  # drifted -> needs re-anchor/retire
}
_BAD = {"conflict", "missing-symbol", "missing-target", "orphaned", "drifted-copy"}


def _ensure_clone(target: str, repo_arg: str | None):
    """Return (checkout_path, is_temporary). Uses --repo if given, else a fresh
    shallow clone of vllm@target."""
    if repo_arg:
        return Path(repo_arg), False
    tmp = Path(tempfile.mkdtemp(prefix="musa-verify-"))
    clone = tmp / "vllm"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", target, VLLM_URL, str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    return clone, True


def _verify_rows(clone: Path) -> list[tuple]:
    rows: list[tuple] = []
    for e in manifest.ENTRIES:
        if e.category in manifest.BUILD_APPLIED_CATEGORIES:
            patch = ROOT / e.path
            if not patch.is_file():
                rows.append((e.id, e.category, "orphaned", "patch file missing"))
                continue
            status = build_apply.apply_patch(clone, patch, check_only=True)
            rows.append((e.id, e.category, _VERIFY_STATUS.get(status, status), ""))
        elif e.category == "4a":
            # drift tripwire: re-diff the upstream-vs-shadow and compare to the
            # stored tripwire. A mismatch means upstream changed under the copy.
            cur = _module_tripwire(clone, e)
            stored = _tripwire_path(e)
            if cur is None:
                rows.append((e.id, e.category, "missing-target", e.upstream_path or ""))
            elif not stored.is_file():
                rows.append(
                    (e.id, e.category, "no-tripwire", "run `regen --area module`")
                )
            elif stored.read_text() == cur:
                rows.append((e.id, e.category, "clean", "tripwire matches upstream"))
            else:
                rows.append(
                    (
                        e.id,
                        e.category,
                        "drifted-copy",
                        "upstream changed under the copy",
                    )
                )
        elif e.category == "5":
            ok = _probe_upstream(clone, e.upstream_path)
            rows.append(
                (
                    e.id,
                    e.category,
                    "present" if ok else "missing-symbol",
                    e.upstream_path or "",
                )
            )
        elif e.category == "6":
            ok = _probe_upstream(clone, e.upstream_path)
            rows.append(
                (
                    e.id,
                    e.category,
                    "present" if ok else "missing-target",
                    e.upstream_path or "",
                )
            )
    return rows


def cmd_verify(args) -> int:
    target = args.target or read_pin("VLLM_TAG")
    if not target:
        print(
            "ERROR: no target tag (pass --target or set VLLM_TAG in third_party/PINS)"
        )
        return 2
    clone, temp = _ensure_clone(target, args.repo)
    try:
        rows = _verify_rows(clone)
    finally:
        if temp:
            shutil.rmtree(clone.parent, ignore_errors=True)
    print(f"=== musa_sync verify: {len(rows)} divergences vs vllm@{target} ===")
    for did, cat, status, detail in rows:
        line = f"  [{cat:>2}] {status:<14} {did}"
        if detail:
            line += f"   ({detail})"
        print(line)
    bad = [r for r in rows if r[2] in _BAD]
    n_clean = sum(1 for r in rows if r[2] == "clean")
    print(
        f"--- {n_clean} clean / {len(rows)} total / {len(bad)} need attention "
        f"({', '.join(sorted({r[2] for r in bad})) or 'none'}) ---"
    )
    return 1 if bad else 0


# -------------------------------------------------------------------- rebase / regen
def _checkout(target: str) -> int:
    if not WORKDIR.exists():
        r = subprocess.run(
            ["git", "clone", VLLM_URL, str(WORKDIR)], capture_output=True, text=True
        )
        if r.returncode:
            print(r.stderr)
            return 1
    _git(WORKDIR, "fetch", "--tags")
    r = _git(WORKDIR, "checkout", "-f", target)
    if r.returncode:
        print(r.stderr)
        return 1
    return 0


def cmd_rebase(args) -> int:
    target = args.tag or read_pin("VLLM_TAG")
    if _checkout(target):
        return 1
    order = manifest.series_apply_order()
    for patch in order:
        r = _git(WORKDIR, "am", "-3", str(patch))
        if r.returncode != 0:
            print(f"CONFLICT: {patch.name}\n{r.stdout}\n{r.stderr}")
            print(
                "Resolve in third_party/vllm, then "
                "`git -C third_party/vllm am --continue`, then re-run; or "
                "`git -C third_party/vllm am --abort` to back out."
            )
            return 1
    print(f"rebased {len(order)} patches onto vllm@{target} (git am -3)")
    return 0


def _regen_module_tripwires() -> int:
    """(re)generate the cat-4a drift tripwires from a PRISTINE upstream
    checkout (the tripwire is the MUSA shadow's delta vs the pinned upstream)."""
    target = read_pin("VLLM_TAG")
    if _checkout(target):  # reset WORKDIR to pristine vllm@<pin>
        return 1
    MODULE_DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in manifest.ENTRIES:
        if e.category != "4a":
            continue
        tw = _module_tripwire(WORKDIR, e)
        if tw is None:
            print(f"  WARN no upstream for {e.id} ({e.upstream_path})")
            continue
        _tripwire_path(e).write_text(tw)
        print(f"  tripwire: {e.id} ({len(tw.splitlines())} diff lines)")
        n += 1
    print(f"regenerated {n} cat-4a module tripwires in {MODULE_DRIFT_DIR}")
    return 0


def cmd_regen(args) -> int:
    if args.area == "module":
        return _regen_module_tripwires()
    target = read_pin("VLLM_TAG")
    SERIES_DIR.mkdir(parents=True, exist_ok=True)
    r = _git(
        WORKDIR,
        "format-patch",
        "--no-signature",
        "--no-numbered",
        "--zero-commit",
        "-o",
        str(SERIES_DIR.resolve()),
        target,
    )
    if r.returncode:
        print(r.stderr)
        return 1
    print(
        r.stdout.strip()
        or f"regenerated series in {SERIES_DIR} from vllm@{target}..HEAD"
    )
    return 0


# -------------------------------------------------------------------------- report
def cmd_report(args) -> int:
    by_cat: dict[str, int] = {}
    for e in manifest.ENTRIES:
        by_cat[e.category] = by_cat.get(e.category, 0) + 1
    if args.doc:
        print("| id | cat | phase | required | upstream_path | intent |")
        print("|---|---|---|---|---|---|")
        for e in manifest.ENTRIES:
            print(
                f"| {e.id} | {e.category} | {e.apply_phase} | {e.required} | "
                f"{e.upstream_path or ''} | {e.intent} |"
            )
        return 0
    print(
        f"MDM manifest: {len(manifest.ENTRIES)} divergences by category {dict(sorted(by_cat.items()))}"
    )
    for e in manifest.ENTRIES:
        print(f"  [{e.category:>2}] {e.apply_phase:<11} {e.id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="musa_sync", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "apply", help="build-time: apply the diff series to a cloned vLLM"
    )
    p.add_argument("repo")
    p.add_argument("--phase", default=None, help="restrict to one apply_phase")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--no-strict", action="store_true")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser(
        "verify", help="offline pre-bump gate: status of every divergence"
    )
    p.add_argument(
        "--target", default=None, help="vLLM tag (default: VLLM_TAG from PINS)"
    )
    p.add_argument(
        "--repo", default=None, help="use an existing checkout instead of cloning"
    )
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "rebase", help="git am -3 the series onto vllm@<tag> in third_party/vllm"
    )
    p.add_argument("tag", nargs="?", default=None)
    p.set_defaults(func=cmd_rebase)

    p = sub.add_parser("regen", help="regenerate the series from the clone's commits")
    p.add_argument("--area", default=None, help="(reserved) py|csrc|module")
    p.set_defaults(func=cmd_regen)

    p = sub.add_parser("report", help="manifest census")
    p.add_argument(
        "--doc", action="store_true", help="render the Markdown census table"
    )
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
