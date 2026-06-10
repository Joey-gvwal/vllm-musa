# SPDX-License-Identifier: Apache-2.0
"""third_party/PINS is the single source of truth for the upstream pin.

Asserts setup.py and Makefile.sync BOTH resolve the upstream vLLM tag from
`third_party/PINS` (and neither hardcodes it), so a version bump edits one place
and the generation base can never desync from the build base. Pure file parsing +
`sed` — no MUSA hardware, no heavy setup.py import.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / "third_party" / "PINS"


def _parse_pins() -> dict:
    """The same KEY=VALUE parse setup.py::_read_pins uses."""
    pins = {}
    for line in PINS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pins[key.strip()] = value.split("#", 1)[0].strip()
    return pins


def test_pins_exists_with_required_keys():
    assert PINS.is_file(), "third_party/PINS missing"
    pins = _parse_pins()
    assert pins.get("VLLM_TAG"), pins
    assert pins.get("FLASHINFER_COMMIT"), pins


def test_pins_is_tracked_not_ignored():
    # third_party/* is ignored but PINS must be re-included (tracked).
    r = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "third_party/PINS"],
        capture_output=True,
        text=True,
    )
    # check-ignore exits 0 (and echoes the path) when the path IS ignored.
    assert r.returncode != 0, "third_party/PINS is gitignored — it must be tracked"


def test_setup_py_reads_pins_not_literal():
    src = (ROOT / "setup.py").read_text()
    assert (
        "_read_pins(" in src and '_PINS["VLLM_TAG"]' in src
    ), "setup.py must read VLLM_TAG from PINS"
    assert '_PINS["FLASHINFER_COMMIT"]' in src
    assert 'git_tag="v0.22.0"' not in src, "setup.py still hardcodes the vLLM tag"


def test_makefile_sync_reads_pins_not_literal():
    mk = (ROOT / "Makefile.sync").read_text()
    assert "third_party/PINS" in mk, "Makefile.sync must read the pin from PINS"
    assert (
        "FETCH_HEAD" not in mk
    ), "Makefile.sync still references the old FETCH_HEAD pin"


@pytest.mark.skipif(shutil.which("sed") is None, reason="sed unavailable")
def test_setup_and_makefile_resolve_same_tag():
    pins_tag = _parse_pins()["VLLM_TAG"]
    # Replicate Makefile.sync's exact `$(shell sed -n 's/^VLLM_TAG=//p' ...)`.
    sed_tag = subprocess.run(
        ["sed", "-n", "s/^VLLM_TAG=//p", str(PINS)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert sed_tag == pins_tag, (sed_tag, pins_tag)
    assert pins_tag, "VLLM_TAG resolved empty"
