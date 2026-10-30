# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards the DeepSelect/MUSA integration without needing torch or a MUSA box.

Three things can silently regress and none of them fails at runtime on a CUDA
build: the series patch could keep the CUDA-only probe, the plugin could stop
registering the ops, or the vendored backend could lose one of the two MUSA
workarounds (write only the first ``k`` columns, restore the NaN marker after the
final copy). Source-level so it runs in CI without hardware.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERIES = ROOT / "vllm_musa" / "patches" / "series"
PACKAGE = ROOT / "vllm_musa" / "deep_select"
BACKEND = PACKAGE / "backend.py"
REGISTRY = PACKAGE / "registry.py"
PLUGIN = ROOT / "vllm_musa" / "__init__.py"
PINS = ROOT / "third_party" / "PINS"
DOC = ROOT / "docs" / "deepselect.md"


def _deep_select_patches() -> list[Path]:
    patches = sorted(SERIES.glob("*-MUSA-*DeepSelect*.patch"))
    assert patches, "no DeepSelect series patch found"
    return patches


def _added_lines(text: str) -> str:
    return "\n".join(
        line[1:]
        for line in text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_series_patch_makes_the_probe_lazy() -> None:
    added = "\n".join(
        _added_lines(p.read_text(encoding="utf-8")) for p in _deep_select_patches()
    )
    assert "def _deep_select_op_available() -> bool:" in added
    # The probe must consult the namespace, not only the in-tree extension.
    assert 'hasattr(torch.ops.deep_select, "topk")' in added
    # The deep_select backend must no longer be rejected on non-CUDA alone.
    assert 'failures.append("requires a CUDA platform")' not in added


def test_series_patch_keeps_the_cuda_capability_check() -> None:
    added = "\n".join(
        _added_lines(p.read_text(encoding="utf-8")) for p in _deep_select_patches()
    )
    # SM100-only stays the rule for CUDA; only the "op exists" half became
    # platform-neutral.
    assert "current_platform.is_device_capability_family(100)" in added
    assert "not self._is_cuda" in added


def test_series_patch_does_not_silently_bypass_explicit_backend() -> None:
    source = (SERIES / "0210-MUSA-route-explicit-DeepSelect-through-DSV4-indexer.patch").read_text(
        encoding="utf-8"
    )
    added = _added_lines(source)
    assert added.count('topk_backend == "deep_select"') >= 3
    assert "_musa_try_fill_decode_topk_from_paged_logits" in added
    assert "deep_select_topk(" in added
    assert "requires zero-based indexer windows" in added


def test_plugin_registers_deep_select() -> None:
    source = PLUGIN.read_text(encoding="utf-8")
    assert "from .deep_select import ensure_registered" in source
    assert "_register_deep_select()" in source
    # Registration must happen in _register_ops (the plugin entry point), not at
    # module import time, so the platform is already initialised.
    tree = ast.parse(source)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_register_ops"
    )
    calls = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_register_deep_select" in calls


def test_registry_defines_both_ops_and_a_meta_kernel() -> None:
    source = REGISTRY.read_text(encoding="utf-8")
    assert '"deep_select::"' in source
    assert "_TOPK_SCHEMA" in source and "_ALIGNMENT_SCHEMA" in source
    assert "get_alignment_requirement() -> int[]" in source
    assert '"Meta"' in source
    # The dispatch key comes from the platform, not a hardcoded "CUDA".
    assert "current_platform.dispatch_key" in source


def test_backend_keeps_the_two_musa_workarounds() -> None:
    source = BACKEND.read_text(encoding="utf-8")
    # (1) a preallocated output may be wider than k: write only the first k
    # columns so the caller's pre-filled tail survives.
    assert "output_idx[:, :k] = " in source
    assert "output_idx[:, :] = " not in source
    # (2) the final copy must not erase the NaN marker.
    assert "output_idx[bad_rows, 0] = 0x3F3F3F3F" in source
    # (3) duplicate indices from torch_musa's bf16 topk are re-derived by an
    # argsort of the affected rows (a permutation by construction).
    assert "torch.argsort(" in source
    assert "work[bad]" in source
    assert "descending=True, stable=True" in source


def test_alignment_contract_is_documented_not_a_placeholder() -> None:
    source = BACKEND.read_text(encoding="utf-8")
    assert "_ALIGNMENT = (32, 32)" in source


def test_installation_documentation_exists() -> None:
    assert DOC.is_file()
    source = DOC.read_text(encoding="utf-8")
    assert "DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA" in source
    assert "sparse_indexer_topk_backend" in source


def test_pin_records_the_vendored_revision() -> None:
    pins = PINS.read_text(encoding="utf-8")
    package = (PACKAGE / "__init__.py").read_text(encoding="utf-8")
    sha = "fac460ce55f1b2e86a30fd60ea892fc782f3a8ea"
    assert f"DEEPSELECT_COMMIT={sha}" in pins
    assert f'DEEPSELECT_COMMIT = "{sha}"' in package
    assert "feat/musa-kernels" in pins and "feat/musa-kernels" in package
