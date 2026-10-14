# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 on MUSA keeps the aux-stream overlap only in FULL CUDA graphs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = next(
    (ROOT / "vllm_musa/patches/series").glob(
        "*-MUSA-keep-DeepSeek-V4-aux-stream-overlap-in-FULL-dec*.patch"
    )
)
GATE = "_musa_deepseek_v4_aux_overlap()"


def _lines(prefix: str) -> list[str]:
    return [
        line[1:]
        for line in PATCH.read_text().splitlines()
        if line.startswith(prefix) and not line.startswith(prefix * 3)
    ]


def test_patch_only_changes_attention() -> None:
    text = PATCH.read_text()
    changed = {
        line[len("+++ b/") :]
        for line in text.splitlines()
        if line.startswith("+++ b/")
    }
    assert changed == {"vllm/models/deepseek_v4/attention.py"}
    assert "From: musa <musa@local>" in text


def test_gate_is_used_at_the_three_overlap_sites() -> None:
    added = _lines("+")
    added_text = "\n".join(added)
    for gate in ("kv_role", "device_type", "is_forward_context_available"):
        assert gate not in added_text
    calls = [line for line in added if GATE in line and "def " not in line]
    assert len(calls) == 3


def test_gate_under_real_graph_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    if getattr(torch.version, "musa", None) is None:
        pytest.skip("needs MUSA")
    from vllm.config import CUDAGraphMode

    attn = pytest.importorskip("vllm.models.deepseek_v4.attention")

    def gate(mode: CUDAGraphMode, capture: bool) -> bool:
        ctx = SimpleNamespace(cudagraph_runtime_mode=mode)
        monkeypatch.setattr(attn, "get_forward_context", lambda: ctx)
        if not capture:
            return attn._musa_deepseek_v4_aux_overlap()
        x = torch.zeros(8, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = attn._musa_deepseek_v4_aux_overlap()
            x.add_(1)
        return result

    # V1 captures FULL graphs under FULL; V2 captures them under NONE.
    assert gate(CUDAGraphMode.FULL, capture=False)
    assert gate(CUDAGraphMode.NONE, capture=True)
    # Breakable piecewise capture of prefill and mixed batches, and eager steps.
    assert not gate(CUDAGraphMode.PIECEWISE, capture=True)
    assert not gate(CUDAGraphMode.NONE, capture=False)
    assert not gate(CUDAGraphMode.PIECEWISE, capture=False)
