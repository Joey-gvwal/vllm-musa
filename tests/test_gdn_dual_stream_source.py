# SPDX-License-Identifier: Apache-2.0
"""Source contract for the MUSA GDN projection overlap."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "vllm_musa"
    / "model_executor"
    / "layers"
    / "mamba"
    / "gdn"
    / "qwen_gdn_linear_attn.py"
)


def test_gdn_dual_stream_projection_contract() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "_GDN_DUAL_STREAM_TOKEN_THRESHOLD = 1024" in source
    assert "torch.musa.Stream()" in source
    assert "create=not is_capturing" in source
    assert "alt_stream.wait_stream(current_stream)" in source
    assert "current_stream.wait_stream(alt_stream)" in source
    assert "VLLM_MUSA_GDN_DUAL_PROJ_STREAM" not in source

    qkvz = "mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)"
    ba = "ba, _ = self.in_proj_ba(hidden_states)"
    aux_context = "with torch.musa.stream(alt_stream):"
    wait_for_aux = "current_stream.wait_stream(alt_stream)"

    overlap_start = source.index("def _forward_input_projections")
    overlap_end = source.index("def _forward_core", overlap_start)
    overlap = source[overlap_start:overlap_end]
    assert overlap.index(qkvz) < overlap.index(aux_context)
    assert overlap.index(aux_context) < overlap.index(ba)
    assert overlap.index(ba) < overlap.index(wait_for_aux)
