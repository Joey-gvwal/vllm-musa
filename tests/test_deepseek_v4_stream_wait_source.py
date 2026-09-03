# SPDX-License-Identifier: Apache-2.0
"""Source contract for the MUSA-safe DSV4 auxiliary overlap hand-off."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0115-MUSA-use-stream-waits-for-DSV4-overlap-hand-off.patch"
)
MTP_PATCH = ROOT / (
    "vllm_musa/patches/series/"
    "0117-MUSA-keep-DSV4-MTP-overlap-with-stream-waits.patch"
)


def test_dsv4_musa_overlap_uses_stream_waits() -> None:
    patch = PATCH.read_text()

    assert "use_stream_waits: bool = False" in patch
    assert "aux_stream.wait_stream(current_stream)" in patch
    assert "current_stream.wait_stream(aux_stream)" in patch
    assert "use_stream_waits=current_platform.is_musa()" in patch


def test_dsv4_no_mtp_does_not_disable_overlap() -> None:
    patch = PATCH.read_text()

    assert "_musa_dsv4_graph_requires_serialized_overlap" not in patch
    assert "enable=aux_streams is not None" in patch


def test_dsv4_mtp_does_not_disable_overlap() -> None:
    patch = MTP_PATCH.read_text()

    assert "+            if not _musa_deepseek_v4_aux_overlap_enabled" in patch
