"""Source contracts for the opt-in SGLang-MUSA packed QSA candidate."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
BRIDGE = ROOT / "vllm_musa/v1/attention/ops/qwen_qsa_sglang_triton.py"
DISPATCH = ROOT / "third_party/vllm/vllm/models/qwen4_exp/nvidia/ops/qsa.py"
OWNER = ROOT / "third_party/vllm/vllm/models/qwen4_exp/nvidia/qsa.py"
AMD_DISPATCH = ROOT / "third_party/vllm/vllm/models/qwen4_exp/amd/ops/qsa.py"
AMD_OWNER = ROOT / "third_party/vllm/vllm/models/qwen4_exp/amd/qsa.py"
MANIFEST = ROOT / "vllm_musa/patches/manifest.py"


def test_packed_qsa_bridge_uses_sglang_ck_kernel():
    source = BRIDGE.read_text()
    assert "def sparse_gqa_fwd_interface_triton_ck" in source
    assert "qwen_sparse_kv_extraction_compact_paged_triton" in source
    assert "def _paged_row_metadata" in source
    assert "tl.cumsum(valid.to(tl.int32), axis=0)" in source
    assert "ranks" in source
    assert "_paged_row_metadata[(rows,)]" in source
    assert source.count("qwen_sparse_fa2_cu_seqlens_triton(") == 1
    assert "result = sparse_gqa_fwd_interface_triton_ck(" in source
    assert "out=out" in source
    assert "out.copy_(result)" in source
    assert "triton.next_power_of_2(rows)" in source
    assert "stride_k_block" in source and "stride_v_block" in source
    assert "k_cache.reshape(-1, kv_heads, head_dim)" not in source


def test_packed_qsa_dispatch_is_opt_in_and_has_sequence_lengths():
    dispatch = DISPATCH.read_text()
    owner = OWNER.read_text()
    assert 'os.environ.get("VLLM_MUSA_QSA_SGLANG_PACKED", "1") == "1"' in dispatch
    assert 'q.shape[-1] == 256' in dispatch
    assert "q.device.type == \"musa\"" in dispatch
    assert "sequence_lengths=attn_metadata.seq_lens" in owner
    assert "scale=self.scale" in owner


def test_amd_qsa_owner_keeps_the_same_metadata_contract():
    dispatch = AMD_DISPATCH.read_text()
    owner = AMD_OWNER.read_text()
    assert 'os.environ.get("VLLM_MUSA_QSA_SGLANG_PACKED", "1") == "1"' in dispatch
    assert 'q.shape[-1] == 256' in dispatch
    assert "q.device.type == \"musa\"" in dispatch
    assert "sequence_lengths=attn_metadata.seq_lens" in owner
    assert "scale=self.scale" in owner


@pytest.mark.parametrize("owner_path", [OWNER, AMD_OWNER])
@pytest.mark.parametrize("query_len", [1, 2, 256])
def test_owner_decode_guard_does_not_depend_on_dcp_counts(owner_path, query_len):
    tree = ast.parse(owner_path.read_text())
    call = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "qsa_sparse_paged_attention"
    )
    guard = next(kw.value for kw in call.keywords if kw.arg == "is_decode")
    # Normal FlashAttention leaves DCP counts at zero even for prefill, and
    # ROCm metadata has no such counts at all.  Both must route using query len.
    for metadata in (
        SimpleNamespace(max_query_len=query_len, num_prefill_tokens=0),
        SimpleNamespace(max_query_len=query_len),
    ):
        enabled = eval(compile(ast.Expression(guard), str(owner_path), "eval"),
                       {"attn_metadata": metadata})
        assert enabled is (query_len == 1)


def test_bridge_is_in_the_musa_divergence_manifest():
    assert (
        "vllm_musa/v1/attention/ops/qwen_qsa_sglang_triton.py"
        in MANIFEST.read_text()
    )
