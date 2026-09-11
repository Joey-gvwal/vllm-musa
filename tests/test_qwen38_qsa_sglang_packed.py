"""MUSA correctness coverage for the paged-to-packed QSA GQA adapter."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_musa")
if not getattr(torch.version, "musa", None) or not torch.musa.is_available():
    pytest.skip("requires a MUSA device", allow_module_level=True)

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qsa_sglang_test_ops",
    ROOT / "vllm_musa/v1/attention/ops/qwen_qsa_sglang_triton.py",
)
OPS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OPS)


@pytest.mark.parametrize("rows,kv_heads,gaps,capture", [
    (3, 1, False, False), (129, 2, False, False),
    (3, 1, True, False), (129, 2, True, False), (3, 1, True, True),
])
def test_paged_qsa_matches_dense_reference(rows, kv_heads, gaps, capture):
    """Cover request remapping, partial tiles, padding and >128 row prefixes."""
    torch.manual_seed(17)
    device = "musa"
    dim, page_size = 256, 16
    topk = 259 if gaps else 35
    q_heads = 3 * kv_heads
    q = torch.randn(rows, q_heads, dim, device=device, dtype=torch.bfloat16)
    # Keep different non-contiguous K/V layouts, including padding between
    # physical pages, so a contiguous-slot addressing assumption cannot pass.
    k = torch.randn(5, page_size, kv_heads, dim + 8, device=device,
                    dtype=torch.bfloat16)[..., :dim]
    v = torch.randn(5, page_size + 1, kv_heads, dim, device=device,
                    dtype=torch.bfloat16)[:, :page_size]
    table = torch.tensor([[3, 1, 4], [0, 2, -1]], device=device, dtype=torch.int32)
    lengths_cpu = [37, 19]
    lengths = torch.tensor(lengths_cpu, device=device, dtype=torch.int32)
    requests_cpu = [i % 2 for i in range(rows)]
    requests_cpu[-1] = -1
    requests = torch.tensor(requests_cpu, device=device, dtype=torch.int32)
    indices_cpu = []
    for req in requests_cpu:
        selected = list(range(33)) if req == 0 else list(range(2, 19))
        if gaps:
            selected = [i % 19 if i % 2 == 0 else -1 for i in range(topk)]
            selected[0] = -1
            selected[128] = 1000  # out-of-range hole at a gather tile boundary
        if req < 0:
            selected = []
        indices_cpu.append(selected + [-1] * (topk - len(selected)))
    # Exercise non-unit column strides in selection and block-table metadata.
    indices_storage = torch.empty(rows, topk * 2, device=device, dtype=torch.int32)
    indices = indices_storage[:, ::2]
    indices.copy_(torch.tensor(indices_cpu, device=device, dtype=torch.int32))
    table_storage = torch.empty(2, 6, device=device, dtype=torch.int32)
    table_storage[:, ::2].copy_(table)
    table = table_storage[:, ::2]
    out = torch.full_like(q, float("nan"))
    scale = 0.071
    result = OPS.sparse_gqa_from_paged_triton(
        q, k, v, table, requests, indices, lengths, scale, out=out
    )
    assert result is out

    def check_reference():
        reference = torch.zeros_like(q, dtype=torch.float32)
        for row, req in enumerate(requests_cpu):
            if req < 0:
                continue
            selected_cpu = [x for x in indices_cpu[row]
                            if 0 <= x < lengths_cpu[req]]
            if not selected_cpu:
                continue
            selected = torch.tensor(selected_cpu, device=device, dtype=torch.int64)
            blocks = table[req].long()[selected // page_size]
            offsets = selected % page_size
            keys = k[blocks, offsets].float().repeat_interleave(3, dim=1)
            values = v[blocks, offsets].float().repeat_interleave(3, dim=1)
            scores = torch.einsum("hd,nhd->hn", q[row].float(), keys) * scale
            reference[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), values)
        torch.musa.synchronize()
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out.float(), reference, atol=0.03, rtol=0.03)

    check_reference()
    if capture:
        torch.musa.synchronize()
        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            OPS.sparse_gqa_from_paged_triton(
                q, k, v, table, requests, indices, lengths, scale, out=out
            )
        for length in (11, 0, 19):
            lengths_cpu[:] = [length, length]
            requests_cpu[:] = [1, -1, 0]
            indices_cpu[2] = list(indices_cpu[0])
            indices_cpu[0] = indices_cpu[0][1:] + indices_cpu[0][:1]
            lengths.copy_(torch.tensor(lengths_cpu, device=device, dtype=torch.int32))
            requests.copy_(torch.tensor(requests_cpu, device=device, dtype=torch.int32))
            indices.copy_(torch.tensor(indices_cpu, device=device, dtype=torch.int32))
            q.mul_(0.9)
            out.fill_(float("nan"))
            graph.replay()
            check_reference()
