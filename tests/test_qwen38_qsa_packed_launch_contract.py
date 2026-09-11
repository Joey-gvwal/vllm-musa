"""Exercise Python launch plumbing without importing a GPU runtime.

This checks argument binding and scratch lifetime, not GPU numerical results.
"""

import ast
import inspect
import os
from pathlib import Path
from types import SimpleNamespace


SOURCE = (Path(__file__).parents[1]
          / "vllm_musa/v1/attention/ops/qwen_qsa_sglang_triton.py")


class Tensor:
    def __init__(self, shape, *, device, dtype):
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.ndim = len(self.shape)

    def numel(self):
        size = 1
        for dim in self.shape:
            size *= dim
        return size

    def stride(self, dim):
        stride = 1
        for size in self.shape[dim + 1:]:
            stride *= size
        return stride

    def to(self, *, device, dtype):
        assert self.device == device and self.dtype == dtype
        return self

    def contiguous(self):
        return self


def test_packed_launch_arguments_and_reused_output(monkeypatch):
    monkeypatch.delenv("VLLM_MUSA_QSA_STAGE_PROFILE", raising=False)
    monkeypatch.delenv("VLLM_MUSA_QSA_SGLANG_FUSED_DECODE", raising=False)
    tree = ast.parse(SOURCE.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    launches = []
    allocations = []

    class Launcher:
        def __init__(self, name):
            self.name = name
            self.signature = inspect.Signature([
                inspect.Parameter(p.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in functions[name].args.args
            ])

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                kwargs.pop("num_warps", None)
                kwargs.pop("num_stages", None)
                bound = self.signature.bind(*args, **kwargs)
                launches.append((self.name, grid, bound.arguments))
            return launch

    def empty(*shape, **kwargs):
        tensor = Tensor(shape, **kwargs)
        allocations.append(tensor)
        return tensor

    def packed(q, k, v, indices, cu_q, cu_k, kv_lens, scale, out=None):
        assert out is output
        assert q.shape == out.shape
        assert cu_k.numel() == cu_q.numel() == q.shape[0] + 1
        assert kv_lens.numel() == q.shape[0]
        assert k.shape == v.shape == (q.shape[0] * indices.shape[1], 1, 256)
        launches.append(("gqa", None, {}))
        return out

    device = SimpleNamespace(type="musa", index=0)
    runtime = {
        "os": os,
        "torch": SimpleNamespace(empty=empty, int32="int32", int64="int64"),
        "triton": SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b,
                                  next_power_of_2=lambda n: 1 << (n - 1).bit_length()),
        "_DECODE_WORKSPACES": {},
        "sparse_gqa_fwd_interface_triton_ck": packed,
    }
    for name in ("_paged_row_metadata", "_fa2_prefix_sum", "_compact_paged_kv"):
        runtime[name] = Launcher(name)
    for name in ("_decode_workspace", "qwen_sparse_kv_extraction_compact_paged_triton",
                 "sparse_gqa_from_paged_triton"):
        module = ast.Module(body=[functions[name]], type_ignores=[])
        exec(compile(module, str(SOURCE), "exec"), runtime)

    def tensor(*shape, dtype="int32"):
        return Tensor(shape, device=device, dtype=dtype)

    q = tensor(3, 3, 256, dtype="bfloat16")
    k = tensor(5, 16, 1, 256, dtype="bfloat16")
    v = tensor(5, 16, 1, 256, dtype="bfloat16")
    output = tensor(3, 3, 256, dtype="bfloat16")
    table, requests, indices, lengths = tensor(2, 3), tensor(3), tensor(3, 259), tensor(2)
    run = runtime["sparse_gqa_from_paged_triton"]
    for iteration in range(2):
        launches.clear()
        assert run(q, k, v, table, requests, indices, lengths, 0.071, out=output) is output
        assert [x[0] for x in launches] == [
            "_paged_row_metadata", "_fa2_prefix_sum", "_compact_paged_kv", "gqa"
        ]
        metadata, gather = launches[0][2], launches[2][2]
        assert metadata["ranks"] is gather["ranks"]
        assert metadata["row_requests"] is gather["req_indices"]
        assert metadata["row_lengths"] is gather["seq_lens"]
        if iteration == 0:
            first_allocation_count = len(allocations)
        else:
            assert len(allocations) == first_allocation_count
