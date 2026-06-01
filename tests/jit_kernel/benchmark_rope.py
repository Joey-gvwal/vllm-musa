"""Microbenchmark vLLM-MUSA JIT RoPE against vLLM native RoPE.

Run on a MUSA host from the vllm-musa repository root:

    python tests/jit_kernel/benchmark_rope.py

The default shape matrix mirrors tests/jit_kernel/test_rope.py so the
benchmark covers the same M2.5 + Eagle3 shapes as the correctness test.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

# torchada redirects torch.cuda symbols to MUSA. Keep this import first.
import torchada  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

# Register the vllm-musa platform hooks and torch.ops.vllm.musa_rotary_embedding.
import vllm_musa  # noqa: F401
from vllm_musa.jit_kernel.csrc import rope as _rope_module  # noqa: F401

_VLLM_CFG = VllmConfig()
_VLLM_CFG_CM = set_current_vllm_config(_VLLM_CFG)
_VLLM_CFG_CM.__enter__()


@dataclass(frozen=True)
class Shape:
    label: str
    num_heads_per_rank: int
    num_kv_heads_per_rank: int
    head_dim: int
    rotary_dim: int
    num_tokens: int
    is_neox: bool = True
    dtype: torch.dtype = torch.bfloat16


SHAPES = [
    Shape("eagle3_draft_tp8_decode", 3, 1, 128, 128, 1),
    Shape("eagle3_draft_tp8_chain8", 3, 1, 128, 128, 8),
    Shape("m25_target_tp8_decode", 6, 1, 128, 128, 1),
    Shape("m25_target_tp8_prefill", 6, 1, 128, 128, 4096),
    Shape("qwen3_8b_tp8_decode", 4, 1, 128, 128, 1),
    Shape("qwen3_8b_tp8_prefill", 4, 1, 128, 128, 1024),
    Shape("qwen3_30b_tp2_decode", 16, 4, 128, 128, 1),
    Shape("min_heads_1_decode", 1, 1, 128, 128, 1),
    Shape("min_heads_2_decode", 2, 1, 128, 128, 1),
]


@dataclass(frozen=True)
class BenchResult:
    label: str
    provider: str
    tokens: int
    q_heads: int
    kv_heads: int
    iters: int
    event_us: float
    wall_us: float


def make_inputs(shape: Shape, max_pos: int, device: str):
    torch.manual_seed(0)
    positions = torch.arange(shape.num_tokens, dtype=torch.int64, device=device)
    query = torch.randn(
        shape.num_tokens,
        shape.num_heads_per_rank * shape.head_dim,
        dtype=shape.dtype,
        device=device,
    )
    key = torch.randn(
        shape.num_tokens,
        shape.num_kv_heads_per_rank * shape.head_dim,
        dtype=shape.dtype,
        device=device,
    )
    rope = RotaryEmbedding(
        head_size=shape.head_dim,
        rotary_dim=shape.rotary_dim,
        max_position_embeddings=max(max_pos, shape.num_tokens),
        base=10000.0,
        is_neox_style=shape.is_neox,
        dtype=shape.dtype,
        init_cache=True,
    ).to(device)
    return positions, query, key, rope


def native_rope(rope, positions, query, key):
    return RotaryEmbedding.forward_static(
        positions,
        query,
        key,
        rope.head_size,
        rope.rotary_dim,
        rope.cos_sin_cache,
        rope.is_neox_style,
    )


def jit_rope(positions, query, key, shape: Shape, cos_sin_cache):
    torch.ops.vllm.musa_rotary_embedding(
        positions,
        query,
        key,
        shape.head_dim,
        cos_sin_cache,
        shape.is_neox,
    )
    return query, key


def assert_close(shape: Shape, rope, positions, query, key) -> None:
    q_native, k_native = native_rope(rope, positions, query.clone(), key.clone())
    q_jit = query.clone()
    k_jit = key.clone()
    cos_sin_cache = rope.cos_sin_cache.to(query.device, dtype=query.dtype)
    jit_rope(positions, q_jit, k_jit, shape, cos_sin_cache)
    q_max = (q_native.float() - q_jit.float()).abs().max().item()
    k_max = (k_native.float() - k_jit.float()).abs().max().item()
    if not torch.allclose(q_native.float(), q_jit.float(), atol=1e-2, rtol=1e-2):
        raise AssertionError(f"{shape.label}: query mismatch max_abs={q_max:.4e}")
    if not torch.allclose(k_native.float(), k_jit.float(), atol=1e-2, rtol=1e-2):
        raise AssertionError(f"{shape.label}: key mismatch max_abs={k_max:.4e}")


def default_iters(num_tokens: int, quick: bool) -> int:
    if quick:
        return 20 if num_tokens >= 1024 else 100
    if num_tokens >= 4096:
        return 50
    if num_tokens >= 1024:
        return 100
    if num_tokens >= 64:
        return 300
    return 1000


def default_warmup(num_tokens: int, quick: bool) -> int:
    if quick:
        return 5
    if num_tokens >= 1024:
        return 20
    return 50


def measure(
    fn: Callable[[], None],
    *,
    warmup: int,
    iters: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter()
    event_us = start.elapsed_time(end) * 1000.0 / iters
    wall_us = (wall_end - wall_start) * 1_000_000.0 / iters
    return event_us, wall_us


def bench_shape(
    shape: Shape,
    *,
    device: str,
    max_pos: int,
    warmup: int | None,
    iters: int | None,
    quick: bool,
    reset_inputs: bool,
    check_correctness: bool,
) -> tuple[BenchResult, BenchResult]:
    positions, query, key, rope = make_inputs(shape, max_pos=max_pos, device=device)
    cos_sin_cache = rope.cos_sin_cache.to(query.device, dtype=query.dtype)

    # Trigger JIT compilation before timing.
    jit_rope(positions, query.clone(), key.clone(), shape, cos_sin_cache)
    torch.cuda.synchronize()

    if check_correctness:
        assert_close(shape, rope, positions, query, key)

    shape_warmup = (
        warmup if warmup is not None else default_warmup(shape.num_tokens, quick)
    )
    shape_iters = iters if iters is not None else default_iters(shape.num_tokens, quick)

    def run_native():
        native_rope(rope, positions, query, key)

    if reset_inputs:
        q_buf = torch.empty_like(query)
        k_buf = torch.empty_like(key)

        def run_jit():
            q_buf.copy_(query)
            k_buf.copy_(key)
            jit_rope(positions, q_buf, k_buf, shape, cos_sin_cache)

    else:
        q_buf = query.clone()
        k_buf = key.clone()

        def run_jit():
            jit_rope(positions, q_buf, k_buf, shape, cos_sin_cache)

    native_event_us, native_wall_us = measure(
        run_native, warmup=shape_warmup, iters=shape_iters
    )
    jit_event_us, jit_wall_us = measure(run_jit, warmup=shape_warmup, iters=shape_iters)

    native = BenchResult(
        shape.label,
        "native",
        shape.num_tokens,
        shape.num_heads_per_rank,
        shape.num_kv_heads_per_rank,
        shape_iters,
        native_event_us,
        native_wall_us,
    )
    jit = BenchResult(
        shape.label,
        "jit",
        shape.num_tokens,
        shape.num_heads_per_rank,
        shape.num_kv_heads_per_rank,
        shape_iters,
        jit_event_us,
        jit_wall_us,
    )
    return native, jit


def parse_shape_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def print_results(results: list[tuple[BenchResult, BenchResult]]) -> None:
    print(
        "shape                          toks  qh kvh iters  "
        "native_event_us  jit_event_us  event_speedup  "
        "native_wall_us   jit_wall_us   wall_speedup"
    )
    for native, jit in results:
        event_speedup = native.event_us / jit.event_us if jit.event_us else math.nan
        wall_speedup = native.wall_us / jit.wall_us if jit.wall_us else math.nan
        print(
            f"{native.label:<30} {native.tokens:>4} "
            f"{native.q_heads:>3} {native.kv_heads:>3} {native.iters:>5} "
            f"{native.event_us:>15.3f} {jit.event_us:>13.3f} "
            f"{event_speedup:>14.2f} "
            f"{native.wall_us:>15.3f} {jit.wall_us:>13.3f} "
            f"{wall_speedup:>12.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark vllm-musa JIT RoPE against vLLM native RoPE."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pos", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument(
        "--shapes",
        default=None,
        help="Comma-separated shape labels to run. Defaults to all shapes.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use short warmup/iteration counts for smoke validation.",
    )
    parser.add_argument(
        "--reset-inputs",
        action="store_true",
        help="Copy pristine inputs into the JIT buffers before every iteration.",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the one-shot native-vs-JIT output check before timing.",
    )
    args = parser.parse_args()

    if not torch.musa.is_available():
        print("FAIL no MUSA device available")
        return 1

    shape_filter = parse_shape_filter(args.shapes)
    unknown = (
        shape_filter - {shape.label for shape in SHAPES} if shape_filter else set()
    )
    if unknown:
        print(f"FAIL unknown shape label(s): {', '.join(sorted(unknown))}")
        return 1

    torch.cuda.set_device(args.device)
    print(
        f"=== JIT RoPE microbenchmark on {args.device}; "
        f"reset_inputs={args.reset_inputs} ==="
    )
    results = []
    for shape in SHAPES:
        if shape_filter and shape.label not in shape_filter:
            continue
        native, jit = bench_shape(
            shape,
            device=args.device,
            max_pos=args.max_pos,
            warmup=args.warmup,
            iters=args.iters,
            quick=args.quick,
            reset_inputs=args.reset_inputs,
            check_correctness=not args.skip_correctness,
        )
        results.append((native, jit))
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
