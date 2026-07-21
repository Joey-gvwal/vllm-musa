# SPDX-License-Identifier: Apache-2.0
"""Correctness and eager latency A/B for the DSV4 FlashMLA cache pack."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from vllm_musa import _custom_ops as ops


PACK_IMPL_ENV = "VLLM_MUSA_DEEPSEEK_V4_QNORM_ROPE_KV_PACK_IMPL"
FUSED_INSERT_ENV = "VLLM_MUSA_DEEPSEEK_V4_QNORM_ROPE_KV_INSERT_FUSED"


def make_inputs(num_tokens: int, num_heads: int, block_size: int):
    torch.manual_seed(0)
    device = torch.device("musa")
    q = torch.randn(
        (num_tokens, num_heads, 512), device=device, dtype=torch.bfloat16
    )
    kv = torch.randn((num_tokens, 512), device=device, dtype=torch.bfloat16)
    token_bytes = 448 + 64 * 2 + 8
    cache = torch.full(
        (2, block_size, token_bytes), 0xA5, device=device, dtype=torch.uint8
    )
    slots = torch.arange(num_tokens, device=device, dtype=torch.int64)
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64) + 17
    cos_sin = torch.randn((256, 64), device=device, dtype=torch.float32)
    return q, kv, cache, slots, positions, cos_sin


def run_once(impl: str, inputs, block_size: int):
    q, kv, cache, slots, positions, cos_sin = inputs
    q = q.clone()
    cache = cache.clone()
    os.environ[PACK_IMPL_ENV] = impl
    ops.deepseek_v4_qnorm_rope_kv_insert(
        q, kv, cache, slots, positions, cos_sin, 1.0e-6, block_size
    )
    torch.musa.synchronize()
    return q, cache


def bench(impl: str, inputs, block_size: int, warmups: int, iterations: int):
    q, kv, cache, slots, positions, cos_sin = inputs
    q = q.clone()
    cache = cache.clone()
    os.environ[PACK_IMPL_ENV] = impl
    for _ in range(warmups):
        ops.deepseek_v4_qnorm_rope_kv_insert(
            q, kv, cache, slots, positions, cos_sin, 1.0e-6, block_size
        )
    torch.musa.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        ops.deepseek_v4_qnorm_rope_kv_insert(
            q, kv, cache, slots, positions, cos_sin, 1.0e-6, block_size
        )
    torch.musa.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    os.environ[FUSED_INSERT_ENV] = "1"
    inputs = make_inputs(args.tokens, args.heads, args.block_size)
    legacy_q, legacy_cache = run_once("legacy", inputs, args.block_size)
    optimized_q, optimized_cache = run_once("optimized", inputs, args.block_size)
    result = {
        "tokens": args.tokens,
        "heads": args.heads,
        "q_equal": bool(torch.equal(legacy_q, optimized_q)),
        "cache_equal": bool(torch.equal(legacy_cache, optimized_cache)),
    }
    if not result["q_equal"] or not result["cache_equal"]:
        result["cache_mismatch_count"] = int(
            torch.count_nonzero(legacy_cache != optimized_cache).item()
        )
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(1)

    result["legacy_ms"] = bench(
        "legacy", inputs, args.block_size, args.warmups, args.iterations
    )
    result["optimized_ms"] = bench(
        "optimized", inputs, args.block_size, args.warmups, args.iterations
    )
    result["speedup"] = result["legacy_ms"] / result["optimized_ms"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
