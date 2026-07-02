#!/usr/bin/env python3
"""Sweep Qwen FP8 MoE dispatch sizes for the MUSA DeepGEMM threshold.

This is a synthetic single-layer benchmark for the vLLM-MUSA implementation. It
uses Qwen3.5-35B-A3B-FP8 TP4 local shapes and compares the native MUSA fused-MoE
fallback path against the grouped DeepGEMM prefill path across token counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch
import torch_musa  # noqa: F401


DEFAULT_M_LIST = (
    "1024,2048,4096,8192,11755,16384,20000,32768,"
    "60001,65536,68936,80012,88936,131072"
)
DEFAULT_OUT_DIR = Path("/tmp/vllm_musa_qwen_fp8_moe_threshold")
DISABLE_QWEN_PREFILL_TOKENS = 1 << 60


def parse_int_list(value: str) -> list[int]:
    items = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError(f"empty integer list: {value!r}")
    return items


def sync() -> None:
    torch.musa.synchronize()


@contextmanager
def patched_qwen_prefill_threshold(fused_moe_module, min_tokens: int):
    # Force the unified dispatch to grouped DeepGEMM at a chosen crossover via
    # the global env override, or disable it entirely for the native baseline.
    from vllm_musa.model_executor.layers.fused_moe import moe_dispatch

    names = ("VLLM_MUSA_MOE_DEEPGEMM", "VLLM_MUSA_MOE_DEEPGEMM_MIN_TOKENS")
    saved = {name: os.environ.get(name) for name in names}
    try:
        if min_tokens >= DISABLE_QWEN_PREFILL_TOKENS:
            os.environ["VLLM_MUSA_MOE_DEEPGEMM"] = "0"
        else:
            os.environ["VLLM_MUSA_MOE_DEEPGEMM"] = "1"
            os.environ["VLLM_MUSA_MOE_DEEPGEMM_MIN_TOKENS"] = str(min_tokens)
        moe_dispatch.reset_tuned_cache()
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        moe_dispatch.reset_tuned_cache()


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip()
    except Exception:
        return ""


def make_qwen_weights(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("musa")
    # Keep synthetic weights in a moderate range so FP8 random data does not
    # saturate and obscure path-to-path comparison metrics.
    w1 = (torch.randn(
        (num_experts, 2 * intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    ) * 0.05).to(torch.float8_e4m3fn)
    w2 = (torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    ) * 0.05).to(torch.float8_e4m3fn)
    w1_scale = torch.full(
        (num_experts, (2 * intermediate_size) // 128, hidden_size // 128),
        0.01,
        device=device,
        dtype=torch.float32,
    )
    w2_scale = torch.full(
        (num_experts, hidden_size // 128, intermediate_size // 128),
        0.01,
        device=device,
        dtype=torch.float32,
    )
    return {
        "w1": w1.contiguous(),
        "w2": w2.contiguous(),
        "w1_scale": w1_scale.contiguous(),
        "w2_scale": w2_scale.contiguous(),
    }


def make_topk_ids(
    m: int,
    *,
    num_experts: int,
    topk: int,
    routing: str,
    seed: int,
) -> torch.Tensor:
    device = torch.device("musa")
    if routing == "uniform":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + m)
        return torch.randint(
            0,
            num_experts,
            (m, topk),
            device=device,
            dtype=torch.int32,
            generator=generator,
        ).contiguous()

    if routing == "deterministic-skew":
        base = torch.arange(m, device=device, dtype=torch.int32)
        ids = torch.empty((m, topk), device=device, dtype=torch.int32)
        multipliers = [1, 7, 13, 29, 53, 97, 113, 193]
        offsets = [35, 53, 96, 146, 200, 236, 17, 89]
        for slot in range(topk):
            ids[:, slot] = (
                base * multipliers[slot % len(multipliers)]
                + offsets[slot % len(offsets)]
            ) % num_experts
        return ids.contiguous()

    raise ValueError(f"unknown routing mode: {routing}")


def make_qwen_inputs(
    m: int,
    *,
    weights: dict[str, torch.Tensor],
    hidden_size: int,
    num_experts: int,
    topk: int,
    seed: int,
    routing: str,
) -> dict[str, torch.Tensor | list[int] | bool | str | None]:
    torch.manual_seed(seed + m)
    device = torch.device("musa")
    hidden_states = (
        torch.randn((m, hidden_size), device=device, dtype=torch.bfloat16) * 0.05
    ).contiguous()
    topk_ids = make_topk_ids(
        m, num_experts=num_experts, topk=topk, routing=routing, seed=seed
    )
    topk_weights = torch.rand((m, topk), device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return {
        "hidden_states": hidden_states,
        "w1": weights["w1"],
        "w2": weights["w2"],
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "activation": "silu",
        "apply_router_weight_on_input": False,
        "use_fp8_w8a8": True,
        "use_int8_w8a8": False,
        "use_int8_w8a16": False,
        "use_int4_w4a16": False,
        "ocp_mx_scheme": None,
        "per_channel_quant": False,
        "global_num_experts": num_experts,
        "expert_map": None,
        "w1_scale": weights["w1_scale"],
        "w2_scale": weights["w2_scale"],
        "w1_zp": None,
        "w2_zp": None,
        "a1_scale": None,
        "a2_scale": None,
        "block_shape": [128, 128],
        "w1_bias": None,
        "w2_bias": None,
        "inplace": False,
    }


def benchmark_us(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        del out
    sync()

    samples = []
    for _ in range(iters):
        sync()
        start = time.perf_counter()
        out = fn()
        sync()
        del out
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def measure_one(
    fused_moe_module,
    tensors: dict,
    *,
    mode: str,
    warmup: int,
    iters: int,
) -> tuple[float, str]:
    if mode == "native":
        min_tokens = DISABLE_QWEN_PREFILL_TOKENS
    elif mode == "deepgemm":
        min_tokens = 0
    else:
        raise ValueError(f"unknown mode: {mode}")

    with patched_qwen_prefill_threshold(fused_moe_module, min_tokens):
        try:
            latency = benchmark_us(
                lambda: fused_moe_module.fused_experts_impl(**tensors),
                warmup=warmup,
                iters=iters,
            )
            status = "ok"
        except Exception as exc:
            sync()
            latency = float("nan")
            status = repr(exc)
    return latency, status


def run_once(fused_moe_module, tensors: dict, *, mode: str) -> torch.Tensor:
    if mode == "native":
        min_tokens = DISABLE_QWEN_PREFILL_TOKENS
    elif mode == "deepgemm":
        min_tokens = 0
    else:
        raise ValueError(f"unknown mode: {mode}")
    with patched_qwen_prefill_threshold(fused_moe_module, min_tokens):
        return fused_moe_module.fused_experts_impl(**tensors)


def compare_outputs(
    fused_moe_module,
    tensors: dict,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    native_out = run_once(fused_moe_module, tensors, mode="native")
    sync()
    deepgemm_out = run_once(fused_moe_module, tensors, mode="deepgemm")
    sync()
    native_f = native_out.float()
    deepgemm_f = deepgemm_out.float()
    diff = (native_f - deepgemm_f).abs()
    denom = native_f.abs().mean().clamp_min(1e-6)
    allclose = bool(torch.allclose(deepgemm_f, native_f, atol=atol, rtol=rtol))
    finite = bool(torch.isfinite(native_f).all().item() and torch.isfinite(deepgemm_f).all().item())
    return {
        "status": "ok" if finite else "nonfinite",
        "allclose": allclose,
        "atol": atol,
        "rtol": rtol,
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "relative_mean_abs": float((diff.mean() / denom).item()),
        "native_finite": bool(torch.isfinite(native_f).all().item()),
        "deepgemm_finite": bool(torch.isfinite(deepgemm_f).all().item()),
    }


def _smooth_winners(points: list[dict]) -> list[dict]:
    if len(points) < 3:
        return points
    smoothed = [dict(point) for point in points]
    for index in range(1, len(points) - 1):
        prev_winner = smoothed[index - 1]["winner"]
        next_winner = points[index + 1]["winner"]
        if prev_winner == next_winner and smoothed[index]["winner"] != prev_winner:
            smoothed[index]["winner"] = prev_winner
            smoothed[index]["smoothed"] = True
    return smoothed


def choose_threshold(rows: Iterable[dict], *, margin: float) -> dict:
    by_m: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["status"] == "ok":
            by_m[int(row["m"])][row["mode"]] = row

    compared = []
    for m in sorted(by_m):
        modes = by_m[m]
        if "native" not in modes or "deepgemm" not in modes:
            continue
        native_us = float(modes["native"]["latency_us"])
        deepgemm_us = float(modes["deepgemm"]["latency_us"])
        winner = "deepgemm" if deepgemm_us <= native_us * margin else "native"
        compared.append(
            {
                "m": m,
                "native_us": native_us,
                "deepgemm_us": deepgemm_us,
                "ratio_deepgemm_over_native": deepgemm_us / native_us,
                "winner": winner,
            }
        )

    compared = _smooth_winners(compared)

    if not compared:
        return {
            "recommended_min_tokens": None,
            "policy": "insufficient data",
            "points": compared,
        }

    min_tokens = None
    for point in compared:
        if point["winner"] == "deepgemm":
            min_tokens = point["m"]
            break

    if min_tokens is None:
        policy = (
            "native for all measured token counts; extend --m-list above "
            f"{compared[-1]['m']} before selecting a production threshold"
        )
    else:
        policy = f"native < {min_tokens}, deepgemm >= {min_tokens}"

    return {
        "recommended_min_tokens": min_tokens,
        "policy": policy,
        "margin": margin,
        "points": compared,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-list", default=DEFAULT_M_LIST)
    parser.add_argument("--correctness-m-list", default="1024,4096")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--margin", type=float, default=0.98)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--routing",
        choices=["uniform", "deterministic-skew"],
        default="uniform",
        help="Synthetic top-k route distribution for the sweep.",
    )
    parser.add_argument("--correctness-atol", type=float, default=2e-2)
    parser.add_argument("--correctness-rtol", type=float, default=2e-2)
    parser.add_argument("--fail-on-correctness-mismatch", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available")

    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "1")
    os.environ.setdefault("VLLM_USE_DEEP_GEMM_E8M0", "0")
    os.environ.setdefault("VLLM_MUSA_SILU_FP8_QUANT_MAX_TOKENS", "1000000")

    try:
        from vllm_musa.patches import apply_patches

        apply_patches(force=True)
    except Exception as exc:
        print(f"WARN apply_patches failed: {exc!r}", flush=True)

    from vllm_musa.model_executor.layers.fused_moe import fused_moe

    torch.musa.set_device(0)
    weights = make_qwen_weights(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        seed=args.seed,
    )
    rows = []
    m_list = parse_int_list(args.m_list)
    correctness_m_list = parse_int_list(args.correctness_m_list)
    for m in m_list:
        tensors = make_qwen_inputs(
            m,
            weights=weights,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            topk=args.topk,
            seed=args.seed,
            routing=args.routing,
        )
        for mode in ("native", "deepgemm"):
            latency_us, status = measure_one(
                fused_moe,
                tensors,
                mode=mode,
                warmup=args.warmup,
                iters=args.iters,
            )
            row = {
                "m": m,
                "mode": mode,
                "latency_us": latency_us,
                "tokens_per_s": (m * 1e6 / latency_us)
                if status == "ok" and latency_us > 0
                else "",
                "status": status,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_experts": args.num_experts,
                "topk": args.topk,
                "routing": args.routing,
            }
            rows.append(row)
            if args.progress:
                print(json.dumps(row, sort_keys=True), flush=True)
        del tensors
        torch.musa.empty_cache()

    correctness_rows = []
    for m in correctness_m_list:
        tensors = make_qwen_inputs(
            m,
            weights=weights,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            topk=args.topk,
            seed=args.seed,
            routing=args.routing,
        )
        try:
            comparison = compare_outputs(
                fused_moe,
                tensors,
                atol=args.correctness_atol,
                rtol=args.correctness_rtol,
            )
        except Exception as exc:
            sync()
            comparison = {"status": repr(exc), "allclose": False}
        comparison.update(
            {
                "m": m,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_experts": args.num_experts,
                "topk": args.topk,
                "routing": args.routing,
            }
        )
        correctness_rows.append(comparison)
        if args.progress:
            print(json.dumps(comparison, sort_keys=True), flush=True)
        del tensors
        torch.musa.empty_cache()

    recommendation = choose_threshold(rows, margin=args.margin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "qwen_fp8_moe_threshold_raw.csv", rows)
    write_csv(
        args.out_dir / "qwen_fp8_moe_threshold_correctness.csv",
        correctness_rows,
    )
    (args.out_dir / "qwen_fp8_moe_threshold_raw.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "qwen_fp8_moe_threshold_correctness.json").write_text(
        json.dumps(correctness_rows, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "qwen_fp8_moe_threshold_recommendation.json").write_text(
        json.dumps(recommendation, indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "qwen_fp8_moe_threshold_metadata.json").write_text(
        json.dumps(
            {
                "argv": os.sys.argv,
                "git_head": _git_value(["rev-parse", "HEAD"]),
                "git_branch": _git_value(["branch", "--show-current"]),
                "git_status_short": _git_value(["status", "--short"]),
                "m_list": m_list,
                "correctness_m_list": correctness_m_list,
                "warmup": args.warmup,
                "iters": args.iters,
                "seed": args.seed,
                "margin": args.margin,
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_experts": args.num_experts,
                "topk": args.topk,
                "routing": args.routing,
                "env": {
                    key: os.environ.get(key, "")
                    for key in (
                        "VLLM_USE_DEEP_GEMM",
                        "VLLM_USE_DEEP_GEMM_E8M0",
                        "VLLM_MUSA_SILU_FP8_QUANT_MAX_TOKENS",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if args.fail_on_correctness_mismatch:
        failed = [
            row
            for row in correctness_rows
            if row.get("status") != "ok" or not row.get("allclose")
        ]
        if failed:
            raise SystemExit(
                "FAIL correctness mismatch: "
                + json.dumps(failed, sort_keys=True)
            )

    print(json.dumps(recommendation, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
