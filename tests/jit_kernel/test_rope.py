"""MUSA-0203 rope kernel unit + integration tests.

Compare the PR #47 csrc-JIT rope kernel against vLLM's upstream
`RotaryEmbedding.forward_native` (pure PyTorch reference) for the
shape matrix relevant to the M2.5 + Eagle3 SOTA failure.

Three test layers:

  1. Eager parity   — call the JIT directly and the native ref, compare.
  2. CUDAGraph parity (single capture) — wrap each in a single
     CUDAGraph capture and compare replay outputs to the eager refs.
  3. Multi-replay   — capture once, replay many times with the same
     position values, confirm no firmware reboot / NaN drift.

If 1 passes but 2 fails -> bug in the JIT kernel's CUDAGraph
compatibility (the most likely culprit per our bisect).
If 1 fails -> bug in the kernel itself even in eager mode.
If 1+2 pass but 3 fails -> bug in repeated replay state.

Run on the authorized MUSA container (TP=1 / single rank to isolate
from any distributed-side effects).
"""

import sys
import traceback
from dataclasses import dataclass

import torch

# torchada redirects torch.cuda symbols to MUSA — must be first.
import torchada  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

# Plug into vllm_musa so torch.ops.vllm.musa_rotary_embedding is registered
# and the MusaRotaryEmbedding OOT class is registered.
import vllm_musa  # noqa: F401

# Explicit import: register `torch.ops.vllm.musa_rotary_embedding`.
from vllm_musa.jit_kernel.csrc import rope as _rope_module  # noqa: F401

# RotaryEmbedding.__init__ -> CustomOp.__init__ -> dispatch_forward() reads the
# current vllm config. We enter set_current_vllm_config() inside main() (NOT at
# import time) so importing this module -- e.g. during pytest collection -- does
# not leak a global vLLM config into other tests in the same process. (PR #50)


# ---- shape matrix ----------------------------------------------------------


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


# The "did this work for the user matrix" column tracks what we observed
# in MUSA-0202 / MUSA-0203 testing on yeahdongcn70.
SHAPES = [
    # Eagle3 draft, the failing case at TP=8 + captured chain
    Shape(
        "eagle3_draft_tp8_decode",
        num_heads_per_rank=3,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1,
    ),
    Shape(
        "eagle3_draft_tp8_chain8",
        num_heads_per_rank=3,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=8,
    ),
    # M2.5 target at TP=8
    Shape(
        "m25_target_tp8_decode",
        num_heads_per_rank=6,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1,
    ),
    Shape(
        "m25_target_tp8_prefill",
        num_heads_per_rank=6,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=4096,
    ),
    # Qwen3-8B-FP8 at TP=8 (known-working control)
    Shape(
        "qwen3_8b_tp8_decode",
        num_heads_per_rank=4,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1,
    ),
    Shape(
        "qwen3_8b_tp8_prefill",
        num_heads_per_rank=4,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1024,
    ),
    # Qwen3-30B-A3B-FP8 at TP=2 (known-working control)
    Shape(
        "qwen3_30b_tp2_decode",
        num_heads_per_rank=16,
        num_kv_heads_per_rank=4,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1,
    ),
    # Edge: smallest possible num_heads (boundary case)
    Shape(
        "min_heads_1_decode",
        num_heads_per_rank=1,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1,
    ),
    Shape(
        "min_heads_2_decode",
        num_heads_per_rank=2,
        num_kv_heads_per_rank=1,
        head_dim=128,
        rotary_dim=128,
        num_tokens=1,
    ),
]


# ---- harness ---------------------------------------------------------------


def make_inputs(shape: Shape, max_pos: int = 8192, device: str = "cuda"):
    """Returns (positions, query, key, cos_sin_cache, jit_module).

    The same inputs are used for both the JIT call and the native ref.
    Query/key are cloned before each call so the in-place mutation
    doesn't contaminate the reference.
    """
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

    # Build a RotaryEmbedding to inherit its forward_native + cache shape.
    rope = RotaryEmbedding(
        head_size=shape.head_dim,
        rotary_dim=shape.rotary_dim,
        max_position_embeddings=max_pos,
        base=10000.0,
        is_neox_style=shape.is_neox,
        dtype=shape.dtype,
        init_cache=True,
    ).to(device)
    return positions, query, key, rope


def eager_native(rope, positions, query, key):
    """Call upstream forward_static (the pure-PyTorch reference).

    `forward_static` is a @staticmethod on `RotaryEmbedding` but
    `nn.Module.__getattr__` shadows class attribute lookup, so we
    invoke it directly through the class.
    """
    q = query.clone()
    k = key.clone()
    out_q, out_k = RotaryEmbedding.forward_static(
        positions,
        q,
        k,
        rope.head_size,
        rope.rotary_dim,
        rope.cos_sin_cache,
        rope.is_neox_style,
    )
    return out_q, out_k


def eager_jit(positions, query, key, head_size, cos_sin_cache, is_neox):
    """Call the JIT rope via the registered custom op."""
    q = query.clone()
    k = key.clone()
    # Match dtype / device of cos_sin_cache to query (the wrapper does this).
    csc = cos_sin_cache.to(query.device, dtype=query.dtype)
    torch.ops.vllm.musa_rotary_embedding(positions, q, k, head_size, csc, is_neox)
    return q, k


def compare(label: str, kind: str, q_ref, k_ref, q_test, k_test, atol=1e-2, rtol=1e-2):
    """Compare two (q, k) outputs and print a single PASS/FAIL line."""
    if q_test.shape != q_ref.shape:
        print(
            f"FAIL {label} {kind} SHAPE_MISMATCH q_ref={q_ref.shape} q_test={q_test.shape}"
        )
        return False
    if k_test.shape != k_ref.shape:
        print(
            f"FAIL {label} {kind} SHAPE_MISMATCH k_ref={k_ref.shape} k_test={k_test.shape}"
        )
        return False
    q_close = torch.allclose(q_ref.float(), q_test.float(), atol=atol, rtol=rtol)
    k_close = torch.allclose(k_ref.float(), k_test.float(), atol=atol, rtol=rtol)
    q_max = (q_ref.float() - q_test.float()).abs().max().item()
    k_max = (k_ref.float() - k_test.float()).abs().max().item()
    if q_close and k_close:
        print(f"PASS {label} {kind} q_max_diff={q_max:.4e} k_max_diff={k_max:.4e}")
        return True
    else:
        print(
            f"FAIL {label} {kind} q_close={q_close} k_close={k_close} "
            f"q_max_diff={q_max:.4e} k_max_diff={k_max:.4e}"
        )
        return False


def check_eager_parity(shape: Shape) -> bool:
    """Layer 1: JIT vs native, both in eager mode."""
    try:
        positions, query, key, rope = make_inputs(shape)
        q_native, k_native = eager_native(rope, positions, query, key)
        q_jit, k_jit = eager_jit(
            positions,
            query,
            key,
            shape.head_dim,
            rope.cos_sin_cache,
            shape.is_neox,
        )
        return compare(shape.label, "EAGER", q_native, k_native, q_jit, k_jit)
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL {shape.label} EAGER EXC={e}")
        return False


def _capture_replay(callable_fn, *args):
    """Capture a CUDAGraph that runs `callable_fn(*args)` and replay it once.

    Returns the output tensors (which are pool-resident — clone them out
    if you want to retain across replays).
    """
    # Warm-up to allocate any lazy state.
    out = callable_fn(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            out = callable_fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    g.replay()
    torch.cuda.synchronize()
    return out


def check_captured_jit_parity(shape: Shape) -> bool:
    """Layer 2: JIT inside captured CUDAGraph vs native eager."""
    try:
        positions, query, key, rope = make_inputs(shape)
        q_native, k_native = eager_native(rope, positions, query, key)

        # Pre-allocate stable buffers that the captured graph will write into.
        q_buf = query.clone()
        k_buf = key.clone()
        csc = rope.cos_sin_cache.to(query.device, dtype=query.dtype)

        def fn():
            torch.ops.vllm.musa_rotary_embedding(
                positions,
                q_buf,
                k_buf,
                shape.head_dim,
                csc,
                shape.is_neox,
            )

        # Reset buffers, capture + replay.
        q_buf.copy_(query)
        k_buf.copy_(key)
        # Warm-up call (not captured).
        fn()
        # Reset for the captured replay so the comparison is clean.
        q_buf.copy_(query)
        k_buf.copy_(key)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        # Now reset inputs again and replay.
        q_buf.copy_(query)
        k_buf.copy_(key)
        g.replay()
        torch.cuda.synchronize()
        return compare(shape.label, "CAPTURED", q_native, k_native, q_buf, k_buf)
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL {shape.label} CAPTURED EXC={e}")
        return False


def check_multi_replay(shape: Shape, n_replays: int = 32) -> bool:
    """Layer 3: capture once, then replay N times with the same captured
    inputs, checking for NaN/Inf after each replay.

    If the kernel has a stale-pointer or first-call init issue, repeated
    replays will surface it as NaN/Inf or a crash.
    """
    try:
        positions, query, key, rope = make_inputs(shape)
        q_buf = query.clone()
        k_buf = key.clone()
        csc = rope.cos_sin_cache.to(query.device, dtype=query.dtype)

        def fn():
            torch.ops.vllm.musa_rotary_embedding(
                positions,
                q_buf,
                k_buf,
                shape.head_dim,
                csc,
                shape.is_neox,
            )

        # Capture
        fn()
        q_buf.copy_(query)
        k_buf.copy_(key)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                fn()
        torch.cuda.current_stream().wait_stream(s)

        # Run n_replays times, checking NaN/Inf each time.
        for i in range(n_replays):
            q_buf.copy_(query)
            k_buf.copy_(key)
            g.replay()
            torch.cuda.synchronize()
            if torch.isnan(q_buf).any() or torch.isnan(k_buf).any():
                print(f"FAIL {shape.label} MULTI_REPLAY NaN at replay #{i}")
                return False
            if torch.isinf(q_buf).any() or torch.isinf(k_buf).any():
                print(f"FAIL {shape.label} MULTI_REPLAY Inf at replay #{i}")
                return False

        print(f"PASS {shape.label} MULTI_REPLAY n={n_replays}")
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"FAIL {shape.label} MULTI_REPLAY EXC={e}")
        return False


# ---- main ------------------------------------------------------------------


def main() -> int:
    if not torch.musa.is_available():
        print("FAIL no MUSA device available")
        return 1
    torch.musa.set_device(0)
    # PR #50 review: enter the vLLM config here (not at import) -- see note above.
    set_current_vllm_config(VllmConfig()).__enter__()
    print(f"=== MUSA-0203 rope unit tests on device {torch.musa.current_device()} ===")
    print(f"shapes tested: {len(SHAPES)}\n")

    results = []
    for shape in SHAPES:
        print(
            f"--- {shape.label}  (q={shape.num_heads_per_rank}, "
            f"kv={shape.num_kv_heads_per_rank}, head_dim={shape.head_dim}, "
            f"rot_dim={shape.rotary_dim}, n_tok={shape.num_tokens}) ---"
        )
        r1 = check_eager_parity(shape)
        r2 = check_captured_jit_parity(shape)
        r3 = check_multi_replay(shape, n_replays=32)
        results.append((shape.label, r1, r2, r3))

    print("\n=== Summary ===")
    print(f"{'shape':<35} {'EAGER':>6} {'CAPTURED':>9} {'MULTI':>7}")
    fails = 0
    for label, r1, r2, r3 in results:
        flags = (
            "PASS" if r1 else "FAIL",
            "PASS" if r2 else "FAIL",
            "PASS" if r3 else "FAIL",
        )
        if not (r1 and r2 and r3):
            fails += 1
        print(f"{label:<35} {flags[0]:>6} {flags[1]:>9} {flags[2]:>7}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} shape(s) FAILED'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
