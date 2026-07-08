# Ready-to-apply port: MUSA sparse-MLA attention via SGLang's TileLang kernel.
#
# Deploy as: vllm-musa/vllm_musa/v1/attention/ops/sparse_mla_tilelang.py
# Then swap the call in vllm_musa/v1/attention/ops/flashmla.py::flash_mla_sparse_fwd
# (snippet at the bottom of this file).
#
# The kernel `sparse_attention_fwd_kernel_v1` is copied VERBATIM from
# sglang/srt/layers/attention/nsa/tilelang_kernel.py:233-399 — the exact kernel
# SGLang-MUSA runs for GLM-5.2 DSA. num_stages defaults to 2 there; the MUSA
# call site passes num_stages=0 (tilelang_kernel.py:1591). Keep num_stages=0.

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}
if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

# MUSA: TileLang disk cache is unreliable; recompile each process.
tilelang.disable_cache()


# --- BEGIN verbatim copy from sglang nsa/tilelang_kernel.py:233-399 ---
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd_kernel_v1(
    num_heads, dim, tail_dim, topk, *, kv_group=1, sm_scale=None,
    is_causal=True, block_I=64, num_stages=2, threads=256,
):
    assert dim == tilelang.math.next_power_of_2(dim), f"dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"tail_dim={tail_dim}"
    assert is_causal is True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise loads index=0 -> wrong kv"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")
    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    dtype = "bfloat16"
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim
    if head_kv > 64:
        assert head_kv % 64 == 0
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, "int32"),
        Output: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            mask = T.alloc_fragment([BI], "bool")
            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0
                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]
                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

    return main
# --- END verbatim copy ---


def sparse_mla_fwd_bf16(q, kv, indices, sm_scale, d_v=512):
    """MUSA bf16 sparse-MLA forward — mate-`flash_mla_sparse_fwd`-compatible.

    q       [T, H, 576] bf16   (nope 512 + rope 64)
    kv      [slots, 1, 576] bf16
    indices [T, 1, 2048] int32 (-1 = invalid, absolute rows into kv)
    returns [T, H, 512] bf16
    """
    num_heads = q.shape[1]
    dim = q.shape[2]
    tail_dim = dim - d_v
    topk = indices.shape[-1]
    assert topk == 2048, f"kernel requires topk==2048, got {topk}; right-pad indices with -1"
    kernel = sparse_attention_fwd_kernel_v1(
        num_heads, d_v, tail_dim, topk, sm_scale=sm_scale, num_stages=0
    )
    out = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))  # [1, T, H, 512]
    return out.squeeze(0)


_PREWARMED = set()


def prewarm(num_heads=64, d_v=512, tail_dim=64, topk=2048, device="musa", sm_scale=1.0):
    """Compile the kernel OUTSIDE CUDAGraph capture (first-compile-in-capture deadlocks)."""
    key = (num_heads, d_v, tail_dim, topk)
    if key in _PREWARMED:
        return
    q = torch.zeros(1, num_heads, d_v + tail_dim, dtype=torch.bfloat16, device=device)
    kv = torch.zeros(1, 1, d_v + tail_dim, dtype=torch.bfloat16, device=device)
    idx = torch.full((1, 1, topk), -1, dtype=torch.int32, device=device)
    sparse_mla_fwd_bf16(q, kv, idx, sm_scale, d_v=d_v)
    _PREWARMED.add(key)


