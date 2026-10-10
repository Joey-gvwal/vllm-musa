# vLLM v0.30.0 build-time patch series

The files in this directory apply to the immutable vLLM commit named by
`VLLM_COMMIT` in `third_party/PINS` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`,
release tag `v0.30.0`). `setup.py` applies them in filename order before the
editable or wheel install. They are source patches; runtime registrations in
`vllm_musa/patches/` are separate.

| Patch | Purpose |
| --- | --- |
| `0001-MUSA-v0.30.0-faithful-port.patch` | Base v0.30 MUSA adaptation, including the native build, attention, model runner, and KV cache paths. |
| `0002-MUSA-skip-unsafe-v2-warmup-for-Qwen3.5-family.patch` | Skip the v0.30 V2 dummy-decode warmup on MUSA Qwen3.5-family models; preserve CUDAGraph capture and semantic output. |
| `0003-MUSA-preserve-dedicated-Mamba-prefix-cache-on-v0.30.patch` | Port the applicable v0.28 `0138` dedicated Mamba pool prefix-cache changes. v0.30 already routes block frees by `block.pool`; this patch preserves the remaining cache-order, hashed-state, and null-state behavior. |
| `0004-MUSA-rotate-V2-output-copy-streams-on-v0.30.patch` | Port the v0.28 `0172` per-step V2 output-copy streams to v0.30. |
| `0005-MUSA-skip-Prometheus-observe-for-connectors-without-.patch` | Port the v0.28 `0173` `MultiConnector` metrics fix: skip the Prometheus observe for a connector that reports transfer stats without registering metrics (e.g. `MooncakeConnector`). |
| `0006-perf-musa-extend-DSV4-score-DeepGEMM-to-decode-batch.patch` | Port the v0.28 `0174` DeepSeek-V4 FP32-output score projections to DeepGEMM up to the multi-stream token threshold. |
| `0007-MUSA-replay-the-decode-graph-for-padded-final-prompt.patch` | Port the v0.28 `0175` contract-gated V2 decode-graph replay for a remote-prefilled final prompt token padded to the verify shape. |
| `0008-MUSA-pass-the-SwiGLU-clamp-through-functional-fused_.patch` | Port the v0.28 `0176` `gemm1_clamp_limit` forwarding from the functional `fused_experts` entry point to the SiLU-and-mul activation. |
| `0009-MUSA-route-DSV4-decode-indexer-top-k-to-its-own-op.patch` | Port the v0.28 `0177` DeepSeek-V4 decode indexer top-k op; GLM-5.2 rows keep the shared op. |
| `0010-MUSA-shorten-the-DFlash-input-prep-padding-tail.patch` | Port the v0.28 `0178` fixed 1024-wide DFlash input-prep padding blocks. |
| `0011-MUSA-route-DeepSeek-V4-prefill-score-GEMM-to-DeepGEM.patch` | Port the v0.28 `0179` contract-gated DeepGEMM route for DeepSeek-V4 score GEMMs above the multi-stream token threshold. |
| `0012-MUSA-split-multi-request-indexer-prefill-chunks-per-.patch` | Port the v0.28 `0180` split of multi-request sparse-indexer prefill chunks into single-request chunks before the per-row fallback. |
| `0013-MUSA-free-deferred-KV-blocks-through-the-v0.30-block.patch` | Route the scheduler's deferred and copy-on-write block frees through `BlockPool.free_blocks`, which returns each block to its owning pool; `KVCacheManager` has no `free_blocks` on v0.30. |
| `0014-MUSA-split-chained-boolean-operators-in-v0.30-Triton.patch` | Nest the three-operand conditions in `compute_tile_loop_bounds` and the DeepSeek-V3.2 fused norm-RoPE kernel, which MUSA Triton 3.2 rejects as chained boolean operators. |
| `0015-MUSA-zero-padded-heads-in-the-v0.30-BF16-sparse-MLA-.patch` | Zero the padded query heads of the BF16 sparse MLA prefill so stale values cannot reach the MUSA kernel. |
| `0016-MUSA-carry-the-DeepSeek-V4-MUSA-paths-onto-v0.30-att.patch` | Apply the MUSA DeepSeek-V4 attention and sparse-indexer paths on the v0.30 structure: v0.30's q-lora split, wq_b projection, `compress_ratio` and `skip_compressor` are kept; graph decode selects the learned indexer top-k with the native kernel; C4 indexer compression strips `launch_pdl`. |
| `0017-MUSA-fix-v0.30-warmup-and-runner-helper-references.patch` | Model Runner V1 kernel warmup, the Kimi-K3 KDA autotune helper, the V1 KV-cache allocation context, the V2 DCP sequence-length helper and the MUSA sparse-indexer schedule metadata (`num_states`) follow the v0.30 definitions. |

Currently **17 patches**.

Keep each patch as a `git format-patch` artifact with its `index` lines. To
verify the stack, use a **pristine** checkout or archive of the pinned vLLM
commit with `python3.10 tools/musa_sync.py verify --repo <pristine-vllm>`.
The verifier checks every patch in order; an already-patched tree is not a
valid input for that command. A strict replay is `git apply --check` followed
by `git apply` for each patch in filename order. Compare the resulting source
hashes with the runtime import tree when reporting hardware evidence.
