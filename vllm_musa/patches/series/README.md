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
| `0018-MUSA-keep-MoE-shared-experts-on-the-compute-stream.patch` | Keep MoE shared experts on the compute stream on MUSA: graph capture runs on a torch_musa-owned stream that the cached `current_stream()` does not track, so the aux-stream fork would escape the capture. |
| `0019-MUSA-keep-V2-only-configurations-on-Model-Runner-V2.patch` | DeepSeek-V4 and the other MRV1-default architectures stay on Model Runner V1 on MUSA unless the configuration needs a V2-only feature (DSpark, prefill context parallel, ...), which keeps the V2 default. |
| `0020-MUSA-compile-the-watermark-Philox-helper-on-Triton-3.patch` | The V2 rejection sampler's resample kernel references the watermark Philox helper; MUSA Triton 3.2 has no `tl.mul`, so the helper uses the wrapping `uint32` product. |
| `0021-MUSA-enable-DeepSeek-V4.1-Engram-on-MUSA.patch` | Create and accept the Engram configuration on MUSA, and load power-of-two float32 Engram scales as their ue8m0 exponent bytes. |
| `0022-MUSA-route-DeepSeek-V4.1-attention-through-the-MUSA-.patch` | DeepSeek-V4.1 attention takes the MUSA FlashMLA ops, stream-wait overlaps, serialized long prefill, the fp32-output score helper, and the MUSA fused RoPE/KV insert without the per-head q norm. |
| `0023-MUSA-route-the-delayed-mHC-pre-blocks-to-the-MUSA-pr.patch` | The delayed mHC pre block and the fused post + delayed pre pair go through the MUSA MHC provider. |
| `0024-MUSA-load-the-DeepSeek-V4.1-ratio-2-compressor-rows-.patch` | The ratio-2 compressor loads its two pooled rows as separate tiles on MUSA instead of joining their pointers. |
| `0025-MUSA-load-DeepSeek-V4-FP8-experts-with-their-own-blo.patch` | FP8 routed experts use the block size of the checkpoint (or `expert_weight_block_size`), so 128x128 experts load beside 32x32 linears; DeepSeek-V4.1 resolves the expert dtype through the quant config. |
| `0026-MUSA-bound-the-exact-indexer-cache-rows-to-each-bloc.patch` | The MUSA exact indexer views its fp8 cache one row per block over that block's own bytes; a shared-page block stride (DeepSeek-V4.1) no longer runs the last row past the cache. |
| `0027-MUSA-claim-DeepSeek-V4.1-checkpoints-for-the-DeepSee.patch` | The DeepSeek-V4 FP8 config also claims DeepSeek-V4.1 model types, so their FP8 checkpoints get the DeepSeek expert dispatch instead of the generic FP8 config. |
| `0028-MUSA-load-the-MoE-top-k-gating-rows-element-by-eleme.patch` | The softmax and softplus-sqrt top-k gating kernels load each row element by element on MUSA; the vectorized loads ignore the scores for 320, 384 and 448 experts (DeepSeek-V4.1). |
| `0029-MUSA-score-decode-indexer-keys-of-any-cache-block-si.patch` | Decode indexer top-k scores cache blocks the materialized scorer does not take (DeepSeek-V4.1's 128-entry ratio-1 blocks) with a graph-safe Triton paged-logits kernel plus the radix top-k; a capture with no graph-safe path fails instead of selecting no keys. |
| `0030-MUSA-score-prefill-indexer-rows-in-batches.patch` | The exact MUSA prefill indexer fallback scores row chunks against all keys with each row's window masked and one top-k per chunk, instead of one einsum, top-k and two host syncs per query row. |
| `0031-MUSA-run-fine-block-FP8-linears-through-the-MUSA-blo.patch` | The MUSA block-FP8 linear list tries the MUSA block-GEMV kernel before Triton; it takes scale blocks finer than DeepGEMM's 128 (DeepSeek-V4.1's 32x32) and runs a GEMV for small M. |
| `0032-MUSA-score-prefill-indexer-rows-with-the-FP8-MQA-log.patch` | The MUSA prefill indexer fallback gathers the FP8 keys and scales without dequantizing, scores them with the FP8 MQA logits kernel and selects each row's window with the windowed top-k kernel; the FP32 path remains the fallback. |
| `0033-MUSA-score-decode-indexer-keys-with-FP8-dot-products.patch` | The graph-safe MUSA decode paged-logits kernel scores each key tile against all heads with one dot product on the FP8 operands (any cache block size); fewer than 16 heads keep the per-head reduction. |
| `0034-MUSA-select-indexer-top-k-with-the-staged-radix-kern.patch` | MUSA indexer top-k up to 512 uses the staged radix select for decode rows and eager prefill windows that start at zero; it returns the same index set and order as the generic kernel, which the other windows keep. |
| `0035-MUSA-insert-the-DeepSeek-V4.1-DSpark-context-KV-with.patch` | DSpark inserts the target's context KV into the draft layers' sliding-window cache with the MUSA fused q-norm/RoPE/KV insert; the fused CUDA insert ops are not built on MUSA. |
| `0036-MUSA-fold-the-DeepSeek-V4-shared-expert-into-the-rou.patch` | A TP-only DeepSeek-V4 MoE with sqrt-softplus routing passes its shared expert to vllm-musa instead of the MoE runner: the shared projections join the routed FP8 expert stack after loading and the router gives every token one extra slot on them at weight 1.0. |
| `0037-MUSA-run-the-DeepSeek-V4.1-Engram-wkv-projection-row.patch` | Each tensor-parallel rank already looks up a contiguous slice of the Engram hash heads, so on MUSA the Engram `wkv` projection runs row-parallel on that slice and one all-reduce sums the partial key and value rows; no rank gathers every head or runs the full projection. |
| `0038-MUSA-share-the-DeepSeek-V4.1-decode-plan-across-KV-c.patch` | Every KV-cache group's sparse-SWA metadata builder makes its own FlashMLA plan objects; the DeepSeek-V4.1 decode attention reuses the forward pass's first plan for a layer type and cache layout, so the tile-scheduler planner runs once per layer type instead of once per group. |
| `0039-MUSA-let-the-DeepSeek-V4-router-read-the-BF16-gate-o.patch` | On MUSA the non-hash DeepSeek-V4 MoE gates keep their BF16 output; the MUSA biased router widens it exactly, so routing runs no FP32 cast kernel. Hash layers and the upstream fallback route on FP32 logits. |
| `0040-MUSA-map-the-DeepSeek-V4.1-decode-top-k-once-per-ind.patch` | The DeepSeek-V4.1 layers after an index source read its top-k indices through the same compressed cache until the next source; the MUSA decode attention maps them to global KV slots once per index source and cache in a forward pass instead of once per layer. |

Currently **40 patches**.

Keep each patch as a `git format-patch` artifact with its `index` lines. To
verify the stack, use a **pristine** checkout or archive of the pinned vLLM
commit with `python3.10 tools/musa_sync.py verify --repo <pristine-vllm>`.
The verifier checks every patch in order; an already-patched tree is not a
valid input for that command. A strict replay is `git apply --check` followed
by `git apply` for each patch in filename order. Compare the resulting source
hashes with the runtime import tree when reporting hardware evidence.
