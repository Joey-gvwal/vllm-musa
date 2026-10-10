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

Keep each patch as a `git format-patch` artifact with its `index` lines. To
verify the stack, use a **pristine** checkout or archive of the pinned vLLM
commit with `python3.10 tools/musa_sync.py verify --repo <pristine-vllm>`.
The verifier checks every patch in order; an already-patched tree is not a
valid input for that command. A strict replay is `git apply --check` followed
by `git apply` for each patch in filename order. Compare the resulting source
hashes with the runtime import tree when reporting hardware evidence.
