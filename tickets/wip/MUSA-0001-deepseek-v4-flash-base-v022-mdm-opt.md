# MUSA-0001 DeepSeek-V4 Flash Base v0.22 MDM optimization migration

## Status

wip

## Summary

Migrate DeepSeek-V4 Flash Base optimization commits from `joey/deepseek-v4-flash-base-vllm-22-pr` after `3f655e224f3fe53f50603e392f0a8ddecbefb3b4` onto `upstream/v0.22.0-dev`, preserving the new MDM git patch/apply workflow.

## Scope

- Port optimization-only or optimization-focused MUSA-owned changes into the v0.22 MDM branch.
- Convert old runtime string patch edits for upstream vLLM files into `vllm_musa/patches/series/*.patch` generated through `third_party/vllm`.
- Keep upstream functional enablement already present on `upstream/v0.22.0-dev`; focus on the new optimization commits.

## Verification

- `python3 tools/musa_sync.py verify`
- `python3 tools/patch_validate.py`
- `python3 -m pytest tests/test_patches.py tests/test_fused_moe_chunking.py -q`
- Remote MUSA rebuild and DeepSeek-V4 Flash Base smoke after local verification.
