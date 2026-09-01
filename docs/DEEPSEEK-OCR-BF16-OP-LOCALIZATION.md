# DeepSeek-OCR MUSA BF16 precision localization

## Finding

The catastrophic MUSA BF16 output failure is isolated to the SAM vision
encoder's `torch.nn.functional.scaled_dot_product_attention` path at the
current op-level granularity. It is not a tokenizer dtype conversion and it is
not evidence that the full model must run in FP32.

The production patch keeps SAM, CLIP, and projector weights/activations in the
configured BF16 dtype. Only the masked SAM SDPA call is scoped to
`SDPBackend.MATH` on MUSA; q/k/v remain BF16 and the runtime handles stable
accumulation. Non-MUSA execution is unchanged.

## Evidence

The run used the code represented by commit `652d2b5794b6aba2d5e3e781d0f6b19625ed65d0` (based on `v0.28.0-dev`), the same
OmniDocBench prompt and post-processing as the archived benchmark, and 8 pages
uniformly selected from a 32-page sample:

| Mode | Non-empty pages | Tokens | Wall time | Match to all-vision FP32 |
|---|---:|---:|---:|---:|
| All vision BF16 | 1/8 | 193 | 46.529 s | failed baseline |
| SAM SDPA FP32 only | 8/8 | 5682 | 29.037 s | superseded diagnostic |
| SAM masked SDPA MATH + BF16 q/k/v | 8/8 | 5684 | 28.787 s | 7/8 exact, mean sequence ratio 0.999836 |
| All vision FP32 | 8/8 | 5690 | 30.501 s | correctness reference |

The targeted default and diagnostic selective mode produced identical files on
all 8 pages. The targeted path was about 4.1% faster than the all-vision FP32
path for this matched-token run, while avoiding FP32 conversion of the rest of
the vision stack.

The MUSA default Flash SDPA dispatch produces a large error for the additive
relative-position mask (cosine `0.683207` against an FP32 reference), while
the MATH backend with BF16 q/k/v produces cosine `0.999989` in the isolated
kernel test. This avoids the FP32 q/k/v conversion and is the current targeted
vLLM fix. A native MUSA Flash SDPA BF16 kernel correction would still be a
torch-musa/MUDNN change outside this vLLM patch.

## Reproduction

The complete scripts and raw diagnostic outputs are in:

```text
~/Documents/DeepSeek-OCR-OmniDocBench-20260901/results/op_localization_20260901/
```

The patch series applies cleanly to the vendored upstream vLLM base:

```bash
git -C third_party/vllm archive HEAD | tar -x -C /tmp/vllm-base
git -C /tmp/vllm-base apply --check \
  vllm_musa/patches/series/0135-MUSA-fix-deepseek-ocr-tokenizer-and-vision-precision.patch
```

The full 1,651-page OmniDocBench regression should be run after installing
the patch and rebuilding/reinstalling vLLM-MUSA. The 8-page diagnostic is a
localization gate, not a replacement for the full benchmark.
