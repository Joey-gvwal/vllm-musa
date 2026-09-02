# PR: Fix DeepSeek-OCR BF16 precision on MUSA without full-vision FP32

## Summary

DeepSeek-OCR produced empty or severely corrupted output when its vision tower
ran in BF16 on MUSA. The previous workaround converted the complete vision
stack to FP32, which restored output at a substantial memory and throughput
cost. This change localizes the workaround to the one failing operation:
DeepSeek-OCR's SAM masked scaled-dot-product attention (SDPA).

The patch is in commit `565ceb5a` and the follow-up regression evidence is in
`ddee6228`, on branch
`fix/deepseek-ocr-sam-sdpa-bf16` (pushed to the `joey` remote).

## Root cause

SAM `RelPosAttention` supplies an additive relative-position mask to
`torch.nn.functional.scaled_dot_product_attention`. On the tested MUSA
runtime, the default Flash SDPA path is numerically wrong for BF16 inputs with
this mask. The no-mask BF16 path is not implicated.

An isolated MUSA test with shape `[1, 12, 4096, 64]` measured the following
against an FP32 reference:

| Backend and inputs | Cosine similarity | Mean absolute error |
|---|---:|---:|
| Default Flash, BF16 q/k/v + additive mask | 0.683207 | 0.023911 |
| MATH, BF16 q/k/v + additive mask | 0.999989 | near-zero |

This identifies the failing boundary as the MUSA Flash SDPA masked path, not
the tokenizer and not a requirement to run all vision layers in FP32. A fix to
the native MUSA Flash/MUDNN kernel itself is outside this vLLM repository.

## Code change

- Add a `musa_masked_sdpa_math` opt-in flag to `RelPosAttention`.
- Enable the flag only on SAM blocks constructed by `DeepseekOCRForCausalLM`
  and only when the MUSA runtime is present.
- Select `SDPBackend.MATH` only around the masked SDPA call. q/k/v, SAM, CLIP,
  projector, image tensors, and patch tensors remain in the configured BF16
  dtype.
- Leave unmasked SDPA and all non-MUSA model paths unchanged.

Changed through the patch series:

```text
vllm_musa/patches/series/0135-MUSA-fix-deepseek-ocr-tokenizer-and-vision-precision.patch
```

## Full OmniDocBench result

The same 1,651-page dataset, prompt, post-processing, four-GPU data-parallel
script, and OmniDocBench evaluator were used for both rows. The baseline is
the archived all-vision-FP32 run in
`~/Documents/DeepSeek-OCR-OmniDocBench-20260901/results/main_regression_20260901/`.

| Metric | All-vision FP32 baseline | Masked SDPA MATH + BF16 | Change |
|---|---:|---:|---:|
| Overall | 84.2056 | **85.0948** | **+0.8892** |
| Text Edit | 0.103909 | **0.103018** | -0.000891 |
| Formula CDM | 84.5772% | **87.8162%** | **+3.2390 pp** |
| Table TEDS | **78.4306%** | 77.7700% | -0.6606 pp |
| Table TEDS-S | **81.7559%** | 80.9526% | -0.8033 pp |
| Table Edit | 0.194344 | 0.201177 | +0.006833 |
| Reading Order Edit | 0.179140 | **0.178171** | -0.000968 |

Evaluation gates: 1,651 pages; 2,352/2,352 CDM samples; 665 table samples;
zero CDM exceptions; zero page-level timeouts; one evaluator quick-match
timeout fallback (the built-in chunked fallback handled it).

## Performance result

On four MTT S5000 GPUs with batch size 8, the targeted path completed in
`1950.859 s` (`0.846294 page/s`) versus `2060.555 s`
(`0.801241 page/s`) for the all-vision-FP32 baseline: **5.32% lower wall
time** and **5.62% higher aggregate throughput**. The isolated MATH backend
also avoids the FP32 q/k/v conversion used by the earlier selective
diagnostic. These are end-to-end image results, not a claim that Flash SDPA
itself is faster than MATH.

## Reproduction commands

The commands below assume the vLLM-MUSA checkout is installed in the same
container as the model and use password-only remote access plus the GPU
Dashboard lease workflow. Do not put an API key or lease token in a script or
log.

### Build and install

```bash
cd /workspace/vllm-musa
export VLLM_MUSA_USE_CCACHE=1
export CCACHE_DIR=/home/dist/ccache-vllm-musa
unset CXX PYTORCH_MCC
python3 -m pip install --no-build-isolation -e .
python3 - <<'PY'
import torch, vllm, vllm_musa
print(torch.__version__, getattr(torch.version, "musa", None))
print(vllm.__version__, vllm.__file__)
print(vllm_musa.__file__)
PY
```

### Text-only server smoke and `vllm bench serve`

This is a control path; it does not exercise the image encoder or masked SAM
SDPA.

```bash
export MUSA_VISIBLE_DEVICES=2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
vllm serve /home/dist/models/DeepSeek-OCR \
  --served-model-name deepseek-ocr \
  --trust-remote-code --dtype bfloat16 --max-model-len 8192 \
  --host 0.0.0.0 --port 8000
```

In another shell after the readiness probe succeeds:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-ocr \
  --dataset-name random \
  --random-input-len 4096 --random-output-len 1024 \
  --num-prompts 100 --request-rate inf \
  --seed 0 --percentile-metrics ttft,tpot,itl,e2el
```

### Image server smoke

Use an image request to exercise the patched path; a text-only request is not
evidence for this fix.

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @image_request.json
```

`image_request.json` must contain an OpenAI-compatible `image_url` data URI,
the model name `deepseek-ocr`, and the same prompt used by the official
benchmark (`<image>\n<|grounding|>Convert the document to markdown.`).

### Official image benchmark

The reproducible full-data command is the archived script, not a newly
assembled benchmark:

```bash
cd /workspace/odb_bundle
export MODEL=/home/dist/models/DeepSeek-OCR
export DATASET=/home/dist/datasets/OmniDocBench
export GPU_INDICES=2,3,4,5
export OUTPUT=/tmp/deepseek_ocr_predictions_math_full
export FORCE=1
bash scripts/run_inference_4gpu.sh
find "$OUTPUT" -maxdepth 1 -type f -name '*.md' | wc -l  # 1651
cat "$OUTPUT/benchmark_result_4gpu.json"
```

Then run the existing `prepare_eval_shards.py`, `run_eval_shards.sh`,
`run_cdm_shards.sh`, and `aggregate_results.py` commands in
`~/Documents/DeepSeek-OCR-OmniDocBench-20260901/README.md`. The complete
latest artifacts are in
`~/Documents/DeepSeek-OCR-OmniDocBench-20260901/results/musa_math_full_20260902/`.

## Scope and follow-up

The vLLM fix is deliberately narrow and preserves BF16 everywhere else. The
remaining upstream follow-up is to repair MUSA/MUDNN Flash SDPA for additive
relative-position masks, then benchmark that native path against this MATH
fallback with the same end-to-end gate.
