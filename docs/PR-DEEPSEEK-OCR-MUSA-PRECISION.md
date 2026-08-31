# [MUSA] Fix DeepSeek-OCR output precision and tokenizer decoding

## PR summary

This patch fixes two independent DeepSeek-OCR correctness failures on MUSA:

1. vLLM selected the slow `LlamaTokenizer` for the normalized `deepseek_ocr`
   config type. Its cached wrapper concatenated byte-level token markers,
   exposing `Ġ`/`Ċ` in generated text.
2. The DeepSeek-OCR vision tower produced incorrect OCR under BF16 on MUSA.
   The same checkpoint and official Hugging Face reference produced correct
   OCR with FP32 vision computation and on CPU FP32. The vLLM-MUSA model now
   keeps the language model in BF16 but runs SAM/CLIP/projector and image
   inputs in FP32 on MUSA.

The source patch is tracked in:

```text
vllm_musa/patches/series/0135-MUSA-fix-deepseek-ocr-tokenizer-and-vision-precision.patch
```

## Codebase and baseline

- Repository: `/home/user/Documents/develop/vllm-workspace/vllm-musa-ds-ocr`
- Branch: `fix/deepseek-ocr-musa-precision`
- PR commit: `1235effd fix(musa): restore DeepSeek-OCR vision precision`
- Base branch: `v0.28.0-dev`
- Base commit: `55647d007b1e6d6570ad47b7e1595220bb983a4b`
- Nested vLLM source baseline: `2cf0a6915ce544dc493a0990f2ea38d81601128a`
- Nested vLLM source is generated/ignored by the top-level repository; the
  patch-series file is therefore the canonical top-level change.

## Validation environment

- Image: `sh-harbor.mthreads.com/mcctest/vllm:v0.28.0-ph1-5.2.0-torch2.11.0.post1-20260831`
- Image digest: `sha256:9f523d0c843156c45ed24bcc6a180a77fe3cc9ad1be7f8602429b316a89a7758`
- Host: MTT S5000, MUSA device 0
- NFS mount: host `/mnt/nfs` → container `/home/dist`
- Model: `/home/dist/models/DeepSeek-OCR`
- Packages after reinstall: `vllm 0.28.0`, `vllm-musa 0.1.28`, `torch 2.11.0.post1+musa5.2.0`, `torch_musa 2.11.0.post1+musa5.2.0`, `torchada 0.1.83`, `mate 0.2.4`, `transformers 5.5.3`, `huggingface-hub 1.29.0`

## Problem and root-cause evidence

Before the patch:

- text generation could expose `TheĠcapitalĠofĠChina...` instead of normal
  spaces;
- `Free OCR.` returned repeated `-Ġ1Ċ`/table-like output and grounding often
  returned all-zero boxes;
- 18 OmniDocBench demo pages returned HTTP-successful responses but had
  diagnostic page NED mean `0.9573` (0 is better);
- the official HF reference in the image's MUSA BF16 path showed the same
  visual corruption, while an isolated FP32 reference and CPU FP32 reference
  produced correct Chinese OCR;
- startup selected `TRITON_ATTN` for the vision encoder and warned that the
  S5000 `E=64,N=896` fused-MoE autotune configuration was missing.

## Changes

### Tokenizer selection

`deepseek_ocr` is added to vLLM's incorrect-tokenizer-class override set. The
generic fast `TokenizersBackend` now honors `tokenizer.json`'s ByteLevel
decoder instead of concatenating token strings.

### MUSA vision precision fallback

On MUSA only:

- `sam_model`, `vision_model`, and `projector` are converted to FP32;
- global and local image tensors are converted to the same vision dtype;
- the language model remains in its configured BF16 dtype.

This is intentionally a correctness fallback. The next optimization should
bisect SAM attention, CLIP attention, and projector to recover BF16 where a
numerical threshold proves it safe.

## Build/install command

The source was staged under `/workspace/vllm-musa` in the task container. The
nested vLLM source was restored to `2cf0a691` and the top-level patch series
was applied by the build before installation.

```bash
cd /workspace/vllm-musa
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export CCACHE_DIR=/home/dist/ccache-vllm-musa
export VLLM_MUSA_CCACHE_DIR=/tmp/vllm-musa-ccache
export VLLM_MUSA_CCACHE_SOURCE_DIR=$VLLM_MUSA_CCACHE_DIR/musa-sources
export CCACHE_BASEDIR=/workspace/vllm-musa
export CCACHE_COMPILERCHECK=content
export CCACHE_NOHASHDIR=true
export CCACHE_TEMPDIR=$VLLM_MUSA_CCACHE_DIR/tmp
export CCACHE_LOGFILE=$VLLM_MUSA_CCACHE_DIR/ccache.log
export CCACHE_MAXSIZE=20G
export VLLM_MUSA_CCACHE_MAXSIZE=$CCACHE_MAXSIZE
export VLLM_MUSA_USE_CCACHE=1
unset CXX PYTORCH_MCC
mkdir -p $VLLM_MUSA_CCACHE_DIR/tmp
SKIP_THIRD_PARTY=1 python3 -m pip install -e . --no-build-isolation -v
```

Build result: editable build/install succeeded. The ccache task report recorded
51 cacheable requests, 51 direct hits, 0 misses, and 100% task hit rate.

## Server launch command

```bash
cd /workspace/vllm-musa
export MUSA_VISIBLE_DEVICES=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
vllm serve /home/dist/models/DeepSeek-OCR \
  --host 127.0.0.1 --port 18888 \
  --served-model-name DeepSeek-OCR \
  --tensor-parallel-size 1 \
  --max-model-len 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.5 \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --logits-processors \
    vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor
```

Post-install checks:

```bash
curl -fsS http://127.0.0.1:18888/health
curl -fsS http://127.0.0.1:18888/v1/models
```

Both checks passed and `/v1/models` listed `DeepSeek-OCR` with max model length
4096.

## Text-only command and result

```bash
curl -fsS http://127.0.0.1:18888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-OCR","messages":[{"role":"user","content":"What is the capital of China? Answer in one word."}],"temperature":0,"max_tokens":32}'
```

Result after reinstall and patch:

```text
content: Beijing
completion_tokens: 2
elapsed: 0.032 s
```

The previous tokenizer path could expose literal `Ġ` markers in equivalent
text output; the patched fast backend produces ordinary spaces.

## Image OCR and grounding commands/results

The following command uses the official model demo image
`DeepSeek-OCR/assets/show1.jpg`:

```bash
IMAGE_B64=$(base64 -w0 /tmp/deepseek_show1.jpg)
curl -fsS http://127.0.0.1:18888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"DeepSeek-OCR\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,$IMAGE_B64\"}},{\"type\":\"text\",\"text\":\"Free OCR.\"}]}],\"temperature\":0,\"max_tokens\":256}"
```

For grounding, change the text to:

```text
<|grounding|>Convert the document to markdown.
```

After the patch, `Free OCR.` returned the correct title and Chinese geometry
problem text, including:

```text
# 八年级数学下册几何证明题练习
1. 已知：△ABC 的两条高 BD、CE 交于点 F ...
```

Grounding returned `sub_title`, `text`, and `image` blocks with non-zero boxes.
Before the patch it returned repeated numeric/table output or all-zero boxes.

## Official vLLM benchmark command and result

The official vLLM text-only benchmark shape was run twice; the first run
included one-time post-reinstall compilation and had a long TTFT tail. The
second run is the warm steady-state result:

```bash
cd /workspace/vllm-musa
export MUSA_VISIBLE_DEVICES=0
vllm bench serve \
  --model /home/dist/models/DeepSeek-OCR \
  --tokenizer /home/dist/models/DeepSeek-OCR \
  --served-model-name DeepSeek-OCR \
  --base-url http://127.0.0.1:18888 \
  --backend openai-chat --endpoint /v1/chat/completions \
  --dataset-name random \
  --random-input-len 64 --random-output-len 128 \
  --request-rate inf --num-prompts 16 --max-concurrency 4 \
  --ignore-eos --trust-remote-code --temperature 0 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99
```

Warm result:

| Metric | Result |
|---|---:|
| Successful requests | 16/16 |
| Request throughput | 4.79 req/s |
| Output throughput | 612.61 tok/s |
| Mean TTFT | 49.09 ms |
| P99 TTFT | 56.88 ms |
| Mean TPOT | 6.19 ms |
| Mean E2E | 835.07 ms |

The cold first run measured 0.34 req/s because it included initial graph/JIT
work; it is retained in the evidence bundle and is not the steady-state number.

## OmniDocBench demo batch result

The 18 downloaded OmniDocBench demo images were sent concurrently (4 workers)
using the same `Free OCR.` prompt and `max_tokens=1024`:

```bash
BASE_URL=http://127.0.0.1:18888 \
IMAGE_DIR=/tmp/deepseek_ocr_fix_images/images \
MAX_TOKENS=1024 \
PRED_OUT=/tmp/deepseek_ocr_pr_predictions_1024.jsonl \
python3 /tmp/deepseek_ocr_omnidoc_batch.py
```

Result after rebuild/install:

| Metric | Result |
|---|---:|
| Successful pages | 18/18 |
| Batch wall time | 29.484 s |
| Mean per-request latency | 6.134 s |
| Diagnostic NED mean | **0.4173** |
| Diagnostic NED median | 0.3647 |
| Diagnostic NED min/max | 0.0476 / 0.9217 |

For comparison, the pre-fix BF16 path with `max_tokens=256` produced repeated
or empty OCR and diagnostic NED mean `0.9573`. A post-fix `max_tokens=256`
run measured NED mean `0.5845`; the `max_tokens=1024` run above is the more
useful quality result for long document pages.

The NED calculation is a local page-level diagnostic. The official
OmniDocBench end2end/ocr scripts and their final Overall score were not run in
this PR validation because the pre-fix output was not a valid prediction; the
post-fix JSONL and source images are retained for a follow-up official score
run.

## Validation status and trade-offs

- `/health`: pass
- `/v1/models`: pass
- text-only semantic request: pass
- official image OCR demo: pass after FP32 vision fallback
- grounding image request: pass after FP32 vision fallback
- tokenizer regression test: `1 passed`
- rebuilt editable install: pass
- ccache gate: pass, 100% task hit rate
- official OmniDocBench final score: not run; diagnostic NED retained
- performance optimization of FP32 vision: not yet done

FP32 vision is intentionally a correctness-first fallback. It is slower than
the broken BF16 path; follow-up work should identify the smallest subset of
vision operators that requires FP32 and add a numerical correctness gate before
restoring BF16.

## Evidence bundle

The raw post-rebuild evidence is under:

```text
/tmp/deepseek_ocr_pr_build_20260831/
```

Important files include `deepseek_ocr_pr_install_retry.log`,
`ccache-report.txt`, `deepseek_ocr_pr_server.log`,
`health_models.log`, `text_image_smoke_rebuilt.log`,
`vllm_bench_text_rebuilt_warm.log`,
`omnidoc_predictions_rebuilt_1024.jsonl`, and
`omnidoc_diagnostic_rebuilt_1024.json`.
