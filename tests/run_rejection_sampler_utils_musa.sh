#!/usr/bin/env bash
set -euo pipefail

# Run the upstream v0.26 rejection-sampler statistical suite on MUSA. Keeping
# this as a small adapter avoids copying and drifting the upstream test logic.
ROOT=${VLLM_MUSA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VLLM_ROOT=${VLLM_ROOT:-$ROOT/third_party/vllm}
VOCAB_SIZE=${VOCAB_SIZE:-1024}
SOURCE=$VLLM_ROOT/tests/v1/spec_decode/test_rejection_sampler_utils.py

if [[ ! -f "$SOURCE" ]]; then
  echo "missing upstream test: $SOURCE" >&2
  exit 2
fi

TMP=$(mktemp /tmp/rejection_sampler_utils_musa_XXXXXX.py)
trap 'rm -f "$TMP"' EXIT

sed \
  -e "s/VOCAB_SIZE = 4096/VOCAB_SIZE = $VOCAB_SIZE/" \
  -e 's/torch.cuda.is_available()/torch.musa.is_available()/' \
  -e 's/CUDA required/MUSA required/' \
  -e 's/device = "cuda"/device = "musa"/g' \
  -e '/^import torch$/a import torch_musa  # noqa: F401' \
  "$SOURCE" >"$TMP"

cd "$(dirname "$TMP")"
PYTHONPATH="$ROOT:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  pytest -q "$TMP" --disable-warnings --maxfail=1 "$@"
