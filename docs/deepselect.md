# DeepSelect on MUSA

vLLM-MUSA registers the `torch.ops.deep_select` interface during the general
plugin hook. The native implementation is provided by the separately built
DeepSelect package; vLLM-MUSA does not build that extension and does not fall
back to `torch.topk` on the MUSA dispatch path.

## Pinned source

Use the repository, branch, and commit recorded in `third_party/PINS`:

```bash
grep -E '^DEEPSELECT_(REPO|BRANCH|COMMIT)=' third_party/PINS
```

Keep the checkout outside the vLLM-MUSA tree when the two projects are built
from separate source mounts. Do not replace the pinned revision without
rerunning `tests/test_deep_select_contract.py` on MUSA.

## Build and install

In the MUSA runtime environment, install DeepSelect before starting vLLM:

```bash
cd /path/to/DeepSelect
DEEP_SELECT_BUILD_TARGET_PLATFORM=MUSA python3 -m pip install -v -e .
python3 -c 'import deep_select.deep_select_musa_cpp'
```

Then install or update vLLM-MUSA normally. The editable DeepSelect install
must resolve to the same source revision on every worker. A missing native
extension is a startup error when the `deep_select` backend is selected.

## Runtime selection

Select the backend through the vLLM kernel configuration:

```text
{"sparse_indexer_topk_backend":"deep_select"}
```

The backend supports the DSV4 sparse-indexer top-k contract for BF16 and FP32
logits, with 32-byte input and output row-stride alignment. The standard vLLM
path requests indices only and passes `abort_when_nan_found=False`; DeepSelect
still checks for NaNs and writes its marker for invalid rows.

For provenance, record the DeepSelect source commit, extension path and hash,
vLLM-MUSA commit, vLLM commit, and the exact serving command with benchmark
artifacts.
