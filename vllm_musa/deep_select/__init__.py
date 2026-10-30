# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSelect on MUSA: the sparse indexer's top-k primitive.

DeepSelect is DeepSeek's top-k selection kernel used by the DSA sparse indexer
(``vllm/model_executor/layers/indexer_topk.py`` and the candidate-logits path in
``vllm/model_executor/kernels/attention/dsa/sparse_mqa_logits.py``). Upstream builds
it as the CUDA extension ``vllm._deepselect_C``; MUSA cannot run those kernels, so
this package registers the same ``torch.ops.deep_select`` surface with the native
MUSA extension from the separately installed DeepSelect package.

The native MUSA extension is built from ``DeepSelect`` branch ``feat/musa-kernels``
at the pinned commit below. The CPU reference backend is vendored from the
upstream ``xd/musa-port`` contract implementation; see ``backend.py``.

The reference backend remains CPU-only for contract tests. MUSA never silently
falls back to it. Native kernel performance requires a separate E2E measurement.
"""

from __future__ import annotations

from vllm_musa.deep_select.backend import get_alignment_requirement, topk
from vllm_musa.deep_select.registry import ensure_registered, is_registered

DEEPSELECT_REPO = "git@sh-code.mthreads.com:zhiguo.qin/DeepSelect.git"
DEEPSELECT_BRANCH = "feat/musa-kernels"
DEEPSELECT_COMMIT = "fac460ce55f1b2e86a30fd60ea892fc782f3a8ea"

__all__ = [
    "DEEPSELECT_BRANCH",
    "DEEPSELECT_COMMIT",
    "DEEPSELECT_REPO",
    "ensure_registered",
    "get_alignment_requirement",
    "is_registered",
    "topk",
]
