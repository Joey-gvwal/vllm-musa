# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA FlashMLA sparse backend shim.

Registers the sparse-MLA backend so the MLA selector can pick it for DSA
(index_topk) models such as GLM-5.2, and routes the sparse FlashMLA ops
(metadata + kernels) from the CUDA `vllm._flashmla_C` path to the MUSA
`flash_mla` library.
"""

import torch
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_musa.v1.attention.ops.flashmla import is_flashmla_sparse_supported

# MUSA: the upstream sparse backend imports FlashMLA ops from the core module,
# which is backed by the CUDA `vllm._flashmla_C` (not built on MUSA). Rebind
# those names in the core sparse module to the MUSA `flash_mla` equivalents so
# the sparse metadata builder and impl execute on MUSA.
import vllm.v1.attention.backends.mla.flashmla_sparse as _core_sparse
from vllm_musa.v1.attention.ops.flashmla import (
    FlashMLASchedMeta as _musa_sched_meta,
    flash_mla_sparse_fwd as _musa_sparse_fwd,
    flash_mla_with_kvcache as _musa_mla_kvcache,
    get_mla_metadata as _musa_get_mla_metadata,
)

_core_sparse.get_mla_metadata = _musa_get_mla_metadata
_core_sparse.flash_mla_sparse_fwd = _musa_sparse_fwd
_core_sparse.flash_mla_with_kvcache = _musa_mla_kvcache
_core_sparse.FlashMLASchedMeta = _musa_sched_meta


@register_backend(AttentionBackendEnum.FLASHMLA_SPARSE)
class MUSAFlashMLASparseBackend(FlashMLASparseBackend):
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 3

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        return is_flashmla_sparse_supported()[1]
