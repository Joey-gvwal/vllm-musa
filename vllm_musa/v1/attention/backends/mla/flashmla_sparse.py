# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA FlashMLA sparse backend shim.

Stays on upstream vLLM's sparse MLA code path and narrows only the platform
capability and MATE availability checks. Registers the backend so the MLA
selector can pick it for DSA (index_topk) models such as GLM-5.2.
"""

import torch
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_musa.v1.attention.ops.flashmla import is_flashmla_sparse_supported


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
