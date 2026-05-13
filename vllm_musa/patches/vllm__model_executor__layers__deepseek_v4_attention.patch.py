# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 attention to use MUSA sparse FlashMLA backend shims.
"""

PATCHES = [
    (
        """from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
""",
        """from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm_musa.v1.attention.backends.mla.flashmla_sparse import (
    MUSADeepseekV4FlashMLASparseBackend as DeepseekV4FlashMLASparseBackend,
)
""",
    ),
    (
        'assert cap is not None, "DeepseekV4 attention requires a CUDA device"',
        'assert cap is not None, "DeepseekV4 attention requires a MUSA device"',
    ),
    (
        """        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
""",
        """        fused_insert = getattr(
            getattr(torch.ops, "_C", None),
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            None,
        )
        if fused_insert is None:
            raise NotImplementedError(
                "DeepSeek-V4 on MUSA requires a MUSA implementation of "
                "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert before "
                "model execution can proceed."
            )
        fused_insert(
            q,
            kv,
            swa_kv_cache_2d,
            swa_metadata.slot_mapping,
            positions.to(torch.int64),
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
        )
""",
    ),
]
