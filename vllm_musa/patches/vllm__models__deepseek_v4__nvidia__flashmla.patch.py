# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 FlashMLA imports for MUSA."""

PATCHES = [
    (
        """from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
""",
        """from vllm_musa.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
""",
    ),
    (
        """        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to layer.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)
""",
        """        active_decode_tokens = q.shape[0]
        if topk_indices is not None:
            topk_indices = topk_indices[:active_decode_tokens]
        if topk_lens is not None:
            topk_lens = topk_lens[:active_decode_tokens]

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        if swa_indices is not None:
            swa_indices = swa_indices[:active_decode_tokens]
        if swa_lens is not None:
            swa_lens = swa_lens[:active_decode_tokens]

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to layer.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)
""",
    ),
]
