# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 sparse attention indexer profiling on MUSA.
"""

PATCHES = [
    (
        """import torch
""",
        """import os

import torch
""",
    ),
    (
        """        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
        """        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        elif current_platform.is_musa():
            if (
                os.getenv(
                    "VLLM_MUSA_ENABLE_SPARSE_ATTN_INDEXER_PROFILE_FALLBACK",
                    "0",
                )
                == "1"
            ):
                attn_metadata = get_forward_context().attn_metadata
                if not isinstance(attn_metadata, dict):
                    logger.warning_once(
                        "Using opt-in MUSA SparseAttnIndexer profiling "
                        "fallback. This only exercises the fake/profiling "
                        "path; prefill/decode still require a real MUSA "
                        "sparse attention indexer backend."
                    )
                    return sparse_attn_indexer(
                        hidden_states,
                        _encode_layer_name(self.k_cache.prefix),
                        self.k_cache.kv_cache,
                        q_quant if not isinstance(q_quant, tuple) else q_quant[0],
                        None if not isinstance(q_quant, tuple) else q_quant[1],
                        k,
                        weights,
                        self.quant_block_size,
                        self.scale_fmt,
                        self.topk_tokens,
                        self.head_dim,
                        self.max_model_len,
                        self.max_total_seq_len,
                        self.topk_indices_buffer,
                        self.skip_k_cache_insert,
                        self.use_fp4_cache,
                    )
            raise NotImplementedError(
                "SparseAttnIndexer native forward is not implemented for MUSA. "
                "A MUSA sparse attention indexer backend is required beyond "
                "the opt-in profiling fallback."
            )
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )
""",
    ),
]

RELOAD_AFTER_PATCH = [
    "__TARGET_MODULE__",
]
