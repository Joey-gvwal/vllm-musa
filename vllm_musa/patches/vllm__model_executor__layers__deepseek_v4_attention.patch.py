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
]
