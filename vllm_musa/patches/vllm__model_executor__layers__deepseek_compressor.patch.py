# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 compressor/cache paths with MUSA-specific gates.
"""

PATCHES = [
    (
        """    def forward(
        self,
        # [num_tokens, hidden_size]
        x: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        num_tokens, _ = x.shape
""",
        """    def forward(
        self,
        # [num_tokens, hidden_size]
        x: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        if current_platform.is_musa() or x.device.type == "musa":
            raise NotImplementedError(
                "DeepSeek-V4 compressor/cache updates are not implemented for "
                "MUSA yet. A MUSA implementation of the fused compress, "
                "RMSNorm/RoPE, FP8/MXFP4 quantization, and KV-cache insert "
                "path is required before model execution can proceed."
            )
        num_tokens, _ = x.shape
""",
    ),
]
