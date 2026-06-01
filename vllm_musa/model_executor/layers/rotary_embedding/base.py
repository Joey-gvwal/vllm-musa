import os

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_musa.jit_kernel import rotary_embedding


@RotaryEmbedding.register_oot
class MusaRotaryEmbedding(RotaryEmbedding):
    @classmethod
    def enabled(cls) -> bool:
        # The MUSA RoPE JIT is an experimental generic RotaryEmbedding
        # replacement. Keep it opt-in so DeepSeek-V4-specific enablement does
        # not force Qwen/FlashAttention models through this kernel.
        if os.getenv("VLLM_MUSA_ENABLE_JIT_ROPE", "0") != "1":
            return False
        return super().enabled()

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # MUSA-0202: do NOT reassign self.cos_sin_cache inside forward.
        # CUDAGraph-aware Dynamo flags `self.cos_sin_cache = ...` as a
        # buffer mutation and refuses to compile (RuntimeError: Assigning
        # / modifying buffers of nn.Module during forward pass is not
        # allowed when using cudagraph). Use a local variable instead;
        # .to() is a no-op when device/dtype already match.
        cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

        rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key
