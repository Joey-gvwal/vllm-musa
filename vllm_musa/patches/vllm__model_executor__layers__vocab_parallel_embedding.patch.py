# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch vocab embedding with an opt-in DeepSeek-V4 MUSA graph-safe gather path.
"""

PATCHES = [
    (
        """from collections.abc import Sequence
from dataclasses import dataclass
""",
        """import os
from collections.abc import Sequence
from dataclasses import dataclass
""",
    ),
    (
        """DEFAULT_VOCAB_PADDING_SIZE = 64


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
""",
        """DEFAULT_VOCAB_PADDING_SIZE = 64


def _musa_deepseek_v4_graph_safe_embedding(
    layer: torch.nn.Module,
    input_: torch.Tensor,
) -> torch.Tensor | None:
    mode = os.getenv("VLLM_MUSA_DEEPSEEK_V4_GRAPH_SAFE_EMBED", "0").strip().lower()
    if mode not in {"1", "true", "yes", "workspace", "gather", "graph_safe"}:
        return None
    if not (
        current_platform.is_musa()
        or getattr(torch.version, "musa", None) is not None
    ):
        return None
    workspace = getattr(layer, "_musa_deepseek_v4_embedding_workspace", None)
    if workspace is None:
        return None
    if input_.dtype != torch.long:
        raise RuntimeError(
            "VLLM_MUSA_DEEPSEEK_V4_GRAPH_SAFE_EMBED requires int64 input ids; "
            "allocate-free casting is not available in this path."
        )

    flat_input = input_.reshape(-1)
    num_tokens = flat_input.shape[0]
    if num_tokens > workspace.shape[0]:
        raise RuntimeError(
            "VLLM_MUSA_DEEPSEEK_V4_GRAPH_SAFE_EMBED workspace is too small: "
            f"num_tokens={num_tokens}, workspace_tokens={workspace.shape[0]}"
        )

    embedding_dim = layer.weight.shape[1]
    out = workspace[:num_tokens].reshape(num_tokens, 1, embedding_dim)
    weight_view = layer.weight.unsqueeze(0).expand(num_tokens, -1, -1)
    index_view = flat_input.reshape(num_tokens, 1, 1).expand(-1, 1, embedding_dim)
    torch.gather(weight_view, 1, index_view, out=out)
    return workspace[:num_tokens].reshape(*input_.shape, embedding_dim)


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
""",
    ),
    (
        """    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)
""",
        """    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        musa_output = _musa_deepseek_v4_graph_safe_embedding(layer, input_)
        if musa_output is not None:
            return musa_output
        return F.embedding(input_, layer.weight)
""",
    ),
]
