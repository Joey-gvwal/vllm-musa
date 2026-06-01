# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA compatibility patch for v0.22 MoE activation helpers."""

PATCHES = [
    (
        "        torch.ops._C.silu_and_mul(output, input)\n",
        """        if input.device.type == "musa":
            d = input.shape[-1] // 2
            output.copy_(F.silu(input[..., :d]) * input[..., d:])
        else:
            torch.ops._C.silu_and_mul(output, input)
""",
    ),
]
