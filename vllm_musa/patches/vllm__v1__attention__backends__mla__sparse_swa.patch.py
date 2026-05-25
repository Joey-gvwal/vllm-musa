# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch DeepSeek-V4 sparse SWA metadata kernel for MUSA Triton.
"""

PATCHES = [
    (
        """from vllm.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata
""",
        """from vllm_musa.v1.attention.ops.flashmla import FlashMLASchedMeta, get_mla_metadata
""",
    ),
    (
        """    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + token_idx, 0)
        return
""",
        """    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if is_valid == 0:
        tl.store(swa_lens_ptr + token_idx, 0)
        return
""",
    ),
    (
        """        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility.
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)

        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)
""",
        """        # NOTE: Ensure all metadata tensors maintain fixed memory addresses
        # for CUDA graph compatibility. Build token-to-request and validity
        # metadata on device: the upstream CPU repeat_interleave + pin_memory +
        # H2D copy path serializes MTP drafting on MUSA.
        num_tokens = slot_mapping.shape[0]
        token_to_req_indices = self.token_to_req_indices[:num_tokens]
        is_valid_token = self.is_valid_token[:num_tokens]
        if num_tokens > 0:
            _compute_token_to_req_and_valid_kernel[(num_tokens,)](
                token_to_req_indices,
                is_valid_token,
                slot_mapping,
                query_start_loc,
                num_reqs,
                BLOCK_SIZE=triton.next_power_of_2(max(num_reqs, 1)),
            )
""",
    ),
    (
        """

@triton.jit
def _compute_swa_indices_and_lens_kernel(
""",
        """

@triton.jit
def _compute_token_to_req_and_valid_kernel(
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    slot_mapping_ptr,
    query_start_loc_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = tl.full((), 0, tl.int32)
    for i in range(0, BLOCK_SIZE):
        q_start = tl.load(query_start_loc_ptr + i, mask=i < num_reqs, other=0)
        q_end = tl.load(query_start_loc_ptr + i + 1, mask=i < num_reqs, other=0)
        in_req = (token_idx >= q_start) & (token_idx < q_end) & (i < num_reqs)
        req_idx = tl.where(in_req, i, req_idx)
    tl.store(token_to_req_indices_ptr + token_idx, req_idx)

    slot = tl.load(slot_mapping_ptr + token_idx)
    tl.store(is_valid_token_ptr + token_idx, slot >= 0)


@triton.jit
def _compute_swa_indices_and_lens_kernel(
""",
    ),
]
