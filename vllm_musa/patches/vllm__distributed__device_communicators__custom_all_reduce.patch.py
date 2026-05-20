# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.distributed.device.communicators.custom_all_reduce.
"""

PATCHES = [
    # Patch CustomAllreduce.max_size.
    (
        "max_size=8192 * 1024,",
        "max_size=16 * 8192 * 1024,",
    ),
    # Use ray lead to the env MUSA_VISIBLE_DEVICES has some problem, and the patch can be deleted after fixed
    (
        "if cuda_visible_devices:",
        "if cuda_visible_devices and current_platform.is_cuda():",
    ),
    # Patch CustomAllreduce enable musa's custom_allreduce.
    (
        "if not current_platform.is_rocm() and not _can_p2p(rank, world_size):",
        "if not current_platform.is_rocm() and not current_platform.is_musa() and not _can_p2p(rank, world_size):",
    ),
    # Upgrade the previous MUSA patch if it was already persisted on disk.
    (
        "if ( not current_platform.is_rocm() or not current_platform.is_musa() ) and not _can_p2p(rank, world_size):",
        "if not current_platform.is_rocm() and not current_platform.is_musa() and not _can_p2p(rank, world_size):",
    ),
    # MUSA-0062 (torch 2.7.1): removed MUSA-0052's `world_size > 2` CAR
    # gate, re-enabling custom_all_reduce on MUSA for TP>2. The
    # compile-path safety (Inductor lowering past the Python alignment
    # gate) was handled at the kernel level; see generated/musa0062/.
    #
    # MUSA-0069 (torch >= 2.9): added a torch-version-aware `world_size
    # > 2` disable because the kernel rejected non-vector-aligned
    # numel ("input length must be multiple of 4") produced by torch
    # 2.9 Inductor's compile-mode lowering.
    #
    # MUSA-0075 (torch >= 2.9): removed the MUSA-0069 gate. The C++
    # wrapper `vllm-musa/csrc/custom_all_reduce.cu` now zero-pads the
    # tail of reg_buffer when numel is not a multiple of d_T (the
    # kernel vector width = 16 / element_size). Each rank pads
    # identically; the sum of zero peer tails is zero; the kernel
    # writes zeros into out's tail (within PyTorch's allocator slack).
    # CAR is re-enabled at TP>2 on torch_musa 2.9.0. See
    # generated/musa0075/.
]
