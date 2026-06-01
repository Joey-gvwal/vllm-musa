# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA Triton compatibility patch for v0.22 block table kernels.

MUSA Triton rejects ``tl.pointer_type(elem_dtype)`` when the dtype is threaded
through a helper argument. The v0.22 block-table kernels only use int32 pointer
loads, so specialize the helper instead of carrying the dtype parameter.
"""


def normalize_source(source: str) -> str:
    return source.replace(
        "def _load_ptr(ptr_to_ptr, elem_dtype: tl.constexpr):",
        "def _load_ptr(ptr_to_ptr, elem_dtype):",
    )


PATCHES = [
    (
        "dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)",
        "dst_block_table_ptr = _load_int32_ptr(dst_block_table_ptrs + group_id)",
    ),
    (
        "src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)",
        "src_block_table_ptr = _load_int32_ptr(src_block_table_ptrs + group_id)",
    ),
    (
        "block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)",
        "block_table_ptr = _load_int32_ptr(block_table_ptrs + group_id)",
    ),
    (
        """@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)
""",
        """@triton.jit
def _load_int32_ptr(ptr_to_ptr):
    ptr = tl.load(ptr_to_ptr)
    ptr = ptr.to(tl.pointer_type(tl.int32))
    return tl.multiple_of(ptr, 16)
""",
    ),
]
