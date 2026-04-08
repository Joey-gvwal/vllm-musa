# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import logger

if current_platform.is_out_of_tree():
    from flash_attn import (  # noqa: F401
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
        get_scheduler_metadata,
    )
    from vllm import _custom_ops as ops

    reshape_and_cache_flash = ops.reshape_and_cache_flash


def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None:
    logger.info_once("Musa platform use FLASH_ATTN with version 3.")
    return 3


def flash_attn_supports_fp8() -> bool:
    logger.info_once("Cannot use FLASH_ATTN with FP8 on Musa platform")
    return False


def flash_attn_supports_sinks() -> bool:
    return True


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
