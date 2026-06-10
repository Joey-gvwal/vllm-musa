# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: filter vLLM functorch config overrides for the
installed Torch version; was inline ``_patch_vllm_functorch_config``)."""

import importlib
from functools import wraps

from vllm.logger import init_logger

from vllm_musa.patches._shared import filter_existing_config

logger = init_logger(__name__)

PATCHES: list = []


def apply() -> None:
    try:
        compiler_interface = importlib.import_module(
            "vllm.compilation.compiler_interface"
        )
        from torch._functorch import config as functorch_config
    except Exception as e:
        logger.debug("Skipping functorch config patch: %s", e)
        return

    original_get_config = compiler_interface._get_vllm_functorch_config
    if getattr(original_get_config, "_musa_filters_functorch_config", False):
        return

    @wraps(original_get_config)
    def get_existing_functorch_config() -> dict:
        return filter_existing_config(original_get_config(), functorch_config)

    get_existing_functorch_config._musa_filters_functorch_config = True
    compiler_interface._get_vllm_functorch_config = get_existing_functorch_config
