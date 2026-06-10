# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: ignore missing functorch config keys in vLLM compile
contexts; was inline ``_patch_functorch_config_patch``)."""

from vllm.logger import init_logger

from vllm_musa.patches._shared import make_config_patch_filter

logger = init_logger(__name__)

PATCHES: list = []


def apply() -> None:
    try:
        from torch._functorch import config as functorch_config
    except Exception as e:
        logger.debug("Skipping functorch config.patch patch: %s", e)
        return

    original_patch = functorch_config.__dict__.get("patch", functorch_config.patch)
    if getattr(original_patch, "_musa_filters_missing_config_keys", False):
        return

    functorch_config.__dict__["patch"] = make_config_patch_filter(
        original_patch, functorch_config
    )
