# SPDX-License-Identifier: Apache-2.0
"""MUSA cat-6 object patch: ignore missing inductor config keys in vLLM compile
contexts; was inline ``_patch_inductor_config_patch``)."""

from vllm.logger import init_logger

from vllm_musa.patches._shared import make_config_patch_filter

logger = init_logger(__name__)

PATCHES: list = []


def apply() -> None:
    try:
        from torch._inductor import config as inductor_config
    except Exception as e:
        logger.debug("Skipping inductor config.patch patch: %s", e)
        return

    original_patch = inductor_config.__dict__.get("patch", inductor_config.patch)
    if getattr(original_patch, "_musa_filters_missing_config_keys", False):
        return

    inductor_config.__dict__["patch"] = make_config_patch_filter(
        original_patch, inductor_config
    )
