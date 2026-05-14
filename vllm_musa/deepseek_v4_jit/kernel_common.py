# SPDX-License-Identifier: Apache-2.0
"""Small TileLang compatibility helpers used by DeepSeek-V4 MUSA JIT paths."""

from __future__ import annotations

import os
from typing import Any


_TILELANG_MUSA_OPT_FLAGS = [
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-opt-level=1",
]


def _tilelang_musa_pass_configs(tilelang):
    """Return optional MUSA pass configs without requiring a specific TileLang build."""
    if os.environ.get("VLLM_MUSA_DEEPSEEK_V4_TILELANG_PASS_CONFIG") != "1":
        return None

    pass_configs = {}
    for key_name, value in (
        ("TL_ENABLE_MUSA_BURST", True),
        ("TL_ENABLE_REDUCE_BURST", True),
        ("TL_DEVICE_COMPILE_FLAGS", _TILELANG_MUSA_OPT_FLAGS),
    ):
        key = getattr(tilelang.PassConfigKey, key_name, None)
        if key is not None:
            pass_configs[key] = value
    return pass_configs or None


def _tilelang_jit(tilelang, name: str, pass_configs=None):
    if pass_configs is None:
        pass_configs = _tilelang_musa_pass_configs(tilelang)
    try:
        if pass_configs is None:
            return tilelang.jit()
        return tilelang.jit(pass_configs=pass_configs)
    except TypeError as exc:
        if "pass_configs" not in str(exc):
            raise
        return tilelang.jit()


def _patch_tilelang_musa_wrapper() -> bool:
    """Patch TileLang's MUSA wrapper for host IR without ``T.call_packed``.

    The TileLang build in the current S5000 container can lower simple MUSA
    kernels, but its wrapper sorts device functions by searching the generated
    host IR for ``T.call_packed("<kernel>")``. Some lowered kernels do not
    contain that string, so compilation fails with ``ValueError: substring not
    found`` before the generated MUSA source is wrapped. Falling back to the
    device-module order preserves the upstream behavior for the single-kernel
    JIT helpers used here and keeps fixed TileLang builds untouched.
    """

    try:
        from tilelang import tvm
        from tilelang.jit.adapter.musa_wrapper import TLMUSASourceWrapper
        from tilelang.jit.adapter.utils import get_annotated_mod
    except Exception:
        return False

    if getattr(TLMUSASourceWrapper, "_vllm_musa_no_call_packed_patch", False):
        return True

    original_parse = TLMUSASourceWrapper.parse_source_information

    def patched_parse(self: Any):
        try:
            return original_parse(self)
        except ValueError as exc:
            if "substring not found" not in str(exc):
                raise

            if self.device_mod is None or self.host_mod is None:
                with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
                    device_mod, host_mod = get_annotated_mod(self.mod, self.target)
                self.device_mod = device_mod
                self.host_mod = host_mod

            assert len(self.device_mod.functions) >= 1, (
                "Device module should have at least one function."
            )
            assert len(self.host_mod.functions) == 1, (
                "Only support one function in host module."
            )

            block_info_map = {}
            grid_info_map = {}
            dynamic_smem_buf_map = {}
            use_cooperative_groups_map = {}
            function_names = []

            for g_var, func in self.device_mod.functions.items():
                block_info = [1, 1, 1]
                grid_info = [1, 1, 1]
                function_name = g_var.name_hint
                attrs = func.attrs
                dynamic_smem_buf = None
                use_cooperative_groups = False

                if "use_cooperative_groups" in attrs:
                    use_cooperative_groups = attrs["use_cooperative_groups"]
                if "dyn_shared_memory_buf" in attrs:
                    dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
                if "thread_extent" in attrs:
                    for tag, extent in attrs["thread_extent"].items():
                        if "threadIdx" in tag:
                            block_info["xyz".index(tag[-1])] = extent
                        elif "blockIdx" in tag:
                            grid_info["xyz".index(tag[-1])] = extent

                block_info_map[function_name] = block_info
                grid_info_map[function_name] = grid_info
                dynamic_smem_buf_map[function_name] = dynamic_smem_buf
                use_cooperative_groups_map[function_name] = use_cooperative_groups
                function_names.append(function_name)

            self.block_info = block_info_map
            self.grid_info = grid_info_map
            self.dynamic_smem_buf = dynamic_smem_buf_map
            self.use_cooperative_groups = use_cooperative_groups_map

            for _, func in self.host_mod.functions.items():
                if "tma_descriptor_args" in func.attrs:
                    self.tma_descriptor_args = func.attrs["tma_descriptor_args"]
                if "l2_persistent_map" in func.attrs:
                    for function_name in function_names:
                        self.l2_persistent_map[function_name] = func.attrs[
                            "l2_persistent_map"
                        ]

            self.function_names = function_names

    patched_parse._vllm_musa_original = original_parse
    TLMUSASourceWrapper.parse_source_information = patched_parse
    TLMUSASourceWrapper._vllm_musa_no_call_packed_patch = True
    return True
