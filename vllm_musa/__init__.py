# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
vLLM MUSA Platform Plugin

This plugin enables vLLM to run on Moore Threads MUSA GPUs.
It provides a MUSAPlatform implementation that integrates with vLLM's
platform abstraction layer.

Usage:
    Install this package alongside vLLM, and the MUSA platform will be
    automatically detected and used when running on Moore Threads hardware.
"""

import logging
import os
from pathlib import Path

__all__ = [
    "MUSAPlatform",
    "musa_platform_plugin",
    "register",
    "register_custom_ops",
    "collect_env",
]
__version__ = "0.1.1"

logger = logging.getLogger(__name__)

# Import torchada early to ensure torch.device patching happens before
# any torch.device("cuda:X") calls in vLLM. This is critical for MUSA
# to work correctly - it patches torch.cuda to redirect to MUSA.
try:
    # isort: off
    import torchada  # noqa: F401
    import torch

    # isort: on
    _torchada_available = True
except ImportError:
    _torchada_available = False


def _patch_tvm_ffi_musa_extension() -> None:
    """Provide MUSA helpers expected by MATE on older TVM-FFI builds."""
    try:
        import tvm_ffi.cpp.extension as tvm_ffi_ext
    except Exception:
        return

    if not hasattr(tvm_ffi_ext, "_find_musa_home"):

        def _find_musa_home() -> str:
            for env_name in ("MUSA_HOME", "MUSA_PATH"):
                musa_home = os.environ.get(env_name)
                if musa_home:
                    return musa_home

            for candidate in ("/usr/local/musa", "/opt/musa"):
                if (Path(candidate) / "bin" / "mcc").exists():
                    return candidate

            raise RuntimeError(
                "Could not find MUSA installation. Please set MUSA_HOME."
            )

        tvm_ffi_ext._find_musa_home = _find_musa_home

    if not hasattr(tvm_ffi_ext, "_get_musa_target"):

        def _normalize_musa_arch(arch: str) -> str:
            arch = arch.strip()
            if arch.startswith("mp_"):
                return arch
            return f"mp_{arch.replace('.', '')}"

        def _get_musa_target() -> list[str]:
            arch_list = (
                os.environ.get("TVM_FFI_MUSA_ARCH_LIST")
                or os.environ.get("MUSA_ARCH_LIST")
                or os.environ.get("TORCH_MUSA_ARCH_LIST")
            )
            if arch_list:
                arches = arch_list.replace(",", " ").split()
            else:
                arches = []
                try:
                    import torch

                    if hasattr(torch, "musa") and torch.musa.is_available():
                        get_arch_list = getattr(torch.musa, "get_arch_list", None)
                        if callable(get_arch_list):
                            arches = list(get_arch_list())
                        if not arches:
                            major, minor = torch.musa.get_device_capability()
                            arches = [f"{major}{minor}"]
                except Exception:
                    arches = []

            return [
                f"--offload-arch={_normalize_musa_arch(str(arch))}"
                for arch in arches
                if str(arch).strip()
            ]

        tvm_ffi_ext._get_musa_target = _get_musa_target


_patch_tvm_ffi_musa_extension()


def _patch_mate_tvm_ffi_musa_include() -> None:
    """Expose bundled TVM-FFI MUSA compatibility headers to MATE JIT."""
    try:
        import mate.jit.cpp_ext as mate_cpp_ext
    except Exception:
        return

    if getattr(mate_cpp_ext, "_vllm_musa_include_patched", False):
        return

    include_dir = Path(__file__).resolve().parent / "include"
    header = include_dir / "tvm" / "ffi" / "extra" / "musa" / "device_guard.h"
    if not header.exists():
        return

    original_resolve_include_paths = mate_cpp_ext._resolve_include_paths

    def _resolve_include_paths(extra_include_dirs):
        include_paths = original_resolve_include_paths(extra_include_dirs)
        include_paths.append(str(include_dir))
        return list(dict.fromkeys(include_paths))

    mate_cpp_ext._resolve_include_paths = _resolve_include_paths
    mate_cpp_ext._vllm_musa_include_patched = True


_patch_mate_tvm_ffi_musa_include()

# Track whether patches have been applied in this process
_patches_applied = False


########### platform plugin ###########


def musa_platform_plugin() -> str | None:
    """Register the MUSA platform.

    vLLM platform plugin entry point. Called by vLLM to check if the MUSA
    platform is available. Returns the qualified class name if available.

    Note: We intentionally do NOT apply patches here because this function
    is called during vLLM module initialization which can cause circular
    import issues. Patches are applied via the general plugin mechanism.
    """
    # Check if torchada detected MUSA platform
    if _torchada_available:
        import torchada

        if torchada.is_musa_platform():
            return "vllm_musa.platform.MUSAPlatform"

    # Fallback: check if torch_musa is available
    try:
        import torch_musa  # noqa: F401

        return "vllm_musa.platform.MUSAPlatform"
    except ImportError:
        pass

    return None


########### general plugins ###########


def _apply_vllm_patches() -> None:
    """Apply vLLM source patches for MUSA compatibility.

    This function is idempotent - it only applies patches once per process.
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    try:
        from .patches import apply_patches

        apply_patches()
    except Exception as e:
        logger.error(f"Failed to apply vLLM patches: {e}")


def _register_patches() -> None:
    """Apply vLLM source patches for MUSA compatibility."""
    _apply_vllm_patches()


def _register_ops() -> None:
    """Register OOT custom ops (activation, layernorm, fused_moe, etc.)."""
    import vllm_musa.model_executor  # noqa: F401


def _register_modules() -> None:
    """Register distributed connectors, utils, and v1 attention backends."""
    import vllm_musa.distributed  # noqa: F401
    import vllm_musa.utils  # noqa: F401
    import vllm_musa.v1  # noqa: F401


def register_custom_ops() -> None:
    """
    vLLM general plugin entry point for MUSA customizations.

    This function is called by vLLM's general plugin mechanism after the
    platform is initialized, which avoids circular import issues.
    It applies vLLM source patches and registers all MUSA-specific ops,
    distributed connectors, and attention backends.
    """
    _register_patches()
    _register_ops()
    _register_modules()
    logger.info("MUSA patches and custom ops registered")


def register() -> str | None:
    """Compatibility platform entry point used by older vLLM plugin metadata."""
    platform = musa_platform_plugin()
    if platform is not None:
        _apply_vllm_patches()
    return platform


########### console scripts ###########


def collect_env() -> None:
    """Entry point for vllm_collect_env console script."""
    from .collect_env import main

    main()


########### lazy imports ###########


def __getattr__(name: str):
    """Lazy import module components."""
    if name == "MUSAPlatform":
        from .platform import MUSAPlatform

        return MUSAPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
