# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patches for vLLM compatibility with MUSA platform.

This module contains patches that modify vLLM source files at runtime
to ensure compatibility with the MUSA Triton version.
"""

import contextlib
import fcntl
import importlib.util
import os
import sys
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

_patches_applied = False


def _patch_is_needed(source: str, old: str, new: str) -> bool:
    """Return whether a text replacement still needs to be applied."""
    if old not in source:
        return False
    if new == "":
        return True
    return new not in source


@contextlib.contextmanager
def _patch_file_lock():
    lock_path = Path(os.getenv("VLLM_MUSA_PATCH_LOCK", "/tmp/vllm_musa_patches.lock"))
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _get_patch_files():
    """Get all patch files in the patches directory."""
    patches_dir = Path(__file__).parent
    patch_files = []

    for patch_file in patches_dir.glob("*.patch.py"):
        # Extract module name from filename
        # Format: module.name.patch.py -> module.name
        module_name = patch_file.stem.rsplit(".patch", 1)[0]
        # Convert filename format to module format
        # vllm__attention__ops__triton_unified_attention -> vllm.attention.ops.triton_unified_attention
        module_name = module_name.replace("__", ".")
        patch_files.append((module_name, patch_file))

    return patch_files


def _load_patch_config(patch_file: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Load patch configuration from a patch file.

    Patch files should define a PATCHES list of (old_str, new_str) tuples.
    """
    spec = importlib.util.spec_from_file_location("patch_config", patch_file)
    if spec is None or spec.loader is None:
        return [], []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        reload_after_patch = getattr(module, "RELOAD_AFTER_PATCH", False)
        if reload_after_patch is True:
            reload_targets = ["__TARGET_MODULE__"]
        elif isinstance(reload_after_patch, str):
            reload_targets = [reload_after_patch]
        elif reload_after_patch:
            reload_targets = list(reload_after_patch)
        else:
            reload_targets = []
        return getattr(module, "PATCHES", []), reload_targets
    except Exception as e:
        logger.warning(f"Failed to load patch config from {patch_file}: {e}")
        return [], []


def apply_patches():
    """Apply all patches for MUSA compatibility.

    This function should be called early during platform initialization.
    """
    global _patches_applied
    if _patches_applied:
        return

    with _patch_file_lock():
        if _patches_applied:
            return
        _patches_applied = True
        _apply_patches_unlocked()


def _apply_patches_unlocked():
    """Apply patches while holding the cross-process patch lock."""

    patch_files = _get_patch_files()

    for module_name, patch_file in patch_files:
        try:
            # Find the module spec
            try:
                spec = importlib.util.find_spec(module_name)
            except (ModuleNotFoundError, ImportError) as e:
                # Module doesn't exist in this vLLM version (e.g., vllm.worker.worker
                # exists in vLLM 0.10.x but not in 0.13.0 where V0 engine was removed)
                # or has circular import issues during spec discovery
                logger.debug(
                    f"Module {module_name} not found or has import issues: {e}, "
                    "skipping patch (this is expected for version-specific patches "
                    "or when modules are not yet fully initialized)"
                )
                continue
            if spec is None or spec.origin is None:
                logger.debug(f"Module {module_name} not found, skipping patch")
                continue

            # Read the source file
            try:
                with open(spec.origin, "r") as f:
                    source = f.read()
            except (IOError, OSError) as e:
                logger.debug(f"Cannot read {spec.origin}: {e}, skipping patch")
                continue

            # Load patches from patch file
            patches, reload_targets = _load_patch_config(patch_file)
            if not patches:
                continue

            # Check if any patches are needed
            needs_patch = any(
                _patch_is_needed(source, old, new) for old, new in patches
            )
            if not needs_patch:
                logger.debug(f"No patches needed for {module_name}")
                continue

            # Apply patches
            patched_source = source
            applied_count = 0
            for old, new in patches:
                if _patch_is_needed(patched_source, old, new):
                    patched_source = patched_source.replace(old, new)
                    applied_count += 1

            # Write back the patched source. Do not evict an already-imported
            # module from sys.modules: some vLLM modules register torch custom
            # ops at import time, and re-importing them would register the same
            # schema twice in the current process.
            with open(spec.origin, "w") as f:
                f.write(patched_source)

            logger.info(f"Applied {applied_count} patch(es) to {module_name}")
            for reload_target in reload_targets:
                if reload_target == "__TARGET_MODULE__":
                    reload_target = module_name
                loaded_module = sys.modules.get(reload_target)
                if loaded_module is None:
                    continue
                importlib.reload(loaded_module)
                logger.info(f"Reloaded patched module {reload_target}")

        except Exception as e:
            # More detailed error handling for circular imports
            if "circular import" in str(e) or "partially initialized" in str(e):
                logger.debug(
                    f"Skipping patch for {module_name} due to circular import "
                    f"during initialization: {e}"
                )
            else:
                logger.warning(f"Failed to apply patches to {module_name}: {e}")
