# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.distributed.device.communicators.custom_all_reduce.
"""

_TRACE_IMPORTS = (
    "from contextlib import contextmanager\nfrom typing import cast\n",
    "import atexit\nimport os\nimport traceback\n"
    "from contextlib import contextmanager\nfrom typing import cast\n",
)

_TRACE_HELPER = (
    "logger = init_logger(__name__)\n\n\n"
    "def _musa_custom_ar_env_int(name: str, default: int) -> int:\n"
    "    try:\n"
    "        return int(os.environ.get(name, str(default)))\n"
    "    except ValueError:\n"
    "        return default\n\n\n"
    "_MUSA_CUSTOM_AR_TRACE = os.environ.get(\n"
    "    \"VLLM_MUSA_CUSTOM_AR_TRACE\", \"\"\n"
    ").lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "_MUSA_CUSTOM_AR_TRACE_LIMIT = _musa_custom_ar_env_int(\n"
    "    \"VLLM_MUSA_CUSTOM_AR_TRACE_LIMIT\", 256\n"
    ")\n"
    "_MUSA_CUSTOM_AR_TRACE_STACK_LIMIT = _musa_custom_ar_env_int(\n"
    "    \"VLLM_MUSA_CUSTOM_AR_TRACE_STACK_LIMIT\", 24\n"
    ")\n"
    "_MUSA_CUSTOM_AR_TRACE_SUMMARY_LIMIT = _musa_custom_ar_env_int(\n"
    "    \"VLLM_MUSA_CUSTOM_AR_TRACE_SUMMARY_LIMIT\", 64\n"
    ")\n"
    "_MUSA_CUSTOM_AR_TRACE_STATE = {\"emitted\": 0, \"total\": 0, \"counts\": {}}\n\n\n"
    "def _musa_custom_ar_callsite() -> str:\n"
    "    frames = traceback.extract_stack(limit=_MUSA_CUSTOM_AR_TRACE_STACK_LIMIT)[:-2]\n"
    "    callsites: list[str] = []\n"
    "    for frame in reversed(frames):\n"
    "        filename = frame.filename\n"
    "        if filename.endswith((\n"
    "            \"custom_all_reduce.py\",\n"
    "            \"cuda_communicator.py\",\n"
    "            \"parallel_state.py\",\n"
    "            \"communication_op.py\",\n"
    "        )):\n"
    "            continue\n"
    "        try:\n"
    "            if filename.startswith(\"/ws/\"):\n"
    "                filename = os.path.relpath(filename, \"/ws\")\n"
    "        except ValueError:\n"
    "            pass\n"
    "        callsites.append(f\"{filename}:{frame.lineno}:{frame.name}\")\n"
    "        if len(callsites) >= 3:\n"
    "            break\n"
    "    return \" <- \".join(callsites) if callsites else \"unknown\"\n\n\n"
    "def _musa_trace_custom_ar_call(comm, inp: torch.Tensor) -> None:\n"
    "    if not _MUSA_CUSTOM_AR_TRACE:\n"
    "        return\n"
    "    numel = inp.numel()\n"
    "    nbytes = numel * inp.element_size()\n"
    "    shape = tuple(inp.shape)\n"
    "    callsite = _musa_custom_ar_callsite()\n"
    "    key = (\n"
    "        f\"rank={getattr(comm, 'rank', 'unknown')} \"\n"
    "        f\"world_size={getattr(comm, 'world_size', 'unknown')} \"\n"
    "        f\"shape={shape} dtype={inp.dtype} numel={numel} bytes={nbytes} \"\n"
    "        f\"capturing={getattr(comm, '_IS_CAPTURING', False)} site={callsite}\"\n"
    "    )\n"
    "    counts = _MUSA_CUSTOM_AR_TRACE_STATE[\"counts\"]\n"
    "    counts[key] = counts.get(key, 0) + 1\n"
    "    _MUSA_CUSTOM_AR_TRACE_STATE[\"total\"] += 1\n"
    "    if _MUSA_CUSTOM_AR_TRACE_STATE[\"emitted\"] < _MUSA_CUSTOM_AR_TRACE_LIMIT:\n"
    "        logger.info(\n"
    "            \"MUSA_CUSTOM_AR_TRACE total_calls=%d group_count=%d %s\",\n"
    "            _MUSA_CUSTOM_AR_TRACE_STATE[\"total\"],\n"
    "            counts[key],\n"
    "            key,\n"
    "        )\n"
    "        _MUSA_CUSTOM_AR_TRACE_STATE[\"emitted\"] += 1\n\n\n"
    "def _musa_dump_custom_ar_trace_summary() -> None:\n"
    "    if not _MUSA_CUSTOM_AR_TRACE:\n"
    "        return\n"
    "    counts = _MUSA_CUSTOM_AR_TRACE_STATE[\"counts\"]\n"
    "    if not counts:\n"
    "        logger.info(\"MUSA_CUSTOM_AR_TRACE_SUMMARY total_calls=0\")\n"
    "        return\n"
    "    total = sum(counts.values())\n"
    "    logger.info(\n"
    "        \"MUSA_CUSTOM_AR_TRACE_SUMMARY total_calls=%d unique_groups=%d\",\n"
    "        total,\n"
    "        len(counts),\n"
    "    )\n"
    "    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)\n"
    "    for key, count in sorted_counts[:_MUSA_CUSTOM_AR_TRACE_SUMMARY_LIMIT]:\n"
    "        logger.info(\"MUSA_CUSTOM_AR_TRACE_SUMMARY count=%d %s\", count, key)\n\n\n"
    "atexit.register(_musa_dump_custom_ar_trace_summary)\n\n\n"
)

PATCHES = [
    # Add an opt-in trace hook for TP all-reduce shape/callsite analysis.
    # This is disabled by default and used by MUSA-3013 profiling only.
    _TRACE_IMPORTS,
    ("logger = init_logger(__name__)\n\n\n", _TRACE_HELPER),
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
    (
        "    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:\n"
        "        \"\"\"The main allreduce API that provides support for cuda graph.\"\"\"\n",
        "    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:\n"
        "        \"\"\"The main allreduce API that provides support for cuda graph.\"\"\"\n"
        "        _musa_trace_custom_ar_call(self, input)\n",
    ),
]
