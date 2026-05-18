# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.backends diagnostics on MUSA.

MUSA DeepSeek-V4 compile/no-graph diagnostics showed that fresh compilation can
serve requests while replaying vLLM's serialized compile-cache artifacts can
kill workers before readiness. Keep upstream behavior by default, but expose a
narrow opt-in gate to skip only vLLM's `vllm_compile_cache.py` replay path while
leaving the rest of torch/Inductor compilation behavior intact.
"""

PATCHES = [
    (
        """        if (compile_range, graph_index, self.compiler.name) not in self.cache:
            return None
""",
        """        if os.getenv("VLLM_MUSA_SKIP_VLLM_COMPILE_CACHE_REPLAY", "0").strip().lower() in {"1", "true", "yes", "on"}:
            if (compile_range, graph_index, self.compiler.name) in self.cache:
                logger.info_once(
                    "Skipping vLLM serialized compile-cache replay because "
                    "VLLM_MUSA_SKIP_VLLM_COMPILE_CACHE_REPLAY is enabled."
                )
            return None
        if (compile_range, graph_index, self.compiler.name) not in self.cache:
            return None
""",
    ),
    (
        """    # We're using Dynamo-based piecewise splitting, so we wrap
    # the whole subgraph with a static graph wrapper.
    from .cuda_graph import CUDAGraphOptions
""",
        """    skip_first_piecewise_cg = os.getenv(
        "VLLM_MUSA_SKIP_FIRST_PIECEWISE_CUDAGRAPH", "0"
    ).strip().lower()
    if (
        skip_first_piecewise_cg in {"1", "true", "yes", "on"}
        and getattr(current_platform, "is_musa", lambda: False)()
        and is_first_graph
    ):
        logger.info_once(
            "Skipping first piecewise CUDAGraph wrapper on MUSA because "
            "VLLM_MUSA_SKIP_FIRST_PIECEWISE_CUDAGRAPH is enabled."
        )
        return piecewise_backend

    # We're using Dynamo-based piecewise splitting, so we wrap
    # the whole subgraph with a static graph wrapper.
    from .cuda_graph import CUDAGraphOptions
""",
    ),
]
