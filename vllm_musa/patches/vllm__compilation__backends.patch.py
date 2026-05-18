# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.compilation.backends diagnostics on MUSA.

MUSA DeepSeek-V4 compile/no-graph diagnostics showed that fresh compilation can
serve requests while replaying vLLM's serialized compile-cache artifacts can
kill workers before readiness. Keep upstream behavior by default, but expose a
narrow opt-in gate to skip only vLLM's `vllm_compile_cache.py` replay path while
leaving the rest of torch/Inductor compilation behavior intact.

MUSA DeepSeek-V4 force-graph diagnostics also showed that early piecewise
CUDAGraph wrappers can fail on generated Inductor allocations during stream
capture. Keep upstream behavior by default, but expose a second opt-in gate to
skip the first N piecewise CUDAGraph wrappers for source-map diagnostics.

The later post-attention graph blocker is operator-local rather than
index-local, so expose a third default-off diagnostic gate that skips wrapping
piecewise subgraphs whose FX nodes match a comma-separated list of operator
name fragments.
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
    skip_initial_piecewise_cg_env = os.getenv(
        "VLLM_MUSA_SKIP_INITIAL_PIECEWISE_CUDAGRAPHS"
    )
    skip_initial_piecewise_cg_count = 0
    if skip_initial_piecewise_cg_env is not None:
        try:
            skip_initial_piecewise_cg_count = max(
                0, int(skip_initial_piecewise_cg_env)
            )
        except ValueError:
            logger.warning_once(
                "Ignoring invalid VLLM_MUSA_SKIP_INITIAL_PIECEWISE_CUDAGRAPHS=%r; "
                "expected a non-negative integer.",
                skip_initial_piecewise_cg_env,
            )
    elif skip_first_piecewise_cg in {"1", "true", "yes", "on"}:
        skip_initial_piecewise_cg_count = 1

    piecewise_compile_index = getattr(piecewise_backend, "piecewise_compile_index", -1)
    if (
        skip_initial_piecewise_cg_count > 0
        and getattr(current_platform, "is_musa", lambda: False)()
        and 0 <= piecewise_compile_index < skip_initial_piecewise_cg_count
    ):
        if skip_initial_piecewise_cg_count == 1 and is_first_graph:
            logger.info_once(
                "Skipping first piecewise CUDAGraph wrapper on MUSA because "
                "VLLM_MUSA_SKIP_FIRST_PIECEWISE_CUDAGRAPH is enabled."
            )
        else:
            logger.info(
                "Skipping piecewise CUDAGraph wrapper %s/%s on MUSA because "
                "VLLM_MUSA_SKIP_INITIAL_PIECEWISE_CUDAGRAPHS=%s.",
                piecewise_compile_index,
                getattr(piecewise_backend, "total_piecewise_compiles", "?"),
                skip_initial_piecewise_cg_count,
            )
        return piecewise_backend

    skip_piecewise_op_env = os.getenv(
        "VLLM_MUSA_SKIP_PIECEWISE_CUDAGRAPH_OPS", ""
    ).strip()
    if (
        skip_piecewise_op_env
        and getattr(current_platform, "is_musa", lambda: False)()
        and getattr(piecewise_backend, "graph", None) is not None
    ):
        skip_op_tokens = [
            token.strip().lower()
            for token in skip_piecewise_op_env.replace(";", ",").split(",")
            if token.strip()
        ]
        matched_skip_ops = []
        for node in piecewise_backend.graph.graph.nodes:
            target = getattr(node, "target", "")
            target_text = " ".join(
                str(part).lower()
                for part in (
                    getattr(node, "name", ""),
                    target,
                    getattr(target, "__name__", ""),
                    getattr(target, "name", ""),
                )
                if part
            )
            for token in skip_op_tokens:
                if token in target_text:
                    matched_skip_ops.append(token)
                    break
        if matched_skip_ops:
            logger.info(
                "Skipping piecewise CUDAGraph wrapper %s/%s on MUSA because "
                "VLLM_MUSA_SKIP_PIECEWISE_CUDAGRAPH_OPS matched %s.",
                piecewise_compile_index,
                getattr(piecewise_backend, "total_piecewise_compiles", "?"),
                sorted(set(matched_skip_ops)),
            )
            return piecewise_backend

    # We're using Dynamo-based piecewise splitting, so we wrap
    # the whole subgraph with a static graph wrapper.
    from .cuda_graph import CUDAGraphOptions
""",
    ),
]
