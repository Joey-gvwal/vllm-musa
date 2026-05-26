# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MUSA-0109 / MUSA-0090 layer-3 attempt: bypass `draft_token_ids_copy_stream`
on MUSA so the H2D copy of draft tokens runs on the default stream (same
stream the runner's CUDAGraph replay used).

Why: with VLLM_MUSA_EAGLE_RUNNER=1, vllm's `_copy_draft_token_ids_to_cpu`
fails with "MUDNN err 999 = unknown error" because it copies from
runner-output (potentially CUDAGraph pool memory) on a DEDICATED stream
(`self.draft_token_ids_copy_stream`). Forcing default-stream avoids the
cross-stream interaction.

This is a non-destructive change: same-stream copy is semantically
identical to cross-stream copy + event-record + wait. The async-overlap
benefit goes away, but at BS=1 the copy is ~12 bytes (int32 [bs=1, 3])
so it doesn't matter.

Gated via VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM=1 (default OFF).
Enable only when running with VLLM_MUSA_EAGLE_RUNNER=1.
"""

PATCHES = [
    (
        """            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
""",
        """            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            if (
                current_platform.is_musa()
                and not self.input_batch.sampling_metadata.all_greedy
                and __import__("os").environ.get(
                    "VLLM_MUSA_SPEC_DECODE_RANDOM_FALLBACK",
                    "1",
                ).lower()
                not in ("0", "false", "no", "off")
            ):
                # The MUSA random rejection-sampling kernel is not
                # correctness-stable yet. Do not run the drafter for
                # non-greedy requests; the target model already sampled one
                # correct token for this step.
                input_fits_in_drafter = False
""",
    ),
    (
        """        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)
""",
        """        if (
            spec_config is not None
            and current_platform.is_musa()
            and spec_config.method == "mtp"
            and spec_decode_metadata is not None
            and not spec_config.disable_padded_drafter_batch
            and hasattr(self.get_model(), "get_mtp_target_hidden_states")
        ):
            # DeepSeek V4 MTP consumes the target pre-hc_head residual. After
            # a rejection, the verifier recovers a target token but the drafter
            # may already have generated next-step drafts from the pre-recovery
            # residual. Bookkeeping has already materialized valid sampled
            # token IDs on CPU, so suppress those drafts here without adding a
            # graph-unsafe D2H sync inside the decode/capture path.
            max_valid_count = self.num_spec_tokens + 1
            if any(
                0 < len(tokens) < max_valid_count
                for tokens in valid_sampled_token_ids
            ):
                self._draft_token_ids = [
                    [] for _ in self.input_batch.req_ids
                ]
                self._draft_token_req_ids = self.input_batch.req_ids.copy()

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)
""",
    ),
    (
        """        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support,
            min_cg_attn_backend,
            self.uniform_decode_query_len,
            self.parallel_config.tensor_parallel_size,
            self.kv_cache_config,
            self.max_num_reqs,
            is_profiling=is_profiling,
        )
""",
        """        if (
            current_platform.is_musa()
            and self.speculative_config is not None
            and self.speculative_config.method == "mtp"
            and getattr(self.model_config.hf_config, "model_type", None)
            == "deepseek_v4"
            and __import__("os").environ.get(
                "VLLM_MUSA_DEEPSEEK_V4_MTP_ALLOW_CUDAGRAPH",
                "0",
            ).lower()
            not in ("1", "true", "yes", "on")
        ):
            # DeepSeek V4 MTP verifier rows need the eager/prefill metadata
            # path on MUSA for greedy token parity. Capturing the verifier
            # shape as a FULL_DECODE_ONLY graph currently trips a MUSA graph
            # replay failure before output parsing. Keep graph+MTP service
            # commands correctness-safe by falling back to eager MTP unless a
            # diagnostic run explicitly opts into the graph path.
            if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                logger.warning(
                    "Disabling CUDA Graph for DeepSeek V4 MTP on MUSA. "
                    "Set VLLM_MUSA_DEEPSEEK_V4_MTP_ALLOW_CUDAGRAPH=1 "
                    "only for diagnostic graph replay runs."
                )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            self.compilation_config.cudagraph_capture_sizes = []
            self.compilation_config.max_cudagraph_capture_size = 0

        cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
            min_cg_support,
            min_cg_attn_backend,
            self.uniform_decode_query_len,
            self.parallel_config.tensor_parallel_size,
            self.kv_cache_config,
            self.max_num_reqs,
            is_profiling=is_profiling,
        )
""",
    ),
]

import logging
import os

_log = logging.getLogger(__name__)

_ENABLED = os.environ.get("VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM", "0") == "1"

if not _ENABLED:
    _log.info(
        "MUSA-0109: VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM=0; "
        "draft_token_ids_copy_stream stays on a dedicated stream (default vllm)"
    )
else:
    try:
        import torch
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError as exc:
        _log.warning(
            "MUSA-0109 draft_copy_stream: import failed (%s); patch disabled",
            exc,
        )
        GPUModelRunner = None

    if GPUModelRunner is not None and not getattr(
        GPUModelRunner, "_musa_draft_copy_stream_patched", False
    ):
        _ORIGINAL_INIT = GPUModelRunner.__init__

        def _musa_patched_init(self, *args, **kwargs):
            _ORIGINAL_INIT(self, *args, **kwargs)
            # Override the dedicated copy stream with the current default stream
            # AFTER the original init has constructed everything else.
            if getattr(self, "draft_token_ids_copy_stream", None) is not None:
                # Replace with current stream so the H2D copy stays on it.
                self.draft_token_ids_copy_stream = torch.cuda.current_stream()
                _log.info(
                    "MUSA-0109: draft_token_ids_copy_stream redirected to "
                    "torch.cuda.current_stream() to avoid CUDAGraph pool "
                    "cross-stream interaction"
                )
            if getattr(self, "valid_sampled_token_count_copy_stream", None) is not None:
                self.valid_sampled_token_count_copy_stream = torch.cuda.current_stream()
                _log.info(
                    "MUSA-0109: valid_sampled_token_count_copy_stream redirected "
                    "to torch.cuda.current_stream()"
                )

        GPUModelRunner.__init__ = _musa_patched_init
        GPUModelRunner._musa_draft_copy_stream_patched = True
        _log.info(
            "MUSA-0109: GPUModelRunner.__init__ monkey-patched to redirect "
            "copy streams to default stream (gate "
            "VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM=1)"
        )
