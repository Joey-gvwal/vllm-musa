# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA DeepSeek/MTP source patches for gpu_model_runner.py.

The base patch is MUSA-0203, a backport of vllm-project/vllm#34880 that makes
the draft model's CUDAGraphWrapper(FULL) capture entries during boot.

MUSA-3046 is kept as an additional source patch tuple so it composes with the
upstream MUSA-0203 backport: skip the drafter for non-greedy MUSA requests.

History: this filename previously held the MUSA-0109
``VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM`` default-stream workaround that targeted
the now-deleted EagleFullLoopRunner. That monkey patch is no longer reachable.
"""

# ---- MUSA-0203 / PR #34880: initialize supports_sd_full_graph = False ----
_OLD_INIT_FLAG = """        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding."""

_NEW_INIT_FLAG = """        self.use_aux_hidden_state_outputs = False
        # MUSA-0203 / PR #34880: tracks whether the draft model supports
        # FULL-mode CUDA-graph capture (Eagle + padded drafter batch).
        self.supports_sd_full_graph = False
        # Set up speculative decoding."""

# ---- MUSA-0203 / PR #34880: set the flag for Eagle proposers ----
_OLD_EAGLE_INIT = """            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )"""

_NEW_EAGLE_INIT = """            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
                # MUSA-0203 / PR #34880: enable FULL-mode draft capture when
                # padded drafter batch is enabled.
                self.supports_sd_full_graph = (
                    not self.speculative_config.disable_padded_drafter_batch
                )"""

# ---- MUSA-0203 / PR #34880: declare spec_decode_cm None ----
_OLD_ATTN_DECL = """        attn_metadata: PerLayerAttnMetadata | None = None"""

_NEW_ATTN_DECL = """        attn_metadata: PerLayerAttnMetadata | None = None
        spec_decode_cm: 'CommonAttentionMetadata | None' = None"""

# ---- MUSA-0203 / PR #34880: capture spec_decode_cm ----
_OLD_BUILD_ATTN = (
    """                attn_metadata, _ = self._build_attention_metadata("""
)

_NEW_BUILD_ATTN = """                attn_metadata, spec_decode_cm = self._build_attention_metadata("""

# ---- MUSA-0203 / PR #34880: extend use_cudagraphs predicate ----
_OLD_USE_CG = """                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager"""

_NEW_USE_CG = """                # MUSA-0203 / PR #34880: Eagle now supports FULL cudagraphs via
                # CUDAGraphWrapper around the draft model (gated on
                # supports_sd_full_graph in __init__).
                use_cudagraphs = (
                    (
                        is_graph_capturing
                        and (
                            cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
                            or self.supports_sd_full_graph
                        )
                    )
                    or (
                        not is_graph_capturing
                        and cudagraph_runtime_mode != CUDAGraphMode.NONE
                    )
                ) and not self.speculative_config.enforce_eager"""

# ---- MUSA-0203 / PR #34880: pass common_attn_metadata to drafter ----
_OLD_DRAFTER_CALL = """                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )"""

_NEW_DRAFTER_CALL = """                self.drafter.dummy_run(
                    num_tokens,
                    common_attn_metadata=spec_decode_cm,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )"""

# ---- Hunk 6 (MUSA-0403): DEFAULT-ON FULL-mode draft capture for dflash ----
# dflash hits use_dflash() (before use_eagle()) so it never set
# supports_sd_full_graph -> draft transformer stays uncaptured -> the +47% eager
# floor is draft-launch-bound. Capturing the draft loop lifts the fair compile
# ratio 1.47x->1.83x (prose) and to 4.09x (predictable workloads); acceptance
# stayed healthy (no per-position collapse) so set_inputs_first_pass buffer-
# stability is verified. DEFAULT-ON; opt out with VLLM_MUSA_DFLASH_FULL_WRAP=0.
# platform.check_and_update_config coerces dflash cudagraph sizes to block-
# aligned + pure FULL so the default capture set does not crash the draft.
_OLD_DFLASH_INIT = """            elif self.speculative_config.use_dflash():
                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True"""

_NEW_DFLASH_INIT = """            elif self.speculative_config.use_dflash():
                self.drafter = DFlashProposer(self.vllm_config, self.device, self)
                self.use_aux_hidden_state_outputs = True
                import os as _dflash_os
                if _dflash_os.environ.get("VLLM_MUSA_DFLASH_FULL_WRAP", "1") != "0":
                    self.supports_sd_full_graph = True"""

# ---- MUSA-3046: avoid unstable random speculative MUSA drafter path ----
_OLD_MUSA_RANDOM_DRAFTER_FITS = """            # Decide whether to run the drafter or zero out draft tokens.
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
"""

_NEW_MUSA_RANDOM_DRAFTER_FITS = """            # Decide whether to run the drafter or zero out draft tokens.
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
"""

# ---- MUSA-3406: reuse pinned spec-decode metadata upload buffers ----
_OLD_SPEC_METADATA_BUFFERS_INIT = """        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)"""

_NEW_SPEC_METADATA_BUFFERS_INIT = """        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # MUSA-3406: reusable CPU/GPU buffers for speculative metadata indices.
        # Avoid per-step torch.from_numpy(...).to(device) pageable uploads in
        # _calc_spec_decode_metadata on the TP8 DeepSeek-V4 decode path.
        self._spec_cu_num_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self._spec_cu_num_sampled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self._spec_logits_indices = self._make_buffer(
            self.max_num_tokens, dtype=torch.int64
        )
        self._spec_target_logits_indices = self._make_buffer(
            self.max_num_tokens, dtype=torch.int32
        )
        self._spec_bonus_logits_indices = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self._spec_single_req_cached_draft_len = -1"""

_OLD_SPEC_METADATA_TO_DEVICE = """        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )"""

_NEW_SPEC_METADATA_TO_DEVICE = """        if current_platform.is_musa():
            # MUSA-3406: copy through persistent pinned CPU buffers so the
            # per-step metadata uploads do not allocate pageable CPU tensors.
            num_reqs = num_draft_tokens.shape[0]
            num_sampled_total = int(cu_num_sampled_tokens[-1])
            num_draft_total = int(cu_num_draft_tokens[-1])
            single_req_cache_enabled = (
                __import__("os").environ.get(
                    "VLLM_MUSA_SPEC_METADATA_SINGLE_REQ_CACHE", "0"
                ).lower()
                in ("1", "true", "yes", "on")
                and num_reqs == 1
                and int(num_draft_tokens[0]) > 0
                and num_sampled_total == int(num_draft_tokens[0]) + 1
                and num_draft_total == int(num_draft_tokens[0])
                and int(cu_num_scheduled_tokens[0]) == num_sampled_total
            )

            if single_req_cache_enabled:
                # MUSA-3463: in the TP8 single-request MTP decode path the
                # metadata indices are constant for each draft length. Upload
                # them once, then reuse the GPU views while still recomputing
                # draft_token_ids from the current input_ids.gpu below.
                draft_len = int(num_draft_tokens[0])
                if self._spec_single_req_cached_draft_len != draft_len:
                    self._spec_cu_num_draft_tokens.np[0] = draft_len
                    self._spec_cu_num_sampled_tokens.np[0] = draft_len + 1
                    self._spec_logits_indices.np[: draft_len + 1] = (
                        self._arange_scratch[: draft_len + 1]
                    )
                    self._spec_target_logits_indices.np[:draft_len] = (
                        self._arange_scratch[:draft_len]
                    )
                    self._spec_bonus_logits_indices.np[0] = draft_len

                    self._spec_cu_num_draft_tokens.copy_to_gpu(1)
                    self._spec_cu_num_sampled_tokens.copy_to_gpu(1)
                    self._spec_logits_indices.copy_to_gpu(draft_len + 1)
                    self._spec_target_logits_indices.copy_to_gpu(draft_len)
                    self._spec_bonus_logits_indices.copy_to_gpu(1)
                    self._spec_single_req_cached_draft_len = draft_len
            else:
                self._spec_cu_num_draft_tokens.np[:num_reqs] = cu_num_draft_tokens
                self._spec_cu_num_sampled_tokens.np[:num_reqs] = cu_num_sampled_tokens
                self._spec_logits_indices.np[:num_sampled_total] = logits_indices
                self._spec_target_logits_indices.np[:num_draft_total] = (
                    target_logits_indices
                )
                self._spec_bonus_logits_indices.np[:num_reqs] = bonus_logits_indices

                self._spec_cu_num_draft_tokens.copy_to_gpu(num_reqs)
                self._spec_cu_num_sampled_tokens.copy_to_gpu(num_reqs)
                self._spec_logits_indices.copy_to_gpu(num_sampled_total)
                self._spec_target_logits_indices.copy_to_gpu(num_draft_total)
                self._spec_bonus_logits_indices.copy_to_gpu(num_reqs)
                self._spec_single_req_cached_draft_len = -1

            cu_num_draft_tokens = self._spec_cu_num_draft_tokens.gpu[:num_reqs]
            cu_num_sampled_tokens = self._spec_cu_num_sampled_tokens.gpu[:num_reqs]
            logits_indices = self._spec_logits_indices.gpu[:num_sampled_total]
            target_logits_indices = self._spec_target_logits_indices.gpu[
                :num_draft_total
            ]
            bonus_logits_indices = self._spec_bonus_logits_indices.gpu[:num_reqs]
        else:
            # TODO: Optimize the CPU -> GPU copy.
            cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
                self.device, non_blocking=True
            )
            cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
                self.device, non_blocking=True
            )
            logits_indices = torch.from_numpy(logits_indices).to(
                self.device, non_blocking=True
            )
            target_logits_indices = torch.from_numpy(target_logits_indices).to(
                self.device, non_blocking=True
            )
            bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
                self.device, non_blocking=True
            )"""

PATCHES = [
    (_OLD_INIT_FLAG, _NEW_INIT_FLAG),
    (_OLD_EAGLE_INIT, _NEW_EAGLE_INIT),
    (_OLD_DFLASH_INIT, _NEW_DFLASH_INIT),
    (_OLD_ATTN_DECL, _NEW_ATTN_DECL),
    (_OLD_BUILD_ATTN, _NEW_BUILD_ATTN),
    (_OLD_USE_CG, _NEW_USE_CG),
    (_OLD_DRAFTER_CALL, _NEW_DRAFTER_CALL),
    (_OLD_MUSA_RANDOM_DRAFTER_FITS, _NEW_MUSA_RANDOM_DRAFTER_FITS),
    (_OLD_SPEC_METADATA_BUFFERS_INIT, _NEW_SPEC_METADATA_BUFFERS_INIT),
    (_OLD_SPEC_METADATA_TO_DEVICE, _NEW_SPEC_METADATA_TO_DEVICE),
]
