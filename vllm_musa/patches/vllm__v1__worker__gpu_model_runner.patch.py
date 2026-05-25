# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 - gpu_model_runner.py.

Wires the caller-side path that makes the draft model's CUDAGraphWrapper(FULL)
actually capture entries during boot (instead of failing at request time with
``validate_cudagraph_capturing_enabled``).

Three changes:
  1. Add ``supports_sd_full_graph`` flag (True for Eagle + padded drafter).
  2. ``_dummy_run`` captures ``spec_decode_cm`` from ``_build_attention_metadata``
     and passes it to ``drafter.dummy_run`` as ``common_attn_metadata``.
  3. ``_dummy_run`` extends the ``use_cudagraphs`` predicate so FULL captures
     are enabled when ``supports_sd_full_graph``.

The later MUSA-3046 random-sampling guard is kept as an additional source patch
in this file so it composes with the upstream MUSA-0203 backport.

Skipped from PR #34880 (lower-priority, not strictly required to make the
draft FULL capture path engage on MUSA's BS=1 SOTA workload):
  - ``ExecuteModelState.batch_desc`` stash + ``propose_draft_token_ids``
    target_model_batch_desc threading. Our llm_base_proposer.patch.py
    derives ``uniform_decode`` locally from common_attn_metadata.max_query_len
    instead of taking it as an external arg.
  - ``input_batch.num_reqs_padded`` setter at execute_model.
  - ``_capture_cudagraphs`` clear_graphs cleanup.
  - ``_build_attn_group_metadata`` supports_sd_full_graph short-circuit
    (only matters for unpadded drafter batch, which we don't use).

History: this filename previously held the MUSA-0109
``VLLM_MUSA_DRAFT_COPY_DEFAULT_STREAM`` default-stream workaround that targeted
the now-deleted EagleFullLoopRunner (MUSA-0203 cleanup). That patch is no
longer reachable.
"""

# ---- Hunk 1: __init__ - initialize supports_sd_full_graph = False ----
_OLD_INIT_FLAG = """        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding."""

_NEW_INIT_FLAG = """        self.use_aux_hidden_state_outputs = False
        # MUSA-0203 / PR #34880: tracks whether the draft model supports
        # FULL-mode CUDA-graph capture (Eagle + padded drafter batch).
        self.supports_sd_full_graph = False
        # Set up speculative decoding."""

# ---- Hunk 2: __init__ - set the flag for Eagle proposers ----
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

# ---- Hunk 3a: _dummy_run - declare spec_decode_cm None at function top ----
# The assignment to spec_decode_cm happens inside an `if force_attention or
# cudagraph_runtime_mode == CUDAGraphMode.FULL:` block. When that branch
# doesn't fire (PIECEWISE without force_attention) the variable was never
# assigned, but the drafter.dummy_run call always references it:
# UnboundLocalError. Match PR #34880's declaration at the top of _dummy_run.
_OLD_ATTN_DECL = """        attn_metadata: PerLayerAttnMetadata | None = None"""

_NEW_ATTN_DECL = """        attn_metadata: PerLayerAttnMetadata | None = None
        spec_decode_cm: 'CommonAttentionMetadata | None' = None"""

# ---- Hunk 3b: _dummy_run - capture spec_decode_cm from _build_attention_metadata ----
_OLD_BUILD_ATTN = """                attn_metadata, _ = self._build_attention_metadata("""

_NEW_BUILD_ATTN = """                attn_metadata, spec_decode_cm = self._build_attention_metadata("""

# ---- Hunk 4: _dummy_run - extend use_cudagraphs predicate ----
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

# ---- Hunk 5: _dummy_run - pass common_attn_metadata to drafter.dummy_run ----
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

PATCHES = [
    (_OLD_INIT_FLAG, _NEW_INIT_FLAG),
    (_OLD_EAGLE_INIT, _NEW_EAGLE_INIT),
    (_OLD_DFLASH_INIT, _NEW_DFLASH_INIT),
    (_OLD_ATTN_DECL, _NEW_ATTN_DECL),
    (_OLD_BUILD_ATTN, _NEW_BUILD_ATTN),
    (_OLD_USE_CG, _NEW_USE_CG),
    (_OLD_DRAFTER_CALL, _NEW_DRAFTER_CALL),
    (_OLD_MUSA_RANDOM_DRAFTER_FITS, _NEW_MUSA_RANDOM_DRAFTER_FITS),
]
