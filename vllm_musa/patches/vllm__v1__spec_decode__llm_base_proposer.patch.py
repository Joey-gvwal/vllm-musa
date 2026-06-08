# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA-0203: backport of vllm-project/vllm#34880 — llm_base_proposer.py.

In upstream PR #34880 these hunks target ``vllm/v1/spec_decode/eagle.py``.
On v0.20.0-dev the EagleProposer implementation lives in
``vllm/v1/spec_decode/llm_base_proposer.py`` (eagle.py is a 22-line
stub). The hunks below re-target the same semantic changes to the
v0.20.0-dev file structure.

Changes:
  1. Import ``CUDAGraphWrapper`` + ``BatchDescriptor``.
  2. ``CudagraphDispatcher(self.vllm_config)`` -> ``(..., for_draft_model=True)``.
  3. ``initialize_cudagraph_keys``: pass through target's
     ``cudagraph_mode`` directly (was restricted to PIECEWISE).
  4. ``load_model``: wrap ``self.model`` with
     ``CUDAGraphWrapper(runtime_mode=FULL)`` when target has FULL
     captures (no-op when target is PIECEWISE-only).
  5. ``_determine_batch_execution_and_padding``: return 4-tuple
     (adds ``batch_desc``); thread ``uniform_decode`` +
     ``uniform_decode_query_len`` through.
  6. ``propose``: pass ``batch_descriptor=batch_desc`` to both
     ``set_forward_context`` calls (first forward + per-step loop).

Skipped from PR #34880 in this version (lower-priority for MUSA
PIECEWISE workload — can be added later):
  - ``target_model_batch_desc`` propose() parameter (gpu_model_runner
    caller change required).
  - ``dummy_run`` two-pass capture variant.
  - ``prepare_next_token_ids_padded`` / ``prepare_inputs_padded``
    padding fixes.

When upstream PR #34880 merges (or v0.20.0-dev rebases onto it), this
patch's anchors will stop matching and the file should be deleted.
"""


def normalize_source(source: str) -> str:
    """Upgrade already-persisted MUSA draft FULL-wrapper patched sources."""
    return source.replace(
        """        # Gated by VLLM_MUSA_DRAFT_FULL_WRAP (default ON since the
        # query_start_loc in-place hunk fixed the captured-replay stale
        # data_ptr issue). Set to "0" to disable for debugging.
        import os as _os
        cudagraph_mode = self.compilation_config.cudagraph_mode
        if (
            _os.environ.get("VLLM_MUSA_DRAFT_FULL_WRAP", "1") == "1"
            and cudagraph_mode.has_full_cudagraphs()
""",
        """        # Gated by VLLM_MUSA_DRAFT_FULL_WRAP. Default OFF on MUSA after the
        # v0.20.0-dev rebase because the wrapped DeepSeek-V4 MTP draft model
        # can fail graph memory profiling; set to "1" to re-enable for
        # diagnostics after the draft FULL graph path is fixed.
        import os as _os
        cudagraph_mode = self.compilation_config.cudagraph_mode
        draft_full_wrap_default = "0" if current_platform.is_musa() else "1"
        if (
            _os.environ.get(
                "VLLM_MUSA_DRAFT_FULL_WRAP", draft_full_wrap_default
            )
            == "1"
            and cudagraph_mode.has_full_cudagraphs()
""",
    )


# ---- Hunk 1: import CUDAGraphWrapper + BatchDescriptor ----
_OLD_IMPORTS = """from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context"""

_NEW_IMPORTS = """from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context"""

# ---- Hunk 2: dispatcher init — for_draft_model=True ----
_OLD_DISPATCHER_INIT = (
    """        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)"""
)

_NEW_DISPATCHER_INIT = """        # MUSA-0203 / PR #34880: dedicated dispatcher for the draft model so
        # FULL-mode capture keys are registered with uniform_decode_query_len=1
        # (each Eagle decode step is 1 token per request).
        self.cudagraph_dispatcher = CudagraphDispatcher(
            self.vllm_config, for_draft_model=True
        )"""

# ---- Hunk 3: initialize_cudagraph_keys — pass target mode through ----
_OLD_INIT_KEYS = """        \"\"\"Initialize cudagraph dispatcher keys for eagle.

        Eagle only supports PIECEWISE cudagraphs (via mixed_mode).
        This should be called after adjust_cudagraph_sizes_for_spec_decode.
        \"\"\"
        if (
            not self.speculative_config.enforce_eager
            and cudagraph_mode.mixed_mode()
            in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]
        ):
            eagle_cudagraph_mode = CUDAGraphMode.PIECEWISE
        else:
            eagle_cudagraph_mode = CUDAGraphMode.NONE

        self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)"""

_NEW_INIT_KEYS = """        \"\"\"Initialize cudagraph dispatcher keys for eagle.

        MUSA-0203 / PR #34880: pass the target's cudagraph_mode through
        directly. With ``for_draft_model=True`` the dispatcher now
        registers both the legacy mixed-mode (PIECEWISE) keys AND the
        new FULL-mode keys at uniform_decode_query_len=1, so the draft
        model can dispatch FULL captures when the target uses FULL or
        FULL_AND_PIECEWISE.
        \"\"\"
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        # MUSA-0403: a parallel-drafting proposer (dflash) sees
        # 1 + num_speculative_tokens query tokens per request, so register +
        # dispatch FULL keys at that query_len (Eagle uses 1). Aligns the
        # boot-captured batch_descriptor with the inference dispatch.
        _draft_qdl = (
            1 + self.num_speculative_tokens if self.parallel_drafting else 1
        )
        self.cudagraph_dispatcher.uniform_decode_query_len = _draft_qdl
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, uniform_decode_query_len=_draft_qdl
        )"""

# ---- Hunk 4: load_model — wrap with CUDAGraphWrapper ----
_OLD_LOAD_MODEL = """self.model = self._get_model()

        # Find draft layers"""

_NEW_LOAD_MODEL = """self.model = self._get_model()
        # MUSA-0203 / PR #34880: wrap the draft model with FULL-mode
        # CUDAGraphWrapper when target uses FULL captures. CUDAGraphWrapper
        # is a no-op when the runtime mode at call time doesn't match FULL,
        # so this is safe when target is PIECEWISE-only.
        # Gated by VLLM_MUSA_DRAFT_FULL_WRAP. Default OFF on MUSA after the
        # v0.20.0-dev rebase because the wrapped DeepSeek-V4 MTP draft model
        # can fail graph memory profiling; set to "1" to re-enable for
        # diagnostics after the draft FULL graph path is fixed.
        import os as _os
        cudagraph_mode = self.compilation_config.cudagraph_mode
        draft_full_wrap_default = "0" if current_platform.is_musa() else "1"
        if (
            _os.environ.get(
                "VLLM_MUSA_DRAFT_FULL_WRAP", draft_full_wrap_default
            )
            == "1"
            and cudagraph_mode.has_full_cudagraphs()
            and not self.vllm_config.parallel_config.use_ubatching
            and not self.speculative_config.disable_padded_drafter_batch
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )

        # Find draft layers"""

# ---- Hunk 5: _determine_batch_execution_and_padding signature + dispatch + return ----
_OLD_DETERMINE = """    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),"""

_NEW_DETERMINE = """    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        use_cudagraphs: bool = True,
        uniform_decode_query_len: int | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens,
            uniform_decode,
            valid_modes=({CUDAGraphMode.NONE} if not use_cudagraphs else None),
            uniform_decode_query_len=uniform_decode_query_len,"""

# Need to also patch the return statement of _determine_batch_execution_and_padding
# to 4-tuple. Find the unique anchor.
_OLD_DETERMINE_RETURN = (
    """        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp"""
)

_NEW_DETERMINE_RETURN = """        return cudagraph_mode, batch_desc, num_tokens_padded, num_tokens_across_dp"""

# ---- Hunk 6: propose() first-forward _determine_batch unpack ----
# In v0.20.0-dev the propose() first forward is:
#   cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
#       self._determine_batch_execution_and_padding(num_tokens)
#   )
# We need 4-tuple unpack now.
_OLD_PROPOSE_FIRST_DET = """        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )"""

_NEW_PROPOSE_FIRST_DET = """        # MUSA-0203 / PR #34880: derive uniform_decode locally (PR #34880 plumbs
        # `target_model_batch_desc.uniform` from gpu_model_runner; we read it off
        # common_attn_metadata.max_query_len == 1 to skip the caller-side hunks).
        uniform_decode = common_attn_metadata.max_query_len == 1
        cudagraph_runtime_mode, batch_desc, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens, uniform_decode)
        )"""

# ---- Hunk 7: propose() first-forward set_forward_context — thread batch_desc ----
_OLD_FFC_FIRST = """            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),"""

_NEW_FFC_FIRST = """            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_desc,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),"""

# ---- Hunk 8: propose() loop _determine_batch unpack ----
# The second occurrence is inside the loop, with `batch_size` not `num_tokens`.
_OLD_PROPOSE_LOOP_DET = """        cudagraph_runtime_mode, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(batch_size)
        )"""

_NEW_PROPOSE_LOOP_DET = """        # MUSA-0203 / PR #34880: subsequent decode passes dispatch with
        # (batch_size, uniform=True, qdl=1) so the draft's FULL-mode capture
        # keys are exercised.
        cudagraph_runtime_mode, batch_desc, input_batch_size, batch_size_across_dp = (
            self._determine_batch_execution_and_padding(
                batch_size, uniform_decode=True, uniform_decode_query_len=1
            )
        )"""

# ---- Hunk 9: propose() loop set_forward_context — thread batch_desc ----
_OLD_FFC_LOOP = """                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(input_batch_size),"""

_NEW_FFC_LOOP = """                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                slot_mapping=self._get_slot_mapping(input_batch_size),"""

# ---- Hunk 9b: propose() loop — in-place query_start_loc/query_start_loc_cpu ----
# PR #34880 changes reassignment to in-place write so the captured loop graph's
# data_ptr stays valid at replay. Reassignment created a fresh tensor each call
# and the captured FA kernel read from the stale pre-capture pointer, which is
# the root cause of Position 1-4 accept-rate collapse on Step 4.
_OLD_QSL_REASSIGN = """        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()"""

_NEW_QSL_REASSIGN = """        # MUSA-0203 / PR #34880: in-place write keeps the captured graph's
        # data_ptr valid across replays (the previous reassignment created a
        # fresh tensor each call, stranding the captured FA kernel on a stale
        # pointer and collapsing accept rate at positions > 0).
        common_attn_metadata.query_start_loc[: batch_size + 1] = self.arange[
            : batch_size + 1
        ]
        common_attn_metadata.query_start_loc_cpu[: batch_size + 1] = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()"""

# ---- Hunk 11: full dummy_run rewrite — accept common_attn_metadata and
# capture FULL-mode entries during boot. Without this, CUDAGraphWrapper
# tries to capture at request time and trips validate_cudagraph_capturing_enabled.
_OLD_DUMMY_RUN = """    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # FIXME: when using tree-based specdec, adjust number of forward-passes
        # according to the depth of the tree.
        only_one_forward_pass = is_graph_capturing or self.parallel_drafting
        for fwd_idx in range(
            1 if only_one_forward_pass else self.num_speculative_tokens
        ):
            if fwd_idx <= 1:
                cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                    self._determine_batch_execution_and_padding(
                        num_tokens, use_cudagraphs=use_cudagraphs
                    )
                )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):"""

_NEW_DUMMY_RUN = """    def dummy_run(
        self,
        num_tokens: int,
        common_attn_metadata: 'CommonAttentionMetadata | None' = None,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # MUSA-0203 / PR #34880: when capturing FULL graphs for the draft,
        # run 2 fwd passes (initial step + 1 decode step) so both the
        # variable-len first-forward and the uniform decode step get
        # captured at boot.
        if self.parallel_drafting:
            num_fwd_passes = 1
        elif is_graph_capturing:
            num_fwd_passes = min(self.num_speculative_tokens, 2)
        else:
            num_fwd_passes = self.num_speculative_tokens
        for fwd_idx in range(num_fwd_passes):
            if fwd_idx > 0 and common_attn_metadata is not None:
                # Match propose()'s subsequent decode passes which dispatch
                # with (batch_size, uniform=True, qdl=1).
                uniform_decode = True
                num_tokens = common_attn_metadata.num_reqs
                uniform_decode_query_len = 1
            else:
                mode = self.cudagraph_dispatcher.cudagraph_mode
                is_full_sep = (
                    mode.decode_mode() == CUDAGraphMode.FULL
                    and mode.separate_routine()
                )
                uniform_decode = is_full_sep and common_attn_metadata is not None
                uniform_decode_query_len = None

            (
                cudagraph_runtime_mode,
                batch_desc,
                num_input_tokens,
                num_tokens_across_dp,
            ) = self._determine_batch_execution_and_padding(
                num_tokens,
                uniform_decode,
                use_cudagraphs=use_cudagraphs,
                uniform_decode_query_len=uniform_decode_query_len,
            )

            # Make sure to use EAGLE's own buffer during cudagraph capture.
            if (
                self._draft_attn_layer_names
                and slot_mappings is not None
                and next(iter(self._draft_attn_layer_names)) in slot_mappings
            ):
                slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
            else:
                slot_mapping_dict = slot_mappings or {}

            # MUSA-0203 / PR #34880: build per-layer attn metadata for the
            # captured forward so attention backends see the right shape
            # at capture time.
            if common_attn_metadata is not None:
                dummy_attn_metadata = {}
                for attn_group in self.draft_attn_groups:
                    attn_metadata = (
                        attn_group.get_metadata_builder().build_for_drafting(
                            common_attn_metadata=common_attn_metadata,
                            draft_index=0,
                        )
                    )
                    for layer_name in attn_group.layer_names:
                        dummy_attn_metadata[layer_name] = attn_metadata
            else:
                dummy_attn_metadata = None

            with set_forward_context(
                dummy_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_desc,
                slot_mapping=slot_mapping_dict,
            ):"""

# ---- Hunk 12: propose() isinstance check — unwrap CUDAGraphWrapper ----
# When load_model wraps self.model with CUDAGraphWrapper, the isinstance check
# fails because self.model is now a CUDAGraphWrapper, not an Eagle3* class.
# Unwrap before checking. This is PR #34880's matching fix for the load_model
# wrap.
_OLD_ISINSTANCE = """        if self.method in ("eagle3", "dflash"):
            assert isinstance(
                self.model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                ),
            )"""

_NEW_ISINSTANCE = """        if self.method in ("eagle3", "dflash"):
            # MUSA-0203 / PR #34880: self.model may be wrapped by
            # CUDAGraphWrapper(FULL) at load_model time; unwrap for the
            # isinstance check.
            if isinstance(self.model, CUDAGraphWrapper):
                _draft_model = self.model.unwrap()
            else:
                _draft_model = self.model
            assert isinstance(
                _draft_model,
                (
                    Eagle3LlamaForCausalLM,
                    Eagle3DeepseekV2ForCausalLM,
                    DFlashQwen3ForCausalLM,
                ),
            )"""

PATCHES = [
    (_OLD_IMPORTS, _NEW_IMPORTS),
    (_OLD_DISPATCHER_INIT, _NEW_DISPATCHER_INIT),
    (_OLD_INIT_KEYS, _NEW_INIT_KEYS),
    (_OLD_LOAD_MODEL, _NEW_LOAD_MODEL),
    (_OLD_DETERMINE, _NEW_DETERMINE),
    (_OLD_DETERMINE_RETURN, _NEW_DETERMINE_RETURN),
    (_OLD_PROPOSE_FIRST_DET, _NEW_PROPOSE_FIRST_DET),
    (_OLD_FFC_FIRST, _NEW_FFC_FIRST),
    (_OLD_PROPOSE_LOOP_DET, _NEW_PROPOSE_LOOP_DET),
    (_OLD_FFC_LOOP, _NEW_FFC_LOOP),
    (_OLD_QSL_REASSIGN, _NEW_QSL_REASSIGN),
    (_OLD_DUMMY_RUN, _NEW_DUMMY_RUN),
    (_OLD_ISINSTANCE, _NEW_ISINSTANCE),
]
