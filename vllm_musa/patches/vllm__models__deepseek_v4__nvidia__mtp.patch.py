# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Patch vLLM v0.22 DeepSeek-V4 MTP hidden-state shape handling for MUSA."""

PATCHES = [
    (
        """        if current_platform.is_cuda():
            hidden_states = self.mtp_block.hc_post(
                hidden_states, residual, post_mix, res_mix
            )
""",
        """        if current_platform.is_cuda() or current_platform.is_musa():
            hidden_states = self.mtp_block.hc_post(
                hidden_states, residual, post_mix, res_mix
            )
""",
    ),
]


def normalize_source(source: str) -> str:
    """Remove an obsolete logits-side guard from earlier MUSA triage.

    The correct contract is that the MTP forward path returns the flat
    pre-hc_head residual with width ``hc_mult * hidden_size``. Fixing logits to
    accept already dense states lets the first draft sample run, but breaks the
    next draft step's hidden-state buffer.
    """
    stale = """        if not (
            current_platform.is_musa()
            and hidden_states.dim() == 2
            and hidden_states.shape[-1] == mtp_layer.config.hidden_size
        ):
            # MTP forward returns the pre-hc_head residual (T, hc_mult * D);
            # apply hc_head here so logits are computed from the dense hidden
            # state. MUSA request-time propose can also receive an already
            # dense hidden state from the target path, in which case this
            # projection has already happened.
            hidden_states = hidden_states.view(
                -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
            )
            hidden_states = mtp_layer.hc_head_op(
                hidden_states,
                mtp_layer.hc_head_fn,
                mtp_layer.hc_head_scale,
                mtp_layer.hc_head_base,
                mtp_layer.rms_norm_eps,
                mtp_layer.hc_eps,
            )
        logits = self.logits_processor(
"""
    original = """        # MTP forward returns the pre-hc_head residual (T, hc_mult * D); apply
        # hc_head here so logits are computed from the dense hidden state.
        hidden_states = hidden_states.view(
            -1, mtp_layer.hc_mult, mtp_layer.config.hidden_size
        )
        hidden_states = mtp_layer.hc_head_op(
            hidden_states,
            mtp_layer.hc_head_fn,
            mtp_layer.hc_head_scale,
            mtp_layer.hc_head_base,
            mtp_layer.rms_norm_eps,
            mtp_layer.hc_eps,
        )
        logits = self.logits_processor(
"""
    return source.replace(stale, original)
