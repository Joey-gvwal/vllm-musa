# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep DeepSeek V4 MTP verifier rows off FlashMLA sparse decode on MUSA."""

PATCHES = [
    (
        """        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
""",
        """        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        if (
            current_platform.is_musa()
            and self.vllm_config.speculative_config is not None
            and self.vllm_config.speculative_config.method == "mtp"
            and getattr(
                self.vllm_config.model_config.hf_config,
                "model_type",
                None,
            ) == "deepseek_v4"
        ):
            # MUSA FlashMLA sparse decode for DeepSeek V4 MTP verifier rows
            # (query_len > 1) is not greedy-token-parity safe. Keep only the
            # ordinary single-token decode path classified as decode.
            self.reorder_batch_threshold = 1
""",
    ),
]
