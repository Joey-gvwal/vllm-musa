# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep DeepSeek V4 MTP verifier rows off the MUSA sparse decode path."""

PATCHES = [
    (
        """        self.reorder_batch_threshold += self.num_speculative_tokens
""",
        """        self.reorder_batch_threshold += self.num_speculative_tokens
        if (
            current_platform.is_musa()
            and self.num_speculative_tokens > 0
            and getattr(
                self.vllm_config.model_config.hf_config,
                "model_type",
                None,
            ) == "deepseek_v4"
        ):
            # MUSA sparse decode for DeepSeek V4 MTP verifier rows
            # (query_len > 1) is not greedy-token-parity safe. Treat only
            # ordinary single-token requests as decode so verifier rows use
            # the prefill metadata path.
            self.reorder_batch_threshold = 1
""",
    ),
]
