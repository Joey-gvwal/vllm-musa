# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.v1.sample.rejection_sampler.

MUSA-0064: vLLM's spec-decode rejection-sampler Triton kernels do not
compile on MUSA's Triton (3.1.0), crashing engine init whenever
speculative decoding is enabled (observed with the MiniMax-M2.5 Eagle3
draft). The triton.compiler.errors.CompilationError caret points at
`if not is_greedy:` in rejection_greedy_sample_kernel:

    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
       ^
    ValueError('Cannot bitcast data-type of size 8 to data-type of size 1')

Root cause: `is_greedy_ptr` points to a `torch.bool` tensor; MUSA Triton
3.1.0 mishandles `not` / truthiness applied to a bool-typed `tl.load`'d
value (it tries to bitcast the size-8 pointer/value to the size-1 bool
type). CUDA Triton handles it; MUSA Triton 3.1.0 does not.

Fix (three coordinated replacements):
  1. Call site `rejection_sample()` — pass `is_greedy` as int32 instead
     of torch.bool so the kernels load a plain integer.
  2. rejection_greedy_sample_kernel — `True if ... else tl.load` becomes
     `1 if ... else tl.load`, and `if not is_greedy:` becomes the
     explicit `if is_greedy == 0:`.
  3. rejection_random_sample_kernel — `if is_greedy:` becomes the
     explicit `if is_greedy != 0:`.
Plus a defensive 1-element dummy tensor for the `synthetic_conditional_
rates` None pointer (kernels gate every read of it behind
`SYNTHETIC_MODE: tl.constexpr`, so the dummy is never read).

All replacements are behaviourally identical on CUDA; they only avoid
the MUSA-Triton-3.1.0 bool-bitcast path. `device` / `torch` are in
scope at the `rejection_sample()` insertion point.
"""

PATCHES = [
    (
        """def rejection_sample(
""",
        """def _musa_spec_decode_random_fallback_enabled() -> bool:
    value = __import__("os").environ.get(
        "VLLM_MUSA_SPEC_DECODE_RANDOM_FALLBACK",
        "1",
    )
    if value.lower() in ("0", "false", "no", "off"):
        return False
    try:
        from vllm.platforms import current_platform

        return current_platform.is_musa()
    except Exception:
        return False


def _musa_sample_first_target_token(
    sampler: Sampler,
    target_logits: torch.Tensor,
    metadata: SpecDecodeMetadata,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor | None:
    # MUSA random rejection sampling is not correctness-stable yet. Preserve
    # the distribution by sampling only the first target-token logits and
    # returning placeholders for all speculative positions, effectively
    # disabling speculative acceptance for non-greedy requests.
    if any(num_tokens <= 0 for num_tokens in metadata.num_draft_tokens):
        return None

    starts: list[int] = []
    offset = 0
    for num_tokens in metadata.num_draft_tokens:
        starts.append(offset)
        offset += num_tokens

    start_indices = torch.tensor(
        starts,
        dtype=torch.long,
        device=target_logits.device,
    )
    first_target_logits = target_logits.index_select(0, start_indices)
    sampled, _ = sampler.sample(first_target_logits, sampling_metadata)

    output_token_ids = torch.full(
        (len(metadata.num_draft_tokens), metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=target_logits.device,
    )
    output_token_ids[:, 0] = sampled.to(torch.int32)
    return output_token_ids


def rejection_sample(
""",
    ),
    (
        """        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
""",
        """        target_logits = self.apply_logits_processors(
            target_logits, sampling_metadata, metadata
        )
        if (
            not sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs is None
            and _musa_spec_decode_random_fallback_enabled()
        ):
            output_token_ids = _musa_sample_first_target_token(
                self.sampler,
                target_logits,
                metadata,
                sampling_metadata,
            )
            if output_token_ids is not None:
                return SamplerOutput(
                    sampled_token_ids=output_token_ids,
                    logprobs_tensors=None,
                )

        # [num_tokens, vocab_size]
        # NOTE(woosuk): `target_logits` can be updated in place inside the
        # `apply_sampling_constraints` function.
        target_logits = apply_sampling_constraints(
            target_logits,
            metadata.cu_num_draft_tokens,
            sampling_metadata,
        )

        output_token_ids = rejection_sample(
""",
    ),
    # 1 + dummy: call site -- is_greedy as int32 + synthetic dummy tensor.
    (
        """    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
""",
        """    if sampling_metadata.all_greedy:
        is_greedy = None
    else:
        # MUSA-0064: int32, not torch.bool -- MUSA Triton 3.1.0 chokes on
        # `not` / truthiness of a bool-typed tl.load'd value.
        is_greedy = (
            sampling_metadata.temperature == GREEDY_TEMPERATURE
        ).to(torch.int32)

    # MUSA-0064: MUSA Triton rejects the None-pointer handling for the
    # `synthetic_conditional_rates_ptr` kernel arg. Pass a 1-element
    # dummy tensor when synthetic mode is off; both rejection-sampler
    # kernels gate every read of it behind `SYNTHETIC_MODE: tl.constexpr`
    # so the dummy is never read.
    if synthetic_conditional_rates is None:
        synthetic_conditional_rates = torch.empty(
            1, dtype=torch.float32, device=device
        )
""",
    ),
    # 2: rejection_greedy_sample_kernel -- avoid `not` on a loaded value.
    (
        """    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if not is_greedy:
        # Early exit for non-greedy sampling requests.
        return""",
        """    # MUSA-0064: `1`/`== 0` instead of `True`/`not` -- MUSA Triton
    # 3.1.0 cannot bitcast the bool-typed loaded value.
    is_greedy = 1 if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    if is_greedy == 0:
        # Early exit for non-greedy sampling requests.
        return""",
    ),
    # 3: rejection_random_sample_kernel -- avoid truthiness on a loaded value.
    (
        """    req_idx = tl.program_id(0)
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy:
        # Early exit for greedy sampling requests.
        return""",
        """    req_idx = tl.program_id(0)
    # MUSA-0064: explicit `!= 0` instead of truthiness on the loaded
    # value -- MUSA Triton 3.1.0 cannot bitcast the bool-typed value.
    is_greedy = tl.load(is_greedy_ptr + req_idx)
    if is_greedy != 0:
        # Early exit for greedy sampling requests.
        return""",
    ),
]
