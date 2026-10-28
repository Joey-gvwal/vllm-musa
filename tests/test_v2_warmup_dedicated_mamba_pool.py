# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = next(
    (ROOT / "vllm_musa" / "patches" / "series").glob(
        "*-number-warmup-blocks-within-each-dedicated-Mamb.patch"
    )
)

# Nemotron-3.5 MTP6 shape: four Mamba groups and one attention group.
NUM_MAMBA_GROUPS = 4
NUM_SPEC = 6
MAX_MODEL_LEN = 6144


def test_series_routes_all_warmup_blocks_through_the_allocator() -> None:
    text = PATCH.read_text()
    added = [line for line in text.splitlines() if line.startswith("+")]
    call = "alloc_blocks(g, n) for g, n in enumerate("
    assert "+def _warmup_block_allocator(" in added
    assert not any("_alloc_blocks(" in line for line in added)
    assert sum(call in line for line in added) == 5


def _runner(torch, kv, max_num_seqs: int, musa_mamba_num_blocks: int | None):
    mamba = kv.MambaSpec(
        block_size=MAX_MODEL_LEN,
        shapes=((4, 8),),
        dtypes=(torch.bfloat16,),
        num_speculative_blocks=NUM_SPEC,
    )
    attn = kv.FullAttentionSpec(
        block_size=64, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
    )
    groups = [
        kv.KVCacheGroupSpec([f"mamba.{i}"], mamba) for i in range(NUM_MAMBA_GROUPS)
    ]
    groups.append(kv.KVCacheGroupSpec(["attn.0"], attn))
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            is_mm_encoder_only=False, num_lookahead_tokens=NUM_SPEC
        ),
        num_speculative_steps=NUM_SPEC,
        decode_query_len=NUM_SPEC + 1,
        is_pooling_model=False,
        is_encoder_decoder=False,
        max_model_len=MAX_MODEL_LEN,
        model_state=SimpleNamespace(),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=groups,
            num_blocks=4096,
            musa_mamba_num_blocks=musa_mamba_num_blocks,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=512
        ),
        block_tables=None,
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 1024),
    )


def _warmup_block_ids(runner) -> list[list[int]]:
    """Run the V2 kernel warmup and collect the block IDs of every group."""
    # The MUSA plugin loads before the V2 runner, as in a server process.
    pytest.importorskip("vllm_musa.v1")
    warmup = pytest.importorskip("vllm.v1.worker.gpu.warmup")
    outputs = []
    warmup._warmup_kernels(runner, outputs.append, lambda grammar_output: None)
    per_group: list[list[int]] = [[] for _ in runner.kv_cache_config.kv_cache_groups]
    for output in outputs:
        new_ids = [req.block_ids for req in output.scheduled_new_reqs]
        cached = output.scheduled_cached_reqs
        new_ids += [ids for ids in cached.new_block_ids if ids is not None]
        for ids in new_ids:
            for group_id, group_ids in enumerate(ids):
                per_group[group_id].extend(group_ids)
    return per_group


@pytest.mark.parametrize("max_num_seqs", [1, 2])
def test_warmup_keeps_mamba_blocks_inside_their_dedicated_pools(
    max_num_seqs: int,
) -> None:
    torch = pytest.importorskip("torch")
    kv = pytest.importorskip("vllm.v1.kv_cache_interface")
    # The MUSA planner sizes each pool for max_num_seqs requests plus the null block.
    pool_blocks = max_num_seqs * (1 + NUM_SPEC) + 1
    per_group = _warmup_block_ids(_runner(torch, kv, max_num_seqs, pool_blocks))

    for ids in per_group[:NUM_MAMBA_GROUPS]:
        assert ids
        assert len(set(ids)) == len(ids)
        assert all(0 < block_id < pool_blocks for block_id in ids)
    assert per_group[NUM_MAMBA_GROUPS]
    assert 0 not in per_group[NUM_MAMBA_GROUPS]


def test_warmup_numbers_shared_pool_groups_from_one_counter() -> None:
    torch = pytest.importorskip("torch")
    kv = pytest.importorskip("vllm.v1.kv_cache_interface")
    per_group = _warmup_block_ids(_runner(torch, kv, 1, None))

    all_ids = [block_id for ids in per_group for block_id in ids]
    assert len(set(all_ids)) == len(all_ids)
    assert 0 not in all_ids
