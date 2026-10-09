from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = (
    ROOT
    / "vllm_musa/patches/series/0174-perf-musa-extend-DSV4-score-DeepGEMM-to-decode-batch.patch"
)


def test_dsv4_score_deepgemm_covers_every_dspark4_decode_batch() -> None:
    source = PATCH.read_text(encoding="utf-8")
    # 64 DSpark-4 requests are 320 tokens; the bound matches the multi-stream
    # token threshold, above which the prefill route takes over.
    assert "_MUSA_DEEPSEEK_V4_SCORE_FP32_DEEPGEMM_MAX_TOKENS = 1024" in source
    assert "multi-stream" in source
    assert "prefill route" in source
