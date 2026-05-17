# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import os
import sys
import threading
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass


_ENV_ENABLE = "VLLM_MUSA_DEEPSEEK_V4_FALLBACK_TIMER"
_ENV_SYNC = "VLLM_MUSA_DEEPSEEK_V4_FALLBACK_TIMER_SYNC"
_ENV_LOG_EVERY = "VLLM_MUSA_DEEPSEEK_V4_FALLBACK_TIMER_LOG_EVERY"


@dataclass
class _TimerRecord:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0


_LOCK = threading.Lock()
_RECORDS: defaultdict[str, _TimerRecord] = defaultdict(_TimerRecord)
_PRINTED_ATEXIT = False


def enabled() -> bool:
    return os.getenv(_ENV_ENABLE, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _sync_enabled() -> bool:
    return os.getenv(_ENV_SYNC, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _log_every() -> int:
    raw_value = os.getenv(_ENV_LOG_EVERY, "50").strip()
    try:
        value = int(raw_value)
    except ValueError:
        return 50
    return max(value, 1)


def _sync_device() -> None:
    if not _sync_enabled():
        return
    try:
        import torch
    except Exception:
        return

    musa = getattr(torch, "musa", None)
    if musa is not None and hasattr(musa, "synchronize"):
        try:
            musa.synchronize()
            return
        except Exception:
            pass

    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "synchronize"):
        try:
            cuda.synchronize()
        except Exception:
            pass


def _emit(prefix: str, name: str, record: _TimerRecord, last_ms: float) -> None:
    mean_ms = record.total_ms / max(record.count, 1)
    print(
        " ".join(
            (
                prefix,
                f"pid={os.getpid()}",
                f"name={name}",
                f"count={record.count}",
                f"total_ms={record.total_ms:.3f}",
                f"mean_ms={mean_ms:.3f}",
                f"last_ms={last_ms:.3f}",
                f"max_ms={record.max_ms:.3f}",
            )
        ),
        file=sys.stderr,
        flush=True,
    )


class _ScopeTimer:

    def __init__(self, name: str):
        self.name = name
        self.start_ns = 0

    def __enter__(self):
        _sync_device()
        self.start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _sync_device()
        elapsed_ms = (time.perf_counter_ns() - self.start_ns) / 1_000_000.0
        with _LOCK:
            record = _RECORDS[self.name]
            record.count += 1
            record.total_ms += elapsed_ms
            record.max_ms = max(record.max_ms, elapsed_ms)
            if record.count % _log_every() == 0:
                _emit("MUSA_DSV4_TIMER", self.name, record, elapsed_ms)


def timed(name: str):
    if not enabled():
        return nullcontext()
    return _ScopeTimer(name)


def snapshot() -> dict[str, dict[str, float | int]]:
    with _LOCK:
        return {
            name: {
                "count": record.count,
                "total_ms": record.total_ms,
                "mean_ms": record.total_ms / max(record.count, 1),
                "max_ms": record.max_ms,
            }
            for name, record in _RECORDS.items()
        }


def _flush_summary() -> None:
    global _PRINTED_ATEXIT
    if _PRINTED_ATEXIT or not enabled():
        return
    _PRINTED_ATEXIT = True
    with _LOCK:
        items = list(_RECORDS.items())
    for name, record in items:
        _emit("MUSA_DSV4_TIMER_SUMMARY", name, record, 0.0)


atexit.register(_flush_summary)
