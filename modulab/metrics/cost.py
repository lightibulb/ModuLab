from __future__ import annotations
import time
import tracemalloc
from typing import Callable, Any


def wall_time_ms(fn: Callable[[], Any]):
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    result = fn()
    t1 = time.perf_counter_ns()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, (t1 - t0) / 1e6, peak / 1e6
