"""
TimingChannelProtector

Context manager to help mitigate simple timing side-channels by enforcing minimum execution time.
Use sparingly: constant-time adds latency; apply only on code paths that must be hardened.
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from .. import constants, logger

DEFAULT_MAX_VARIATION = 0.001  # 1ms default (kept as small fallback)


@contextmanager
def constant_time_execution(min_time: float | None = None):
    """
    Usage:
        with constant_time_execution(0.002):
            do_critical_work()
    """
    if min_time is None:
        min_time = float(getattr(constants, "DEFAULT_MIN_TIME_PROTECT", DEFAULT_MAX_VARIATION))
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed < min_time:
            to_sleep = min_time - elapsed
            # in prod you'd want a jitter-free sleep primitive or OS-level pacing
            time.sleep(to_sleep)
            logger.secure_log("debug", "constant_time padding applied", padded=to_sleep)
