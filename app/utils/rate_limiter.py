from __future__ import annotations

import threading
import time


class MinIntervalLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()
