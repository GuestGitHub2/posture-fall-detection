"""Rolling rates measured in monotonic time."""

import time
from collections import deque


class FPS:
    def __init__(self, window: int = 60) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, timestamp: float | None = None) -> None:
        self._times.append(time.monotonic() if timestamp is None else timestamp)

    @property
    def value(self) -> float:
        if len(self._times) < 2:
            return 0.0
        duration = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / duration if duration > 0 else 0.0
