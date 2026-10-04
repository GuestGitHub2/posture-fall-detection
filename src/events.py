"""Local event callbacks, isolated from rendering and pose inference."""

import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class EventBus:
    def __init__(self, path: str | Path | None = None) -> None:
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._callbacks.append(callback)

    def emit(
        self,
        event_type: str,
        track_id: int,
        confidence: float,
        posture: str,
        source_timestamp: float,
    ) -> dict[str, Any]:
        event = {
            "type": event_type,
            "track_id": track_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_timestamp": source_timestamp,
            "confidence": float(confidence),
            "posture": posture,
        }
        LOGGER.warning("EVENT %s", json.dumps(event))
        if self.path:
            try:
                with self._lock, self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event) + "\n")
            except OSError as exc:
                LOGGER.error("Could not write events to %s: %s", self.path, exc)
        for callback in tuple(self._callbacks):
            try:
                callback(event.copy())
            except Exception:
                LOGGER.exception("Event callback failed")
        return event
