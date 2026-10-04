"""OpenCV capture with a single latest-frame slot for live cameras."""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.utils.fps import FPS

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    index: int
    timestamp: float
    captured_at: float


def open_capture(source: int | str, config: dict[str, Any]) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open {'camera' if isinstance(source, int) else 'video'} {source}. "
            "Check the device/path and camera permissions."
        )
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
        cap.set(cv2.CAP_PROP_FPS, config["fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class CameraCapture:
    """Capture never waits for inference; a new frame replaces the previous slot."""

    def __init__(self, index: int, config: dict[str, Any]) -> None:
        self.cap = open_capture(index, config)
        self.rate = FPS()
        self.error: str | None = None
        self._condition = threading.Condition()
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self) -> "CameraCapture":
        self._thread.start()
        return self

    def _run(self) -> None:
        index = 0
        try:
            while not self._stop.is_set():
                ok, image = self.cap.read()
                if not ok:
                    self.error = (
                        "Camera stopped returning frames; check connection and permissions."
                    )
                    break
                now = time.monotonic()
                self.rate.tick(now)
                with self._condition:
                    self._frame = Frame(image, index, now, now)
                    self._condition.notify_all()
                index += 1
        except Exception as exc:
            self.error = f"Camera capture failed: {exc}"
        finally:
            self._stop.set()
            with self._condition:
                self._condition.notify_all()

    def latest(self, after: int = -1, timeout: float = 0.1) -> Frame | None:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._stop.is_set() or (self._frame is not None and self._frame.index > after)
                ),
                timeout,
            )
            return self._frame if self._frame is not None and self._frame.index > after else None

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoCapture:
    """Sequential video frames use media time, independent of processing speed."""

    def __init__(self, path: str, config: dict[str, Any]) -> None:
        self.cap = open_capture(path, config)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(self.fps) or self.fps <= 0:
            self.fps = float(config["fps"])
            LOGGER.warning("Video FPS unavailable; using %.1f", self.fps)
        self.index = 0

    def read(self) -> Frame | None:
        ok, image = self.cap.read()
        if not ok:
            return None
        result = Frame(image, self.index, self.index / self.fps, time.monotonic())
        self.index += 1
        return result

    def close(self) -> None:
        self.cap.release()
