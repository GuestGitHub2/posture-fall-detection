import time
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import CameraCapture, VideoCapture


def test_media_timestamps_preserve_fps(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    capture = VideoCapture(str(path), {"fps": 30})
    frames = [capture.read() for _ in range(5)]
    assert [packet.timestamp for packet in frames] == [0, 0.04, 0.08, 0.12, 0.16]
    assert capture.read() is None
    capture.close()


def test_latest_frame_slot_drops_backlog(monkeypatch) -> None:
    class FakeCapture:
        index = 0

        def read(self):
            time.sleep(0.002)
            self.index += 1
            return True, np.full((4, 4, 3), self.index % 255, np.uint8)

        def release(self):
            pass

    monkeypatch.setattr("src.capture.camera.open_capture", lambda *args: FakeCapture())
    capture = CameraCapture(0, {}).start()
    first = capture.latest(timeout=0.2)
    time.sleep(0.05)
    latest = capture.latest(first.index)
    assert latest.index > first.index + 5
    capture.close()
