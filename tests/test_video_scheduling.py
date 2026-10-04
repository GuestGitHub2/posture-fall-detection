from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from src.app import run_video
from src.capture.camera import Frame
from src.config import load_config


@pytest.mark.parametrize("source_fps,inference_fps", [(25, 15), (24, 10), (30, 15), (10, 30)])
def test_video_inference_keeps_requested_average_rate(monkeypatch, source_fps, inference_fps):
    config = load_config()
    config["pose"]["inference_fps"] = inference_fps
    timestamps = []

    class Capture:
        fps = source_fps
        index = 0

        def __init__(self, *args):
            pass

        def read(self):
            if self.index >= source_fps * 2:
                return None
            packet = Frame(np.zeros((4, 4, 3), np.uint8), self.index, self.index / self.fps, 0)
            self.index += 1
            return packet

        def close(self):
            pass

    class FakeDisplay:
        paused = False

        def __init__(self, *args):
            pass

        def show(self, *args):
            return ""

        def close(self):
            pass

    class FakePipeline:
        provider = "CPUExecutionProvider"

        def process(self, frame, timestamp):
            timestamps.append(timestamp)
            return SimpleNamespace(timestamp=timestamp)

    monkeypatch.setattr("src.app.VideoCapture", Capture)
    monkeypatch.setattr("src.app.Display", FakeDisplay)
    args = Namespace(video="test.mp4", headless=True, no_realtime=False, output=None, max_frames=0)
    run_video(args, config, FakePipeline())
    assert len(timestamps) == min(source_fps, inference_fps) * 2
