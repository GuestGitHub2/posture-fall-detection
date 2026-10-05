from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.app import InferenceWorker
from src.capture.camera import Frame, MediaClock, VideoCapture
from src.pipeline import PipelineResult
from src.visualization.renderer import overlay_is_current
from tests.test_pose_runtime import decoder


@pytest.mark.parametrize(
    "positions,expected",
    [
        ([0, 40, 95, 120, 210], [0, 0.04, 0.095, 0.12, 0.21]),
        ([0, 0, 0, 0], [0, 0.04, 0.08, 0.12]),
        ([0, float("nan"), -10, float("inf")], [0, 0.04, 0.08, 0.12]),
        ([0, 40, 39, 125, 200], [0, 0.04, 0.08, 0.125, 0.2]),
        ([0, 100, 100, 105, 210], [0, 0.1, 0.14, 0.18, 0.21]),
    ],
)
def test_media_clock_valid_vfr_jitter_and_invalid_pts(positions, expected):
    clock = MediaClock(25)
    values = [clock.timestamp(index, pos) for index, pos in enumerate(positions)]
    np.testing.assert_allclose(values, expected)
    assert np.all(np.diff(values) > 0)


def test_video_reads_opencv_pts_not_frame_number(monkeypatch):
    class Capture:
        index = -1

        def read(self):
            self.index += 1
            return self.index < 4, np.zeros((5, 5, 3), np.uint8)

        def get(self, prop):
            return 30 if prop == cv2.CAP_PROP_FPS else [0, 38, 99, 202][self.index]

        def release(self):
            pass

    monkeypatch.setattr("src.capture.camera.open_capture", lambda *_: Capture())
    capture = VideoCapture("fixture", {"fps": 30})
    frames = [capture.read() for _ in range(4)]
    np.testing.assert_allclose([frame.timestamp for frame in frames], [0, 0.038, 0.099, 0.202])
    assert [frame.index for frame in frames] == [0, 1, 2, 3]
    capture.close()


def test_rtmo_unclipped_nms_retains_boundary_people():
    boxes = np.array([[[-1000, 0, 100, 100, 0.95], [-20, 0, 100, 100, 0.9]]], np.float32)
    joints = np.ones((1, 2, 17, 3), np.float32)
    joints[..., :2] = [50, 50]
    people = decoder().decode([boxes, joints], 1, (100, 100), 0)
    assert len(people) == 2
    assert all(np.array_equal(person.bbox, [0, 0, 100, 100]) for person in people)


def test_rtmo_invalid_joint_does_not_discard_person():
    joints = np.full((1, 1, 17, 3), 0.95, np.float32)
    joints[..., :2] = [50, 50]
    joints[0, 0, 0, 0] = float("nan")
    joints[0, 0, 1, 2] = float("inf")
    people = decoder().decode([np.array([[[10, 10, 90, 90, 0.95]]]), joints], 1, (100, 100), 0)
    assert len(people) == 1
    assert (people[0].keypoints[:2, 2] == 0).all()
    assert np.isfinite(people[0].keypoints).all()
    assert (people[0].keypoints[2:, 2] > 0.9).all()


def test_result_age_and_frame_index_gate_stale_and_future_overlays():
    result = PipelineResult([], 1.0, {"pose_ms": 0}, "CPUExecutionProvider", frame_index=30)
    config = {"max_overlay_age_seconds": 0.25, "max_overlay_frame_lag": 8}
    assert overlay_is_current(result, 1.1, 33, config)
    assert not overlay_is_current(result, 1.1, 39, config)
    assert not overlay_is_current(result, 1.3, 35, config)
    assert not overlay_is_current(result, 0.9, 29, config)
    assert not overlay_is_current(result, 1.1, 29, config)
    assert overlay_is_current(replace(result, frame_index=-1), 1.1, 33, config)


def test_inference_worker_fetches_latest_after_rate_wait_and_stamps_index():
    calls = []

    class Stop:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, delay):
            calls.append(("wait", delay))
            return self.stopped

    class Capture:
        error = None
        index = 100

        def latest(self, after):
            calls.append(("latest", after))
            self.index += 11  # The old frames have already been overwritten.
            return Frame(np.zeros((4, 4, 3), np.uint8), self.index, float(self.index), 0)

    class Pipeline:
        def process(self, frame, timestamp):
            calls.append(("process", timestamp))
            if timestamp >= 122:
                stop.stopped = True
            return PipelineResult([], timestamp, {}, "CPUExecutionProvider")

    stop = Stop()
    worker = InferenceWorker(Capture(), Pipeline(), 15)
    worker.stop = stop
    worker._run()
    assert worker.result.frame_index == 122
    assert worker.result.timestamp == 122
    assert [kind for kind, _ in calls] == ["wait", "latest", "process", "wait", "latest", "process"]
    assert [value for kind, value in calls if kind == "process"] == [111, 122]
