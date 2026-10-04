"""Pose geometry contracts and runtime fallback are independent of model weights."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.pose.rtmo import RTMOEstimator, letterbox, nms
from src.pose.rtmpose import affine_crop, decode_simcc
from src.runtime.providers import provider_candidates


def decoder() -> RTMOEstimator:
    estimator = RTMOEstimator.__new__(RTMOEstimator)
    estimator.score_threshold = 0.3
    estimator.nms_threshold = 0.45
    estimator.max_people = 30
    return estimator


def test_letterbox_nonsquare_coordinates_round_trip() -> None:
    tensor, ratio = letterbox(np.zeros((720, 1280, 3), np.uint8), (640, 640))
    assert tensor.shape == (1, 3, 640, 640)
    assert ratio == 0.5
    assert np.all(tensor[0, :, 360:, :] == 114)
    joints = np.zeros((1, 1, 17, 3), np.float32)
    joints[..., 0] = 250
    joints[..., 1] = 125
    joints[..., 2] = 0.9
    boxes = np.array([[[200, 50, 300, 200, 0.95]]], dtype=np.float32)
    result = decoder().decode([boxes, joints], ratio, (720, 1280), 2.5)
    assert len(result) == 1
    assert result[0].timestamp == 2.5
    np.testing.assert_allclose(result[0].keypoints[:, :2], np.tile([500, 250], (17, 1)))
    np.testing.assert_allclose(result[0].bbox, [400, 100, 600, 400])


def test_empty_and_low_confidence_scene_returns_no_people() -> None:
    estimate = decoder()
    assert estimate.decode([np.zeros((1, 0, 5)), np.zeros((1, 0, 17, 3))], 1, (640, 640), 0) == []
    assert estimate.decode([np.zeros((1, 1, 5)), np.zeros((1, 1, 17, 3))], 1, (640, 640), 0) == []


def test_off_frame_joints_are_invalidated_without_geometry_clipping() -> None:
    joints = np.ones((1, 1, 17, 3), np.float32)
    joints[0, 0, 0] = [-50, 20, 0.9]
    result = decoder().decode([np.array([[[0, 0, 100, 100, 0.9]]]), joints], 1, (100, 100), 0)
    assert result[0].keypoints[0, 0] == -50
    assert result[0].keypoints[0, 2] == 0


def test_nms_preserves_distinct_people() -> None:
    boxes = np.array([[0, 0, 50, 100], [2, 0, 52, 100], [100, 0, 150, 100]])
    assert nms(boxes, np.array([0.9, 0.8, 0.7]), 0.45) == [0, 2]


def test_wrong_rtmo_export_gives_actionable_error() -> None:
    with pytest.raises(ValueError, match="official ONNX SDK"):
        decoder().decode([np.zeros((1, 17, 512))], 1, (640, 640), 0)


def test_rtmpose_crop_and_simcc_roundtrip() -> None:
    _, center, scale = affine_crop(
        np.zeros((480, 640, 3), np.uint8),
        np.array([100, 100, 200, 300], np.float32),
        (192, 256),
        1.25,
    )
    x = np.zeros((1, 17, 384), np.float32)
    y = np.zeros((1, 17, 512), np.float32)
    x[:, :, 192], y[:, :, 256] = 0.9, 0.8
    joints = decode_simcc([x, y], center, scale, (192, 256), 2.0)
    np.testing.assert_allclose(joints[:, :2], np.tile(center, (17, 1)))
    np.testing.assert_allclose(joints[:, 2], 0.8)


def test_platform_provider_priority_and_unavailable_explicit() -> None:
    available = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
        "CoreMLExecutionProvider",
    ]
    assert provider_candidates("auto", available, "Darwin") == [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    assert provider_candidates("auto", available, "Windows") == available[:4]
    assert provider_candidates("cuda", ["CPUExecutionProvider"], "Windows") == [
        "CPUExecutionProvider"
    ]


@pytest.mark.parametrize("failure", ["initialization", "execution"])
def test_provider_failures_retry_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    import src.runtime.onnx_runtime as runtime

    calls = []

    class Session:
        def __init__(self, _path, sess_options, providers):
            provider = providers[0][0]
            calls.append(provider)
            self.provider = provider
            if provider == "CoreMLExecutionProvider" and failure == "initialization":
                raise RuntimeError("Unsupported operator")

        def disable_fallback(self):
            pass

        def get_providers(self):
            return [self.provider]

        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=[1, 3, 640, 640], type="tensor(float)")]

        def run(self, _names, _feed):
            if self.provider == "CoreMLExecutionProvider":
                raise RuntimeError("Hardware failure")
            return [np.array([42])]

    fake = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(add_free_dimension_override_by_name=lambda *_: None),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=Session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    monkeypatch.setattr(
        runtime,
        "provider_candidates",
        lambda _: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        runtime, "available_providers", lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    )
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"test fixture")
    session = runtime.ORTSession(model)
    assert session.run(np.zeros((1,)))[0][0] == 42
    assert session.provider == "CPUExecutionProvider"
    assert calls == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_missing_model_explains_offline_install(tmp_path: Path) -> None:
    from src.runtime.onnx_runtime import ORTSession

    with pytest.raises(FileNotFoundError, match="download_models.py"):
        ORTSession(tmp_path / "missing.onnx")


def test_silent_accelerator_fallback_tries_other_windows_providers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import src.runtime.onnx_runtime as runtime

    calls = []

    class Session:
        def __init__(self, _path, sess_options, providers):
            self.preferred = providers[0][0]
            calls.append(self.preferred)
            if self.preferred == "DmlExecutionProvider":
                assert sess_options.enable_mem_pattern is False
                assert sess_options.execution_mode == 0

        def disable_fallback(self):
            pass

        def get_providers(self):
            return (
                ["CPUExecutionProvider"]
                if self.preferred == "CUDAExecutionProvider"
                else [self.preferred]
            )

    fake = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(add_free_dimension_override_by_name=lambda *_: None),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=Session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    order = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setattr(runtime, "provider_candidates", lambda _: order)
    monkeypatch.setattr(runtime, "available_providers", lambda: order)
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"test fixture")
    session = runtime.ORTSession(model)
    assert session.provider == "DmlExecutionProvider"
    assert calls == order[:2]


def test_coreml_defaults_keep_batch_static_and_dynamic_person_counts_on_cpu() -> None:
    from src.runtime.onnx_runtime import ORTSession

    session = ORTSession.__new__(ORTSession)
    session.config = {}
    assert session._provider_options("CoreMLExecutionProvider")["RequireStaticInputShapes"] == "1"
    session.config = {
        "provider_options": {"CoreMLExecutionProvider": {"RequireStaticInputShapes": "0"}}
    }
    assert session._provider_options("CoreMLExecutionProvider")["RequireStaticInputShapes"] == "0"


def test_installed_rtmo_coreml_empty_scene_regression() -> None:
    """When the official local model exists, empty detections stay on CoreML."""
    import onnxruntime as ort

    from src.pose.base import create_pose_estimator

    model = Path(__file__).resolve().parents[1] / "models" / "rtmo_m.onnx"
    if not model.is_file() or "CoreMLExecutionProvider" not in ort.get_available_providers():
        pytest.skip("Optional real model/CoreML integration is unavailable")
    estimator = create_pose_estimator(
        {"pose": {"model": "rtmo_m", "model_path": str(model)}, "runtime": {"provider": "coreml"}}
    )
    for frame in [
        np.zeros((360, 640, 3), np.uint8),
        np.full((720, 1280, 3), 114, np.uint8),
        np.zeros((640, 640, 3), np.uint8),
    ]:
        assert estimator.infer(frame, 1.0) == []
        assert estimator.provider == "CoreMLExecutionProvider"
