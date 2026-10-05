"""Portable provider failover logic, without requiring Windows GPU hardware."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.runtime.onnx_runtime import ORTSession


def test_windows_inference_failures_walk_all_installed_local_providers(monkeypatch, tmp_path):
    order = [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    calls = []

    class Session:
        def __init__(self, path, sess_options, providers):
            self.provider = providers[0][0]
            calls.append(self.provider)
            if self.provider == "DmlExecutionProvider":
                assert not sess_options.enable_mem_pattern
                assert sess_options.execution_mode == 0

        def disable_fallback(self):
            pass

        def get_providers(self):
            return [self.provider]

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, names, feed):
            if self.provider != "CPUExecutionProvider":
                raise RuntimeError("Fixture device/operator failure")
            return [feed["image"]]

    ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=Session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr("src.runtime.onnx_runtime.provider_candidates", lambda _: order)
    monkeypatch.setattr("src.runtime.onnx_runtime.available_providers", lambda: order)
    model = tmp_path / "fixture.onnx"
    model.write_bytes(b"mock session, no real weights")
    session = ORTSession(model)
    data = np.array([1], np.float32)
    np.testing.assert_array_equal(session.run(data)[0], data)
    assert calls == order
    assert session.provider == order[-1]


def test_final_cpu_failure_reports_model_and_provider(monkeypatch, tmp_path):
    class Session:
        def __init__(self, *args, **kwargs):
            pass

        def disable_fallback(self):
            pass

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, *args):
            raise RuntimeError("Fixture corrupt model")

    ort = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        InferenceSession=Session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setattr(
        "src.runtime.onnx_runtime.provider_candidates", lambda _: ["CPUExecutionProvider"]
    )
    model = tmp_path / "bad.onnx"
    model.write_bytes(b"mock")
    session = ORTSession(model)
    with pytest.raises(RuntimeError, match="bad.onnx on CPUExecutionProvider"):
        session.run(np.zeros(1))


def test_nonlocal_provider_cannot_be_requested_even_if_installed():
    from src.runtime.providers import provider_candidates

    with pytest.raises(ValueError, match="only supported local providers"):
        provider_candidates(
            "AzureExecutionProvider", ["AzureExecutionProvider", "CPUExecutionProvider"], "Darwin"
        )
