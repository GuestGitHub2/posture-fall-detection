"""Installed ONNX execution providers, ordered for low-latency local use."""

from __future__ import annotations

import logging
import platform

LOG = logging.getLogger(__name__)
ALIASES = {
    "cpu": "CPUExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}


def available_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is missing. Install requirements.txt first.") from exc
    return ort.get_available_providers()


def provider_candidates(
    requested: str = "auto", available: list[str] | None = None, system: str | None = None
) -> list[str]:
    """Return only installed providers, always retaining a CPU fallback."""
    installed = available if available is not None else available_providers()
    os_name = system or platform.system()
    if os_name == "Darwin":
        order = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    elif os_name == "Windows":
        order = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        order = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
    if requested.lower() != "auto":
        explicit = ALIASES.get(requested.lower(), requested)
        if explicit not in set(ALIASES.values()):
            raise ValueError(
                f"Unsupported execution provider {requested!r}. Choose auto, coreml, cuda, "
                "directml, openvino or cpu; only supported local providers are allowed."
            )
        if explicit not in installed:
            LOG.warning(
                "Requested provider %s is unavailable; using installed CPU fallback.", explicit
            )
        order = [explicit, "CPUExecutionProvider"]
    result = [name for name in order if name in installed]
    if not result:
        raise RuntimeError(f"No supported local ONNX provider installed. Available: {installed}")
    return result
