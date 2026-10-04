"""Local ONNX sessions with provider initialization and inference fallback."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.providers import available_providers, provider_candidates

LOG = logging.getLogger(__name__)


class ORTSession:
    """An inference session which retries failed acceleration on the next provider."""

    def __init__(self, model_path: str | Path, config: dict[str, Any] | None = None) -> None:
        import onnxruntime as ort

        self.ort = ort
        self.path = Path(model_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Model missing: {self.path}. Run python tools/download_models.py first. "
                "Inference never downloads models automatically."
            )
        self.config = config or {}
        self.candidates = provider_candidates(str(self.config.get("provider", "auto")))
        self._index = -1
        self._lock = threading.Lock()
        self.session: Any = None
        self._load_next()
        LOG.info(
            "Loaded %s; selected provider %s; session providers %s",
            self.path.name,
            self.provider,
            self.session.get_providers(),
        )

    def _options(self, provider: str) -> Any:
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = int(self.config.get("intra_op_threads", 4))
        options.inter_op_num_threads = 1
        options.log_severity_level = int(self.config.get("log_severity_level", 3))
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        if provider == "CoreMLExecutionProvider":
            # Batch-one video sessions keep the convolution backbone static.
            # Dynamic person-count/NMS subgraphs remain on CPU, including empty
            # outputs which CoreML cannot represent. This does not alter weights.
            overrides = self.config.get("free_dimension_overrides", {"batch": 1})
            for name, value in overrides.items():
                options.add_free_dimension_override_by_name(str(name), int(value))
        if provider == "DmlExecutionProvider":
            # Required by DirectML: no memory-pattern optimization, no parallel execution.
            options.enable_mem_pattern = False
        if bool(self.config.get("profile", False)):
            options.enable_profiling = True
        return options

    def _provider_options(self, provider: str) -> dict[str, str]:
        user = self.config.get("provider_options", {}).get(provider, {})
        options: dict[str, str] = {}
        if provider == "CoreMLExecutionProvider":
            options = {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1",
            }
        options.update({str(k): str(v) for k, v in user.items()})
        return options

    def _load_next(self) -> None:
        errors: list[str] = []
        for index in range(self._index + 1, len(self.candidates)):
            self._index = index
            preferred = self.candidates[index]
            providers: list[Any] = [(preferred, self._provider_options(preferred))]
            if (
                preferred != "CPUExecutionProvider"
                and "CPUExecutionProvider" in available_providers()
            ):
                providers.append("CPUExecutionProvider")
            try:
                session = self.ort.InferenceSession(
                    str(self.path), sess_options=self._options(preferred), providers=providers
                )
                session.disable_fallback()
                actual = session.get_providers()
                if preferred not in actual:
                    LOG.warning(
                        "ONNX Runtime could not activate %s. Active providers: %s",
                        preferred,
                        actual,
                    )
                    errors.append(f"{preferred}: silently unavailable after initialization")
                    continue
                self.session = session
                self.provider = actual[0]
                # Report preferred provider, not an assertion that all nodes run on it.
                return
            except Exception as exc:
                errors.append(f"{preferred}: {exc}")
                LOG.warning(
                    "Provider initialization failed (%s), trying fallback: %s", preferred, exc
                )
        raise RuntimeError(f"Could not load ONNX model {self.path.name}: {'; '.join(errors)}")

    @property
    def input_name(self) -> str:
        return str(self.session.get_inputs()[0].name)

    @property
    def input_shape(self) -> list[Any]:
        return list(self.session.get_inputs()[0].shape)

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.path.stat().st_size,
            "provider": self.provider,
            "session_providers": self.session.get_providers(),
            "inputs": [
                {"name": x.name, "shape": x.shape, "type": x.type}
                for x in self.session.get_inputs()
            ],
            "outputs": [
                {"name": x.name, "shape": x.shape, "type": x.type}
                for x in self.session.get_outputs()
            ],
        }

    def run(self, inputs: np.ndarray | dict[str, np.ndarray]) -> list[np.ndarray]:
        """Serialize calls (needed by DirectML) and retry provider errors locally."""
        with self._lock:
            feed = {self.input_name: inputs} if isinstance(inputs, np.ndarray) else inputs
            while True:
                try:
                    return self.session.run(None, feed)
                except Exception as exc:
                    if (
                        self._index + 1 >= len(self.candidates)
                        or self.provider == "CPUExecutionProvider"
                    ):
                        raise RuntimeError(
                            f"ONNX inference failed for {self.path.name} on {self.provider}: {exc}"
                        ) from exc
                    LOG.warning(
                        "Inference failed on %s, retrying next installed provider: %s",
                        self.provider,
                        exc,
                    )
                    self._load_next()
