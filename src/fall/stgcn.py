"""ONNX adapter for compatible COCO-17 ST-GCN++ or lightweight temporal models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.fall.detector import FallResult
from src.fall.normalization import FALL_NORMALIZATION
from src.fall.temporal_buffer import TemporalBuffer
from src.pose.pose_types import Pose
from src.posture.classifier import PostureResult
from src.runtime.onnx_runtime import ORTSession


def skeleton_tensor(sequence: np.ndarray) -> np.ndarray:
    """Convert [T,17,3] x/y/confidence to contiguous float32 [1,3,T,17,1]."""
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (17, 3):
        raise ValueError("ST-GCN requires a normalized COCO-17 sequence [T,17,3]")
    if not np.isfinite(sequence).all():
        raise ValueError("ST-GCN skeleton contains non-finite values")
    return np.ascontiguousarray(sequence.transpose(2, 0, 1)[None, ..., None])


class STGCNFallDetector:
    """Causal timestamp-sampled inference with independent per-track buffers.

    Input normalization uses a fixed window root and robust body scale. Models
    trained with another joint order, normalization, layout or label order require
    explicit conversion; arbitrary action-recognition checkpoints are incompatible.
    """

    def __init__(self, config: dict[str, Any], session: Any | None = None) -> None:
        self.config = config.get("fall", config)
        self.samples = int(self.config.get("samples", self.config.get("sample_count", 48)))
        self.history_seconds = float(self.config.get("history_seconds", 2.0))
        self.max_gap_seconds = float(self.config.get("max_gap_seconds", 0.5))
        self.min_history_seconds = float(
            self.config.get("min_history_seconds", self.history_seconds * 0.8)
        )
        self.classes = list(self.config.get("classes", ["normal", "fall"]))
        self.normalization = self.config.get("normalization", FALL_NORMALIZATION)
        if self.normalization != FALL_NORMALIZATION:
            raise ValueError(
                f"Fall normalization must be {FALL_NORMALIZATION}; old per-frame centered weights require retraining"
            )
        if self.samples < 2 or not self.classes or "normal" not in self.classes:
            raise ValueError("ST-GCN requires >=2 samples and a class order containing normal")
        if not any(label in self.classes for label in ("fall", "falling", "fallen")):
            raise ValueError("ST-GCN class order must identify fall, falling or fallen")
        self.output_kind = str(self.config.get("output_kind", "logits"))
        if self.output_kind not in ("logits", "probabilities"):
            raise ValueError("fall.output_kind must be logits or probabilities")
        if session is None:
            model_path = self.config.get("model_path")
            if not model_path or not Path(model_path).expanduser().is_file():
                raise FileNotFoundError(
                    f"Trained temporal fall model missing: {model_path}. "
                    "Train/supply compatible ONNX weights, or set fall.detector: heuristic. "
                    "download_models.py installs pose weights only."
                )
            session = ORTSession(model_path, config.get("runtime", {}))
        model_path = self.config.get("model_path")
        if model_path:
            metadata_path = Path(model_path).with_suffix(".metadata.json")
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("Fall metadata must be a JSON object")
                if metadata.get("normalization") != FALL_NORMALIZATION:
                    raise ValueError(
                        f"Incompatible fall normalization in {metadata_path}; retrain with {FALL_NORMALIZATION}"
                    )
                if (
                    metadata.get("classes") != self.classes
                    or metadata.get("samples") != self.samples
                ):
                    raise ValueError(
                        "Fall metadata class order/sample count conflicts with configuration"
                    )
                if metadata.get("layout") != "NCTVM":
                    raise ValueError("Fall metadata layout must be NCTVM")
                expected = {
                    "history_seconds": self.history_seconds,
                    "minimum_joint_confidence": float(
                        self.config.get("minimum_joint_confidence", 0.3)
                    ),
                    "minimum_scale_quality": float(self.config.get("minimum_scale_quality", 0.6)),
                    "max_gap_seconds": self.max_gap_seconds,
                }
                for key, value in expected.items():
                    actual = metadata.get(key)
                    if key != "history_seconds" and key not in metadata:
                        continue
                    if (
                        isinstance(actual, bool)
                        or not isinstance(actual, (int, float))
                        or not np.isfinite(actual)
                        or not np.isclose(actual, value)
                    ):
                        raise ValueError(f"Fall metadata {key} conflicts with configuration")
        self.session = session
        shape = self.session.input_shape
        if len(shape) != 5:
            raise ValueError(f"Expected NCTVM input [1,3,T,17,1], got {shape}")
        for actual, expected in zip(shape, (1, 3, self.samples, 17, 1), strict=True):
            if isinstance(actual, int) and actual != expected:
                raise ValueError(
                    f"Model input {shape} conflicts with configured [1,3,{self.samples},17,1]"
                )
        self.buffers: dict[int, TemporalBuffer] = {}
        self.unreliable_since: dict[int, float] = {}
        self.last_updates: dict[int, float] = {}

    @property
    def provider(self) -> str:
        return str(self.session.provider)

    def remove(self, track_id: int) -> None:
        self.buffers.pop(track_id, None)
        self.unreliable_since.pop(track_id, None)
        self.last_updates.pop(track_id, None)

    def update(
        self, track_id: int, pose: Pose, timestamp: float, posture: PostureResult | None = None
    ) -> FallResult:
        if track_id not in self.buffers:
            self.buffers[track_id] = TemporalBuffer(
                self.history_seconds, max(self.samples * 3, 90), self.max_gap_seconds
            )
        buffer = self.buffers[track_id]
        previous = self.last_updates.get(track_id)
        if previous is not None and (
            timestamp <= previous or timestamp - previous > self.max_gap_seconds
        ):
            buffer.clear()
            self.unreliable_since.pop(track_id, None)
        self.last_updates[track_id] = timestamp
        if buffer.append(pose, timestamp):
            self.unreliable_since.pop(track_id, None)
        threshold = float(self.config.get("minimum_joint_confidence", 0.3))
        current_quality = float((pose.keypoints[[5, 6, 11, 12, 15, 16], 2] >= threshold).mean())
        reliable = current_quality >= float(
            self.config.get("min_current_pose_fraction", 0.5)
        ) and all(
            int((pose.keypoints[list(pair), 2] >= threshold).sum())
            >= self.config.get("minimum_confident_joints_per_pair", 1)
            for pair in ((5, 6), (11, 12), (15, 16))
        )
        if not reliable:
            self.unreliable_since.setdefault(track_id, timestamp)
            duration = timestamp - self.unreliable_since[track_id]
            if duration >= float(self.config.get("max_unreliable_seconds", 0.45)):
                buffer.clear()
            return FallResult(
                "normal", 0.0, {"measurement_valid": 0.0, "unreliable_duration": duration}
            )
        if track_id in self.unreliable_since:
            if timestamp - self.unreliable_since[track_id] >= float(
                self.config.get("max_unreliable_seconds", 0.45)
            ):
                buffer.clear()
                buffer.append(pose, timestamp)
            self.unreliable_since.pop(track_id)
        history = buffer.samples
        duration = history[-1].timestamp - history[0].timestamp
        if duration < self.min_history_seconds:
            return FallResult("normal", 0.0, {"history_seconds": duration, "warming_up": 1.0})
        tensor = skeleton_tensor(
            buffer.resample_fall(
                self.samples,
                timestamp,
                threshold,
                float(self.config.get("minimum_scale_quality", 0.6)),
            )
        )
        visible_fraction = float((tensor[0, 2, :, :, 0] >= threshold).mean())
        if visible_fraction < float(self.config.get("min_visible_fraction", 0.35)):
            return FallResult(
                "normal", 0.0, {"visible_fraction": visible_fraction, "insufficient_pose": 1.0}
            )
        raw = np.asarray(self.session.run(tensor)[0], dtype=np.float64).reshape(-1)
        if raw.size != len(self.classes) or not np.isfinite(raw).all():
            raise ValueError(
                f"Model output must contain {len(self.classes)} finite class scores, got {raw.shape}"
            )
        if self.output_kind == "logits":
            scores = np.exp(raw - raw.max())
            scores /= scores.sum()
        else:
            if np.any(raw < 0) or np.any(raw > 1) or not np.isclose(raw.sum(), 1.0, atol=0.01):
                raise ValueError("Probability output must lie in [0,1] and sum to one")
            scores = raw / raw.sum()
        probability = float(
            sum(
                scores[index]
                for index, name in enumerate(self.classes)
                if name in ("fall", "falling", "fallen")
            )
        )
        confirmed = float(self.config.get("confirmed_threshold", 0.8))
        possible = float(self.config.get("possible_threshold", 0.55))
        status = (
            "fall"
            if probability >= confirmed
            else "possible_fall"
            if probability >= possible
            else "normal"
        )
        features = {
            f"probability_{name}": float(value)
            for name, value in zip(self.classes, scores, strict=True)
        }
        features.update(history_seconds=duration, visible_fraction=visible_fraction)
        return FallResult(status, probability, features)
