"""Optional trained ONNX MLP using the shared normalized COCO-17 contract."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.pose.normalization import normalize_pose
from src.pose.pose_types import Pose
from src.posture.classifier import PostureResult
from src.posture.geometry import extract_features
from src.runtime.onnx_runtime import ORTSession

MLP_DEFAULTS = {
    "model_path": "models/posture_mlp.onnx",
    "classes": ["standing", "sitting", "bending", "lying", "squatting", "kneeling", "unknown"],
    "confidence_threshold": 0.3,
    "minimum_visible_joints": 6,
    "minimum_confidence": 0.55,
}


class MLPPostureClassifier:
    """Consume float32 [N,17,3] skeletons and [N,num_classes] logits.

    Classes must exactly match the order used during training. No pretrained
    posture model is bundled; constructing without weights raises a clear error.
    """

    def __init__(
        self,
        config: dict | None = None,
        runtime_config: dict | None = None,
        session: Any | None = None,
    ) -> None:
        self.config = MLP_DEFAULTS | (config or {})
        self.classes = list(self.config["classes"])
        if len(self.classes) < 2 or len(set(self.classes)) != len(self.classes):
            raise ValueError(
                "MLP classes must contain at least two distinct labels in training order"
            )
        self.session = (
            session
            if session is not None
            else ORTSession(self.config["model_path"], runtime_config)
        )
        shape = self.session.input_shape
        if len(shape) != 3 or shape[1:] != [17, 3]:
            raise ValueError(f"Posture MLP must accept [N,17,3], got {shape}")

    def classify(self, pose: Pose) -> PostureResult:
        normalized = normalize_pose(pose, self.config["confidence_threshold"])
        features = extract_features(normalized, self.config["confidence_threshold"])
        if features["visible_joints"] < self.config["minimum_visible_joints"]:
            return PostureResult("unknown", 0.0, features)
        logits = np.asarray(self.session.run(normalized.normalized[None].astype(np.float32))[0])
        if logits.shape != (1, len(self.classes)) or not np.isfinite(logits).all():
            raise ValueError(
                f"Posture MLP must output finite [1,{len(self.classes)}] logits, got {logits.shape}"
            )
        probabilities = np.exp(logits[0] - logits[0].max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        label = (
            self.classes[index] if confidence >= self.config["minimum_confidence"] else "unknown"
        )
        return PostureResult(label, confidence, features)
