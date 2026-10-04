"""Replaceable posture classifier interface and geometric implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.pose.pose_types import Pose
from src.posture.geometry import extract_features
from src.posture.rules import POSTURE_DEFAULTS, classify_features


@dataclass(frozen=True)
class PostureResult:
    label: str
    confidence: float
    features: dict[str, float] = field(default_factory=dict)


class Classifier(Protocol):
    def classify(self, pose: Pose) -> PostureResult: ...


class PostureClassifier:
    def __init__(self, config: dict | None = None) -> None:
        self.config = POSTURE_DEFAULTS | (config or {})

    def classify(self, pose: Pose) -> PostureResult:
        features = extract_features(pose, self.config["confidence_threshold"])
        label, confidence = classify_features(features, self.config)
        return PostureResult(label, confidence, features)
