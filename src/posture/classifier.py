"""Replaceable posture classifier interface and geometric implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.pose.pose_types import Pose
from src.pose.visibility import BodyVisibility, assess_visibility
from src.posture.geometry import extract_features
from src.posture.rules import POSTURE_DEFAULTS, classify_features


@dataclass(frozen=True)
class PostureResult:
    label: str
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    pose_quality: float | None = None
    visibility: BodyVisibility | None = None

    def __post_init__(self) -> None:
        if self.pose_quality is None:
            object.__setattr__(
                self, "pose_quality", self.features.get("pose_quality", self.confidence)
            )


class Classifier(Protocol):
    def classify(self, pose: Pose) -> PostureResult: ...


class PostureClassifier:
    def __init__(self, config: dict | None = None) -> None:
        self.config = POSTURE_DEFAULTS | (config or {})
        self._labels: dict[int, str] = {}

    def remove(self, track_id: int) -> None:
        self._labels.pop(track_id, None)

    def classify_tracked(self, pose: Pose, track_id: int) -> PostureResult:
        result = self._classify(pose, self._labels.get(track_id))
        self._labels[track_id] = result.label
        return result

    def classify(self, pose: Pose) -> PostureResult:
        return self._classify(pose)

    def _classify(self, pose: Pose, preferred: str | None = None) -> PostureResult:
        features = extract_features(pose, self.config["confidence_threshold"])
        label, confidence = classify_features(features, self.config, preferred)
        return PostureResult(
            label,
            confidence,
            features,
            visibility=assess_visibility(pose, self.config["confidence_threshold"]),
        )
