"""Portable temporal detector contract, independent of final person state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.pose.pose_types import Pose
from src.posture.classifier import PostureResult


@dataclass(frozen=True)
class FallResult:
    status: str = "normal"
    probability: float = 0.0
    features: dict[str, float] = field(default_factory=dict)


class FallDetector(Protocol):
    def update(
        self, track_id: int, pose: Pose, timestamp: float, posture: PostureResult | None = None
    ) -> FallResult: ...

    def remove(self, track_id: int) -> None: ...
