"""Confidence-aware body measurements and stable per-person normalization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.pose.pose_types import Pose

SCALE_DEFAULTS = {
    "scale_ema_seconds": 0.6,
    "scale_window_seconds": 1.0,
    "minimum_scale_quality": 0.6,
    "scale_max_change_ratio": 1.25,
}


def joint_center(pose: Pose, indices: tuple[int, ...], threshold: float = 0.3) -> np.ndarray | None:
    """Return the mean of visible joints, or None if none are reliable."""
    joints = pose.keypoints[list(indices)]
    valid = joints[:, 2] >= threshold
    return joints[valid, :2].mean(axis=0) if valid.any() else None


@dataclass(frozen=True)
class ScaleMeasurement:
    value: float
    source: str
    quality: float


def measure_body_scale(pose: Pose, threshold: float = 0.3) -> ScaleMeasurement:
    hips = joint_center(pose, (11, 12), threshold)
    shoulders = joint_center(pose, (5, 6), threshold)
    torso = (
        float(np.linalg.norm(hips - shoulders))
        if hips is not None and shoulders is not None
        else 0.0
    )
    legs = []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        joints = pose.keypoints[[hip, knee, ankle]]
        if np.all(joints[:, 2] >= threshold):
            legs.append(
                float(
                    np.linalg.norm(joints[0, :2] - joints[1, :2])
                    + np.linalg.norm(joints[1, :2] - joints[2, :2])
                )
            )
    if torso > 1e-6 and legs and np.median(legs) > 1e-6:
        quality = float(pose.keypoints[[5, 6, 11, 12, 13, 14, 15, 16], 2].mean())
        return ScaleMeasurement(torso + float(np.median(legs)), "torso_legs", quality)
    if torso > 1e-6:
        return ScaleMeasurement(torso * 2.8, "torso", 0.35)
    return ScaleMeasurement(float(max(pose.bbox[2:] - pose.bbox[:2])), "bbox", 0.1)


def normalize_pose(
    pose: Pose, confidence_threshold: float = 0.3, *, body_scale: float | None = None
) -> Pose:
    """Hip-centered coordinates for posture; optional fixed scale for temporal use."""
    measurement = measure_body_scale(pose, confidence_threshold)
    scale = measurement.value if body_scale is None else float(body_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Body scale must be finite and positive")
    center = joint_center(pose, (11, 12), confidence_threshold)
    if center is None:
        visible = pose.keypoints[:, 2] >= confidence_threshold
        center = (
            pose.keypoints[visible, :2].mean(axis=0)
            if visible.any()
            else (pose.bbox[:2] + pose.bbox[2:]) / 2
        )
    normalized = pose.keypoints.copy()
    normalized[:, :2] = (normalized[:, :2] - center) / scale
    normalized[normalized[:, 2] < confidence_threshold, :2] = 0.0
    return Pose(
        pose.keypoints,
        pose.bbox,
        pose.score,
        normalized,
        pose.timestamp,
        scale,
        measurement.source,
        measurement.quality,
        body_scale is not None,
    )


class StableBodyScale:
    """Time-based EMA of reliable rolling medians; weak fallback scales never replace it."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = SCALE_DEFAULTS | (config or {})
        self.value: float | None = None
        self._measurements: deque[tuple[float, float]] = deque()
        self._last_reliable: float | None = None

    def apply(self, pose: Pose, timestamp: float, threshold: float = 0.3) -> Pose:
        measurement = measure_body_scale(pose, threshold)
        while (
            self._measurements
            and timestamp - self._measurements[0][0] > self.config["scale_window_seconds"]
        ):
            self._measurements.popleft()
        if measurement.quality >= self.config["minimum_scale_quality"]:
            self._measurements.append((timestamp, measurement.value))
            target = float(np.median([value for _, value in self._measurements]))
            if self.value is None:
                self.value = target
            else:
                ratio = self.config["scale_max_change_ratio"]
                target = float(np.clip(target, self.value / ratio, self.value * ratio))
                elapsed = max(
                    0.0,
                    timestamp
                    - (self._last_reliable if self._last_reliable is not None else timestamp),
                )
                alpha = 1 - np.exp(-elapsed / self.config["scale_ema_seconds"])
                self.value += float(alpha * (target - self.value))
            self._last_reliable = timestamp
        return normalize_pose(
            pose, threshold, body_scale=self.value if self.value is not None else measurement.value
        )
