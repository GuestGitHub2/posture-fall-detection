"""Shared training/inference contract retaining root translation within a window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pose.normalization import joint_center, measure_body_scale
from src.pose.pose_types import Pose

FALL_NORMALIZATION = "window_root_v1"


@dataclass(frozen=True)
class WindowTransform:
    origin: np.ndarray
    scale: float
    reliable: bool


def normalize_fall_sequence(
    poses: list[Pose], confidence_threshold: float = 0.3, minimum_scale_quality: float = 0.6
) -> tuple[np.ndarray, WindowTransform]:
    """First visible hip origin and a fixed median reliable body scale for all frames.

    Never use independently centered Pose.normalized coordinates for a fall model.
    Without a reliable root/scale, return zero-confidence data rather than fabricate motion.
    """
    output = np.zeros((len(poses), 17, 3), dtype=np.float32)
    roots = [joint_center(pose, (11, 12), confidence_threshold) for pose in poses]
    measurements = [measure_body_scale(pose, confidence_threshold) for pose in poses]
    scales = [m.value for m in measurements if m.quality >= minimum_scale_quality]
    origin = next((root for root in roots if root is not None), None)
    reliable = origin is not None and bool(scales)
    transform = WindowTransform(
        np.zeros(2, np.float32) if origin is None else origin.copy(),
        float(np.median(scales)) if scales else 1.0,
        reliable,
    )
    if not reliable:
        return output, transform
    for index, pose in enumerate(poses):
        output[index] = pose.keypoints
        output[index, :, :2] = (pose.keypoints[:, :2] - transform.origin) / transform.scale
        output[index, output[index, :, 2] < confidence_threshold, :2] = 0
    return output, transform


def interpolate_skeletons(
    points: np.ndarray,
    timestamps: np.ndarray,
    targets: np.ndarray,
    max_gap_seconds: float = 0.5,
    confidence_threshold: float = 0.3,
) -> np.ndarray:
    """Short joint gaps only; no extrapolation, zero-confidence padding at boundaries."""
    output = np.zeros((len(targets), 17, 3), np.float32)
    if not len(points):
        return output
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("Skeleton timestamps must be finite and strictly increasing")
    for joint in range(17):
        visible = (points[:, joint, 2] >= confidence_threshold) & np.isfinite(points[:, joint]).all(
            axis=1
        )
        times, values = timestamps[visible], points[visible, joint]
        for index, moment in enumerate(targets):
            right = int(np.searchsorted(times, moment))
            # Floating-point jitter must not drop endpoints on a uniform grid.
            exact = right if right < len(times) and abs(times[right] - moment) < 1e-6 else right - 1
            if 0 <= exact < len(times) and abs(times[exact] - moment) < 1e-6:
                output[index, joint] = values[exact]
            elif 0 < right < len(times):
                left = right - 1
                gap = times[right] - times[left]
                if gap <= max_gap_seconds:
                    weight = (moment - times[left]) / gap
                    output[index, joint] = values[left] * (1 - weight) + values[right] * weight
    return output
