"""Bounding-box and keypoint matching primitives."""

from __future__ import annotations

import numpy as np

from src.pose.pose_types import Pose


def iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.maximum(
        0.0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])
    )
    overlap = float(np.prod(intersection))
    union = float(np.prod(first[2:] - first[:2]) + np.prod(second[2:] - second[:2]) - overlap)
    return overlap / union if union > 0 else 0.0


def skeleton_distance(
    first: Pose, second: Pose, translation: np.ndarray, threshold: float
) -> float:
    visible = (first.keypoints[:, 2] >= threshold) & (second.keypoints[:, 2] >= threshold)
    scale = max(first.body_scale, second.body_scale, 1e-6)
    if not visible.any():
        centers = (
            (first.bbox[:2] + first.bbox[2:]) / 2 + translation,
            (second.bbox[:2] + second.bbox[2:]) / 2,
        )
        return float(np.linalg.norm(centers[0] - centers[1]) / scale)
    differences = first.keypoints[visible, :2] + translation - second.keypoints[visible, :2]
    return float(np.median(np.linalg.norm(differences, axis=1)) / scale)


def center_distance(first: Pose, second: Pose, translation: np.ndarray) -> float:
    """Predicted box-center distance in body lengths."""
    delta = (
        (first.bbox[:2] + first.bbox[2:]) / 2
        + translation
        - (second.bbox[:2] + second.bbox[2:]) / 2
    )
    return float(np.linalg.norm(delta) / max(first.body_scale, second.body_scale, 1e-6))


def scale_distance(
    first: Pose, second: Pose, minimum_quality: float, *, reliable_reference: bool = False
) -> float:
    """Log-scale discrepancy only when both body measurements are reliable."""
    if second.scale_quality < minimum_quality or (
        not reliable_reference and first.scale_quality < minimum_quality
    ):
        return 0.0
    return float(abs(np.log(first.body_scale / second.body_scale)))
