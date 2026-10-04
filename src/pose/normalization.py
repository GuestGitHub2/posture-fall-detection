"""Resolution and position independent skeleton coordinates."""

from __future__ import annotations

import numpy as np

from src.pose.pose_types import Pose


def joint_center(pose: Pose, indices: tuple[int, ...], threshold: float = 0.3) -> np.ndarray | None:
    """Return the mean of visible joints, or None if none are reliable."""
    joints = pose.keypoints[list(indices)]
    valid = joints[:, 2] >= threshold
    return joints[valid, :2].mean(axis=0) if valid.any() else None


def normalize_pose(pose: Pose, confidence_threshold: float = 0.3) -> Pose:
    """Center at hips and scale by torso plus leg lengths, preserving confidence.

    Limb lengths are more stable during rotation than the axis-aligned box height.
    Missing skeletons use their box extent only as a normalization fallback.
    """
    center = joint_center(pose, (11, 12), confidence_threshold)
    shoulders = joint_center(pose, (5, 6), confidence_threshold)
    torso = (
        float(np.linalg.norm(center - shoulders))
        if center is not None and shoulders is not None
        else 0.0
    )
    leg_lengths = []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        points = pose.keypoints[[hip, knee, ankle]]
        if np.all(points[:, 2] >= confidence_threshold):
            leg_lengths.append(
                float(
                    np.linalg.norm(points[0, :2] - points[1, :2])
                    + np.linalg.norm(points[1, :2] - points[2, :2])
                )
            )
    body_scale = torso + (float(np.median(leg_lengths)) if leg_lengths else torso * 1.8)
    if body_scale <= np.finfo(np.float32).eps:
        body_scale = float(max(pose.bbox[2] - pose.bbox[0], pose.bbox[3] - pose.bbox[1]))
    if center is None:
        visible = pose.keypoints[:, 2] >= confidence_threshold
        center = (
            pose.keypoints[visible, :2].mean(axis=0)
            if visible.any()
            else (pose.bbox[:2] + pose.bbox[2:]) / 2
        )
    normalized = pose.keypoints.copy()
    normalized[:, :2] = (normalized[:, :2] - center) / body_scale
    normalized[normalized[:, 2] < confidence_threshold, :2] = 0.0
    return Pose(pose.keypoints, pose.bbox, pose.score, normalized, pose.timestamp, body_scale)
