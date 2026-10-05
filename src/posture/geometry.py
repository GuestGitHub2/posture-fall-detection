"""Confidence-aware COCO skeleton geometry."""

from __future__ import annotations

import math

import numpy as np

from src.pose.normalization import joint_center, normalize_pose
from src.pose.pose_types import Pose


def joint_angle(
    points: np.ndarray, indices: tuple[int, int, int], threshold: float
) -> float | None:
    joints = points[list(indices)]
    if np.any(joints[:, 2] < threshold):
        return None
    first, second = joints[0, :2] - joints[1, :2], joints[2, :2] - joints[1, :2]
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= np.finfo(np.float32).eps:
        return None
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1, 1))))


def extract_features(pose: Pose, threshold: float = 0.3) -> dict[str, float]:
    pose = normalize_pose(pose, threshold) if pose.normalized is None else pose
    shoulders = joint_center(pose, (5, 6), threshold)
    hips = joint_center(pose, (11, 12), threshold)
    knees = joint_center(pose, (13, 14), threshold)
    ankles = joint_center(pose, (15, 16), threshold)
    features = {
        "bbox_aspect_ratio": float((pose.bbox[2] - pose.bbox[0]) / (pose.bbox[3] - pose.bbox[1])),
        "body_scale": float(pose.body_scale),
        "visible_joints": float(np.sum(pose.keypoints[:, 2] >= threshold)),
        "joint_confidence": float(pose.keypoints[[5, 6, 11, 12, 13, 14, 15, 16], 2].mean()),
        "pose_quality": float(pose.keypoints[[5, 6, 11, 12, 13, 14, 15, 16], 2].mean()),
        "scale_quality": pose.scale_quality,
    }
    if hips is not None:
        features.update(hip_x=float(hips[0]), hip_y=float(hips[1]))
    if shoulders is not None:
        features.update(shoulder_x=float(shoulders[0]), shoulder_y=float(shoulders[1]))
    if hips is not None and shoulders is not None:
        vector = hips - shoulders
        if np.linalg.norm(vector) > np.finfo(np.float32).eps:
            features["torso_angle"] = math.degrees(
                math.atan2(abs(float(vector[0])), abs(float(vector[1])))
            )
            features["shoulder_hip_height"] = float(vector[1] / pose.body_scale)
    knee_angles = [
        joint_angle(pose.keypoints, indices, threshold) for indices in ((11, 13, 15), (12, 14, 16))
    ]
    valid_angles = [angle for angle in knee_angles if angle is not None]
    if valid_angles:
        features["knee_angle"] = float(np.mean(valid_angles))
    hip_angles = [
        joint_angle(pose.keypoints, indices, threshold) for indices in ((5, 11, 13), (6, 12, 14))
    ]
    if any(angle is not None for angle in hip_angles):
        features["hip_angle"] = float(np.mean([angle for angle in hip_angles if angle is not None]))
    if hips is not None and knees is not None:
        features["hip_knee_height"] = float((knees[1] - hips[1]) / pose.body_scale)
    if hips is not None and ankles is not None:
        features["hip_ankle_height"] = float((ankles[1] - hips[1]) / pose.body_scale)
        features["ankle_y"] = float(ankles[1])
    return features
