"""Interpretable baseline; every classification threshold is configurable."""

from __future__ import annotations

POSTURE_DEFAULTS = {
    "confidence_threshold": 0.3,
    "minimum_visible_joints": 6,
    "minimum_confidence": 0.4,
    "lying_torso_angle": 65.0,
    "lying_aspect_ratio": 1.0,
    "lying_hip_ankle_height": 0.18,
    "standing_torso_angle": 28.0,
    "standing_knee_angle": 155.0,
    "bending_torso_angle": 35.0,
    "bending_knee_angle": 140.0,
    "sitting_knee_angle": 140.0,
    "sitting_hip_knee_height": 0.2,
    "enable_extra_postures": False,
    "squatting_knee_angle": 95.0,
    "squatting_hip_knee_height": 0.05,
    "kneeling_hip_ankle_height": 0.22,
    "confidence_ceiling": 0.97,
    "smoothing_frames": 5,
}


def classify_features(features: dict[str, float], config: dict) -> tuple[str, float]:
    confidence = min(config["confidence_ceiling"], features["joint_confidence"])
    if (
        features["visible_joints"] < config["minimum_visible_joints"]
        or confidence < config["minimum_confidence"]
    ):
        return "unknown", confidence
    torso, knee = features.get("torso_angle"), features.get("knee_angle")
    if torso is None:
        return "unknown", confidence
    lying = torso >= config["lying_torso_angle"] and (
        features["bbox_aspect_ratio"] >= config["lying_aspect_ratio"]
        or abs(features.get("hip_ankle_height", 1.0)) <= config["lying_hip_ankle_height"]
    )
    if lying:
        return "lying", confidence
    if knee is None:
        return "unknown", confidence
    if torso >= config["bending_torso_angle"] and knee >= config["bending_knee_angle"]:
        return "bending", confidence
    if config["enable_extra_postures"] and knee < config["squatting_knee_angle"]:
        if features.get("hip_knee_height", 1.0) <= config["squatting_hip_knee_height"]:
            return "squatting", confidence
        if features.get("hip_ankle_height", 1.0) <= config["kneeling_hip_ankle_height"]:
            return "kneeling", confidence
    if torso <= config["standing_torso_angle"] and knee >= config["standing_knee_angle"]:
        return "standing", confidence
    if (
        knee <= config["sitting_knee_angle"]
        or features.get("hip_knee_height", 1.0) <= config["sitting_hip_knee_height"]
    ):
        return "sitting", confidence
    return "unknown", confidence
