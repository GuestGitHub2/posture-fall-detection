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
    "confidence_boundary_fraction": 0.55,
    "confidence_angle_margin": 20.0,
    "confidence_knee_margin": 25.0,
    "confidence_height_margin": 0.15,
    "confidence_aspect_margin": 0.5,
}


def classify_features(features: dict[str, float], config: dict) -> tuple[str, float]:
    quality = features.get("pose_quality", features["joint_confidence"])
    confidence = min(config["confidence_ceiling"], quality)

    def score(*margins: float) -> float:
        strength = max(0.0, min(1.0, min(margins)))
        boundary = config["confidence_boundary_fraction"]
        return min(config["confidence_ceiling"], quality * (boundary + (1 - boundary) * strength))

    angle = config["confidence_angle_margin"]
    knee_margin = config["confidence_knee_margin"]
    height = config["confidence_height_margin"]
    if (
        features["visible_joints"] < config["minimum_visible_joints"]
        or confidence < config["minimum_confidence"]
    ):
        return "unknown", 0.0
    torso, knee = features.get("torso_angle"), features.get("knee_angle")
    if torso is None:
        return "unknown", 0.0
    lying = torso >= config["lying_torso_angle"] and (
        features["bbox_aspect_ratio"] >= config["lying_aspect_ratio"]
        or abs(features.get("hip_ankle_height", 1.0)) <= config["lying_hip_ankle_height"]
    )
    if lying:
        support = max(
            (features["bbox_aspect_ratio"] - config["lying_aspect_ratio"])
            / config["confidence_aspect_margin"],
            (config["lying_hip_ankle_height"] - abs(features.get("hip_ankle_height", 1.0)))
            / height,
        )
        return "lying", score((torso - config["lying_torso_angle"]) / angle, support)
    if knee is None:
        return "unknown", 0.0
    if torso >= config["bending_torso_angle"] and knee >= config["bending_knee_angle"]:
        return "bending", score(
            (torso - config["bending_torso_angle"]) / angle,
            (knee - config["bending_knee_angle"]) / knee_margin,
        )
    if config["enable_extra_postures"] and knee < config["squatting_knee_angle"]:
        if features.get("hip_knee_height", 1.0) <= config["squatting_hip_knee_height"]:
            return "squatting", score(
                (config["squatting_knee_angle"] - knee) / knee_margin,
                (config["squatting_hip_knee_height"] - features["hip_knee_height"]) / height,
            )
        if features.get("hip_ankle_height", 1.0) <= config["kneeling_hip_ankle_height"]:
            return "kneeling", score(
                (config["squatting_knee_angle"] - knee) / knee_margin,
                (config["kneeling_hip_ankle_height"] - features["hip_ankle_height"]) / height,
            )
    if torso <= config["standing_torso_angle"] and knee >= config["standing_knee_angle"]:
        return "standing", score(
            (config["standing_torso_angle"] - torso) / angle,
            (knee - config["standing_knee_angle"]) / knee_margin,
        )
    if (
        knee <= config["sitting_knee_angle"]
        or features.get("hip_knee_height", 1.0) <= config["sitting_hip_knee_height"]
    ):
        return "sitting", score(
            max(
                (config["sitting_knee_angle"] - knee) / knee_margin,
                (config["sitting_hip_knee_height"] - features.get("hip_knee_height", 1.0)) / height,
            )
        )
    return "unknown", 0.0
