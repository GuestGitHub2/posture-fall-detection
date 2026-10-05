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
    "label_hysteresis_margin": 0.08,
}


def posture_scores(features: dict[str, float], config: dict) -> dict[str, float]:
    """Overlapping rule memberships; absent evidence cannot vote for a precise class."""
    quality = features.get(
        "torso_quality", features.get("pose_quality", features["joint_confidence"])
    )
    if quality < config["minimum_confidence"] or "torso_angle" not in features:
        return {}
    boundary = config["confidence_boundary_fraction"]

    def membership(margin: float) -> float:
        if margin < 0:
            return max(0.0, boundary * (1 + margin))
        return min(1.0, boundary + (1 - boundary) * margin)

    def score(*margins: float) -> float:
        return min(config["confidence_ceiling"], quality * min(map(membership, margins)))

    torso = features["torso_angle"]
    angle, km = config["confidence_angle_margin"], config["confidence_knee_margin"]
    height = config["confidence_height_margin"]
    coarse = {
        "upright_partial": score((config["standing_torso_angle"] - torso) / angle),
        "bent_partial": score(
            (torso - config["bending_torso_angle"]) / angle,
            (config["lying_torso_angle"] - torso) / angle,
        ),
        "horizontal_partial": score((torso - config["lying_torso_angle"]) / angle),
    }
    if not features.get("complete_leg", float("knee_angle" in features)) or (
        features["visible_joints"] < config["minimum_visible_joints"]
    ):
        return coarse
    knee = features.get("knee_angle")
    if knee is None:
        return coarse
    scores = {
        "standing": score(
            (config["standing_torso_angle"] - torso) / angle,
            (knee - config["standing_knee_angle"]) / km,
        ),
        "bending": score(
            (torso - config["bending_torso_angle"]) / angle,
            (knee - config["bending_knee_angle"]) / km,
        ),
        "sitting": score(
            (config["lying_torso_angle"] - torso) / angle,
            max(
                (config["sitting_knee_angle"] - knee) / km,
                (config["sitting_hip_knee_height"] - features.get("hip_knee_height", 1)) / height,
            ),
        ),
        "lying": score(
            (torso - config["lying_torso_angle"]) / angle,
            max(
                (features["bbox_aspect_ratio"] - config["lying_aspect_ratio"])
                / config["confidence_aspect_margin"],
                (config["lying_hip_ankle_height"] - abs(features.get("hip_ankle_height", 1)))
                / height,
            ),
        ),
    }
    scores["bending"] *= 1 - scores["lying"] / max(quality, 1e-9)
    if config["enable_extra_postures"]:
        scores["squatting"] = score(
            (config["squatting_knee_angle"] - knee) / km,
            (config["squatting_hip_knee_height"] - features.get("hip_knee_height", 1)) / height,
        )
        scores["kneeling"] = score(
            (config["squatting_knee_angle"] - knee) / km,
            (config["kneeling_hip_ankle_height"] - features.get("hip_ankle_height", 1)) / height,
        )
        if max(scores["squatting"], scores["kneeling"]) >= config["minimum_confidence"]:
            scores.pop("sitting")
    # Coarse classes are a fallback only when precise evidence is inconclusive.
    if max(scores.values()) < config["minimum_confidence"]:
        return coarse
    return scores


def classify_features(
    features: dict[str, float], config: dict, preferred: str | None = None
) -> tuple[str, float]:
    scores = posture_scores(features, config)
    if not scores:
        return "unknown", 0.0
    priority = {
        name: index
        for index, name in enumerate(
            ("sitting", "standing", "bending", "kneeling", "squatting", "lying")
        )
    }
    label = max(scores, key=lambda name: (scores[name], priority.get(name, 0)))
    if (
        preferred in scores
        and scores[preferred] >= config["minimum_confidence"]
        and (scores[label] - scores[preferred] <= config["label_hysteresis_margin"])
    ):
        label = preferred
    confidence = scores[label]
    return (label, confidence) if confidence >= config["minimum_confidence"] else ("unknown", 0.0)
