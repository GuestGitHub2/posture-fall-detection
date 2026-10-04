"""Temporal motion baseline; lying alone never constitutes a fall."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.fall.detector import FallResult
from src.fall.temporal_buffer import TemporalBuffer
from src.pose.pose_types import Pose
from src.posture.classifier import PostureClassifier, PostureResult
from src.posture.geometry import extract_features

FALL_DEFAULTS = {
    "history_seconds": 2.0,
    "max_samples": 90,
    "max_gap_seconds": 0.5,
    "minimum_joint_confidence": 0.4,
    "motion_window_seconds": 0.9,
    "minimum_confident_joints_per_pair": 1,
    "minimum_motion_seconds": 0.12,
    "upright_torso_angle": 30.0,
    "minimum_hip_drop": 0.24,
    "minimum_shoulder_drop": 0.2,
    "minimum_hip_velocity": 0.65,
    "maximum_hip_velocity": 5.0,
    "minimum_torso_rotation": 35.0,
    "possible_torso_angle": 40.0,
    "ground_hip_ankle_height": 0.24,
    "lying_persistence_seconds": 0.35,
    "candidate_timeout_seconds": 1.8,
    "possible_threshold": 0.55,
    "confirmed_threshold": 0.8,
    "normal_probability": 0.03,
    "possible_probability": 0.65,
    "fall_probability": 0.92,
    "confirmation_frames": 3,
    "recovery_seconds": 3.0,
    "event_cooldown_seconds": 10.0,
}


@dataclass
class _MotionSample:
    timestamp: float
    posture: PostureResult


@dataclass
class _PersonHistory:
    buffer: TemporalBuffer
    motion: deque[_MotionSample] = field(default_factory=deque)
    candidate_at: float | None = None
    lying_since: float | None = None
    candidate_features: dict[str, float] = field(default_factory=dict)


class HeuristicFallDetector:
    """Detect a rapid descent and rotation followed by sustained low-lying pose.

    Confidence is an interpretable heuristic score, not a calibrated probability.
    Each track has independent history, reset after gaps or clock discontinuities.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = FALL_DEFAULTS | (config or {})
        self.histories: dict[int, _PersonHistory] = {}
        self.classifier = PostureClassifier()

    def remove(self, track_id: int) -> None:
        self.histories.pop(track_id, None)

    def update(
        self, track_id: int, pose: Pose, timestamp: float, posture: PostureResult | None = None
    ) -> FallResult:
        config = self.config
        if track_id not in self.histories:
            self.histories[track_id] = _PersonHistory(
                TemporalBuffer(
                    config["history_seconds"], config["max_samples"], config["max_gap_seconds"]
                )
            )
        person = self.histories[track_id]
        if person.buffer.append(pose, timestamp):
            person.motion.clear()
            person.candidate_at = person.lying_since = None
            person.candidate_features.clear()
        posture = posture or self.classifier.classify(pose)
        # Use the same confidence threshold for geometry and reliability, so a weak
        # ankle cannot quietly alter the estimated ground relationship.
        current = extract_features(pose, config["minimum_joint_confidence"])
        reliable = (
            all(
                key in current for key in ("hip_y", "shoulder_y", "torso_angle", "hip_ankle_height")
            )
            and posture.confidence >= config["minimum_joint_confidence"]
            and all(
                int((pose.keypoints[list(pair), 2] >= config["minimum_joint_confidence"]).sum())
                >= config["minimum_confident_joints_per_pair"]
                for pair in ((5, 6), (11, 12), (15, 16))
            )
        )
        person.motion.append(
            _MotionSample(timestamp, PostureResult(posture.label, posture.confidence, current))
        )
        while person.motion and (
            timestamp - person.motion[0].timestamp > config["history_seconds"]
            or len(person.motion) > config["max_samples"]
        ):
            person.motion.popleft()
        metrics = {
            "hip_velocity": 0.0,
            "normalized_vertical_velocity": 0.0,
            "hip_drop": 0.0,
            "shoulder_drop": 0.0,
            "torso_rotation": 0.0,
        }
        if not reliable:
            person.motion.clear()
            person.candidate_at = None
            person.lying_since = None
            person.candidate_features.clear()
            return FallResult("normal", config["normal_probability"], metrics)
        best_score = 0.0
        for sample in person.motion:
            elapsed = timestamp - sample.timestamp
            if not config["minimum_motion_seconds"] <= elapsed <= config["motion_window_seconds"]:
                continue
            origin = sample.posture.features
            if (
                sample.posture.label not in {"standing", "sitting", "unknown"}
                or sample.posture.confidence < config["minimum_joint_confidence"]
            ):
                continue
            if origin.get("torso_angle", 90.0) > config["upright_torso_angle"] or not all(
                key in origin for key in ("hip_y", "shoulder_y", "body_scale")
            ):
                continue
            scale = max(origin["body_scale"], 1e-6)
            hip_drop = (current["hip_y"] - origin["hip_y"]) / scale
            shoulder_drop = (current["shoulder_y"] - origin["shoulder_y"]) / scale
            velocity = hip_drop / elapsed
            rotation = current["torso_angle"] - origin["torso_angle"]
            if velocity > metrics["hip_velocity"]:
                metrics = {
                    "hip_velocity": velocity,
                    "normalized_vertical_velocity": velocity,
                    "hip_drop": hip_drop,
                    "shoulder_drop": shoulder_drop,
                    "torso_rotation": rotation,
                }
            motion_evidence = (
                hip_drop >= config["minimum_hip_drop"]
                and shoulder_drop >= config["minimum_shoulder_drop"]
                and config["minimum_hip_velocity"] <= velocity <= config["maximum_hip_velocity"]
                and rotation >= config["minimum_torso_rotation"]
                and current["torso_angle"] >= config["possible_torso_angle"]
            )
            if motion_evidence and hip_drop > best_score:
                best_score = hip_drop
                person.candidate_features = metrics.copy()
                if person.candidate_at is None:
                    person.candidate_at = timestamp
        if (
            person.candidate_at is not None
            and timestamp - person.candidate_at > config["candidate_timeout_seconds"]
        ):
            person.candidate_at = person.lying_since = None
            person.candidate_features.clear()
        if person.candidate_at is None:
            return FallResult("normal", config["normal_probability"], metrics)
        grounded_lying = (
            posture.label == "lying"
            and abs(current["hip_ankle_height"]) <= config["ground_hip_ankle_height"]
        )
        if grounded_lying:
            if person.lying_since is None:
                person.lying_since = timestamp
            duration = timestamp - person.lying_since
            metrics |= person.candidate_features | {"lying_duration": duration}
            if duration >= config["lying_persistence_seconds"]:
                return FallResult("fall", config["fall_probability"], metrics)
        else:
            person.lying_since = None
            if posture.label == "standing":
                person.candidate_at = None
                return FallResult("normal", config["normal_probability"], metrics)
        return FallResult("possible_fall", config["possible_probability"], metrics)
