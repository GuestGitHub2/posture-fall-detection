"""Conservative rapid-fall and slow-collapse paths over valid per-person motion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.fall.detector import FallResult
from src.fall.temporal_buffer import TemporalBuffer
from src.pose.normalization import StableBodyScale
from src.pose.pose_types import Pose
from src.posture.classifier import PostureClassifier, PostureResult
from src.posture.geometry import extract_features

FALL_DEFAULTS = {
    "history_seconds": 2.5,
    "max_samples": 90,
    "max_gap_seconds": 0.5,
    "max_unreliable_seconds": 0.45,
    "minimum_joint_confidence": 0.4,
    "minimum_pose_quality": 0.4,
    "motion_window_seconds": 0.9,
    "minimum_confident_joints_per_pair": 1,
    "minimum_motion_seconds": 0.12,
    "upright_torso_angle": 30.0,
    "maximum_origin_torso_angle": 75.0,
    "minimum_hip_drop": 0.24,
    "minimum_shoulder_drop": 0.2,
    "minimum_hip_velocity": 0.65,
    "maximum_hip_velocity": 5.0,
    "minimum_torso_rotation": 35.0,
    "bent_origin_torso_rotation": 18.0,
    "low_origin_hip_ankle_height": 0.25,
    "low_origin_minimum_hip_drop": 0.14,
    "low_origin_minimum_shoulder_drop": 0.3,
    "low_origin_minimum_hip_velocity": 0.35,
    "low_origin_minimum_shoulder_velocity": 0.8,
    "possible_torso_angle": 40.0,
    "ground_hip_ankle_height": 0.24,
    "maximum_floor_distance": 0.2,
    "lying_persistence_seconds": 0.35,
    "candidate_timeout_seconds": 2.0,
    "slow_enabled": True,
    "slow_window_seconds": 2.5,
    "slow_minimum_motion_seconds": 1.2,
    "slow_minimum_hip_drop": 0.58,
    "slow_minimum_shoulder_drop": 0.5,
    "slow_minimum_hip_velocity": 0.18,
    "slow_maximum_hip_velocity": 0.65,
    "slow_minimum_torso_rotation": 35.0,
    "slow_minimum_downward_fraction": 0.8,
    "slow_jitter_tolerance": 0.015,
    "slow_lying_persistence_seconds": 0.6,
    "controlled_lowering_enabled": True,
    "controlled_origin_knee_angle": 155.0,
    "controlled_knee_angle": 130.0,
    "controlled_minimum_hip_drop": 0.18,
    "controlled_confirmation_seconds": 0.12,
    "controlled_window_seconds": 1.2,
    "controlled_hold_seconds": 2.5,
    "possible_threshold": 0.55,
    "confirmed_threshold": 0.8,
    "normal_probability": 0.03,
    "possible_probability": 0.65,
    "fall_probability": 0.92,
    "possible_confirmation_seconds": 0.08,
    "fall_confirmation_seconds": 0.2,
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
    scale_filter: StableBodyScale
    motion: deque[_MotionSample] = field(default_factory=deque)
    candidate_at: float | None = None
    lying_since: float | None = None
    lying_duration: float = 0.0
    candidate_features: dict[str, float] = field(default_factory=dict)
    unreliable_since: float | None = None
    last_update: float | None = None
    last_valid: float | None = None
    interrupted: bool = False
    controlled_since: float | None = None
    controlled_at: float | None = None

    def reset(self) -> None:
        self.buffer.clear()
        self.motion.clear()
        self.candidate_at = self.lying_since = None
        self.lying_duration = 0.0
        self.candidate_features.clear()
        self.last_valid = None
        self.controlled_since = self.controlled_at = None


class HeuristicFallDetector:
    """Scores are heuristic evidence, not calibrated accident probabilities.

    Short unreliable intervals suspend measurements and confirmation. A floor
    proxy from the origin's ankles rejects elevated bed/sofa transitions. It is
    deliberately conservative and cannot infer intent from identical 2D motion.
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
        c = self.config
        if not np.isfinite(timestamp):
            raise ValueError("Fall timestamp must be finite")
        if track_id not in self.histories:
            self.histories[track_id] = _PersonHistory(
                TemporalBuffer(c["history_seconds"], c["max_samples"], c["max_gap_seconds"]),
                StableBodyScale(c),
            )
        person = self.histories[track_id]
        delta = timestamp - person.last_update if person.last_update is not None else 0.0
        if person.last_update is not None and (delta <= 0 or delta > c["max_gap_seconds"]):
            person.reset()
            person.scale_filter = StableBodyScale(c)
            person.unreliable_since = None
        person.last_update = timestamp
        if not pose.scale_is_stable:
            pose = person.scale_filter.apply(pose, timestamp, c["minimum_joint_confidence"])
        posture = posture or self.classifier.classify(pose)
        current = extract_features(pose, c["minimum_joint_confidence"])
        reliable = (
            all(
                key in current
                for key in ("hip_y", "shoulder_y", "torso_angle", "hip_ankle_height", "ankle_y")
            )
            and current["pose_quality"] >= c["minimum_pose_quality"]
            and all(
                int((pose.keypoints[list(pair), 2] >= c["minimum_joint_confidence"]).sum())
                >= c["minimum_confident_joints_per_pair"]
                for pair in ((5, 6), (11, 12), (15, 16))
            )
        )
        metrics = {
            "hip_velocity": 0.0,
            "normalized_vertical_velocity": 0.0,
            "hip_drop": 0.0,
            "shoulder_drop": 0.0,
            "torso_rotation": 0.0,
            "measurement_valid": float(reliable),
        }
        if not reliable:
            if person.unreliable_since is None:
                person.unreliable_since = timestamp
            duration = timestamp - person.unreliable_since
            if duration >= c["max_unreliable_seconds"]:
                person.reset()
            person.interrupted = True
            return FallResult(
                "normal", c["normal_probability"], metrics | {"unreliable_duration": duration}
            )
        if (
            person.unreliable_since is not None
            and timestamp - person.unreliable_since >= c["max_unreliable_seconds"]
        ):
            person.reset()
        person.unreliable_since = None
        # Even if invalid inputs arrived meanwhile, do not bridge a long valid-pose gap.
        if person.buffer.append(pose, timestamp):
            person.reset()
            person.buffer.append(pose, timestamp)
        person.motion.append(
            _MotionSample(
                timestamp,
                PostureResult(posture.label, posture.confidence, current, posture.pose_quality),
            )
        )
        while person.motion and (
            timestamp - person.motion[0].timestamp > c["history_seconds"]
            or len(person.motion) > c["max_samples"]
        ):
            person.motion.popleft()
        # A sustained knee-flexion/lowering phase while the torso stays upright
        # is evidence for a controlled transition, rather than an immediate fall.
        # Initially kneeling people remain eligible: this requires a recent change
        # from straight knees, not simply the kneeling posture itself.
        if c["controlled_lowering_enabled"]:
            preparatory = (
                current.get("knee_angle", 180) <= c["controlled_knee_angle"]
                and current["torso_angle"] <= c["upright_torso_angle"]
                and any(
                    timestamp - sample.timestamp <= c["controlled_window_seconds"]
                    and sample.posture.features.get("knee_angle", 0)
                    >= c["controlled_origin_knee_angle"]
                    and sample.posture.features["torso_angle"] <= c["upright_torso_angle"]
                    and (current["hip_y"] - sample.posture.features["hip_y"])
                    / sample.posture.features["body_scale"]
                    >= c["controlled_minimum_hip_drop"]
                    for sample in person.motion
                )
            )
            if preparatory:
                if person.controlled_since is None:
                    person.controlled_since = timestamp
                if (
                    timestamp - person.controlled_since + 1e-9
                    >= c["controlled_confirmation_seconds"]
                ):
                    person.controlled_at = timestamp
            else:
                person.controlled_since = None
            if (
                person.controlled_at is not None
                and timestamp - person.controlled_at <= c["controlled_hold_seconds"]
            ):
                person.candidate_at = person.lying_since = None
                person.lying_duration = 0.0
                person.candidate_features.clear()
                person.last_valid = timestamp
                person.interrupted = False
                return FallResult(
                    "normal", c["normal_probability"], metrics | {"controlled_lowering": 1.0}
                )
        best_score = -1.0
        for sample in person.motion:
            elapsed = timestamp - sample.timestamp
            origin = sample.posture.features
            if (
                elapsed < c["minimum_motion_seconds"]
                or origin.get("torso_angle", 90) > c["maximum_origin_torso_angle"]
                or sample.posture.label == "lying"
            ):
                continue
            scale = origin["body_scale"]
            hip_drop = (current["hip_y"] - origin["hip_y"]) / scale
            shoulder_drop = (current["shoulder_y"] - origin["shoulder_y"]) / scale
            velocity = hip_drop / elapsed
            rotation = current["torso_angle"] - origin["torso_angle"]
            evidence = {
                "hip_velocity": velocity,
                "normalized_vertical_velocity": velocity,
                "hip_drop": hip_drop,
                "shoulder_drop": shoulder_drop,
                "torso_rotation": rotation,
                "origin_floor_y": origin["ankle_y"],
                "origin_scale": scale,
            }
            if velocity > metrics["hip_velocity"]:
                metrics.update(evidence)
            minimum_rotation = (
                c["minimum_torso_rotation"]
                if origin["torso_angle"] <= c["upright_torso_angle"]
                else c["bent_origin_torso_rotation"]
            )
            low_origin = abs(origin["hip_ankle_height"]) <= c["low_origin_hip_ankle_height"]
            minimum_drop = c["low_origin_minimum_hip_drop"] if low_origin else c["minimum_hip_drop"]
            minimum_shoulder = (
                c["low_origin_minimum_shoulder_drop"] if low_origin else c["minimum_shoulder_drop"]
            )
            minimum_velocity = (
                c["low_origin_minimum_hip_velocity"] if low_origin else c["minimum_hip_velocity"]
            )
            rapid = (
                elapsed <= c["motion_window_seconds"]
                and hip_drop >= minimum_drop
                and shoulder_drop >= minimum_shoulder
                and minimum_velocity <= velocity <= c["maximum_hip_velocity"]
                and (
                    not low_origin
                    or shoulder_drop / elapsed >= c["low_origin_minimum_shoulder_velocity"]
                )
                and rotation >= minimum_rotation
                and current["torso_angle"] >= c["possible_torso_angle"]
            )
            slow = False
            if (
                c["slow_enabled"]
                and c["slow_minimum_motion_seconds"] <= elapsed <= c["slow_window_seconds"]
            ):
                path = [
                    item.posture.features["hip_y"]
                    for item in person.motion
                    if item.timestamp >= sample.timestamp
                ]
                downward = (
                    float(np.mean(np.diff(path) >= -c["slow_jitter_tolerance"] * scale))
                    if len(path) > 1
                    else 0.0
                )
                slow = (
                    hip_drop >= c["slow_minimum_hip_drop"]
                    and shoulder_drop >= c["slow_minimum_shoulder_drop"]
                    and c["slow_minimum_hip_velocity"] <= velocity <= c["slow_maximum_hip_velocity"]
                    and rotation >= c["slow_minimum_torso_rotation"]
                    and downward >= c["slow_minimum_downward_fraction"]
                    and posture.label == "lying"
                )
                evidence["downward_fraction"] = downward
            if (rapid or slow) and hip_drop > best_score:
                best_score = hip_drop
                if person.candidate_at is None:
                    person.candidate_at = timestamp
                # Preserve the earliest floor/scale and strongest descent in an episode.
                if hip_drop >= person.candidate_features.get("hip_drop", -1):
                    person.candidate_features = evidence | {
                        "slow_collapse": float(slow and not rapid),
                        "low_origin": float(low_origin),
                    }
        if (
            person.candidate_at is not None
            and timestamp - person.candidate_at > c["candidate_timeout_seconds"]
        ):
            person.candidate_at = person.lying_since = None
            person.lying_duration = 0.0
            person.candidate_features.clear()
        previous_valid = person.last_valid
        person.last_valid = timestamp
        interrupted, person.interrupted = person.interrupted, False
        if person.candidate_at is None:
            return FallResult("normal", c["normal_probability"], metrics)
        floor_distance = (
            abs(current["hip_y"] - person.candidate_features["origin_floor_y"])
            / person.candidate_features["origin_scale"]
        )
        grounded = (
            posture.label == "lying"
            and abs(current["hip_ankle_height"]) <= c["ground_hip_ankle_height"]
            and floor_distance <= c["maximum_floor_distance"]
        )
        if grounded:
            if person.lying_since is None:
                person.lying_since = timestamp
                person.lying_duration = 0.0
            elif not interrupted and previous_valid is not None:
                person.lying_duration += timestamp - previous_valid
            metrics.update(
                person.candidate_features
                | {"lying_duration": person.lying_duration, "floor_distance": floor_distance}
            )
            persistence = (
                c["slow_lying_persistence_seconds"]
                if person.candidate_features["slow_collapse"]
                else c["lying_persistence_seconds"]
            )
            if person.lying_duration + 1e-9 >= persistence:
                return FallResult("fall", c["fall_probability"], metrics)
        else:
            person.lying_since = None
            person.lying_duration = 0.0
            if posture.label == "standing":
                person.candidate_at = None
                person.candidate_features.clear()
                return FallResult("normal", c["normal_probability"], metrics)
        return FallResult("possible_fall", c["possible_probability"], metrics)
