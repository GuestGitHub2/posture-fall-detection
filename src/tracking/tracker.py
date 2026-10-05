"""Conservative motion/geometry association; uncertain matches never update tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pose.normalization import SCALE_DEFAULTS, StableBodyScale, normalize_pose
from src.pose.pose_types import Pose
from src.tracking.matching import center_distance, iou, scale_distance, skeleton_distance

TRACKING_DEFAULTS = {
    "max_missing_frames": 0,
    "max_missing_seconds": 1.5,
    "matching_threshold": 0.75,
    "confidence_threshold": 0.3,
    "iou_weight": 0.3,
    "center_weight": 0.25,
    "skeleton_weight": 0.35,
    "scale_weight": 0.1,
    "max_center_distance": 0.9,
    "max_skeleton_distance": 0.8,
    "max_scale_ratio": 1.8,
    "minimum_common_joints": 4,
    "minimum_sparse_iou": 0.3,
    "ambiguity_margin": 0.035,
    "minimum_association_confidence": 0.2,
    "velocity_smoothing": 0.6,
    "prediction_seconds": 0.5,
    **SCALE_DEFAULTS,
}


@dataclass
class Track:
    track_id: int
    pose: Pose
    last_seen: float
    missing: int = 0
    observed: bool = True
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    association_confidence: float = 1.0
    association_ambiguous: bool = False
    history_epoch: int = 0
    scale_filter: StableBodyScale = field(default_factory=StableBodyScale)


class Tracker:
    """IDs are monotonic. History epochs invalidate evidence after ambiguous identity."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = TRACKING_DEFAULTS | (config or {})
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._last_timestamp: float | None = None

    def _cost(self, track: Track, pose: Pose, timestamp: float) -> float:
        c = self.config
        elapsed = max(0.0, timestamp - track.last_seen)
        translation = track.velocity * min(elapsed, c["prediction_seconds"])
        overlap = iou(track.pose.bbox + np.tile(translation, 2), pose.bbox)
        center = center_distance(track.pose, pose, translation)
        skeleton = skeleton_distance(track.pose, pose, translation, c["confidence_threshold"])
        scale = scale_distance(
            track.pose,
            pose,
            c["minimum_scale_quality"],
            reliable_reference=track.scale_filter.value is not None,
        )
        common = int(
            (
                (track.pose.keypoints[:, 2] >= c["confidence_threshold"])
                & (pose.keypoints[:, 2] >= c["confidence_threshold"])
            ).sum()
        )
        if (
            center > c["max_center_distance"]
            or skeleton > c["max_skeleton_distance"]
            or scale > np.log(c["max_scale_ratio"])
            or (common < c["minimum_common_joints"] and overlap < c["minimum_sparse_iou"])
        ):
            return float("inf")
        weights = [
            c[name] for name in ("iou_weight", "center_weight", "skeleton_weight", "scale_weight")
        ]
        return float(np.dot(weights, (1 - overlap, center, skeleton, scale)) / sum(weights))

    def update(self, poses: list[Pose], timestamp: float) -> list[Track]:
        if not np.isfinite(timestamp):
            raise ValueError("Tracking timestamp must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            # A discontinuity cannot safely carry identities or their state forward.
            self.tracks.clear()
        self._last_timestamp = timestamp
        for track_id, track in list(self.tracks.items()):
            if timestamp - track.last_seen > self.config["max_missing_seconds"]:
                self.tracks.pop(track_id)
        poses = [normalize_pose(pose, self.config["confidence_threshold"]) for pose in poses]
        existing = list(self.tracks.values())
        costs = np.array(
            [[self._cost(track, pose, timestamp) for pose in poses] for track in existing],
            dtype=np.float64,
        ).reshape(len(existing), len(poses))
        feasible = np.isfinite(costs) & (
            costs
            <= self.config["matching_threshold"]
            * (1 - self.config["minimum_association_confidence"])
        )
        ambiguous_rows, blocked_poses = set(), set()
        # Gate both sides: even Hungarian's globally best solution may be a near tie.
        for row in range(len(existing)):
            candidates = np.flatnonzero(feasible[row])
            if len(candidates) > 1:
                ranked = candidates[np.argsort(costs[row, candidates])]
                near = ranked[
                    costs[row, ranked] - costs[row, ranked[0]] <= self.config["ambiguity_margin"]
                ]
                if len(near) > 1:
                    ambiguous_rows.add(row)
                    blocked_poses.update(map(int, near))
        for column in range(len(poses)):
            candidates = np.flatnonzero(feasible[:, column])
            if len(candidates) > 1:
                ranked = candidates[np.argsort(costs[candidates, column])]
                near = ranked[
                    costs[ranked, column] - costs[ranked[0], column]
                    <= self.config["ambiguity_margin"]
                ]
                if len(near) > 1:
                    ambiguous_rows.update(map(int, near))
                    blocked_poses.add(column)
        safe = feasible.copy()
        for row in ambiguous_rows:
            safe[row] = False
            blocked_poses.update(map(int, np.flatnonzero(feasible[row])))
        if blocked_poses:
            safe[:, sorted(blocked_poses)] = False
        assignments = linear_sum_assignment(np.where(safe, costs, 1e6)) if costs.size else ([], [])
        matched_tracks, matched_poses = set(), set()
        for row, column in zip(*assignments, strict=True):
            if not safe[row, column]:
                continue
            track, pose = existing[row], poses[column]
            alternatives = np.r_[
                costs[row, np.arange(len(poses)) != column],
                costs[np.arange(len(existing)) != row, column],
            ]
            margin = (
                float(np.min(alternatives) - costs[row, column])
                if len(alternatives)
                else float("inf")
            )
            if margin <= self.config["ambiguity_margin"]:
                ambiguous_rows.add(row)
                blocked_poses.add(column)
                continue
            delta = timestamp - track.last_seen
            if delta > 1e-6:
                previous_center = (track.pose.bbox[:2] + track.pose.bbox[2:]) / 2
                center = (pose.bbox[:2] + pose.bbox[2:]) / 2
                velocity = (center - previous_center) / delta
                smooth = self.config["velocity_smoothing"]
                track.velocity = smooth * velocity + (1 - smooth) * track.velocity
            track.pose = track.scale_filter.apply(
                pose, timestamp, self.config["confidence_threshold"]
            )
            track.last_seen, track.missing, track.observed = timestamp, 0, True
            track.association_ambiguous = False
            track.association_confidence = float(
                np.clip(1 - costs[row, column] / self.config["matching_threshold"], 0, 1)
            )
            matched_tracks.add(track.track_id)
            matched_poses.add(column)
        for row, track in enumerate(existing):
            if track.track_id in matched_tracks:
                continue
            track.missing += 1
            track.observed = False
            if row in ambiguous_rows:
                if not track.association_ambiguous:
                    track.history_epoch += 1
                    track.scale_filter = StableBodyScale(self.config)
                track.association_ambiguous = True
                track.association_confidence = 0.0
            if (
                self.config["max_missing_frames"] > 0
                and track.missing > self.config["max_missing_frames"]
            ) or timestamp - track.last_seen > self.config["max_missing_seconds"]:
                self.tracks.pop(track.track_id, None)
        for index, pose in enumerate(poses):
            if index not in matched_poses and index not in blocked_poses:
                scale_filter = StableBodyScale(self.config)
                pose = scale_filter.apply(pose, timestamp, self.config["confidence_threshold"])
                self.tracks[self._next_id] = Track(
                    self._next_id, pose, timestamp, scale_filter=scale_filter
                )
                self._next_id += 1
        return sorted(self.tracks.values(), key=lambda track: track.track_id)
