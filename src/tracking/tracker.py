"""Hungarian association with short occlusion retention and motion prediction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.pose.normalization import normalize_pose
from src.pose.pose_types import Pose
from src.tracking.matching import iou, skeleton_distance

TRACKING_DEFAULTS = {
    "max_missing_frames": 15,
    "max_missing_seconds": 1.5,
    "matching_threshold": 0.75,
    "confidence_threshold": 0.3,
    "iou_weight": 0.45,
    "max_skeleton_distance": 0.8,
    "velocity_smoothing": 0.6,
    "prediction_seconds": 0.5,
}


@dataclass
class Track:
    track_id: int
    pose: Pose
    last_seen: float
    missing: int = 0
    observed: bool = True
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))


class Tracker:
    def __init__(self, config: dict | None = None) -> None:
        self.config = TRACKING_DEFAULTS | (config or {})
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, poses: list[Pose], timestamp: float) -> list[Track]:
        # Expire before association: a person reappearing after a long silence
        # must not inherit a stale identity and its downstream state.
        for track_id, track in list(self.tracks.items()):
            if timestamp - track.last_seen > self.config["max_missing_seconds"]:
                self.tracks.pop(track_id)
        poses = [
            normalize_pose(pose, self.config["confidence_threshold"])
            if pose.normalized is None
            else pose
            for pose in poses
        ]
        existing = list(self.tracks.values())
        costs = np.full((len(existing), len(poses)), 1e6, dtype=np.float64)
        for row, track in enumerate(existing):
            elapsed = max(0.0, timestamp - track.last_seen)
            translation = track.velocity * min(elapsed, self.config["prediction_seconds"])
            predicted_box = track.pose.bbox + np.tile(translation, 2)
            for column, pose in enumerate(poses):
                distance = skeleton_distance(
                    track.pose, pose, translation, self.config["confidence_threshold"]
                )
                if distance <= self.config["max_skeleton_distance"]:
                    weight = self.config["iou_weight"]
                    costs[row, column] = (
                        weight * (1 - iou(predicted_box, pose.bbox)) + (1 - weight) * distance
                    )
        assignments = linear_sum_assignment(costs) if costs.size else ([], [])
        matched_tracks, matched_poses = set(), set()
        for row, column in zip(*assignments, strict=True):
            if costs[row, column] > self.config["matching_threshold"]:
                continue
            track, pose = existing[row], poses[column]
            delta = timestamp - track.last_seen
            if delta > 1e-6:
                previous_center = (track.pose.bbox[:2] + track.pose.bbox[2:]) / 2
                center = (pose.bbox[:2] + pose.bbox[2:]) / 2
                velocity = (center - previous_center) / delta
                smooth = self.config["velocity_smoothing"]
                track.velocity = smooth * velocity + (1 - smooth) * track.velocity
            track.pose, track.last_seen, track.missing, track.observed = pose, timestamp, 0, True
            matched_tracks.add(track.track_id)
            matched_poses.add(column)
        for track in existing:
            if track.track_id in matched_tracks:
                continue
            track.missing += 1
            track.observed = False
            if (
                track.missing > self.config["max_missing_frames"]
                or timestamp - track.last_seen > self.config["max_missing_seconds"]
            ):
                self.tracks.pop(track.track_id)
        for index, pose in enumerate(poses):
            if index not in matched_poses:
                self.tracks[self._next_id] = Track(self._next_id, pose, timestamp)
                self._next_id += 1
        return sorted(self.tracks.values(), key=lambda track: track.track_id)
