"""Pose-independent per-person processing shared by live, file and training tools."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.events import EventBus
from src.fall.detector import FallResult
from src.fall.heuristic import HeuristicFallDetector
from src.pose.pose_types import Pose
from src.posture.classifier import PostureClassifier, PostureResult
from src.state.state_machine import PersonStateMachine
from src.tracking.tracker import Tracker


@dataclass(frozen=True)
class PersonSnapshot:
    track_id: int
    pose: Pose
    posture: PostureResult
    fall: FallResult
    state: str
    observed: bool
    last_seen: float


@dataclass(frozen=True)
class PipelineResult:
    persons: list[PersonSnapshot]
    timestamp: float
    timings: dict[str, float]
    provider: str


class Pipeline:
    def __init__(
        self,
        config: dict[str, Any],
        estimator: Any = None,
        event_bus: EventBus | None = None,
        load_pose: bool = True,
    ) -> None:
        self.config = config
        if estimator is None and load_pose:
            from src.pose.base import create_pose_estimator

            estimator = create_pose_estimator(config)
        self.estimator = estimator
        self.tracker = Tracker(config["tracking"])
        if config["posture"].get("backend", "rules") == "mlp":
            from src.posture.mlp import MLPPostureClassifier

            self.posture_classifier = MLPPostureClassifier(config["posture"], config["runtime"])
        else:
            self.posture_classifier = PostureClassifier(config["posture"])
        if config["fall"]["detector"] == "stgcn":
            from src.fall.stgcn import STGCNFallDetector

            self.fall_detector = STGCNFallDetector(config)
        else:
            self.fall_detector = HeuristicFallDetector(config["fall"])
        self.event_bus = event_bus if event_bus is not None else EventBus(config["events"]["path"])
        self._machines: dict[int, PersonStateMachine] = {}
        self._snapshots: dict[int, PersonSnapshot] = {}

    @property
    def provider(self) -> str:
        return self.estimator.provider if self.estimator is not None else "skeleton-only"

    def process(self, frame: np.ndarray, timestamp: float) -> PipelineResult:
        if self.estimator is None:
            raise RuntimeError(
                "Pipeline has no pose estimator; use process_poses for skeleton inputs"
            )
        started = time.perf_counter()
        poses = self.estimator.infer(frame, timestamp)
        pose_ms = (time.perf_counter() - started) * 1000
        result = self.process_poses(poses, timestamp)
        result.timings["pose_ms"] = pose_ms
        result.timings["total_ms"] = (time.perf_counter() - started) * 1000
        return result

    def process_poses(self, poses: list[Pose], timestamp: float) -> PipelineResult:
        started = time.perf_counter()
        tracks = self.tracker.update(poses, timestamp)
        tracking_ms = (time.perf_counter() - started) * 1000
        active = {track.track_id for track in tracks}
        for removed in set(self._machines) - active:
            self._machines.pop(removed)
            self._snapshots.pop(removed, None)
            self.fall_detector.remove(removed)
        persons: list[PersonSnapshot] = []
        posture_ms = fall_ms = 0.0
        for track in tracks:
            if track.track_id not in self._machines:
                self._machines[track.track_id] = PersonStateMachine(
                    {**self.config["state"], **self.config["posture"], **self.config["fall"]}
                )
            machine = self._machines[track.track_id]
            if track.observed:
                begin = time.perf_counter()
                posture = self.posture_classifier.classify(track.pose)
                posture_ms += (time.perf_counter() - begin) * 1000
                begin = time.perf_counter()
                fall = self.fall_detector.update(track.track_id, track.pose, timestamp, posture)
                fall_ms += (time.perf_counter() - begin) * 1000
                state = machine.update(posture, fall, timestamp)
                if state.fall_event:
                    self.event_bus.emit(
                        "fall_detected", track.track_id, fall.probability, posture.label, timestamp
                    )
                if state.recovered_event:
                    self.event_bus.emit(
                        "recovered", track.track_id, fall.probability, posture.label, timestamp
                    )
                snapshot = PersonSnapshot(
                    track.track_id, track.pose, posture, fall, state.state, True, track.last_seen
                )
            else:
                previous = self._snapshots.get(track.track_id)
                if previous is None:
                    continue
                state = machine.missing(timestamp)
                snapshot = PersonSnapshot(
                    track.track_id,
                    track.pose,
                    previous.posture,
                    previous.fall,
                    state.state,
                    False,
                    track.last_seen,
                )
            self._snapshots[track.track_id] = snapshot
            persons.append(snapshot)
        timings = {
            "pose_ms": 0.0,
            "tracking_ms": tracking_ms,
            "posture_ms": posture_ms,
            "fall_ms": fall_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
        }
        return PipelineResult(persons, timestamp, timings, self.provider)
