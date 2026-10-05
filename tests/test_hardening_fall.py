"""Projected motion fixtures test policy, not real-world fall accuracy."""

import numpy as np
import pytest

from src.fall.detector import FallResult
from src.fall.heuristic import HeuristicFallDetector
from src.pose.pose_types import Pose
from src.posture.classifier import PostureResult
from src.state.state_machine import PersonStateMachine
from tests.helpers import fall_pose, skeleton


def progress_at(timestamp, duration=0.4):
    return min(1.0, max(0.0, (timestamp - 0.4) / duration))


def reshape_pose(points):
    return Pose(points, np.r_[points[:, :2].min(axis=0) - 10, points[:, :2].max(axis=0) + 10])


def bending_fall(progress):
    initial, final = skeleton("bending"), fall_pose(1)
    return reshape_pose(initial.keypoints * (1 - progress) + final.keypoints * progress)


def low_posture(label):
    pose = skeleton()
    pose.keypoints[:13, 1] += 140
    pose.keypoints[[13, 14], :2] = [[350, 390], [370, 390]]
    pose.keypoints[[15, 16], :2] = [[280, 400], [320, 400]]
    return reshape_pose(pose.keypoints)


def episode(times, make_pose, detector=None):
    detector = detector or HeuristicFallDetector()
    machine = PersonStateMachine()
    predictions, events = [], []
    for timestamp in times:
        pose = make_pose(float(timestamp))
        posture = detector.classifier.classify(pose)
        result = detector.update(1, pose, float(timestamp), posture)
        state = machine.update(posture, result, float(timestamp))
        predictions.append(result)
        if state.fall_event:
            events.append(float(timestamp))
    return predictions, events


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
@pytest.mark.parametrize("direction", ["forward", "side", "back"])
def test_projected_rapid_falls_have_one_event_at_all_rates(fps, direction):
    def projected(timestamp):
        progress = progress_at(timestamp)
        pose = fall_pose(
            progress * (0.95 if direction == "forward" else 1),
            drop=200 / 0.95 if direction == "forward" else 200,
        )
        if direction == "back":
            pose.keypoints[:, 0] = 600 - pose.keypoints[:, 0]
            pose = reshape_pose(pose.keypoints)
        return pose

    _, events = episode(np.arange(0, 2.4, 1 / fps), projected)
    assert len(events) == 1
    assert 1.2 <= events[0] <= 1.6


def test_event_latency_varies_only_by_sampling_resolution():
    detections = []
    for fps in (5, 10, 15, 30):
        _, events = episode(np.arange(0, 2.4, 1 / fps), lambda time: fall_pose(progress_at(time)))
        detections.append(events[0])
    assert max(detections) - min(detections) <= 0.25


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_slow_collapse_has_separate_path(fps):
    results, events = episode(
        np.arange(0, 4.2, 1 / fps), lambda time: fall_pose(progress_at(time, 1.8))
    )
    assert len(events) == 1
    assert any(row.status == "fall" and row.features["slow_collapse"] == 1 for row in results)
    rapid_only = HeuristicFallDetector({"slow_enabled": False})
    _, events = episode(
        np.arange(0, 4.2, 1 / fps), lambda time: fall_pose(progress_at(time, 1.8)), rapid_only
    )
    assert events == []


@pytest.mark.parametrize(
    "negative",
    [
        "static_lying",
        "quick_lie",
        "bed",
        "sofa",
        "slow_intentional",
        "bending",
        "tying_shoes",
        "kneeling",
        "squatting",
    ],
)
def test_hard_negatives_do_not_confirm(negative):
    def projected(timestamp):
        p = progress_at(timestamp)
        if negative == "static_lying":
            return skeleton("lying")
        if negative in ("quick_lie", "bed", "sofa"):
            # Deliberate supported transitions end above the observed foot/floor level.
            # Same torso rotation, but no descent to the original ankle plane.
            return fall_pose(p, drop={"quick_lie": 90, "bed": 65, "sofa": 105}[negative])
        if negative == "slow_intentional":
            return fall_pose(progress_at(timestamp, 3.0))
        if negative in ("kneeling", "squatting"):
            return low_posture(negative) if timestamp >= 0.4 else skeleton()
        return (
            skeleton(
                "bending", offset=(300, 100 if negative == "tying_shoes" and timestamp > 0.4 else 0)
            )
            if timestamp >= 0.4
            else skeleton()
        )

    results, events = episode(np.arange(0, 5, 1 / 15), projected)
    assert events == []
    assert all(result.status != "fall" for result in results)


def test_fall_can_begin_from_bending():
    _, events = episode(np.arange(0, 2.5, 1 / 15), lambda time: bending_fall(progress_at(time)))
    assert len(events) == 1


@pytest.mark.parametrize("joints", [(5, 6), (11, 12), (15, 16)])
@pytest.mark.parametrize("missing_count", [1, 3])
def test_brief_low_quality_burst_preserves_candidate_but_cannot_confirm(joints, missing_count):
    detector = HeuristicFallDetector()
    results = []
    for index in range(38):
        pose = fall_pose(progress_at(index / 15))
        if 12 <= index < 12 + missing_count:
            pose.keypoints[list(joints), 2] = 0.05
        result = detector.update(1, pose, index / 15)
        results.append(result)
        if 12 <= index < 12 + missing_count:
            assert result.status == "normal"
            assert result.features["measurement_valid"] == 0
            assert detector.histories[1].candidate_at is not None
    assert any(result.status == "fall" for result in results)


def test_sustained_occlusion_eventually_clears_evidence():
    detector = HeuristicFallDetector()
    for index in range(12):
        detector.update(1, fall_pose(progress_at(index / 15)), index / 15)
    for index in range(12, 25):
        pose = fall_pose(1)
        pose.keypoints[[5, 6, 11, 12, 15, 16], 2] = 0
        result = detector.update(1, pose, index / 15)
    person = detector.histories[1]
    assert result.features["unreliable_duration"] > 0.45
    assert person.candidate_at is None and not person.motion and not person.buffer.samples
    assert detector.update(1, fall_pose(1), 25 / 15).status == "normal"


def test_short_absent_sample_burst_and_irregular_cadence():
    times = [0, 0.09, 0.22, 0.36, 0.47, 0.57, 0.66, 0.92, 1.03, 1.18, 1.32, 1.44, 1.6, 1.8, 2.0]
    _, events = episode(times, lambda time: fall_pose(progress_at(time)))
    assert len(events) == 1


def test_monotonic_timestamp_jitter_does_not_change_outcome():
    times = np.arange(0, 2.5, 1 / 15)
    times[1:] += np.sin(np.arange(len(times) - 1)) * 0.008
    _, events = episode(times, lambda time: fall_pose(progress_at(time)))
    assert len(events) == 1


def test_timestamp_regression_resets_motion_and_candidate():
    detector = HeuristicFallDetector()
    for index in range(12):
        detector.update(1, fall_pose(progress_at(index / 15)), index / 15)
    assert detector.histories[1].candidate_at is not None
    assert detector.update(1, fall_pose(1), 0.5).status == "normal"
    assert len(detector.histories[1].motion) == 1
    for index in range(10):
        assert detector.update(1, fall_pose(1), 0.6 + index / 15).status == "normal"


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_state_confirmation_is_duration_based(fps):
    machine = PersonStateMachine({"fall_confirmation_seconds": 0.4})
    events = []
    for timestamp in np.arange(0, 1.0, 1 / fps):
        state = machine.update(
            PostureResult("lying", 0.95), FallResult("fall", 0.95), float(timestamp)
        )
        if state.fall_event:
            events.append(float(timestamp))
    assert events == [pytest.approx(0.4)]


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_slow_intentional_lie_remains_negative_at_all_rates(fps):
    _, events = episode(np.arange(0, 5, 1 / fps), lambda time: fall_pose(progress_at(time, 3.0)))
    assert events == []


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_whole_pipeline_event_outcome_at_all_inference_rates(fps):
    from src.config import load_config
    from src.events import EventBus
    from src.pipeline import Pipeline

    pipeline = Pipeline(load_config(), event_bus=EventBus(), load_pose=False)
    events = []
    pipeline.event_bus.subscribe(events.append)
    for timestamp in np.arange(0, 2.5, 1 / fps):
        pipeline.process_poses(
            [fall_pose(progress_at(timestamp)), skeleton("lying", offset=(1100, 0))],
            float(timestamp),
        )
    assert len(events) == 1 and events[0]["track_id"] == 1


def test_invalid_prediction_intervals_do_not_accrue_confirmation_duration():
    machine = PersonStateMachine({"fall_confirmation_seconds": 0.3})
    posture, positive = PostureResult("lying", 0.95), FallResult("fall", 0.95)
    machine.update(posture, positive, 0)
    assert not machine.update(posture, positive, 0.1).fall_event
    for time in (0.2, 0.3):
        assert not machine.update(
            posture, FallResult("normal", 0, {"measurement_valid": 0}), time
        ).fall_event
    assert not machine.update(posture, positive, 0.4).fall_event
    assert not machine.update(posture, positive, 0.5).fall_event
    assert machine.update(posture, positive, 0.6).fall_event


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_rapid_collapse_from_kneeling_with_strong_shoulder_descent(fps):
    initial = skeleton(offset=(300, 145))
    initial.keypoints[[13, 14], :2] = [[320, 370], [360, 370]]
    initial.keypoints[[15, 16], :2] = [[270, 385], [310, 385]]
    initial = reshape_pose(initial.keypoints)
    from src.posture.classifier import PostureClassifier

    assert PostureClassifier({"enable_extra_postures": True}).classify(initial).label == "kneeling"
    final = fall_pose(1)
    final.keypoints[:, 1] -= 15
    final = reshape_pose(final.keypoints)
    _, events = episode(
        np.arange(0, 2.5, 1 / fps),
        lambda time: reshape_pose(
            initial.keypoints * (1 - progress_at(time)) + final.keypoints * progress_at(time)
        ),
    )
    assert len(events) == 1


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_quick_controlled_lie_on_floor_is_a_hard_negative(fps):
    standing, kneeling = skeleton(), skeleton(offset=(300, 145))
    kneeling.keypoints[[13, 14], :2] = [[320, 370], [360, 370]]
    kneeling.keypoints[[15, 16], :2] = [[270, 385], [310, 385]]
    kneeling = reshape_pose(kneeling.keypoints)
    final = fall_pose(1)
    final.keypoints[:, 1] -= 15

    def controlled(time):
        if time <= 0.8:
            p = min(1, max(0, (time - 0.4) / 0.4))
            return reshape_pose(standing.keypoints * (1 - p) + kneeling.keypoints * p)
        p = min(1, max(0, (time - 1.0) / 0.4))
        return reshape_pose(kneeling.keypoints * (1 - p) + final.keypoints * p)

    results, events = episode(np.arange(0, 5, 1 / fps), controlled)
    assert events == []
    assert all(result.status != "fall" for result in results)
    assert any(result.features.get("controlled_lowering") == 1 for result in results)
