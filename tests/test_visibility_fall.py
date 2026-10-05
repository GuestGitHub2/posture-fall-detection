"""Torso-only fall policies are temporal and tested against projected negatives."""

import numpy as np
import pytest

from src.fall.heuristic import HeuristicFallDetector
from src.pose.pose_types import Pose
from tests.helpers import fall_pose, skeleton
from tests.test_hardening_fall import episode, progress_at
from tests.test_visibility_posture import partial, torso_angle


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
@pytest.mark.parametrize("visibility", ["full", "torso", "transition"])
def test_rapid_falls_with_lower_body_loss(fps, visibility):
    def pose_at(time):
        pose = fall_pose(progress_at(time))
        if visibility == "torso" or visibility == "transition" and time >= 0.6:
            partial(pose)
        return pose

    predictions, events = episode(np.arange(0, 3.2, 1 / fps), pose_at)
    assert len(events) == 1
    if visibility != "full":
        assert any(p.status == "fall" and p.features["partial_evidence"] == 1 for p in predictions)


@pytest.mark.parametrize("fps", [5, 10, 15, 30])
def test_slow_collapse_with_no_lower_body(fps):
    predictions, events = episode(
        np.arange(0, 4.5, 1 / fps),
        lambda time: partial(fall_pose(progress_at(time, 1.8), drop=220)),
    )
    assert len(events) == 1
    assert any(p.status == "fall" and p.features["slow_collapse"] == 1 for p in predictions)


@pytest.mark.parametrize(
    "motion",
    ["sitting", "bending", "crouching", "approaching", "leaving", "entering", "lying", "slow_lie"],
)
def test_partial_body_hard_negatives(motion):
    def pose_at(time):
        p = progress_at(time, 0.5)
        if motion == "sitting":
            pose = skeleton("sitting")
            pose.keypoints[:13, 1] += 100 * p
        elif motion == "bending":
            first, last = skeleton(), skeleton("bending")
            points = first.keypoints * (1 - p) + last.keypoints * p
            pose = Pose(points, first.bbox)
        elif motion == "approaching":
            pose = skeleton(offset=(300, 0), scale=1 + 0.8 * p)
        elif motion in {"crouching", "leaving", "entering"}:
            pose = skeleton(offset=(300, 200 * (1 - p if motion == "entering" else p)))
            pose.image_size = (640, 300)
        elif motion == "lying":
            pose = fall_pose(1)
        else:
            pose = fall_pose(progress_at(time, 3.0))
        return partial(pose)

    predictions, events = episode(np.arange(0, 5, 1 / 15), pose_at)
    assert not events and all(p.status != "fall" for p in predictions)


@pytest.mark.parametrize("surface_height", [60, 120])
def test_supported_fast_lying_uses_retained_floor_proxy_when_ankles_disappear(surface_height):
    def pose_at(time):
        p = progress_at(time)
        pose = fall_pose(p, drop=200 - surface_height)
        if time >= 0.6:
            partial(pose)
        return pose

    predictions, events = episode(np.arange(0, 3, 1 / 15), pose_at)
    assert not events and all(p.status != "fall" for p in predictions)


def test_strong_candidate_lower_body_loss_retains_history_and_uses_valid_torso():
    detector = HeuristicFallDetector()
    for index in range(12):
        detector.update(1, fall_pose(progress_at(index / 15)), index / 15)
    person = detector.histories[1]
    origin = person.motion[0].timestamp
    result = detector.update(1, partial(fall_pose(1)), 0.8)
    assert result.features["measurement_valid"] == 1
    assert result.status == "possible_fall" and person.candidate_at is not None
    assert person.motion[0].timestamp == origin
    assert "origin_floor_y" in person.candidate_features
    assert person.unreliable_since is None


def test_knee_loss_alone_uses_partial_tier_with_available_floor_evidence():
    predictions, events = episode(
        np.arange(0, 3.2, 1 / 15),
        lambda time: partial(fall_pose(progress_at(time)), missing=(13, 14)),
    )
    assert len(events) == 1
    positives = [p for p in predictions if p.status == "fall"]
    assert positives
    assert all(
        p.features["partial_evidence"] == 1 and p.features["floor_available"] == 1
        for p in positives
    )


def test_disappearance_after_strong_candidate_is_bounded_possible_never_confirmed():
    detector = HeuristicFallDetector()
    for index in range(12):
        detector.update(1, fall_pose(progress_at(index / 15)), index / 15)
    for time in np.arange(0.8, 2.3, 0.1):
        result = detector.missing(1, float(time))
        assert result.status != "fall" and result.features["measurement_valid"] == 0
        if time < 1.75:
            assert result.status == "possible_fall"
    assert result.status == "normal"
    assert not detector.histories[1].motion
    assert detector.histories[1].candidate_at is None


def test_disappearance_after_weak_motion_has_no_possible_or_fall():
    detector = HeuristicFallDetector()
    for index in range(6):
        detector.update(1, partial(torso_angle(15)), index / 15)
    for time in np.arange(0.4, 1.5, 0.1):
        assert detector.missing(1, float(time)).status == "normal"


def test_disappearing_and_returning_without_new_motion_cannot_confirm():
    detector = HeuristicFallDetector()
    for index in range(12):
        detector.update(1, fall_pose(progress_at(index / 15)), index / 15)
    for time in np.arange(0.8, 1.6, 0.1):
        detector.missing(1, float(time))
    for time in np.arange(1.7, 3, 0.1):
        assert detector.update(1, partial(fall_pose(1)), float(time)).status == "normal"


def test_short_torso_occlusion_preserves_but_does_not_accrue_persistence():
    detector = HeuristicFallDetector()
    for index in range(18):
        detector.update(1, partial(fall_pose(progress_at(index / 15))), index / 15)
    person = detector.histories[1]
    held = person.lying_duration
    for time in (1.2, 1.3):
        pose = partial(fall_pose(1), missing=(5, 6, 13, 14, 15, 16))
        result = detector.update(1, pose, time)
        assert result.status == "possible_fall"
        assert result.features["measurement_valid"] == 0
        assert person.lying_duration == held
    assert detector.update(1, partial(fall_pose(1)), 1.4).status == "possible_fall"


def test_partial_scale_change_during_rotation_does_not_supply_consistent_evidence():
    def pose_at(time):
        pose = partial(fall_pose(progress_at(time)))
        center = pose.keypoints[[11, 12], :2].mean(axis=0)
        factor = 1 + progress_at(time)
        pose.keypoints[:, :2] = center + (pose.keypoints[:, :2] - center) * factor
        return pose

    predictions, events = episode(np.arange(0, 3, 1 / 15), pose_at)
    assert not events and all(p.status != "fall" for p in predictions)
