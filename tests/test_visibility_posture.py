"""Partial observations and rule-boundary noise are normal camera conditions."""

import numpy as np
import pytest

from src.pose.normalization import StableBodyScale, normalize_pose
from src.pose.visibility import VisibilityMode, assess_visibility
from src.posture.classifier import PostureClassifier
from src.posture.mlp import MLPPostureClassifier
from tests.helpers import skeleton
from tests.test_posture_mlp import Session


def partial(pose, missing=(13, 14, 15, 16)):
    pose.keypoints[list(missing), 2] = 0
    return pose


def torso_angle(angle, torso_only=False):
    pose = skeleton()
    center = pose.keypoints[[11, 12], :2].mean(axis=0)
    vector = np.array([np.sin(np.deg2rad(angle)), -np.cos(np.deg2rad(angle))]) * 100
    pose.keypoints[[5, 6], :2] = center + vector + [[-25, 0], [25, 0]]
    return partial(pose) if torso_only else pose


@pytest.mark.parametrize("label", ["standing", "sitting", "bending", "lying"])
def test_full_body_keeps_precise_classes(label):
    result = PostureClassifier().classify(skeleton(label))
    assert result.label == label
    assert result.visibility.mode == VisibilityMode.FULL_BODY


@pytest.mark.parametrize(
    "angle,label",
    [
        (0, "upright_partial"),
        (28.29, "upright_partial"),
        (45, "bent_partial"),
        (90, "horizontal_partial"),
    ],
)
def test_torso_geometry_without_lower_body(angle, label):
    pose = torso_angle(angle, True)
    before = pose.keypoints.copy()
    result = PostureClassifier().classify(pose)
    assert result.visibility.mode == VisibilityMode.TORSO
    assert result.label == label and result.confidence >= 0.4
    assert result.pose_quality == pytest.approx(0.95)
    assert result.features["torso_quality"] == pytest.approx(0.95)
    np.testing.assert_array_equal(pose.keypoints, before)


@pytest.mark.parametrize(
    "missing,mode",
    [
        ((15, 16), VisibilityMode.LOWER_PARTIAL),
        ((13, 14, 15, 16), VisibilityMode.TORSO),
        (tuple(range(11, 17)), VisibilityMode.UPPER_ONLY),
        (tuple(range(17)), VisibilityMode.INSUFFICIENT),
    ],
)
def test_group_observability_is_not_total_joint_count(missing, mode):
    pose = partial(skeleton(), missing)
    result = PostureClassifier().classify(pose)
    assert result.visibility.mode == mode
    if mode in {VisibilityMode.UPPER_ONLY, VisibilityMode.INSUFFICIENT}:
        assert result.label == "unknown" and result.confidence == 0
    else:
        assert result.label == "upright_partial"


@pytest.mark.parametrize("angle", [27.1, 27.9, 28.0, 28.29, 28.9, 29.1])
def test_no_standing_threshold_cliff(angle):
    result = PostureClassifier().classify(torso_angle(angle))
    assert result.label == "standing" and result.confidence >= 0.4


def test_classification_hysteresis_is_per_track_and_precedes_state_smoothing():
    classifier = PostureClassifier()
    labels = [
        classifier.classify_tracked(torso_angle(angle, True), 1).label
        for angle in [27.8, 28.3, 27.9, 32.2, 31.7, 32.1]
    ]
    assert set(labels) == {"upright_partial"}
    assert classifier.classify_tracked(torso_angle(45, True), 2).label == "bent_partial"
    classifier.remove(1)
    assert classifier.classify_tracked(torso_angle(32.2, True), 1).label == "bent_partial"


def test_full_partial_full_changes_evidence_tier_without_holding_precise_label():
    classifier = PostureClassifier()
    labels = [
        classifier.classify_tracked(pose, 1).label
        for pose in (skeleton(), partial(skeleton()), skeleton())
    ]
    assert labels == ["standing", "upright_partial", "standing"]


def test_confident_outside_frame_joints_are_not_observations():
    pose = skeleton()
    pose.image_size = (640, 250)
    result = PostureClassifier().classify(pose)
    assert result.visibility.mode == VisibilityMode.TORSO
    assert result.visibility.counts["ankles"] == 0
    assert result.label == "upright_partial"
    assert pose.keypoints[15, 2] == pytest.approx(0.95)


def test_partial_scale_ema_does_not_become_reliable_full_body_scale():
    scale = StableBodyScale()
    first = scale.apply(partial(skeleton()), 0)
    next_pose = partial(skeleton())
    next_pose.keypoints[[5, 6], 1] -= 40
    second = scale.apply(next_pose, 1 / 30)
    assert second.body_scale / first.body_scale < 1.03
    assert scale.value is None  # Tracking scale gates retain their previous contract.
    assert second.scale_quality == 0.35
    assert normalize_pose(next_pose).image_size == next_pose.image_size


def test_normalization_preserves_low_confidence_and_masks_only_outside_estimates():
    pose = skeleton()
    pose.keypoints[0, 2] = 0.1
    pose.image_size = (640, 250)
    normalized = normalize_pose(pose)
    assert normalized.normalized[0, 2] == pytest.approx(0.1)
    assert (normalized.normalized[13:] == 0).all()
    assert (pose.keypoints[13:, 2] > 0).all()


def test_mlp_uses_observable_rule_fallback_without_fabricating_model_inputs():
    session = Session([[0, 4]])
    classifier = MLPPostureClassifier({"classes": ["sitting", "standing"]}, session=session)
    result = classifier.classify(partial(skeleton()))
    assert result.label == "upright_partial" and not session.called


def test_mlp_cannot_assert_a_precise_posture_without_observable_torso():
    session = Session([[0, 4]])
    classifier = MLPPostureClassifier({"classes": ["sitting", "standing"]}, session=session)
    result = classifier.classify(partial(skeleton(), missing=(5, 6)))
    assert result.label == "unknown" and not session.called


def test_one_sided_reliable_torso_uses_only_that_side():
    pose = partial(skeleton(), (6, 12, 13, 14, 15, 16))
    pose.keypoints[[6, 12], :2] = 10000
    visibility = assess_visibility(pose)
    assert visibility.torso_observable
    assert PostureClassifier().classify(pose).label == "upright_partial"
