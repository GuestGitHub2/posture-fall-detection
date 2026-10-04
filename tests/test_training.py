import numpy as np

from src.pose.normalization import normalize_pose
from src.pose.pose_types import Pose
from training.dataset import (
    SkeletonRecord,
    augment,
    grouped_split,
    normalize_keypoints,
    resample_sequence,
)
from training.evaluate import classification_metrics, fall_sequence_metrics


def test_normalization_matches_inference():
    points = np.column_stack((np.arange(17), np.arange(17) * 2, np.ones(17))).astype(np.float32)
    np.testing.assert_allclose(
        normalize_keypoints(points),
        normalize_pose(Pose(points, np.array([0.0, 0.0, 16.0, 32.0]))).normalized,
    )


def test_grouped_split_never_leaks_subject():
    points = np.zeros((17, 3), dtype=np.float32)
    records = [SkeletonRecord(str(i), 0, 1, 0.0, points, "normal", str(i // 2)) for i in range(6)]
    train, validation = grouped_split(records, 0.3)
    assert {row.subject_id for row in train}.isdisjoint({row.subject_id for row in validation})


def test_shared_multi_person_recording_never_crosses_split():
    points = np.zeros((17, 3), dtype=np.float32)
    rows = [
        SkeletonRecord("shared", 0, 1, 0.0, points, "normal", "a"),
        SkeletonRecord("shared", 0, 2, 0.0, points, "normal", "b"),
        SkeletonRecord("other", 0, 1, 0.0, points, "normal", "c"),
    ]
    train, validation = grouped_split(rows)
    assert {row.sequence_id for row in train}.isdisjoint({row.sequence_id for row in validation})
    assert {row.subject_id for row in train}.isdisjoint({row.subject_id for row in validation})


def test_long_joint_gaps_are_not_interpolated():
    points = np.ones((2, 17, 3), dtype=np.float32)
    points[0, :, 0], points[1, :, 0] = 0.0, 1.0
    result = resample_sequence(points, np.array([0.0, 1.0]), 3, max_gap_seconds=0.2)
    assert (result[1] == 0).all()
    assert (result[[0, 2], :, 2] == 1).all()


def test_augment_never_revives_missing_joint():
    points = np.ones((17, 3), dtype=np.float32)
    points[0] = 0
    result = augment(points, np.random.default_rng(42), flip_probability=0.0, dropout=0.0)
    assert (result[0] == 0).all()


def test_hard_negative_false_positive_metrics():
    result = classification_metrics(
        ["normal", "normal", "fall", "fall"], ["normal", "fall", "fall", "fall"], ["normal", "fall"]
    )
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 0.5
    assert result["precision"] == 2 / 3


def test_alarm_episodes_and_latency():
    rows = [
        {
            "sequence_id": "bend",
            "timestamp": i * 0.5,
            "label": "normal",
            "prediction": "fall" if i in (1, 2) else "normal",
        }
        for i in range(5)
    ]
    rows += [
        {
            "sequence_id": "fall",
            "timestamp": i * 0.5,
            "label": "fall" if i >= 2 else "normal",
            "fall_onset": 1.0,
            "prediction": "fall" if i >= 3 else "normal",
        }
        for i in range(5)
    ]
    result = fall_sequence_metrics(rows)
    assert result["false_alarms"] == 1
    assert result["mean_detection_latency_seconds"] == 0.5
    assert result["sequence_sensitivity"] == 1.0
    assert result["sequence_specificity"] == 0.0


def test_multi_person_exposure_counts_recording_hours_once():
    rows = [
        {
            "sequence_id": "two_people",
            "track_id": track,
            "timestamp": float(time),
            "label": "normal",
            "prediction": "normal",
        }
        for track in (1, 2)
        for time in (0, 1, 2)
    ]
    result = fall_sequence_metrics(rows)
    assert result["observed_normal_hours"] == 2 / 3600
