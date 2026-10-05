import json

import numpy as np
import pytest

from src.fall.normalization import FALL_NORMALIZATION, normalize_fall_sequence
from src.fall.stgcn import STGCNFallDetector
from src.fall.temporal_buffer import TemporalBuffer
from src.pose.normalization import StableBodyScale, normalize_pose
from src.posture.classifier import PostureClassifier
from src.tracking.tracker import Tracker
from tests.helpers import skeleton
from tests.test_stgcn import Session
from training.dataset import SkeletonRecord, fall_windows


@pytest.mark.parametrize("missing", [(15, 16), (5, 6), (11, 12)])
def test_stable_scale_rejects_one_frame_fallback(missing):
    tracker = Tracker()
    first = tracker.update([skeleton()], 0)[0].pose
    partial = skeleton()
    partial.keypoints[list(missing), 2] = 0
    result = tracker.update([partial], 0.1)[0].pose
    assert result.scale_quality < first.scale_quality
    assert result.body_scale == pytest.approx(first.body_scale)
    assert result.scale_is_stable
    assert tracker.update([skeleton()], 0.2)[0].pose.body_scale == pytest.approx(first.body_scale)


def test_no_ankle_normalization_jump_in_visible_joints():
    scale = StableBodyScale()
    first = scale.apply(skeleton(), 0)
    pose = skeleton()
    pose.keypoints[[15, 16], 2] = 0
    weak = scale.apply(pose, 0.1)
    np.testing.assert_allclose(first.normalized[:15], weak.normalized[:15], atol=1e-6)
    assert weak.scale_source == "torso" and weak.scale_quality < 0.6


def test_window_normalization_preserves_root_descent_and_invariance():
    poses = [skeleton(offset=(300, time * 60)) for time in range(4)]
    points, transform = normalize_fall_sequence(poses)
    assert transform.scale == 300 and transform.reliable
    np.testing.assert_allclose(points[:, [11, 12], 1].mean(axis=1), [0, 0.2, 0.4, 0.6], atol=1e-6)
    other = [skeleton(offset=(1600, 250 + time * 60 * 2.5), scale=2.5) for time in range(4)]
    np.testing.assert_allclose(points, normalize_fall_sequence(other)[0], atol=1e-6)
    # Posture normalization remains per-frame centered.
    assert all(
        np.allclose(normalize_pose(pose).normalized[[11, 12], :2].mean(axis=0), 0) for pose in poses
    )


def test_first_reliable_root_and_short_gap_interpolation():
    poses = [skeleton(offset=(300, offset)) for offset in (0, 30, 60)]
    poses[0].keypoints[[11, 12], 2] = 0
    points, transform = normalize_fall_sequence(poses)
    np.testing.assert_allclose(transform.origin, [300, 230])
    np.testing.assert_allclose(points[2, [11, 12], 1], [0.1, 0.1])
    buffer = TemporalBuffer(0.2, 20, 0.3)
    for index, pose in enumerate(poses):
        buffer.append(pose, index / 10)
    assert np.isfinite(buffer.resample_fall(5)).all()


def test_training_and_runtime_windows_are_identical_and_preserve_translation():
    poses = [skeleton(offset=(300, index * 30)) for index in range(11)]
    records = [
        SkeletonRecord("fall", index, 1, index / 10, pose.keypoints, "fall", bbox=pose.bbox)
        for index, pose in enumerate(poses)
    ]
    data, labels, metadata = fall_windows(records, samples=5, history_seconds=1, stride_seconds=1)
    buffer = TemporalBuffer(1, 90, 0.5)
    for index, pose in enumerate(poses):
        buffer.append(pose, index / 10)
    np.testing.assert_allclose(data[0], buffer.resample_fall(5), atol=1e-6)
    np.testing.assert_allclose(
        data[0, :, [11, 12], 1].mean(axis=0), [0, 0.25, 0.5, 0.75, 1.0], atol=1e-6
    )
    assert labels.tolist() == [1]
    assert metadata[0]["normalization"] == FALL_NORMALIZATION


def test_training_windows_never_bridge_identity_epochs_or_long_gaps():
    records = [
        SkeletonRecord(
            "video",
            index,
            1,
            index / 10,
            skeleton().keypoints,
            "normal",
            history_epoch=int(index >= 6),
        )
        for index in range(12)
    ]
    with pytest.raises(ValueError, match="No full temporal"):
        fall_windows(records, history_seconds=1)
    for row in records:
        object.__setattr__(row, "history_epoch", 0)
    for row in records[6:]:
        object.__setattr__(row, "timestamp", row.timestamp + 2)
    with pytest.raises(ValueError, match="No full temporal"):
        fall_windows(records, history_seconds=1, max_gap_seconds=0.2)


def test_old_model_normalization_is_rejected_in_metadata(tmp_path):
    model = tmp_path / "fall.onnx"
    model.with_suffix(".metadata.json").write_text(
        json.dumps({"normalization": "hip_center_torso_plus_leg_length"})
    )
    with pytest.raises(ValueError, match="Incompatible fall normalization"):
        STGCNFallDetector({"samples": 5, "model_path": str(model)}, Session())


def test_rule_confidence_distinguishes_boundary_and_pose_quality():
    classifier = PostureClassifier()
    clear = classifier.classify(skeleton())
    from src.posture.rules import POSTURE_DEFAULTS, classify_features

    features = dict(clear.features)
    features["torso_angle"] = POSTURE_DEFAULTS["standing_torso_angle"] - 0.1
    label, boundary = classify_features(features, POSTURE_DEFAULTS)
    assert label == "standing" and 0.4 < boundary < clear.confidence
    assert clear.pose_quality == pytest.approx(0.95)
    assert clear.features["pose_quality"] == clear.pose_quality


def test_metadata_preprocessing_mismatch_is_rejected(tmp_path):
    model = tmp_path / "fall.onnx"
    metadata = {
        "normalization": FALL_NORMALIZATION,
        "classes": ["normal", "fall"],
        "samples": 5,
        "history_seconds": 1.0,
        "layout": "NCTVM",
        "minimum_joint_confidence": 0.6,
    }
    model.with_suffix(".metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="minimum_joint_confidence conflicts"):
        STGCNFallDetector(
            {"samples": 5, "history_seconds": 1.0, "model_path": str(model)}, Session()
        )


def test_training_windows_follow_frame_order_across_timestamp_regression():
    records = [
        SkeletonRecord("video", index, 1, index / 10, skeleton().keypoints, "normal")
        for index in range(6)
    ]
    records += [
        SkeletonRecord("video", index, 1, 0.21 + (index - 6) * 0.16, skeleton().keypoints, "normal")
        for index in range(6, 13)
    ]
    with pytest.raises(ValueError, match="No full temporal"):
        fall_windows(records, history_seconds=1)


def test_malformed_metadata_has_clear_value_error(tmp_path):
    model = tmp_path / "fall.onnx"
    model.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "normalization": FALL_NORMALIZATION,
                "classes": ["normal", "fall"],
                "samples": 5,
                "layout": "NCTVM",
                "history_seconds": None,
            }
        )
    )
    with pytest.raises(ValueError, match="history_seconds conflicts"):
        STGCNFallDetector({"samples": 5, "model_path": str(model)}, Session())


def test_old_fall_checkpoint_is_rejected_before_evaluation(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from training.evaluate import predict_checkpoint

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            load=lambda *args, **kwargs: {
                "metadata": {"kind": "fall", "normalization": "hip_center_torso_plus_leg_length"}
            }
        ),
    )
    with pytest.raises(ValueError, match="per-frame centered weights must be retrained"):
        predict_checkpoint(tmp_path / "data.jsonl", tmp_path / "old.pt", "fall", 0.8)
