import numpy as np

from src.pose.normalization import normalize_pose
from tests.helpers import skeleton


def test_size_and_translation_invariance():
    first = normalize_pose(skeleton())
    second = normalize_pose(skeleton(offset=(1500, 240), scale=2.7))
    np.testing.assert_allclose(first.normalized, second.normalized, atol=1e-6)
    np.testing.assert_allclose(first.normalized[[11, 12], :2].mean(axis=0), 0, atol=1e-6)


def test_rotation_preserves_body_scale():
    first, second = normalize_pose(skeleton()), normalize_pose(skeleton("lying"))
    assert abs(first.body_scale - second.body_scale) < 1e-5


def test_missing_joints_are_zeroed_and_not_nan():
    pose = skeleton()
    pose.keypoints[:, 2] = 0
    result = normalize_pose(pose)
    assert np.isfinite(result.normalized).all()
    assert np.all(result.normalized[:, :2] == 0)
