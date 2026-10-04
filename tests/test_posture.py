import pytest

from src.posture.classifier import PostureClassifier
from tests.helpers import skeleton


@pytest.mark.parametrize("label", ["standing", "sitting", "bending", "lying"])
@pytest.mark.parametrize("scale,offset", [(1, (300, 0)), (0.4, (100, 20)), (2.4, (800, 350))])
def test_required_postures_at_different_sizes(label, scale, offset):
    result = PostureClassifier().classify(skeleton(label, offset, scale))
    assert result.label == label
    assert result.confidence > 0.8


def test_low_confidence_or_missing_legs_yields_unknown():
    pose = skeleton()
    pose.keypoints[:, 2] = 0.1
    assert PostureClassifier().classify(pose).label == "unknown"
    pose = skeleton()
    pose.keypoints[13:, 2] = 0
    assert PostureClassifier().classify(pose).label == "unknown"
