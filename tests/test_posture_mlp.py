import numpy as np
import pytest

from src.posture.mlp import MLPPostureClassifier
from tests.helpers import skeleton


class Session:
    input_shape = ["batch", 17, 3]

    def __init__(self, outputs):
        self.outputs = outputs
        self.called = False

    def run(self, inputs):
        self.called = True
        assert inputs.shape == (1, 17, 3)
        assert inputs.dtype == np.float32
        np.testing.assert_allclose(inputs[0, [11, 12], :2].mean(axis=0), 0, atol=1e-6)
        return [np.asarray(self.outputs, dtype=np.float32)]


def test_trained_class_order_and_common_pose_contract():
    classifier = MLPPostureClassifier(
        {"classes": ["sitting", "standing"]}, session=Session([[0, 4]])
    )
    assert classifier.classify(skeleton()).label == "standing"


def test_no_usable_pose_skips_learned_inference():
    session = Session([[0, 4]])
    classifier = MLPPostureClassifier({"classes": ["sitting", "standing"]}, session=session)
    pose = skeleton()
    pose.keypoints[:, 2] = 0
    assert classifier.classify(pose).label == "unknown"
    assert not session.called


@pytest.mark.parametrize("logits", [[[float("nan"), 0]], [[0, 1, 2]]])
def test_bad_model_output_fails_with_clear_contract_error(logits):
    classifier = MLPPostureClassifier({"classes": ["sitting", "standing"]}, session=Session(logits))
    with pytest.raises(ValueError, match="finite"):
        classifier.classify(skeleton())
