from types import SimpleNamespace

import numpy as np
import pytest

from src.fall.stgcn import STGCNFallDetector, skeleton_tensor
from tests.helpers import skeleton


def make_pose(x=0.0):
    return skeleton(offset=(300, x * 300))


class Session:
    input_shape = [1, 3, 5, 17, 1]
    provider = "CPUExecutionProvider"

    def __init__(self, output=None):
        self.output = np.array([[0.0, 4.0]], dtype=np.float32) if output is None else output
        self.inputs = []

    def run(self, tensor):
        self.inputs.append(tensor.copy())
        return [self.output]


def test_layout_and_joint_confidence_preserved():
    sequence = np.arange(5 * 17 * 3, dtype=np.float32).reshape(5, 17, 3)
    tensor = skeleton_tensor(sequence)
    assert tensor.shape == (1, 3, 5, 17, 1)
    assert tensor.flags.c_contiguous
    np.testing.assert_array_equal(tensor[0, 2, :, :, 0], sequence[:, :, 2])


def test_inference_and_tracks_are_independent():
    session = Session()
    detector = STGCNFallDetector(
        {"samples": 5, "history_seconds": 1.0, "min_history_seconds": 0.8, "max_gap_seconds": 0.5},
        session,
    )
    for time in (0.0, 0.25, 0.5, 0.75):
        assert detector.update(1, make_pose(time), time).status == "normal"
        detector.update(2, make_pose(time * 2), time)
    result = detector.update(1, make_pose(1.0), 1.0)
    assert result.status == "fall" and result.probability > 0.98
    np.testing.assert_allclose(
        session.inputs[-1][0, 1, :, [11, 12], 0].mean(axis=0), [0, 0.25, 0.5, 0.75, 1.0], atol=1e-6
    )
    detector.update(2, make_pose(2.0), 1.0)
    np.testing.assert_allclose(
        session.inputs[-1][0, 1, :, [11, 12], 0].mean(axis=0), [0, 0.5, 1.0, 1.5, 2.0], atol=1e-6
    )
    detector.remove(1)
    assert 1 not in detector.buffers


def test_missing_person_resets_learned_history():
    detector = STGCNFallDetector({"samples": 5, "history_seconds": 1.0}, Session())
    for time in np.arange(0.0, 1.1, 0.25):
        detector.update(1, make_pose(), float(time))
    assert detector.update(1, make_pose(), 2.0).features["warming_up"] == 1.0


def test_incompatible_layout_rejected():
    session = SimpleNamespace(input_shape=[1, 3, 5, 25, 1])
    with pytest.raises(ValueError, match="conflicts"):
        STGCNFallDetector({"samples": 5}, session)


def test_malformed_probabilities_rejected():
    detector = STGCNFallDetector(
        {"samples": 5, "history_seconds": 0.5, "output_kind": "probabilities"},
        Session(np.array([[0.9, 0.9]])),
    )
    detector.update(1, make_pose(), 0.0)
    with pytest.raises(ValueError, match="sum to one"):
        detector.update(1, make_pose(), 0.5)


def test_real_onnx_runtime_contract(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from src.runtime.onnx_runtime import ORTSession

    input_info = helper.make_tensor_value_info("skeleton", TensorProto.FLOAT, [1, 3, 5, 17, 1])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 2])
    logits = helper.make_tensor("value", TensorProto.FLOAT, [1, 2], [0.0, 4.0])
    node = helper.make_node("Constant", [], ["logits"], value=logits)
    graph = helper.make_graph([node], "test_contract", [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    path = tmp_path / "contract.onnx"
    onnx.save(model, path)
    session = ORTSession(path, {"provider": "cpu", "intra_op_threads": 1})
    detector = STGCNFallDetector({"samples": 5, "history_seconds": 0.5}, session)
    detector.update(1, make_pose(), 0.0)
    assert detector.update(1, make_pose(), 0.5).status == "fall"


@pytest.mark.parametrize("joints", [(5, 6), (11, 12), (15, 16)])
def test_weak_current_pose_cannot_confirm_but_short_gaps_survive(joints):
    session = Session()
    detector = STGCNFallDetector({"samples": 5, "history_seconds": 1.0}, session)
    for time in (0, 0.2, 0.4, 0.6, 0.8):
        detector.update(1, make_pose(time), time)
    before = len(session.inputs)
    weak = make_pose(1)
    weak.keypoints[list(joints), 2] = 0
    result = detector.update(1, weak, 1.0)
    assert result.status == "normal" and result.features["measurement_valid"] == 0
    assert len(session.inputs) == before
    assert detector.update(1, make_pose(1.2), 1.2).status == "fall"


def test_sustained_unreliability_resets_learned_window():
    detector = STGCNFallDetector({"samples": 5, "history_seconds": 1.0}, Session())
    for index in range(6):
        detector.update(1, make_pose(), index / 5)
    weak = make_pose()
    weak.keypoints[:, 2] = 0
    for index in range(6, 10):
        assert detector.update(1, weak, index / 5).features["measurement_valid"] == 0
    assert not detector.buffers[1].samples
    assert detector.update(1, make_pose(), 2.0).features["warming_up"] == 1
    detector.remove(1)
    assert not detector.last_updates and not detector.unreliable_since
