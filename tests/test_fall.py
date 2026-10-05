import pytest

from src.fall.heuristic import HeuristicFallDetector
from tests.helpers import fall_pose, skeleton


def run_sequence(poses, timestamps, track_id=1, detector=None):
    detector = detector or HeuristicFallDetector()
    return [
        detector.update(track_id, pose, timestamp)
        for pose, timestamp in zip(poses, timestamps, strict=True)
    ]


def test_lying_alone_never_triggers_fall():
    results = run_sequence([skeleton("lying")] * 60, [index / 15 for index in range(60)])
    assert all(result.status == "normal" for result in results)


def test_true_descent_rotation_and_ground_persistence_confirm():
    poses = [fall_pose(min(1, max(0, (index / 15 - 0.4) / 0.4))) for index in range(24)]
    results = run_sequence(poses, [index / 15 for index in range(24)])
    assert any(result.status == "possible_fall" for result in results)
    assert any(result.status == "fall" for result in results)


def test_one_missing_inference_sample_does_not_destroy_fall_evidence():
    indices = [index for index in range(24) if index != 10]
    results = run_sequence(
        [fall_pose(min(1, max(0, (index / 15 - 0.4) / 0.4))) for index in indices],
        [index / 15 for index in indices],
    )
    assert any(result.status == "fall" for result in results)


@pytest.mark.parametrize("negative", ["sitting", "bending", "jump", "slow_lying"])
def test_hard_negative_temporal_regressions(negative):
    poses = []
    for index in range(90):
        time = index / 15
        if negative == "slow_lying":
            pose = fall_pose(min(1, max(0, (time - 0.4) / 3)))
        elif negative == "jump":
            pose = skeleton(offset=(300, -100 if 0.4 < time < 0.9 else 0))
        else:
            pose = skeleton(
                negative if time > 0.4 else "standing",
                offset=(300, 130 if time > 0.4 and negative == "sitting" else 0),
            )
        poses.append(pose)
    assert all(
        result.status != "fall"
        for result in run_sequence(poses, [index / 15 for index in range(90)])
    )


def test_disappearance_or_new_track_cannot_reuse_another_person_fall():
    detector = HeuristicFallDetector()
    for index in range(20):
        detector.update(1, fall_pose(min(1, index / 6)), index / 15)
        assert detector.update(2, skeleton("lying"), index / 15).status == "normal"
    assert detector.update(1, skeleton("lying"), 5).status == "normal"
    detector.remove(1)
    assert 1 not in detector.histories


def test_low_confidence_ankles_use_torso_candidate_without_fabricating_ground():
    detector = HeuristicFallDetector()
    for index in range(12):
        detector.update(1, fall_pose(min(1, max(0, (index / 15 - 0.4) / 0.4))), index / 15)
    occluded = fall_pose(1)
    occluded.keypoints[[15, 16], 2] = 0.1
    result = detector.update(1, occluded, 0.8)
    assert result.status == "possible_fall"
    assert result.features["measurement_valid"] == 1
    assert result.features["floor_available"] == 0
    results = [detector.update(1, fall_pose(1), index / 15) for index in range(13, 30)]
    assert results[0].status == "possible_fall"
    assert any(result.status == "fall" for result in results)


def test_bending_to_tie_shoes_does_not_supply_ground_evidence():
    detector = HeuristicFallDetector()
    for index in range(30):
        pose = skeleton(
            "bending" if index > 5 else "standing", offset=(300, 100 if index > 5 else 0)
        )
        assert detector.update(1, pose, index / 15).status != "fall"
