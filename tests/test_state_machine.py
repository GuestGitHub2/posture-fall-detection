from src.fall.detector import FallResult
from src.posture.classifier import PostureResult
from src.state.state_machine import PersonStateMachine


def test_static_lying_is_lying_and_posture_requires_consecutive_votes():
    machine = PersonStateMachine()
    normal, lying = FallResult(), PostureResult("lying", 0.95)
    assert machine.update(lying, normal, 0).state == "UNKNOWN"
    assert machine.update(lying, normal, 0.1).state == "UNKNOWN"
    assert machine.update(lying, normal, 0.2).state == "LYING"


def test_fall_confirmation_alert_latched_and_recovers_once():
    machine = PersonStateMachine({"recovery_seconds": 0.3})
    lying, fall = PostureResult("lying", 0.95), FallResult("fall", 0.95)
    assert not machine.update(lying, fall, 0).fall_event
    assert not machine.update(lying, fall, 0.1).fall_event
    result = machine.update(lying, fall, 0.2)
    assert result.state == "FALLEN" and result.fall_event
    assert not machine.update(lying, FallResult(), 0.3).fall_event
    assert machine.missing(1).state == "FALLEN"
    standing = PostureResult("standing", 0.95)
    assert machine.update(standing, FallResult(), 1.1).state == "RECOVERING"
    machine.update(standing, FallResult(), 1.2)
    machine.update(standing, FallResult(), 1.3)
    result = machine.update(standing, FallResult(), 1.5)
    assert result.state == "STANDING" and result.recovered_event


def test_single_high_prediction_and_large_gap_do_not_confirm():
    machine = PersonStateMachine()
    lying, fall = PostureResult("lying", 0.95), FallResult("fall", 0.95)
    machine.update(lying, fall, 0)
    machine.update(lying, FallResult(), 0.1)
    assert not machine.update(lying, fall, 0.2).fall_event
    assert not machine.update(lying, fall, 3).fall_event


def test_low_confidence_pose_cannot_recover_fallen_person():
    machine = PersonStateMachine({"confirmation_frames": 1, "recovery_seconds": 0.1})
    machine.update(PostureResult("lying", 0.95), FallResult("fall", 0.95), 0)
    for index in range(1, 10):
        assert (
            machine.update(PostureResult("standing", 0.1), FallResult(), index / 10).state
            == "FALLEN"
        )


def test_high_classification_confidence_does_not_override_low_pose_quality():
    machine = PersonStateMachine()
    weak = PostureResult("lying", 0.99, pose_quality=0.1)
    for index in range(12):
        state = machine.update(weak, FallResult("fall", 0.99), index / 10)
        assert not state.fall_event and state.state != "FALLEN"
    machine = PersonStateMachine({"confirmation_frames": 1, "recovery_seconds": 0.1})
    machine.update(PostureResult("lying", 0.95), FallResult("fall", 0.95), 0)
    for index in range(1, 12):
        state = machine.update(
            PostureResult("standing", 0.99, pose_quality=0.1), FallResult(), index / 10
        )
        assert state.state == "FALLEN" and not state.recovered_event


def test_confident_posture_remains_independent_of_unusable_fall_geometry():
    machine = PersonStateMachine()
    for index in range(12):
        result = machine.update(
            PostureResult("lying", 0.9, pose_quality=0.7),
            FallResult("normal", 0, {"measurement_valid": 0}),
            index / 10,
        )
        assert not result.fall_event
    assert result.state == "LYING"
