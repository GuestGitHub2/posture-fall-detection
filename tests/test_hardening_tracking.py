"""Association uncertainty must cut evidence without inventing identity."""

import numpy as np

from src.config import load_config
from src.events import EventBus
from src.fall.detector import FallResult
from src.pipeline import Pipeline
from src.posture.classifier import PostureResult
from src.state.state_machine import PersonStateMachine
from src.tracking.tracker import Tracker
from tests.helpers import fall_pose, skeleton


def test_full_overlap_abstains_and_invalidates_both_histories():
    config = load_config()
    pipeline = Pipeline(config, event_bus=EventBus(), load_pose=False)
    pipeline.process_poses([skeleton(offset=(250, 0)), skeleton(offset=(350, 0))], 0)
    pipeline.process_poses([skeleton(offset=(280, 0)), skeleton(offset=(320, 0))], 0.1)
    assert set(pipeline.fall_detector.histories) == {1, 2}
    result = pipeline.process_poses([skeleton(offset=(300, 0)), skeleton(offset=(300, 0))], 0.2)
    assert len(result.persons) == 2
    assert all(not person.observed and person.association_ambiguous for person in result.persons)
    assert all(person.state == "UNKNOWN" for person in result.persons)
    assert not pipeline.fall_detector.histories
    assert all(track.history_epoch == 1 for track in pipeline.tracker.tracks.values())
    result = pipeline.process_poses([skeleton(offset=(260, 0)), skeleton(offset=(340, 0))], 0.3)
    assert all(person.fall.status == "normal" for person in result.persons)
    for history in pipeline.fall_detector.histories.values():
        assert len(history.motion) == 1


def test_merged_detection_cannot_be_given_to_either_identical_person():
    tracker = Tracker()
    tracker.update([skeleton(offset=(280, 0)), skeleton(offset=(320, 0))], 0)
    tracks = tracker.update([skeleton(offset=(300, 0))], 0.1)
    assert len(tracks) == 2
    assert all(not track.observed and track.association_ambiguous for track in tracks)
    assert tracker._next_id == 3


def test_partial_overlap_with_reliable_prediction_keeps_ids():
    tracker = Tracker()
    for index in range(9):
        x1, x2 = 200 + 25 * index, 400 - 25 * index
        poses = [skeleton(offset=(x1, 0)), skeleton(offset=(x2, 30))]
        if index % 2:
            poses.reverse()
        tracks = tracker.update(poses, index / 10)
        assert len(tracks) == 2 and all(track.observed for track in tracks)
        assert tracks[0].pose.keypoints[11, 0] == x1 - 20
        assert tracks[1].pose.keypoints[11, 0] == x2 - 20


def test_scale_and_center_gates_reject_unrelated_person():
    tracker = Tracker()
    tracker.update([skeleton()], 0)
    tracks = tracker.update([skeleton(offset=(-300, -200), scale=3)], 0.1)
    assert not tracks[0].observed and tracks[1].track_id == 2


def test_expiration_and_timestamp_regression_never_reuse_ids():
    tracker = Tracker({"max_missing_seconds": 0.3})
    tracker.update([skeleton()], 0)
    tracker.update([], 0.4)
    assert tracker.update([skeleton()], 0.5)[0].track_id == 2
    assert tracker.update([skeleton()], 0.1)[0].track_id == 3


def test_one_person_falls_while_another_crosses_and_briefly_occludes():
    pipeline = Pipeline(load_config(), event_bus=EventBus(), load_pose=False)
    events = []
    pipeline.event_bus.subscribe(events.append)
    for index in range(42):
        timestamp = index / 15
        fallen = fall_pose(min(1, max(0, (timestamp - 0.4) / 0.4)))
        walking = skeleton(offset=(750 - index * 15, 10))
        poses = [fallen, walking] if index not in (10, 11) else [walking]
        if index % 2:
            poses.reverse()
        result = pipeline.process_poses(poses, timestamp)
        for person in result.persons:
            if person.observed and np.allclose(
                person.pose.keypoints[:, :2], walking.keypoints[:, :2]
            ):
                assert person.state != "FALLEN"
                assert person.fall.status == "normal"
    assert len(events) <= 1
    assert all(event["track_id"] == 1 for event in events)


def test_recovered_person_rearms_and_second_fall_ignores_old_cooldown():
    machine = PersonStateMachine(
        {
            "recovery_seconds": 0.3,
            "rearm_seconds": 0.3,
            "fall_confirmation_seconds": 0.2,
            "event_cooldown_seconds": 100,
        }
    )
    normal, fall = FallResult(), FallResult("fall", 0.95)
    lying, standing = PostureResult("lying", 0.95), PostureResult("standing", 0.95)
    events, recovered = [], []
    for index in range(35):
        timestamp = index / 10
        is_fall = timestamp < 0.5 or timestamp >= 1.5
        state = machine.update(
            lying if is_fall else standing, fall if is_fall else normal, timestamp
        )
        if state.fall_event:
            events.append(timestamp)
        if state.recovered_event:
            recovered.append(timestamp)
    assert len(events) == 2 and len(recovered) == 1
    assert events[1] - events[0] < 100


def test_relapse_before_upright_rearming_is_the_same_episode():
    machine = PersonStateMachine({"recovery_seconds": 0.2, "rearm_seconds": 1.0})
    events = []
    for index in range(30):
        time = index / 10
        fallen = time < 0.5 or time >= 0.9
        result = machine.update(
            PostureResult("lying" if fallen else "standing", 0.95),
            FallResult("fall", 0.95) if fallen else FallResult(),
            time,
        )
        events.append(result.fall_event)
    assert sum(events) == 1


def test_loss_reacquisition_cannot_duplicate_alert_even_with_new_predictions():
    config = load_config()
    pipeline = Pipeline(config, event_bus=EventBus(), load_pose=False)
    events = []
    pipeline.event_bus.subscribe(events.append)
    for index in range(32):
        timestamp = index / 15
        pipeline.process_poses([fall_pose(min(1, max(0, (timestamp - 0.4) / 0.4)))], timestamp)
    assert len(events) == 1
    pipeline.process_poses([], 4)
    result = pipeline.process_poses([fall_pose(1)], 4.1)
    new_id = result.persons[0].track_id
    assert new_id != 1 and not pipeline._machines[new_id].armed
    # A learned detector could produce repeated positives on the reacquired body.
    # Episode deduplication must still hold, without importing the old skeleton buffer.
    pipeline.fall_detector.update = lambda *args: FallResult("fall", 0.95)
    for index in range(1, 12):
        pipeline.process_poses([fall_pose(1)], 4.1 + index / 15)
    assert len(events) == 1
    assert pipeline._snapshots[new_id].state == "FALLEN"
    assert events[0]["episode_id"] == 1


def test_independent_continuous_people_can_alert_near_one_another():
    from src.state.episodes import EpisodeRegistry

    registry = EpisodeRegistry()
    first, second = fall_pose(1), fall_pose(1)
    assert registry.confirm(1, first, 0) == 1
    registry.mark_owners({1: 0, 2: 0})
    assert not registry.quarantine(2, second)
    assert registry.confirm(2, second, 0.1) == 2
    registry.mark_owners({2: 0})
    # The already established second person cannot rearm the lost first person's episode.
    registry.observe(2, second, 0.2, True)
    assert not registry.episodes[1].rearmed


def test_uncertain_reacquisition_requires_recovery_plus_rearm_hold():
    config = load_config()
    config["fall"]["recovery_seconds"] = 0.3
    config["state"]["rearm_seconds"] = 0.3
    pipeline = Pipeline(config, event_bus=EventBus(), load_pose=False)
    for index in range(32):
        timestamp = index / 15
        pipeline.process_poses([fall_pose(min(1, max(0, (timestamp - 0.4) / 0.4)))], timestamp)
    pipeline.process_poses([], 4)
    result = pipeline.process_poses([skeleton()], 4.1)
    track_id = result.persons[0].track_id
    assert not pipeline._machines[track_id].armed
    for index in range(1, 6):
        pipeline.process_poses([skeleton()], 4.1 + index / 10)
        assert not pipeline._machines[track_id].armed
    pipeline.process_poses([skeleton()], 4.8)
    assert pipeline._machines[track_id].armed


def test_pipeline_recovery_allows_a_second_physical_fall():
    config = load_config()
    config["fall"]["recovery_seconds"] = 0.3
    config["state"]["rearm_seconds"] = 0.3
    pipeline = Pipeline(config, event_bus=EventBus(), load_pose=False)
    events = []
    pipeline.event_bus.subscribe(events.append)
    for index in range(95):
        time = index / 15
        if time < 2:
            progress = min(1, max(0, (time - 0.4) / 0.4))
        elif time < 3:
            progress = 3 - time
        elif time < 4.2:
            progress = 0
        else:
            progress = min(1, (time - 4.2) / 0.4)
        pipeline.process_poses([fall_pose(progress)], time)
    falls = [event for event in events if event["type"] == "fall_detected"]
    assert len(falls) == 2
    assert falls[0]["track_id"] == falls[1]["track_id"] == 1
    assert [event["episode_id"] for event in falls] == [1, 2]
    assert falls[1]["source_timestamp"] - falls[0]["source_timestamp"] < 10


def test_time_based_occlusion_retention_at_all_inference_rates():
    for fps in (5, 10, 15, 30):
        tracker = Tracker(load_config()["tracking"])
        tracker.update([skeleton()], 0)
        for time in np.arange(1 / fps, 0.85, 1 / fps):
            assert tracker.update([], float(time))[0].track_id == 1
        assert tracker.update([skeleton()], 0.9)[0].track_id == 1


def test_partial_lying_body_with_missing_ankles_has_posture_without_fall():
    pipeline = Pipeline(load_config(), event_bus=EventBus(), load_pose=False)
    for index in range(30):
        pose = skeleton("lying")
        pose.keypoints[[15, 16], 2] = 0
        result = pipeline.process_poses([pose], index / 15)
        assert result.persons[0].fall.status == "normal"
    assert result.persons[0].state == "HORIZONTAL_PARTIAL"


def test_scale_gate_retains_reliable_reference_after_one_weak_measurement():
    tracker = Tracker()
    tracker.update([skeleton()], 0)
    weak = skeleton()
    weak.keypoints[[15, 16], 2] = 0
    tracker.update([weak], 0.1)
    # Same root location, but incompatible body size; one weak ankle frame must
    # not disable the reliable scale gate and transfer history to this person.
    larger = skeleton(offset=(300, -220), scale=2.1)
    tracks = tracker.update([larger], 0.2)
    assert not tracks[0].observed
    assert tracks[1].track_id == 2
