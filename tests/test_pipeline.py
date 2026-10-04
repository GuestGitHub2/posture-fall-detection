from src.config import load_config
from src.events import EventBus
from src.pipeline import Pipeline
from tests.helpers import fall_pose, skeleton


def test_tracked_fall_emits_once_and_other_lying_person_is_independent() -> None:
    config = load_config()
    events = []
    bus = EventBus()
    bus.subscribe(events.append)
    pipeline = Pipeline(config, event_bus=bus, load_pose=False)
    states = []
    for index in range(45):
        timestamp = index / 15
        pose = fall_pose(min(1, max(0, (timestamp - 0.4) / 0.4)))
        poses = [pose, skeleton("lying", offset=(1100, 0))]
        if index == 10:
            poses = poses[1:]
        if index % 2:
            poses.reverse()
        result = pipeline.process_poses(poses, timestamp)
        assert {person.track_id for person in result.persons} == {1, 2}
        assert result.persons[1].fall.status == "normal"
        assert result.persons[1].state != "FALLEN"
        states.append(result.persons[0].state)
    assert "FALLEN" in states
    assert len([event for event in events if event["type"] == "fall_detected"]) == 1
    assert events[0]["track_id"] == 1
    assert pipeline.process_poses([], 5).persons == []
    assert not pipeline.fall_detector.histories


def test_configuration_controls_temporal_confirmation() -> None:
    config = load_config()
    config["fall"]["confirmation_frames"] = 100
    pipeline = Pipeline(config, event_bus=EventBus(), load_pose=False)
    for index in range(35):
        timestamp = index / 15
        result = pipeline.process_poses(
            [fall_pose(min(1, max(0, (timestamp - 0.4) / 0.4)))], timestamp
        )
        assert result.persons[0].state != "FALLEN"
