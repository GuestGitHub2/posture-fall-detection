"""Visibility must remain per person and raw/final labels must be explicit."""

import numpy as np
import pytest

from src.config import load_config
from src.events import EventBus
from src.pipeline import Pipeline
from src.visualization.renderer import Renderer
from tests.helpers import fall_pose, skeleton
from tests.test_hardening_fall import progress_at
from tests.test_visibility_posture import partial


def make_pipeline():
    return Pipeline(load_config(), event_bus=EventBus(), load_pose=False)


def test_full_body_and_partial_people_have_independent_outputs_and_histories():
    pipeline = make_pipeline()
    for index in range(12):
        result = pipeline.process_poses(
            [skeleton(offset=(250, 0)), partial(skeleton(offset=(900, 0)))], index / 15
        )
    first, second = result.persons
    assert (first.posture.label, first.state) == ("standing", "STANDING")
    assert (second.posture.label, second.state) == ("upright_partial", "UPRIGHT_PARTIAL")
    assert first.posture.visibility.mode == "FULL_BODY"
    assert second.posture.visibility.mode == "TORSO"
    assert set(pipeline.fall_detector.histories) == {1, 2}
    assert pipeline.fall_detector.histories[1].motion[0].posture.features["hip_x"] == 250
    assert pipeline.fall_detector.histories[2].motion[0].posture.features["hip_x"] == 900


def test_full_to_partial_downgrades_precise_final_posture_immediately():
    pipeline = make_pipeline()
    for index in range(10):
        pipeline.process_poses([skeleton()], index / 15)
    result = pipeline.process_poses([partial(skeleton())], 10 / 15)
    assert result.persons[0].smoothed_posture == "UPRIGHT_PARTIAL"
    for index in range(11, 21):
        result = pipeline.process_poses([skeleton()], index / 15)
    assert result.persons[0].state == "STANDING"


def test_partial_crossing_retains_clear_matches_without_history_mixing():
    pipeline = make_pipeline()
    for index in range(9):
        x1, x2 = 200 + 25 * index, 400 - 25 * index
        poses = [partial(skeleton(offset=(x1, 0))), partial(skeleton(offset=(x2, 30)))]
        if index % 2:
            poses.reverse()
        result = pipeline.process_poses(poses, index / 10)
        assert len(result.persons) == 2
        assert all(p.observed for p in result.persons)
        assert result.persons[0].pose.keypoints[11, 0] == x1 - 20
        assert result.persons[1].pose.keypoints[11, 0] == x2 - 20
    assert pipeline.fall_detector.histories[1].motion[0].posture.features["hip_y"] == 200
    assert pipeline.fall_detector.histories[2].motion[0].posture.features["hip_y"] == 230


def test_ambiguous_partial_overlap_clears_motion_and_classifier_hysteresis():
    pipeline = make_pipeline()
    pipeline.process_poses(
        [partial(skeleton(offset=(280, 0))), partial(skeleton(offset=(320, 0)))], 0
    )
    result = pipeline.process_poses([partial(skeleton()), partial(skeleton())], 0.1)
    assert all(p.association_ambiguous and not p.observed for p in result.persons)
    assert not pipeline.fall_detector.histories
    assert not pipeline.posture_classifier._labels


def test_partial_fall_with_crossing_person_cannot_alert_for_the_passer():
    pipeline = make_pipeline()
    events = []
    pipeline.event_bus.subscribe(events.append)
    for index in range(48):
        time = index / 15
        fallen = partial(fall_pose(progress_at(time)))
        walking = partial(skeleton(offset=(750 - index * 15, 10)))
        poses = [fallen, walking] if index not in (10, 11) else [walking]
        if index % 2:
            poses.reverse()
        result = pipeline.process_poses(poses, time)
        for person in result.persons:
            if person.observed and np.allclose(
                person.pose.keypoints[:, :2], walking.keypoints[:, :2]
            ):
                assert person.state != "FALLEN" and person.fall.status == "normal"
    assert len(events) == 1
    assert all(event["track_id"] == 1 for event in events)


def test_lost_strong_candidate_shows_possible_without_emitting_event():
    pipeline = make_pipeline()
    pipeline.tracker.config["max_missing_seconds"] = 2.0
    events = []
    pipeline.event_bus.subscribe(events.append)
    for index in range(12):
        pipeline.process_poses([fall_pose(progress_at(index / 15))], index / 15)
    for time in np.arange(0.8, 1.4, 0.1):
        result = pipeline.process_poses([], float(time))
        assert result.persons[0].state == "FALLING"
        assert result.persons[0].fall.status == "possible_fall"
    assert not events
    for time in np.arange(1.4, 2.0, 0.1):
        result = pipeline.process_poses([], float(time))
    assert result.persons[0].state == "UNKNOWN"
    assert result.persons[0].fall.status == "normal"


@pytest.mark.parametrize("stale", [False, True])
def test_renderer_distinguishes_raw_final_visibility_and_keeps_age_gate(monkeypatch, stale):
    pipeline = make_pipeline()
    result = pipeline.process_poses([partial(skeleton())], 0)
    labels = []
    renderer = Renderer(load_config()["visualization"] | {"debug": True})
    monkeypatch.setattr(renderer, "_text", lambda image, text, *_: labels.append(text))
    image = np.zeros((720, 1280, 3), np.uint8)
    rendered = renderer.render(image, result, {}, 0.3 if stale else 0)
    np.testing.assert_array_equal(image, np.zeros_like(image))
    assert rendered.shape == image.shape
    if stale:
        assert not any(label.startswith("Raw:") for label in labels)
    else:
        assert any(label.startswith("Raw: UPRIGHT_PARTIAL") for label in labels)
        assert "Final posture: UNKNOWN" in labels
        assert "Visibility: TORSO" in labels
        assert any("Pose quality:" in label and "Match:" in label for label in labels)
        assert any(label.startswith("Scale:") for label in labels)
        assert "Fall: NORMAL 0.03" in labels


def test_missing_overlay_explicitly_labels_the_held_raw_observation(monkeypatch):
    pipeline = make_pipeline()
    pipeline.process_poses([partial(skeleton())], 0)
    result = pipeline.process_poses([], 0.1)
    labels = []
    renderer = Renderer(load_config()["visualization"] | {"debug": True})
    monkeypatch.setattr(renderer, "_text", lambda image, text, *_: labels.append(text))
    renderer.render(np.zeros((720, 1280, 3), np.uint8), result, {}, 0.1)
    assert "Raw observation held; person missing" in labels


def test_saved_frame_diagnostics_include_visibility_without_inventing_joints():
    from tools.diagnose_posture import diagnostic_record

    pipeline = make_pipeline()
    result = pipeline.process_poses([partial(skeleton())], 0)
    record = diagnostic_record(result, 17)
    assert record["frame_index"] == 17
    person = record["people"][0]
    assert person["visibility"]["mode"] == "TORSO"
    assert person["raw_posture"] == "upright_partial"
    assert all(point[2] == 0 for point in person["keypoints"][13:])


def test_torso_only_recovery_can_rearm_for_an_independent_second_fall():
    pipeline = make_pipeline()
    pipeline.config["fall"]["recovery_seconds"] = 0.3
    pipeline.config["state"]["rearm_seconds"] = 0.3
    events = []
    pipeline.event_bus.subscribe(events.append)
    for index in range(110):
        time = index / 15
        if time < 2.4:
            progress = progress_at(time)
        elif time < 3.4:
            progress = 3.4 - time
        elif time < 4.7:
            progress = 0
        else:
            progress = min(1, (time - 4.7) / 0.4)
        pipeline.process_poses([partial(fall_pose(progress))], time)
    falls = [event for event in events if event["type"] == "fall_detected"]
    assert len(falls) == 2
    assert falls[0]["track_id"] == falls[1]["track_id"] == 1
    assert [event["episode_id"] for event in falls] == [1, 2]
