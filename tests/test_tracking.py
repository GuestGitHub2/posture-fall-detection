from src.tracking.tracker import Tracker
from tests.helpers import skeleton


def test_reordered_detections_keep_histories_separate():
    tracker = Tracker()
    tracks = tracker.update([skeleton(offset=(200, 0)), skeleton(offset=(700, 0))], 0)
    assert [track.track_id for track in tracks] == [1, 2]
    tracks = tracker.update([skeleton(offset=(690, 0)), skeleton(offset=(210, 0))], 0.1)
    assert tracks[0].pose.keypoints[11, 0] < tracks[1].pose.keypoints[11, 0]


def test_short_occlusion_preserves_id_and_long_gap_expires():
    tracker = Tracker({"max_missing_seconds": 0.5})
    tracker.update([skeleton()], 0)
    missed = tracker.update([], 0.1)
    assert missed[0].track_id == 1 and not missed[0].observed
    assert tracker.update([skeleton()], 0.2)[0].track_id == 1
    assert tracker.update([], 1.0) == []
    assert tracker.update([skeleton()], 1.1)[0].track_id == 2


def test_motion_prediction_preserves_crossing_ids():
    tracker = Tracker()
    for index, (left, right) in enumerate(
        ((200, 440), (240, 400), (280, 360), (320, 320), (360, 280), (400, 240))
    ):
        poses = [skeleton(offset=(left, 0)), skeleton(offset=(right, 15))]
        if index % 2:
            poses.reverse()
        tracks = tracker.update(poses, index * 0.1)
        assert len(tracks) == 2
        assert abs(tracks[0].pose.keypoints[11, 0] - (left - 20)) < 1


def test_unrelated_new_person_gets_new_id():
    tracker = Tracker()
    tracker.update([skeleton(offset=(200, 0))], 0)
    tracks = tracker.update([skeleton(offset=(1400, 0))], 0.1)
    assert [track.track_id for track in tracks] == [1, 2]
    assert not tracks[0].observed and tracks[1].observed


def test_reappearance_after_processing_gap_gets_fresh_identity():
    tracker = Tracker({"max_missing_seconds": 0.5})
    assert tracker.update([skeleton()], 0)[0].track_id == 1
    assert tracker.update([skeleton()], 2)[0].track_id == 2
