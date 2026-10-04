import numpy as np

from src.fall.temporal_buffer import TemporalBuffer
from tests.helpers import skeleton


def test_time_and_sample_bounds():
    buffer = TemporalBuffer(history_seconds=0.5, max_samples=5)
    for index in range(20):
        buffer.append(skeleton(), index * 0.1)
    assert len(buffer.samples) <= 5
    assert buffer.samples[-1].timestamp - buffer.samples[0].timestamp <= 0.5


def test_long_gap_and_timestamp_reversal_reset_history():
    buffer = TemporalBuffer(max_gap_seconds=0.4)
    assert not buffer.append(skeleton(), 0)
    assert buffer.append(skeleton(), 1)
    assert len(buffer.samples) == 1
    assert buffer.append(skeleton(), 0.9)


def test_interpolate_short_missing_joint_gap_but_not_padding():
    buffer = TemporalBuffer(history_seconds=0.2, max_gap_seconds=0.3)
    buffer.append(skeleton(), 0)
    middle = skeleton()
    middle.keypoints[15, 2] = 0
    buffer.append(middle, 0.1)
    buffer.append(skeleton(), 0.2)
    data = buffer.resample(3)
    assert data.shape == (3, 17, 3)
    assert data[1, 15, 2] > 0.8
    empty = TemporalBuffer().resample(30)
    assert np.all(empty == 0)
