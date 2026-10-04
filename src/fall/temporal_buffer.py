"""Time-based, bounded per-person skeleton history and interpolation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from src.pose.normalization import normalize_pose
from src.pose.pose_types import Pose


@dataclass(frozen=True)
class PoseSample:
    pose: Pose
    timestamp: float


class TemporalBuffer:
    def __init__(
        self, history_seconds: float = 2.0, max_samples: int = 90, max_gap_seconds: float = 0.5
    ) -> None:
        if history_seconds <= 0 or max_samples < 2 or max_gap_seconds <= 0:
            raise ValueError(
                "Temporal buffer requires positive duration/gap and at least two samples"
            )
        self.history_seconds, self.max_gap_seconds = history_seconds, max_gap_seconds
        self._samples: deque[PoseSample] = deque(maxlen=max_samples)

    @property
    def samples(self) -> list[PoseSample]:
        return list(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def append(self, pose: Pose, timestamp: float) -> bool:
        reset = bool(
            self._samples
            and (
                timestamp <= self._samples[-1].timestamp
                or timestamp - self._samples[-1].timestamp > self.max_gap_seconds
            )
        )
        if reset:
            self.clear()
        pose = normalize_pose(pose) if pose.normalized is None else pose
        self._samples.append(PoseSample(pose, timestamp))
        while self._samples and timestamp - self._samples[0].timestamp > self.history_seconds:
            self._samples.popleft()
        return reset

    def resample(
        self,
        num_samples: int,
        end_timestamp: float | None = None,
        confidence_threshold: float = 0.3,
    ) -> np.ndarray:
        """Uniform [T,17,3] samples; absent or long-gap joints have zero confidence."""
        if num_samples < 2:
            raise ValueError("At least two temporal samples are required")
        output = np.zeros((num_samples, 17, 3), dtype=np.float32)
        if not self._samples:
            return output
        samples = self.samples
        end = samples[-1].timestamp if end_timestamp is None else end_timestamp
        target = np.linspace(end - self.history_seconds, end, num_samples)
        times = np.asarray([sample.timestamp for sample in samples])
        data = np.asarray([sample.pose.normalized for sample in samples])
        for joint in range(17):
            valid = data[:, joint, 2] >= confidence_threshold
            if not valid.any():
                continue
            valid_times, values = times[valid], data[valid, joint]
            for index, time in enumerate(target):
                nearest = int(np.argmin(abs(valid_times - time)))
                if abs(valid_times[nearest] - time) < 1e-7:
                    output[index, joint] = values[nearest]
                    continue
                right = int(np.searchsorted(valid_times, time))
                if right == 0 or right == len(valid_times):
                    continue
                left = right - 1
                gap = valid_times[right] - valid_times[left]
                if gap <= self.max_gap_seconds:
                    weight = (time - valid_times[left]) / gap
                    output[index, joint] = (1 - weight) * values[left] + weight * values[right]
        return output
