"""Replaceable pose backends sharing the COCO-17 contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.pose.pose_types import Pose


class PoseEstimator(ABC):
    @abstractmethod
    def infer(self, frame: np.ndarray, timestamp: float) -> list[Pose]:
        """Return real detections, or an empty list when no person is visible."""

    @property
    @abstractmethod
    def provider(self) -> str: ...

    @property
    @abstractmethod
    def model_info(self) -> dict[str, Any]: ...


def create_pose_estimator(config: dict[str, Any]) -> PoseEstimator:
    pose = config.get("pose", config)
    runtime = config.get("runtime", {})
    name = str(pose.get("model", "rtmo_m")).lower()
    if name.startswith("rtmo") and not name.startswith("rtmpose"):
        from src.pose.rtmo import RTMOEstimator

        return RTMOEstimator(pose, runtime)
    if name.startswith("rtmpose"):
        from src.pose.rtmpose import RTMPoseEstimator

        return RTMPoseEstimator(pose, runtime)
    raise ValueError(f"Unsupported pose backend {name!r}; choose rtmo_m or rtmpose_m.")
