"""Common COCO-17 pose contract used by every backend."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

JOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
SKELETON_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
    (0, 1),
    (0, 2),
)


@dataclass
class Pose:
    """Pixel keypoints and xyxy box; normalized keypoints are hip centered."""

    keypoints: NDArray[np.float32]
    bbox: NDArray[np.float32]
    score: float = 1.0
    normalized: NDArray[np.float32] | None = None
    timestamp: float = 0.0
    body_scale: float = 1.0
    scale_source: str = "unmeasured"
    scale_quality: float = 0.0
    scale_is_stable: bool = False
    image_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32).copy()
        self.bbox = np.asarray(self.bbox, dtype=np.float32).copy()
        if self.keypoints.shape != (17, 3) or self.bbox.shape != (4,):
            raise ValueError("Pose requires COCO keypoints (17,3) and xyxy bbox (4,)")
        if not np.isfinite(self.keypoints).all() or not np.isfinite(self.bbox).all():
            raise ValueError("Pose coordinates must be finite")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("Pose bbox must have positive area")
        self.keypoints[:, 2] = np.clip(self.keypoints[:, 2], 0, 1)
