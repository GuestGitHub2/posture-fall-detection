"""Deterministic projected skeletons for meaningful temporal regression tests."""

import numpy as np

from src.pose.pose_types import Pose


def skeleton(
    label: str = "standing", offset: tuple[float, float] = (300, 0), scale: float = 1.0
) -> Pose:
    points = np.zeros((17, 3), np.float32)
    points[:, 2] = 0.95
    coordinates = [
        (0, 50),
        (-10, 45),
        (10, 45),
        (-20, 50),
        (20, 50),
        (-25, 100),
        (25, 100),
        (-40, 150),
        (40, 150),
        (-40, 200),
        (40, 200),
        (-20, 200),
        (20, 200),
        (-20, 300),
        (20, 300),
        (-20, 400),
        (20, 400),
    ]
    points[:, :2] = coordinates
    if label == "sitting":
        points[[13, 14], :2] = ((80, 210), (120, 210))
        points[[15, 16], :2] = ((80, 310), (120, 310))
    elif label == "bending":
        points[:11, 0] += 120
        points[:11, 1] += 50
    elif label == "lying":
        centered = points[:, :2] - (0, 200)
        points[:, :2] = np.stack((-centered[:, 1], centered[:, 0]), axis=1) + (0, 200)
    points[:, :2] = points[:, :2] * scale + offset
    minimum, maximum = (
        points[:, :2].min(axis=0) - 10 * scale,
        points[:, :2].max(axis=0) + 10 * scale,
    )
    return Pose(points, np.concatenate((minimum, maximum)).astype(np.float32))


def fall_pose(progress: float, drop: float = 200.0) -> Pose:
    pose = skeleton()
    centered = pose.keypoints[:, :2] - (300, 200)
    angle = progress * np.pi / 2
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    points = pose.keypoints.copy()
    points[:, :2] = centered @ rotation.T + (300, 200 + drop * progress)
    minimum, maximum = points[:, :2].min(axis=0) - 10, points[:, :2].max(axis=0) + 10
    return Pose(points, np.concatenate((minimum, maximum)).astype(np.float32))
