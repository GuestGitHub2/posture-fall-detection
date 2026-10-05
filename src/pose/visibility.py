"""Observed COCO groups; confidence and coverage are separate quantities."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.pose.pose_types import Pose


class VisibilityMode(StrEnum):
    FULL_BODY = "FULL_BODY"
    LOWER_PARTIAL = "LOWER_PARTIAL"
    TORSO = "TORSO"
    UPPER_ONLY = "UPPER_ONLY"
    INSUFFICIENT = "INSUFFICIENT"


GROUPS = {
    "head": (0, 1, 2, 3, 4),
    "shoulders": (5, 6),
    "hips": (11, 12),
    "knees": (13, 14),
    "ankles": (15, 16),
}


@dataclass(frozen=True)
class BodyVisibility:
    mode: VisibilityMode
    counts: dict[str, int]
    qualities: dict[str, float]
    torso_observable: bool
    complete_leg: bool

    @property
    def torso_quality(self) -> float:
        return min(self.qualities["shoulders"], self.qualities["hips"])


def reliable_joints(pose: Pose, threshold: float) -> np.ndarray:
    """Never count out-of-image estimates as observations when dimensions are known."""
    valid = pose.keypoints[:, 2] >= threshold
    if pose.image_size is not None:
        width, height = pose.image_size
        xy = pose.keypoints[:, :2]
        valid &= (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return valid


def assess_visibility(
    pose: Pose, threshold: float = 0.3, minimum_per_pair: int = 1
) -> BodyVisibility:
    valid = reliable_joints(pose, threshold)
    counts = {name: int(valid[list(indices)].sum()) for name, indices in GROUPS.items()}
    qualities = {
        name: float(pose.keypoints[list(indices), 2][valid[list(indices)]].mean())
        if counts[name]
        else 0.0
        for name, indices in GROUPS.items()
    }
    torso = all(counts[name] >= minimum_per_pair for name in ("shoulders", "hips"))
    complete_leg = any(valid[list(indices)].all() for indices in ((11, 13, 15), (12, 14, 16)))
    if torso and complete_leg:
        mode = VisibilityMode.FULL_BODY
    elif torso and (counts["knees"] or counts["ankles"]):
        mode = VisibilityMode.LOWER_PARTIAL
    elif torso:
        mode = VisibilityMode.TORSO
    elif counts["head"] or counts["shoulders"]:
        mode = VisibilityMode.UPPER_ONLY
    else:
        mode = VisibilityMode.INSUFFICIENT
    return BodyVisibility(mode, counts, qualities, torso, bool(complete_leg))
