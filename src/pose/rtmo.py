"""RTMO ONNX SDK inference using OpenMMLab's exported boxes and COCO-17 poses.

Preprocessing/output conventions were verified against Apache-2.0 RTMLib
(Tau-J/rtmlib); this adapter adds boxes, validation and local provider fallback.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.pose.base import PoseEstimator
from src.pose.pose_types import Pose
from src.runtime.onnx_runtime import ORTSession


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Top-left BGR letterbox (H,W), no pixel normalization for RTMO SDK."""
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
        raise ValueError("Pose inference expects a nonempty BGR image H x W x 3.")
    height, width = size
    ratio = min(height / frame.shape[0], width / frame.shape[1])
    image = np.full((height, width, 3), 114, dtype=np.uint8)
    resized_width = max(1, int(frame.shape[1] * ratio))
    resized_height = max(1, int(frame.shape[0] * ratio))
    image[:resized_height, :resized_width] = cv2.resize(frame, (resized_width, resized_height))
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy person NMS with continuous xyxy coordinates."""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    area = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    keep: list[int] = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        low = np.maximum(boxes[current, :2], boxes[rest, :2])
        high = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        overlap = np.maximum(high - low, 0).prod(axis=1)
        union = area[current] + area[rest] - overlap
        iou = overlap / np.maximum(union, 1e-9)
        order = rest[iou <= threshold]
    return keep


class RTMOEstimator(PoseEstimator):
    def __init__(
        self, pose_config: dict[str, Any], runtime_config: dict[str, Any] | None = None
    ) -> None:
        self.config = pose_config
        self.session = ORTSession(
            pose_config.get("model_path", "models/rtmo_m.onnx"), runtime_config
        )
        shape = self.session.input_shape
        self.size = (
            tuple(int(x) for x in shape[-2:])
            if all(isinstance(x, int) for x in shape[-2:])
            else tuple(pose_config.get("input_size", [640, 640]))
        )
        self.score_threshold = float(pose_config.get("confidence_threshold", 0.3))
        self.nms_threshold = float(pose_config.get("nms_threshold", 0.45))
        self.max_people = int(pose_config.get("max_people", 30))

    @property
    def provider(self) -> str:
        return self.session.provider

    @property
    def model_info(self) -> dict[str, Any]:
        return {"architecture": "RTMO-m Body7 COCO17 (ONNX SDK)", **self.session.model_info}

    def infer(self, frame: np.ndarray, timestamp: float) -> list[Pose]:
        tensor, ratio = letterbox(frame, self.size)
        return self.decode(self.session.run(tensor), ratio, frame.shape[:2], timestamp)

    def decode(
        self, outputs: list[np.ndarray], ratio: float, image_size: tuple[int, int], timestamp: float
    ) -> list[Pose]:
        """Decode the ONNX SDK dets [1,N,5] and keypoints [1,N,17,3]."""
        dets = next((x[0] for x in outputs if x.ndim == 3 and x.shape[-1] == 5), None)
        keypoints = next((x[0] for x in outputs if x.ndim == 4 and x.shape[-2:] == (17, 3)), None)
        if dets is None or keypoints is None or len(dets) != len(keypoints):
            raise ValueError(
                "Incompatible RTMO outputs: expected [1,N,5] boxes and [1,N,17,3] poses from the official ONNX SDK export."
            )
        valid = (
            np.isfinite(dets).all(axis=1)
            & (dets[:, 4] >= self.score_threshold)
            & (dets[:, 2] > dets[:, 0])
            & (dets[:, 3] > dets[:, 1])
        )
        indices = np.flatnonzero(valid)
        boxes = dets[indices, :4].copy() / ratio
        height, width = image_size
        scores = dets[indices, 4]
        poses: list[Pose] = []
        for index in nms(boxes, scores, self.nms_threshold)[: self.max_people]:
            box = boxes[index].copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            joints = keypoints[indices[index]].copy()
            invalid = ~np.isfinite(joints).all(axis=1)
            joints[invalid] = 0
            joints[:, :2] /= ratio
            # Do not clip joint coordinates: partial-frame geometry must retain its
            # structure, but off-frame joints have zero confidence.
            inside = (
                (joints[:, 0] >= 0)
                & (joints[:, 0] < width)
                & (joints[:, 1] >= 0)
                & (joints[:, 1] < height)
            )
            joints[~inside, 2] = 0
            poses.append(Pose(joints, box, float(scores[index]), timestamp=timestamp))
        return poses
