"""Optional RTMPose-m + YOLOX person detector ONNX fallback (Apache-2.0).

SDK preprocessing conventions verified against Tau-J/rtmlib. This implementation
keeps actual detector boxes, makes empty scenes empty and never downloads assets.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.pose.base import PoseEstimator
from src.pose.pose_types import Pose
from src.pose.rtmo import letterbox, nms
from src.runtime.onnx_runtime import ORTSession


def affine_crop(
    frame: np.ndarray, box: np.ndarray, size: tuple[int, int], padding: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop a padded xyxy box without changing the person's aspect ratio."""
    width, height = size
    center = (box[:2] + box[2:]) * 0.5
    scale = (box[2:] - box[:2]) * padding
    aspect = width / height
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    factors = np.array(size, dtype=np.float32) / scale
    matrix = np.array(
        [
            [factors[0], 0, width * 0.5 - factors[0] * center[0]],
            [0, factors[1], height * 0.5 - factors[1] * center[1]],
        ],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(frame, matrix, size, flags=cv2.INTER_LINEAR)
    return crop, center, scale


def decode_simcc(
    outputs: list[np.ndarray],
    center: np.ndarray,
    scale: np.ndarray,
    size: tuple[int, int],
    split_ratio: float,
) -> np.ndarray:
    """Decode [1,17,2W], [1,17,2H] SimCC outputs into pixel COCO-17."""
    if len(outputs) != 2:
        raise ValueError("RTMPose requires two SimCC outputs.")
    x, y = outputs
    if x.ndim != 3 or y.ndim != 3 or x.shape[:2] != (1, 17) or y.shape[:2] != (1, 17):
        raise ValueError("RTMPose requires SimCC [1,17,bins] outputs.")
    scores = np.minimum(x.max(axis=-1), y.max(axis=-1))[0]
    locations = (
        np.stack([x.argmax(axis=-1), y.argmax(axis=-1)], axis=-1)[0].astype(np.float32)
        / split_ratio
    )
    pixels = locations / np.array(size, dtype=np.float32) * scale + center - scale / 2
    return np.concatenate([pixels, np.clip(scores[:, None], 0, 1)], axis=-1).astype(np.float32)


class RTMPoseEstimator(PoseEstimator):
    """Top-down RTMPose with a real, independent multi-person YOLOX detector."""

    def __init__(
        self, pose_config: dict[str, Any], runtime_config: dict[str, Any] | None = None
    ) -> None:
        self.config = pose_config
        self.pose_session = ORTSession(
            pose_config.get("model_path", "models/rtmpose_m.onnx"), runtime_config
        )
        self.detector = ORTSession(
            pose_config.get("detector_path", "models/yolox_tiny.onnx"), runtime_config
        )
        shape = self.pose_session.input_shape
        self.pose_size = (int(shape[-1]), int(shape[-2]))
        shape = self.detector.input_shape
        self.detector_size = (int(shape[-2]), int(shape[-1]))
        self.score_threshold = float(pose_config.get("confidence_threshold", 0.3))
        self.nms_threshold = float(pose_config.get("nms_threshold", 0.45))
        self.mean = np.array(pose_config.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
        self.std = np.array(pose_config.get("std", [58.395, 57.12, 57.375]), dtype=np.float32)

    @property
    def provider(self) -> str:
        if self.pose_session.provider == self.detector.provider:
            return self.pose_session.provider
        return f"pose:{self.pose_session.provider},detector:{self.detector.provider}"

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "architecture": "RTMPose-m Body7 COCO17 + YOLOX-tiny HumanArt",
            "pose": self.pose_session.model_info,
            "detector": self.detector.model_info,
        }

    def detect(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        tensor, ratio = letterbox(frame, self.detector_size)
        outputs = self.detector.run(tensor)[0]
        if outputs.ndim != 3 or outputs.shape[0] != 1:
            raise ValueError("YOLOX detector must return [1,N,5] or [1,N,5+classes].")
        predictions = outputs[0].copy()
        if predictions.shape[-1] == 5:
            boxes, scores = predictions[:, :4] / ratio, predictions[:, 4]
        elif predictions.shape[-1] > 5:
            grids, strides = [], []
            for stride in (8, 16, 32):
                grid_x, grid_y = np.meshgrid(
                    np.arange(self.detector_size[1] // stride),
                    np.arange(self.detector_size[0] // stride),
                )
                grids.append(np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2))
                strides.append(np.full((grid_x.size, 1), stride))
            grid, stride_values = np.concatenate(grids), np.concatenate(strides)
            if len(grid) != len(predictions):
                raise ValueError("YOLOX output grid does not match configured input size.")
            centers = (predictions[:, :2] + grid) * stride_values
            dimensions = np.exp(np.clip(predictions[:, 2:4], -20, 20)) * stride_values
            boxes = (
                np.concatenate([centers - dimensions / 2, centers + dimensions / 2], axis=-1)
                / ratio
            )
            # Official HumanArt export is single-class; COCO person is class 0.
            scores = predictions[:, 4] * predictions[:, 5]
        else:
            raise ValueError("Unsupported YOLOX output width.")
        valid = (
            np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (scores >= self.score_threshold)
        )
        boxes, scores = boxes[valid], scores[valid]
        height, width = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        return [
            (boxes[i], float(scores[i]))
            for i in nms(boxes, scores, self.nms_threshold)[
                : int(self.config.get("max_people", 30))
            ]
            if boxes[i, 2] > boxes[i, 0] and boxes[i, 3] > boxes[i, 1]
        ]

    def infer(self, frame: np.ndarray, timestamp: float) -> list[Pose]:
        poses: list[Pose] = []
        for box, score in self.detect(frame):
            crop, center, scale = affine_crop(
                frame, box, self.pose_size, float(self.config.get("crop_padding", 1.25))
            )
            normalized = (crop.astype(np.float32) - self.mean) / self.std
            tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
            joints = decode_simcc(
                self.pose_session.run(tensor),
                center,
                scale,
                self.pose_size,
                float(self.config.get("simcc_split_ratio", 2.0)),
            )
            height, width = frame.shape[:2]
            inside = (
                (joints[:, 0] >= 0)
                & (joints[:, 0] < width)
                & (joints[:, 1] >= 0)
                & (joints[:, 1] < height)
            )
            joints[~inside, 2] = 0
            poses.append(Pose(joints, box, score, timestamp=timestamp))
        return poses
