"""Compact OpenCV overlay. Drawing never modifies the captured image."""

from typing import Any

import cv2
import numpy as np

from src.pipeline import PipelineResult
from src.pose.pose_types import SKELETON_EDGES


class Renderer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.debug = bool(config["debug"])

    @staticmethod
    def _text(
        image: np.ndarray,
        text: str,
        point: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.55,
    ) -> None:
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        result: PipelineResult | None,
        stats: dict[str, Any],
        timestamp: float,
        paused: bool = False,
    ) -> np.ndarray:
        image = frame.copy()
        height, width = image.shape[:2]
        visible = 0
        if result and timestamp - result.timestamp <= self.config["max_overlay_age_seconds"]:
            for person in result.persons:
                if timestamp - person.last_seen > self.config["max_overlay_age_seconds"]:
                    continue
                visible += int(person.observed)
                state = person.state
                color = (
                    (40, 40, 255)
                    if state == "FALLEN"
                    else ((0, 180, 255) if state in {"FALLING", "RECOVERING"} else (80, 230, 120))
                )
                if not person.observed:
                    color = (150, 150, 150)
                x1, y1, x2, y2 = person.pose.bbox.astype(int)
                x1, x2 = np.clip([x1, x2], 0, width - 1)
                y1, y2 = np.clip([y1, y2], 0, height - 1)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3 if state == "FALLEN" else 2)
                points = person.pose.keypoints
                threshold = self.config["joint_confidence_threshold"]
                for a, b in SKELETON_EDGES:
                    if min(points[a, 2], points[b, 2]) >= threshold:
                        cv2.line(
                            image,
                            tuple(points[a, :2].astype(int)),
                            tuple(points[b, :2].astype(int)),
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                for x, y, confidence in points:
                    if confidence >= threshold:
                        cv2.circle(image, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
                labels = [
                    f"ID {person.track_id}  {state}",
                    f"{person.posture.label.upper()} {person.posture.confidence:.2f}",
                    f"Fall: {person.fall.probability:.2f}"
                    + ("  missing" if not person.observed else ""),
                ]
                if self.debug:
                    features = {**person.posture.features, **person.fall.features}
                    for name in (
                        "torso_angle",
                        "bbox_aspect_ratio",
                        "hip_velocity",
                        "normalized_vertical_velocity",
                    ):
                        if name in features:
                            labels.append(f"{name}: {features[name]:.2f}")
                origin_y = max(100, y1)
                for row, label in enumerate(labels):
                    self._text(
                        image,
                        label,
                        (int(x1), min(height - 8, origin_y + row * 21)),
                        color,
                        0.7 if row == 0 and state == "FALLEN" else 0.55,
                    )
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (min(width, 1000), 83), (15, 20, 25), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
        line = (
            f"Capture {stats.get('capture_fps', 0):.1f} FPS | "
            f"Pose {stats.get('pose_fps', 0):.1f} FPS | "
            f"Process {stats.get('processing_fps', 0):.1f} FPS | People {visible}"
        )
        self._text(image, line, (12, 24), (230, 230, 230))
        latency = result.timings["pose_ms"] if result else 0
        age = stats.get("age_ms", 0)
        provider = result.provider if result else stats.get("provider", "starting")
        self._text(
            image,
            f"Inference {latency:.1f} ms | Age {age:.1f} ms | {provider}",
            (12, 48),
            (230, 230, 230),
        )
        self._text(
            image,
            "q quit | p pause | s snapshot | d debug" + (" | PAUSED" if paused else ""),
            (12, 72),
            (180, 210, 230),
        )
        return image
