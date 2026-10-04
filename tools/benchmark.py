"""Measure local pipeline stage latency; synthetic mode never loads a pose model."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.capture.camera import open_capture
from src.config import load_config
from src.events import EventBus
from src.pipeline import Pipeline
from src.pose.pose_types import Pose


def synthetic_poses(frame_number: int, people: int = 2) -> list[Pose]:
    """Deterministic upright moving skeletons, separated to exercise independent tracks."""
    base = np.array(
        [
            (0, -1),
            (-0.06, -1.02),
            (0.06, -1.02),
            (-0.12, -0.99),
            (0.12, -0.99),
            (-0.2, -0.7),
            (0.2, -0.7),
            (-0.29, -0.35),
            (0.29, -0.35),
            (-0.32, 0),
            (0.32, 0),
            (-0.15, 0),
            (0.15, 0),
            (-0.15, 0.5),
            (0.15, 0.5),
            (-0.15, 1.0),
            (0.15, 1.0),
        ],
        dtype=np.float32,
    )
    result = []
    for person in range(people):
        points = base * 160 + np.array([200 + person * 280 + np.sin(frame_number / 50.0) * 10, 300])
        keypoints = np.column_stack((points, np.full(17, 0.95))).astype(np.float32)
        bbox = np.r_[points.min(0) - 15, points.max(0) + 15].astype(np.float32)
        result.append(Pose(keypoints, bbox))
    return result


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--synthetic", action="store_true", help="Skeleton-only benchmark (default)"
    )
    inputs.add_argument("--video", type=Path)
    inputs.add_argument("--camera", type=int)
    inputs.add_argument(
        "--image", type=Path, help="Repeated actual pose inference on a still image"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--people", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark.json"))
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.people < 1:
        parser.error("frames/people must be positive and warmup non-negative")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    capture = None
    try:
        config = load_config(args.config)
        synthetic = args.video is None and args.camera is None and args.image is None
        pipeline = Pipeline(config, event_bus=EventBus(), load_pose=not synthetic)
        source_fps = float(config["pose"]["inference_fps"])
        image = cv2.imread(str(args.image)) if args.image else None
        if args.image and image is None:
            raise OSError(f"Cannot decode image: {args.image}")
        if not synthetic and args.image is None:
            capture = open_capture(str(args.video) if args.video else args.camera, config["camera"])
            source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or float(config["camera"]["fps"])
        durations: dict[str, list[float]] = {
            name: []
            for name in (
                "pose_ms",
                "posture_ms",
                "fall_ms",
                "tracking_ms",
                "total_ms",
                "capture_ms",
            )
        }
        measured = 0
        observed_people: list[int] = []
        input_dimensions: set[tuple[int, int]] = set()
        wall_start = time.perf_counter()
        measurement_start = None
        for index in range(args.warmup + args.frames):
            beginning = time.perf_counter()
            if synthetic:
                poses = synthetic_poses(index, args.people)
                capture_ms = 0.0
                result = pipeline.process_poses(poses, index / source_fps)
            elif args.image:
                result = pipeline.process(image, index / source_fps)
                capture_ms = 0.0
            else:
                success, frame = capture.read()
                capture_ms = (time.perf_counter() - beginning) * 1000
                if not success:
                    break
                timestamp = index / source_fps if args.video else time.perf_counter() - wall_start
                result = pipeline.process(frame, timestamp)
            if index < args.warmup:
                continue
            if measurement_start is None:
                measurement_start = beginning
            measured += 1
            observed_people.append(sum(person.observed for person in result.persons))
            if not synthetic:
                measured_image = image if args.image else frame
                input_dimensions.add((int(measured_image.shape[1]), int(measured_image.shape[0])))
            for name in durations:
                durations[name].append(capture_ms if name == "capture_ms" else result.timings[name])
        wall_seconds = time.perf_counter() - measurement_start if measurement_start else 0.0
        if not measured:
            raise ValueError("No measured frames. Source shorter than warmup; lower --warmup")
        result = {
            "mode": "synthetic_skeletons"
            if synthetic
            else "image"
            if args.image
            else "video"
            if args.video
            else "camera",
            "provider": pipeline.provider,
            "pose_model": None if synthetic else pipeline.estimator.model_info,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "frames": measured,
            "warmup_frames": args.warmup,
            "people": args.people if synthetic else None,
            "observed_people": {
                "min": min(observed_people),
                "max": max(observed_people),
                "mean": float(np.mean(observed_people)),
            },
            "input_dimensions": [list(size) for size in sorted(input_dimensions)],
            "source_fps": None if synthetic or args.image else source_fps,
            "requested_camera": config["camera"] if args.camera is not None else None,
            "latency_ms": {stage: latency_summary(values) for stage, values in durations.items()},
            "wall_seconds": wall_seconds,
            "average_fps": measured / wall_seconds,
            "pipeline_fps": 1000.0 / np.mean(durations["total_ms"]),
            "notes": "Synchronous unthrottled benchmark; GUI/encoding excluded. Synthetic mode excludes pose inference and camera.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, RuntimeError, ValueError, ImportError, KeyError) as error:
        parser.exit(2, f"Benchmark failed: {error}\n")
    finally:
        if capture:
            capture.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
