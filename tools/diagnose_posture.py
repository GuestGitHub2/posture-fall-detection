"""Inspect saved image/video pose observability without changing production defaults."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.capture.camera import VideoCapture
from src.config import load_config
from src.events import EventBus
from src.pipeline import Pipeline, PipelineResult


def diagnostic_record(result: PipelineResult, frame_index: int) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp": result.timestamp,
        "provider": result.provider,
        "people": [
            {
                "track_id": person.track_id,
                "observed": person.observed,
                "association_confidence": person.association_confidence,
                "association_ambiguous": person.association_ambiguous,
                "raw_posture": person.posture.label,
                "posture_confidence": person.posture.confidence,
                "pose_quality": person.posture.pose_quality,
                "visibility": asdict(person.posture.visibility)
                if person.posture.visibility
                else None,
                "smoothed_posture": person.smoothed_posture,
                "state": person.state,
                "scale_source": person.pose.scale_source,
                "scale_quality": person.pose.scale_quality,
                "features": person.posture.features,
                "fall": asdict(person.fall),
                "keypoints": person.pose.keypoints.tolist(),
                "bbox": person.pose.bbox.tolist(),
            }
            for person in result.persons
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/posture_diagnostics.jsonl"))
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    logging.basicConfig(level=logging.INFO)
    capture = None
    try:
        if not args.input.is_file():
            raise ValueError(f"Input file is missing: {args.input}")
        config = load_config(args.config)
        pipeline = Pipeline(config, event_bus=EventBus())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            if args.input.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                image = cv2.imread(str(args.input))
                if image is None:
                    raise ValueError(f"Cannot decode image: {args.input}")
                output.write(json.dumps(diagnostic_record(pipeline.process(image, 0), 0)) + "\n")
            else:
                capture = VideoCapture(args.input, config["camera"])
                for _ in range(args.max_frames):
                    frame = capture.read()
                    if frame is None:
                        break
                    result = pipeline.process(frame.image, frame.timestamp)
                    output.write(json.dumps(diagnostic_record(result, frame.index)) + "\n")
        logging.info(
            "Diagnostics: %s (pose model %s, provider %s)",
            args.output,
            config["pose"]["model"],
            pipeline.provider,
        )
        return 0
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        logging.error("Diagnostics failed: %s", exc)
        return 1
    finally:
        if capture is not None:
            capture.close()


if __name__ == "__main__":
    raise SystemExit(main())
