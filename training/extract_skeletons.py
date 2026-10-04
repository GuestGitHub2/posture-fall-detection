"""Extract tracked skeleton JSONL once from a folder of videos, entirely locally."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.config import load_config
from src.pose.base import create_pose_estimator
from src.tracking.tracker import Tracker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="JSON mapping video-relative path to label, subject_id, fall_onset and segments",
    )
    parser.add_argument(
        "--label", default="unknown", help="Fallback label; use an annotation file for mixed videos"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    files = (
        [args.input]
        if args.input.is_file()
        else sorted(
            file
            for file in args.input.rglob("*")
            if file.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
    )
    if not files:
        parser.exit(2, f"No video files found: {args.input}\n")
    try:
        annotations = json.loads(args.annotations.read_text()) if args.annotations else {}
        if not isinstance(annotations, dict):
            raise ValueError("Annotations must be a JSON object keyed by relative video path")
        config = load_config(args.config)
        estimator = create_pose_estimator(config)
        args.output.mkdir(parents=True, exist_ok=True)
        for file in files:
            relative = file.relative_to(args.input) if args.input.is_dir() else Path(file.name)
            sequence_id = relative.with_suffix("").as_posix()
            destination = args.output / relative.with_suffix(".jsonl")
            if destination.exists() and not args.overwrite:
                logging.info("Skipping existing %s", destination)
                continue
            capture = cv2.VideoCapture(str(file))
            if not capture.isOpened():
                raise OSError(f"Cannot decode video: {file}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not fps > 0:
                fps = float(config["camera"]["fps"])
            info = annotations.get(relative.as_posix(), {})
            if isinstance(info, str):
                info = {"label": info}
            destination.parent.mkdir(parents=True, exist_ok=True)
            number = 0
            # Fresh tracking per video; no posture/fall models or event side effects are needed.
            tracker = Tracker(config["tracking"])
            temporary = destination.with_suffix(".jsonl.tmp")
            try:
                with temporary.open("w", encoding="utf-8") as stream:
                    while True:
                        success, frame = capture.read()
                        if not success:
                            break
                        timestamp = number / fps
                        tracks = tracker.update(estimator.infer(frame, timestamp), timestamp)
                        label = info.get("label", args.label)
                        for segment in info.get("segments", []):
                            if float(segment["start"]) <= timestamp < float(segment["end"]):
                                label = segment["label"]
                        for person in tracks:
                            if not person.observed:
                                continue
                            row = {
                                "sequence_id": sequence_id,
                                "frame_number": number,
                                "track_id": person.track_id,
                                "timestamp": timestamp,
                                "keypoints": person.pose.keypoints.tolist(),
                                "bbox": person.pose.bbox.tolist(),
                                "label": label,
                                "subject_id": info.get("subject_id"),
                                "fall_onset": info.get("fall_onset"),
                            }
                            stream.write(json.dumps(row) + "\n")
                        number += 1
            finally:
                capture.release()
            temporary.replace(destination)
            logging.info("Extracted %d frames: %s", number, destination)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError, cv2.error) as error:
        parser.exit(2, f"Extraction failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
