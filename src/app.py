"""Run local webcam or video pose/posture/fall detection."""

import argparse
import logging
import math
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from src.capture.camera import CameraCapture, Frame, VideoCapture
from src.config import ROOT, load_config
from src.pipeline import Pipeline, PipelineResult
from src.utils.fps import FPS
from src.visualization.renderer import Renderer

LOGGER = logging.getLogger(__name__)


class InferenceWorker:
    """One sequential inference worker reads the camera's latest frame at target rate."""

    def __init__(self, capture: CameraCapture, pipeline: Pipeline, fps: float) -> None:
        self.capture, self.pipeline = capture, pipeline
        self.interval = 1 / fps
        self.result: PipelineResult | None = None
        self.error: str | None = None
        self.rate = FPS()
        self.paused = threading.Event()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="inference", daemon=True)

    def start(self) -> "InferenceWorker":
        self.thread.start()
        return self

    def _run(self) -> None:
        last_index = -1
        next_inference = 0.0
        try:
            while not self.stop.is_set():
                if self.paused.is_set():
                    self.stop.wait(0.03)
                    continue
                wait = max(0.0, next_inference - time.monotonic())
                if self.stop.wait(wait):
                    break
                packet = self.capture.latest(last_index)
                if packet is None:
                    if self.capture.error:
                        break
                    continue
                started = time.monotonic()
                self.result = self.pipeline.process(packet.image, packet.timestamp)
                self.rate.tick()
                last_index = packet.index
                next_inference = started + self.interval
        except Exception as exc:
            LOGGER.error(
                "Inference worker stopped: %s", exc, exc_info=LOGGER.isEnabledFor(logging.DEBUG)
            )
            self.error = str(exc)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3)


class Display:
    def __init__(
        self, config: dict[str, Any], headless: bool, output: Path | None, fps: float
    ) -> None:
        self.renderer = Renderer(config["visualization"])
        self.headless, self.output, self.fps = headless, output, fps
        self.window = config["visualization"]["window_name"]
        self.writer: cv2.VideoWriter | None = None
        self.rate = FPS()
        self.paused = False
        self.last_image = None
        if not headless:
            try:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                raise RuntimeError("OpenCV display unavailable. Run with --headless.") from exc

    def show(self, packet: Frame, result: PipelineResult | None, stats: dict[str, Any]) -> str:
        self.rate.tick()
        stats["processing_fps"] = self.rate.value
        if self.headless and self.output is None:
            return ""
        image = self.renderer.render(packet.image, result, stats, packet.timestamp, self.paused)
        self.last_image = image
        if self.output:
            if self.writer is None:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                height, width = image.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
                )
                if not self.writer.isOpened():
                    raise RuntimeError(f"Cannot create output video {self.output}")
            self.writer.write(image)
        if self.headless:
            return ""
        cv2.imshow(self.window, image)
        return self.key()

    def key(self) -> str:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            self.renderer.debug = not self.renderer.debug
        elif key == ord("s") and self.last_image is not None:
            path = ROOT / "outputs" / f"frame_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(path), self.last_image):
                LOGGER.info("Saved %s", path)
        elif key == ord("p"):
            self.paused = not self.paused
        return chr(key) if key < 128 else ""

    def close(self) -> None:
        if self.writer:
            self.writer.release()
        if not self.headless:
            cv2.destroyAllWindows()


def run_camera(args: argparse.Namespace, config: dict[str, Any], pipeline: Pipeline) -> None:
    capture = CameraCapture(args.camera, config["camera"]).start()
    worker = InferenceWorker(capture, pipeline, config["pose"]["inference_fps"]).start()
    display = None
    frames = 0
    try:
        display = Display(config, args.headless, args.output, config["camera"]["fps"])
        last_index = -1
        last_frame_wall = time.monotonic()
        while not args.max_frames or frames < args.max_frames:
            packet = capture.latest(last_index)
            if worker.error:
                raise RuntimeError(worker.error)
            if capture.error:
                raise RuntimeError(capture.error)
            if packet is None:
                if time.monotonic() - last_frame_wall > config["camera"]["stall_timeout_seconds"]:
                    raise RuntimeError(
                        "Camera is not returning frames. Check the connection and permissions."
                    )
                if not args.headless and display.key() == "q":
                    break
                continue
            last_frame_wall = time.monotonic()
            last_index = packet.index
            result = worker.result
            stats = {
                "capture_fps": capture.rate.value,
                "pose_fps": worker.rate.value,
                "provider": pipeline.provider,
                "age_ms": (time.monotonic() - result.timestamp) * 1000 if result else 0,
            }
            key = display.show(packet, result, stats)
            if display.paused:
                worker.paused.set()
            else:
                worker.paused.clear()
            frames += 1
            if key == "q":
                break
    finally:
        worker.close()
        capture.close()
        if display:
            display.close()
    LOGGER.info(
        "Camera run completed: %d displayed frames; pose %.1f FPS", frames, worker.rate.value
    )


def run_video(args: argparse.Namespace, config: dict[str, Any], pipeline: Pipeline) -> None:
    capture = VideoCapture(str(args.video), config["camera"])
    display = None
    frames = 0
    pose_rate = FPS()
    last_result = None
    next_pose = 0.0
    wall_start = time.monotonic()
    try:
        display = Display(config, args.headless, args.output, capture.fps)
        while not args.max_frames or frames < args.max_frames:
            if display.paused:
                pause_start = time.monotonic()
                while display.paused:
                    if display.key() == "q":
                        return
                    time.sleep(0.01)
                wall_start += time.monotonic() - pause_start
            packet = capture.read()
            if packet is None:
                break
            if packet.timestamp + 1e-6 >= next_pose:
                last_result = pipeline.process(packet.image, packet.timestamp)
                pose_rate.tick(packet.timestamp)
                interval = 1 / config["pose"]["inference_fps"]
                next_pose = advance_deadline(next_pose, packet.timestamp, interval)
            if not args.headless and not args.no_realtime:
                deadline = wall_start + packet.timestamp
                while not display.paused:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if display.key() == "q":
                        return
                    time.sleep(min(remaining, 0.01))
            stats = {
                "capture_fps": capture.fps,
                "pose_fps": pose_rate.value,
                "provider": pipeline.provider,
                "age_ms": (packet.timestamp - last_result.timestamp) * 1000 if last_result else 0,
            }
            if display.show(packet, last_result, stats) == "q":
                break
            frames += 1
    finally:
        capture.close()
        if display:
            display.close()
    LOGGER.info("Video run completed: %d frames at source %.2f FPS", frames, capture.fps)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int, help="Camera index (default from YAML)")
    source.add_argument("--video", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--provider", help="auto, cpu, coreml, cuda, directml or openvino")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path, help="Record annotated video")
    parser.add_argument(
        "--no-realtime", action="store_true", help="Process files as fast as possible"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0=unlimited)"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        if not args.debug:
            level = getattr(logging, str(config["logging"]["level"]).upper(), None)
            if not isinstance(level, int):
                raise ValueError("logging.level must be a valid logging level")
            logging.getLogger().setLevel(level)
        if args.provider:
            config["runtime"]["provider"] = args.provider
        if args.debug:
            config["visualization"]["debug"] = True
        if args.max_frames < 0:
            raise ValueError("--max-frames must be nonnegative")
        if args.video and (
            not args.video.is_file()
            or (args.output and args.video.resolve() == args.output.resolve())
        ):
            raise ValueError("Input video must exist and output must differ from input")
        if args.camera is None:
            args.camera = config["camera"]["index"]
        pipeline = Pipeline(config)
        LOGGER.info(
            "Pose model: %s | Provider: %s | Fall detector: %s",
            pipeline.estimator.model_info,
            pipeline.provider,
            config["fall"]["detector"],
        )
        if args.video:
            run_video(args, config, pipeline)
        else:
            run_camera(args, config, pipeline)
        return 0
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as exc:
        LOGGER.error("%s", exc, exc_info=args.debug)
        return 2


def advance_deadline(deadline: float, timestamp: float, interval: float) -> float:
    """Advance a fixed schedule without rounding the rate down to a frame divisor."""
    deadline += interval
    if deadline <= timestamp + 1e-6:
        deadline += (math.floor((timestamp + 1e-6 - deadline) / interval) + 1) * interval
    return deadline


if __name__ == "__main__":
    raise SystemExit(main())
