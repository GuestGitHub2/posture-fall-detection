"""Report local hardware, ONNX providers, camera access and model I/O."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _camera_worker(index: int, queue: Any) -> None:
    import cv2

    capture = cv2.VideoCapture(index)
    try:
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        queue.put(
            {
                "works": bool(ok),
                "index": index,
                "frame_size": [int(frame.shape[1]), int(frame.shape[0])] if ok else None,
            }
        )
    finally:
        capture.release()


def check_camera(index: int, timeout: float = 8.0) -> dict[str, Any]:
    """Camera probe in an isolated process with a bounded wait."""
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_camera_worker, args=(index, queue), daemon=True)
    process.start()
    process.join(timeout)
    try:
        if process.is_alive():
            process.terminate()
            process.join(2)
            return {
                "works": False,
                "index": index,
                "reason": f"Camera probe timed out after {timeout:g}s; verify permissions and hardware.",
            }
        if queue.empty():
            return {
                "works": False,
                "index": index,
                "reason": "Camera probe exited without a frame; check camera privacy permissions.",
            }
        result = queue.get(timeout=1)
        if not result["works"]:
            result["reason"] = (
                "Camera unavailable or no frame; close competing apps and check camera/privacy settings."
            )
        return result
    finally:
        queue.close()
        process.close()


def main() -> int:
    import cv2
    import onnxruntime as ort

    from src.runtime.onnx_runtime import ORTSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip-camera", action="store_true", help="Do not open a camera")
    parser.add_argument("--camera-timeout", type=float, default=8.0)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "rtmo_m.onnx")
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--output", type=Path, help="Save the JSON report to a file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report: dict[str, Any] = {
        "os": platform.platform(),
        "cpu_architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "camera": {"checked": False}
        if args.skip_camera
        else check_camera(args.camera, args.camera_timeout),
    }
    failed = False
    try:
        session = ORTSession(args.model, {"provider": args.provider})
        report["loaded_model"] = session.model_info
        report["provider_note"] = (
            "Preferred active provider; unsupported graph operators may execute on CPU."
        )
    except (OSError, ValueError, RuntimeError) as exc:
        report["loaded_model"] = {"error": str(exc)}
        failed = True
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
