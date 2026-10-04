"""Validate ONNX assets or invoke an installed official MMDeploy exporter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser(
        "inspect", help="Load an existing ONNX model locally and report its I/O"
    )
    inspect.add_argument("model", type=Path)
    inspect.add_argument("--provider", default="cpu")
    export = subparsers.add_parser(
        "mmdeploy", help="Run official MMDeploy conversion; requires its separate environment"
    )
    export.add_argument("--mmdeploy-root", type=Path, required=True)
    export.add_argument("--deploy-config", type=Path, required=True)
    export.add_argument("--model-config", type=Path, required=True)
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--image", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            from src.runtime.onnx_runtime import ORTSession

            print(
                json.dumps(ORTSession(args.model, {"provider": args.provider}).model_info, indent=2)
            )
            return 0
        tool = args.mmdeploy_root / "tools" / "deploy.py"
        for path in [tool, args.deploy_config, args.model_config, args.checkpoint, args.image]:
            if not path.is_file():
                raise FileNotFoundError(f"Export input missing: {path}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(tool.resolve()),
            str(args.deploy_config.resolve()),
            str(args.model_config.resolve()),
            str(args.checkpoint.resolve()),
            str(args.image.resolve()),
            "--work-dir",
            str(args.output_dir.resolve()),
            "--device",
            args.device,
            "--dump-info",
        ]
        result = subprocess.run(command, check=False, cwd=args.mmdeploy_root.resolve())
        if result.returncode:
            print(
                "MMDeploy export failed; check the exporter environment and operator support.",
                file=sys.stderr,
            )
            return result.returncode
        models = list(args.output_dir.glob("*.onnx"))
        if not models:
            raise RuntimeError("Exporter returned success but no ONNX asset was produced.")
        print(json.dumps({"exported": [str(path.resolve()) for path in models]}, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Model export/inspection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
