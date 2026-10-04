"""Explicit official model downloads; normal inference has no network activity."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(
    name: str, destination: Path, expected_checksum: str | None = None, force: bool = False
) -> Path:
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text())
    models = manifest["models"]
    if name not in models:
        raise ValueError(f"Unknown model {name!r}; choose {', '.join(models)}")
    model = models[name]
    if model["license"] != "Apache-2.0":
        raise ValueError(f"Model {name} has not been approved for download: unclear license")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / model["filename"]
    record = destination / f"{name}.provenance.json"
    if target.is_file() and not force:
        if record.is_file():
            provenance = json.loads(record.read_text())
            if provenance.get("model_sha256") != sha256_file(target):
                raise ValueError(
                    f"Installed {name} checksum differs from provenance; inspect it or use --force."
                )
        LOG.info("Already installed: %s", target)
        return target
    LOG.info("Downloading %s from %s", name, model["url"])
    LOG.info("Upstream license: %s (%s)", model["license"], model["license_url"])
    with tempfile.TemporaryDirectory(prefix="model-download-", dir=destination) as temporary:
        archive = Path(temporary) / "download.zip"
        request = Request(model["url"], headers={"User-Agent": "PostureFallDetection/0.1"})
        with urlopen(request, timeout=60) as response, archive.open("wb") as output:
            if response.status != 200:
                raise RuntimeError(f"Download failed with HTTP {response.status}")
            shutil.copyfileobj(response, output, 1024 * 1024)
        checksum = sha256_file(archive)
        expected = expected_checksum or model.get("sha256")
        if expected and checksum.lower() != expected.lower():
            raise ValueError(f"Archive checksum mismatch: got {checksum}, expected {expected}")
        with zipfile.ZipFile(archive) as bundle:
            candidates = [info for info in bundle.infolist() if info.filename.endswith(".onnx")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one self-contained ONNX file, found {len(candidates)}")
            model_file = Path(temporary) / "model.onnx"
            # Stream only selected content: no extraction paths from the archive.
            with bundle.open(candidates[0]) as source, model_file.open("wb") as output:
                shutil.copyfileobj(source, output)
            if not model_file.stat().st_size:
                raise ValueError("Downloaded ONNX model is empty")
            metadata = {}
            for info in bundle.infolist():
                if info.filename.endswith(".json") and info.file_size < 2_000_000:
                    metadata[Path(info.filename).name] = json.loads(bundle.read(info))
        model_checksum = sha256_file(model_file)
        provenance = {
            "model": name,
            "source_url": model["url"],
            "source_page": model["source"],
            "license": model["license"],
            "license_url": model["license_url"],
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "archive_sha256": checksum,
            "model_sha256": model_checksum,
            "checksum_verified_against_expected": bool(expected),
            "size_bytes": model_file.stat().st_size,
            "sdk_metadata": metadata,
        }
        # Atomic model and provenance installation after complete validation.
        model_file.replace(target)
        temporary_record = Path(temporary) / "provenance.json"
        temporary_record.write_text(json.dumps(provenance, indent=2) + "\n")
        temporary_record.replace(record)
        LOG.info(
            "Installed %s (%s bytes); ONNX SHA256 %s", target, target.stat().st_size, model_checksum
        )
        if not expected:
            LOG.info(
                "SHA256 recorded for repeatability; no upstream published checksum was available for pre-download validation."
            )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["rtmo_m", "rtmpose_m", "yolox_tiny", "rtmpose_bundle"], default="rtmo_m"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument(
        "--sha256", help="Expected SHA256 of downloaded archive (single model only)"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        names = ["rtmpose_m", "yolox_tiny"] if args.model == "rtmpose_bundle" else [args.model]
        if len(names) > 1 and args.sha256:
            parser.error("--sha256 accepts one model; download the bundle components separately")
        for name in names:
            download_model(name, args.output, args.sha256, args.force)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        LOG.error("Model download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
