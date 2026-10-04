"""YAML configuration with validated defaults and portable project-relative paths."""

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Merge an optional profile over defaults. Models resolve relative to this checkout."""
    config = _read_yaml(ROOT / "configs/default.yaml")
    if path:
        config_path = Path(path).expanduser().resolve()
        override = _read_yaml(config_path)
        if override is None:
            override = {}
        if not isinstance(override, dict):
            raise ValueError("Configuration must be a YAML mapping")
        config = _merge(config, override)
    validate_config(config)
    for section, keys in {
        "pose": ("model_path", "detector_path"),
        "posture": ("model_path",),
        "fall": ("model_path",),
        "events": ("path",),
    }.items():
        for key in keys:
            if config.get(section, {}).get(key):
                candidate = Path(config[section][key]).expanduser()
                config[section][key] = str(
                    candidate if candidate.is_absolute() else ROOT / candidate
                )
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in (
        "camera",
        "pose",
        "tracking",
        "posture",
        "fall",
        "state",
        "runtime",
        "visualization",
        "events",
        "logging",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a YAML mapping")
    positive = (
        ("camera", "width"),
        ("camera", "height"),
        ("camera", "fps"),
        ("pose", "inference_fps"),
        ("fall", "history_seconds"),
        ("fall", "confirmation_frames"),
        ("fall", "sample_count"),
        ("runtime", "intra_op_threads"),
    )
    for section, key in positive:
        value = config[section][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{section}.{key} must be positive")
    for section, key in (
        ("pose", "confidence_threshold"),
        ("fall", "possible_threshold"),
        ("fall", "confirmed_threshold"),
    ):
        value = config[section][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{section}.{key} must be between 0 and 1")
    if config["fall"]["possible_threshold"] >= config["fall"]["confirmed_threshold"]:
        raise ValueError("fall.possible_threshold must be below confirmed_threshold")
    if config["fall"]["detector"] not in {"heuristic", "stgcn"}:
        raise ValueError("fall.detector must be heuristic or stgcn")
    if config["posture"].get("backend", "rules") not in {"rules", "mlp"}:
        raise ValueError("posture.backend must be rules or mlp")
    count_keys = {
        "camera": ("index", "width", "height"),
        "pose": ("max_people",),
        "tracking": ("max_missing_frames",),
        "posture": ("smoothing_frames", "minimum_visible_joints"),
        "fall": ("sample_count", "max_samples", "confirmation_frames"),
        "state": ("posture_confirmation_frames", "possible_confirmation_frames"),
        "runtime": ("intra_op_threads",),
    }
    for section, keys in count_keys.items():
        for key in keys:
            value = config[section][key]
            minimum = 0 if key in {"index", "max_missing_frames"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
    for section, settings in config.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{section} must be a YAML mapping")
        for key, value in settings.items():
            numeric_setting = (
                "confidence" in key
                or key.endswith(
                    ("_seconds", "_angle", "_height", "_velocity", "_drop", "_rotation", "_ratio")
                )
                or key
                in {
                    "iou_weight",
                    "velocity_smoothing",
                    "matching_threshold",
                    "max_skeleton_distance",
                    "nms_threshold",
                }
            )
            if numeric_setting and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{section}.{key} must be numeric")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"{section}.{key} must be finite")
                if key.endswith("_seconds") and value < 0:
                    raise ValueError(f"{section}.{key} cannot be negative")
                if (
                    "confidence" in key
                    or key
                    in {"nms_threshold", "matching_threshold", "iou_weight", "velocity_smoothing"}
                ) and not 0 <= value <= 1:
                    raise ValueError(f"{section}.{key} must be between 0 and 1")
    if config["fall"]["sample_count"] < 2 or config["fall"]["max_samples"] < 2:
        raise ValueError("Fall history requires at least two samples")
    if config["state"]["posture_confirmation_frames"] > config["posture"]["smoothing_frames"]:
        raise ValueError("state.posture_confirmation_frames cannot exceed posture.smoothing_frames")
