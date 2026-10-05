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
        "tracking": ("max_missing_frames", "minimum_common_joints"),
        "posture": ("smoothing_frames", "minimum_visible_joints"),
        "fall": (
            "sample_count",
            "max_samples",
            "confirmation_frames",
            "minimum_confident_joints_per_pair",
        ),
        "state": ("posture_confirmation_frames", "possible_confirmation_frames"),
        "runtime": ("intra_op_threads",),
        "visualization": ("max_overlay_frame_lag",),
    }
    for section, keys in count_keys.items():
        for key in keys:
            value = config[section][key]
            minimum = 0 if key in {"index", "max_missing_frames", "max_overlay_frame_lag"} else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
    for section, settings in config.items():
        if not isinstance(settings, dict):
            raise ValueError(f"{section} must be a YAML mapping")
        for key, value in settings.items():
            numeric_setting = (
                "confidence" in key
                or key.endswith(
                    (
                        "_seconds",
                        "_angle",
                        "_height",
                        "_velocity",
                        "_drop",
                        "_rotation",
                        "_ratio",
                        "_distance",
                        "_tolerance",
                    )
                )
                or key
                in {
                    "iou_weight",
                    "velocity_smoothing",
                    "matching_threshold",
                    "max_skeleton_distance",
                    "nms_threshold",
                    "ambiguity_margin",
                    "max_center_distance",
                    "slow_jitter_tolerance",
                    "episode_match_distance",
                }
                or key.endswith(
                    ("_weight", "_quality", "_fraction", "_margin", "_iou", "_probability")
                )
            )
            if numeric_setting and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{section}.{key} must be numeric")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    raise ValueError(f"{section}.{key} must be finite")
                if key.endswith("_seconds") and value < 0:
                    raise ValueError(f"{section}.{key} cannot be negative")
                if (
                    ("confidence" in key and not key.endswith("_margin"))
                    or key
                    in {"nms_threshold", "matching_threshold", "iou_weight", "velocity_smoothing"}
                ) and not 0 <= value <= 1:
                    raise ValueError(f"{section}.{key} must be between 0 and 1")
                if (
                    key.endswith(("_quality", "_fraction", "_weight", "_iou", "_probability"))
                    or key == "minimum_sparse_iou"
                ) and not 0 <= value <= 1:
                    raise ValueError(f"{section}.{key} must be between 0 and 1")
                if numeric_setting and value < 0:
                    raise ValueError(f"{section}.{key} cannot be negative")
    if config["fall"]["sample_count"] < 2 or config["fall"]["max_samples"] < 2:
        raise ValueError("Fall history requires at least two samples")
    if config["state"]["posture_confirmation_frames"] > config["posture"]["smoothing_frames"]:
        raise ValueError("state.posture_confirmation_frames cannot exceed posture.smoothing_frames")
    for section, keys in {
        "tracking": (
            "matching_threshold",
            "max_missing_seconds",
            "max_center_distance",
            "max_skeleton_distance",
            "ambiguity_margin",
            "scale_ema_seconds",
            "scale_window_seconds",
        ),
        "posture": (
            "confidence_angle_margin",
            "confidence_knee_margin",
            "confidence_height_margin",
            "confidence_aspect_margin",
        ),
        "fall": (
            "max_gap_seconds",
            "max_unreliable_seconds",
            "possible_confirmation_seconds",
            "fall_confirmation_seconds",
            "minimum_motion_seconds",
            "motion_window_seconds",
            "slow_window_seconds",
            "slow_minimum_motion_seconds",
            "controlled_confirmation_seconds",
            "controlled_window_seconds",
            "controlled_hold_seconds",
        ),
        "state": ("rearm_seconds", "max_prediction_gap_seconds"),
        "events": ("episode_retention_seconds", "episode_match_distance"),
    }.items():
        for key in keys:
            if config[section][key] <= 0:
                raise ValueError(f"{section}.{key} must be positive")
    for key in ("max_scale_ratio", "scale_max_change_ratio"):
        if config["tracking"][key] <= 1:
            raise ValueError(f"tracking.{key} must exceed one")
    if (
        sum(
            config["tracking"][key]
            for key in ("iou_weight", "center_weight", "skeleton_weight", "scale_weight")
        )
        <= 0
    ):
        raise ValueError("At least one association weight must be positive")
    if config["tracking"]["minimum_common_joints"] > 17:
        raise ValueError("tracking.minimum_common_joints cannot exceed 17")
    if config["fall"]["minimum_confident_joints_per_pair"] > 2:
        raise ValueError("fall.minimum_confident_joints_per_pair cannot exceed 2")
    fall = config["fall"]
    if (
        fall["detector"] == "heuristic"
        and not fall["minimum_motion_seconds"]
        < fall["motion_window_seconds"]
        <= fall["history_seconds"]
    ):
        raise ValueError("Rapid motion window must fit fall history")
    if (
        fall["detector"] == "heuristic"
        and fall["slow_enabled"]
        and not fall["slow_minimum_motion_seconds"]
        < fall["slow_window_seconds"]
        <= fall["history_seconds"]
    ):
        raise ValueError("Slow motion window must fit fall history")
    if (
        fall["slow_minimum_hip_velocity"] >= fall["slow_maximum_hip_velocity"]
        or fall["minimum_hip_velocity"] >= fall["maximum_hip_velocity"]
    ):
        raise ValueError("Minimum velocity must be below maximum velocity")
    if fall["detector"] == "stgcn" and fall["min_history_seconds"] > fall["history_seconds"]:
        raise ValueError("fall.min_history_seconds cannot exceed history_seconds")
    for key in ("slow_enabled", "controlled_lowering_enabled"):
        if not isinstance(fall[key], bool):
            raise ValueError(f"fall.{key} must be boolean")
    if fall["controlled_origin_knee_angle"] <= fall["controlled_knee_angle"]:
        raise ValueError("Controlled lowering requires distinct straight/flexed knee thresholds")
    if (
        fall["detector"] == "heuristic"
        and fall["controlled_lowering_enabled"]
        and fall["controlled_hold_seconds"] < fall["history_seconds"]
    ):
        raise ValueError("controlled_hold_seconds must cover fall history to avoid delayed alerts")
    if fall["normalization"] != "window_root_v1":
        raise ValueError(
            "Fall models require window_root_v1 normalization; old weights require retraining"
        )
    for key in ("rearm_postures", "recovery_postures"):
        values = config["state"][key]
        if (
            not isinstance(values, list)
            or not values
            or not all(
                isinstance(value, str)
                and value in {"standing", "sitting", "bending", "squatting", "kneeling"}
                for value in values
            )
        ):
            raise ValueError(f"state.{key} must list supported upright/low postures")
