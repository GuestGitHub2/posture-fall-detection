from pathlib import Path

import pytest

from src.config import ROOT, load_config


def test_profile_merges_defaults_and_resolves_paths(tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text("pose:\n  inference_fps: 10\nfall:\n  confirmation_frames: 5\n")
    config = load_config(profile)
    assert config["pose"]["inference_fps"] == 10
    assert config["pose"]["model"] == "rtmo_m"
    assert config["fall"]["confirmation_frames"] == 5
    assert Path(config["pose"]["model_path"]).parent == ROOT / "models"


@pytest.mark.parametrize(
    "text",
    [
        "pose:\n  inference_fps: 0",
        "fall:\n  possible_threshold: 0.9",
        "[]",
        "pose:\n  confidence_threshold: '0.3'",
        "fall:\n  possible_threshold: null",
        "tracking:\n  iou_weight: 2",
        "tracking:\n  velocity_smoothing: 2",
        "fall:\n  history_seconds: .nan",
    ],
)
def test_invalid_settings_fail_clearly(tmp_path: Path, text: str) -> None:
    profile = tmp_path / "bad.yaml"
    profile.write_text(text)
    with pytest.raises(ValueError):
        load_config(profile)


@pytest.mark.parametrize(
    "text",
    [
        "tracking:\n  ambiguity_margin: 0",
        "tracking:\n  max_scale_ratio: 0.9",
        "tracking:\n  scale_max_change_ratio: 1",
        "tracking:\n  minimum_common_joints: 18",
        "tracking:\n  scale_ema_seconds: 0",
        "tracking:\n  minimum_association_confidence: 1.1",
        "posture:\n  confidence_knee_margin: 0",
        "fall:\n  max_unreliable_seconds: 0",
        "fall:\n  fall_confirmation_seconds: 0",
        "fall:\n  possible_confirmation_seconds: -1",
        "fall:\n  slow_window_seconds: 3",
        "fall:\n  slow_minimum_hip_velocity: 1",
        "fall:\n  slow_minimum_downward_fraction: 1.1",
        "fall:\n  slow_enabled: 'yes'",
        "fall:\n  normalization: hip_center_torso_plus_leg_length",
        "state:\n  rearm_postures: [lying]",
        "events:\n  episode_retention_seconds: 0",
        "visualization:\n  max_overlay_frame_lag: 1.5",
        "fall:\n  maximum_floor_distance: '0.2'",
    ],
)
def test_hardening_thresholds_are_validated(tmp_path, text):
    profile = tmp_path / "invalid.yaml"
    profile.write_text(text)
    with pytest.raises(ValueError):
        load_config(profile)


@pytest.mark.parametrize("count", [0, 3])
def test_joint_pair_requirement_is_validated(tmp_path, count):
    profile = tmp_path / "invalid.yaml"
    profile.write_text(f"fall:\n  minimum_confident_joints_per_pair: {count}\n")
    with pytest.raises(ValueError):
        load_config(profile)


@pytest.mark.parametrize(
    "text",
    [
        "fall:\n  detector: stgcn\n  history_seconds: 2.0",
        "fall:\n  history_seconds: 1.0\n  slow_enabled: false",
    ],
)
def test_unused_heuristic_windows_do_not_reject_compatible_profiles(tmp_path, text):
    profile = tmp_path / "valid.yaml"
    profile.write_text(text)
    assert load_config(profile)["fall"]["history_seconds"] > 0
