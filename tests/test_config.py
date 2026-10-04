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
