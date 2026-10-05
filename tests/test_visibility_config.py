import pytest

from src.config import load_config, validate_config


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("posture", "label_hysteresis_margin", -0.1),
        ("posture", "label_hysteresis_margin", 1.1),
        ("posture", "label_hysteresis_margin", "0.1"),
        ("fall", "partial_evidence_multiplier", 1),
        ("fall", "partial_evidence_multiplier", "1.2"),
        ("fall", "partial_evidence_multiplier", float("nan")),
        ("fall", "partial_max_torso_scale_ratio", 1),
        ("fall", "partial_lying_persistence_seconds", 0.2),
        ("fall", "partial_minimum_downward_fraction", 0.5),
        ("fall", "partial_minimum_downward_fraction", 1.1),
        ("fall", "partial_maximum_settling_velocity", 0),
        ("fall", "disappearance_possible_seconds", 0),
        ("fall", "disappearance_possible_seconds", float("inf")),
    ],
)
def test_new_visibility_settings_are_validated(section, key, value):
    config = load_config()
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_visibility_changes_preserve_models_runtime_tracking_and_cli_profiles():
    config = load_config("configs/mac.yaml")
    assert config["pose"]["model"] == "rtmo_m"
    assert config["fall"]["detector"] == "heuristic"
    assert config["runtime"]["provider"] == "auto"
    assert config["tracking"]["ambiguity_margin"] == 0.035
    assert config["posture"]["minimum_visible_joints"] == 6
    assert config["posture"]["confidence_threshold"] == 0.3
