from collections.abc import Callable
from pathlib import Path

import pytest

from bookmatch_ml.config import (
    ConfigError,
    load_feature_config,
    load_ranking_config,
    load_reader_config,
)

ROOT = Path(__file__).parents[1]


def test_feature_config_has_versions_and_content_hash() -> None:
    loaded = load_feature_config(ROOT / "configs" / "features.yaml")

    assert loaded.config.config_version == "features-v1"
    assert loaded.config.concept_profile_version == "concept-v1"
    assert loaded.config.difficulty_profile_version == "difficulty-v1"
    assert loaded.content_hash.startswith("sha256:")


def test_invalid_weight_configuration_fails_visibly(tmp_path: Path) -> None:
    original = (ROOT / "configs" / "features.yaml").read_text(encoding="utf-8")
    invalid = original.replace("mean_token_length: 0.4", "mean_token_length: 0.5")
    path = tmp_path / "features.yaml"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(ConfigError, match="lexical_weights must sum to 1.0"):
        load_feature_config(path)


@pytest.mark.parametrize(
    ("filename", "loader"),
    [
        ("features.yaml", load_feature_config),
        ("reader.yaml", load_reader_config),
        ("ranking.yaml", load_ranking_config),
    ],
)
def test_duplicate_yaml_keys_fail_before_model_validation(
    tmp_path: Path,
    filename: str,
    loader: Callable[[Path], object],
) -> None:
    source = (ROOT / "configs" / filename).read_text(encoding="utf-8")
    first_line = source.splitlines()[0]
    path = tmp_path / filename
    path.write_text(f"{first_line}\n{source}", encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate mapping key 'config_version'"):
        loader(path)


def test_reader_config_is_versioned_and_strict() -> None:
    loaded = load_reader_config(ROOT / "configs" / "reader.yaml")

    assert loaded.config.config_version == "reader-config-v1"
    assert loaded.config.profile_version == "reader-v1"
    assert loaded.config.difficulty_weights == {"easy": 1.0, "medium": 1.25, "hard": 1.5}
    assert loaded.content_hash.startswith("sha256:")


def test_ranking_config_is_versioned_and_weights_are_normalized() -> None:
    loaded = load_ranking_config(ROOT / "configs" / "ranking.yaml")

    assert loaded.config.config_version == "ranking-config-v1"
    assert loaded.config.model_version == "rank-v1"
    assert sum(loaded.config.component_weights.values()) == pytest.approx(1.0)
    assert sum(loaded.config.knowledge_weights.values()) == pytest.approx(1.0)
    assert loaded.content_hash.startswith("sha256:")


def test_invalid_ranking_weights_fail_visibly(tmp_path: Path) -> None:
    source = (ROOT / "configs" / "ranking.yaml").read_text(encoding="utf-8")
    path = tmp_path / "ranking.yaml"
    path.write_text(source.replace("topic_fit: 0.35", "topic_fit: 0.45"), encoding="utf-8")

    with pytest.raises(ConfigError, match="component_weights must sum to 1.0"):
        load_ranking_config(path)
