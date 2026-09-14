from pathlib import Path

import pytest

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.io import write_jsonl
from bookmatch_ml.ranking.loader import (
    RankingInputError,
    load_book_profiles,
    load_reader_profile,
)

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"


def _profiles():
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    return build_book_profiles(evidence, load_feature_config(ROOT / "configs" / "features.yaml"))


def test_generated_profiles_load_strictly(tmp_path: Path) -> None:
    path = tmp_path / "books.jsonl"
    write_jsonl(_profiles(), path)

    books = load_book_profiles(path)
    reader = load_reader_profile(ROOT / "examples" / "reader_profile.json")

    assert len(books) == 2
    assert reader.topic_id == "operating-systems"


def test_duplicate_book_profiles_fail_visibly(tmp_path: Path) -> None:
    path = tmp_path / "books.jsonl"
    profile = _profiles()[0]
    write_jsonl([profile, profile], path)

    with pytest.raises(RankingInputError, match="duplicate book profiles"):
        load_book_profiles(path)


def test_empty_book_profiles_fail_visibly(tmp_path: Path) -> None:
    path = tmp_path / "books.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(RankingInputError, match="book profiles are empty"):
        load_book_profiles(path)
