"""Strict loaders for generated profile artifacts."""

from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from bookmatch_ml.schemas import BookProfile, ReaderProfile


class RankingInputError(ValueError):
    """Generated reader or book profile input is invalid."""


def load_reader_profile(path: Path) -> ReaderProfile:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RankingInputError(f"cannot read reader profile: {path}: {exc}") from exc
    try:
        return ReaderProfile.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise RankingInputError(f"invalid reader profile: {path}: {exc}") from exc


def load_book_profiles(path: Path) -> list[BookProfile]:
    try:
        stream = path.open(encoding="utf-8")
    except OSError as exc:
        raise RankingInputError(f"cannot read book profiles: {path}: {exc}") from exc

    profiles: list[BookProfile] = []
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                profiles.append(BookProfile.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise RankingInputError(
                    f"invalid book profiles {path} line {line_number}: {exc}"
                ) from exc
    duplicates = sorted(
        book_id
        for book_id, count in Counter(profile.book_id for profile in profiles).items()
        if count > 1
    )
    if duplicates:
        raise RankingInputError(f"duplicate book profiles: {', '.join(duplicates)}")
    if not profiles:
        raise RankingInputError(f"book profiles are empty: {path}")
    return profiles
