"""Small deterministic text utilities used by the baseline features."""

import re
import unicodedata

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", flags=re.UNICODE)
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def tokenize(value: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(value)]


def sentences(value: str) -> list[str]:
    return [part.strip() for part in SENTENCE_BOUNDARY_PATTERN.split(value) if tokenize(part)]


def count_alias_mentions(value: str, aliases: list[str]) -> int:
    """Count longest non-overlapping aliases so nested aliases are not double-counted."""

    normalized = normalize_text(value)
    if not normalized:
        return 0

    occupied: list[tuple[int, int]] = []
    alias_patterns = sorted(
        {normalize_text(alias) for alias in aliases if normalize_text(alias)},
        key=lambda alias: (-len(alias.split()), -len(alias), alias),
    )
    for alias in alias_patterns:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for match in pattern.finditer(normalized):
            span = match.span()
            if not any(span[0] < end and start < span[1] for start, end in occupied):
                occupied.append(span)
    return len(occupied)


def contains_any_phrase(value: str, phrases: list[str]) -> bool:
    normalized = normalize_text(value)
    return any(
        re.search(rf"(?<!\w){re.escape(normalize_text(phrase))}(?!\w)", normalized)
        for phrase in phrases
    )
