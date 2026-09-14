"""Human-relative book-difficulty evaluation."""

import hashlib
import math
from pathlib import Path

from pydantic import ValidationError

from bookmatch_ml.config import LoadedEvaluationConfig
from bookmatch_ml.schemas import (
    BookProfile,
    DifficultyComponentName,
    DifficultyEvaluationItem,
    DifficultyEvaluationReport,
    DifficultyJudgments,
    LoadedDifficultyJudgments,
)


class EvaluationDataError(ValueError):
    """Evaluation inputs are invalid or insufficient for a requested metric."""


def load_difficulty_judgments(path: Path) -> LoadedDifficultyJudgments:
    """Load strict human-order JSON while retaining its exact content hash."""

    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvaluationDataError(f"cannot read difficulty judgments: {path}: {exc}") from exc
    try:
        judgments = DifficultyJudgments.model_validate_json(content)
    except (ValidationError, ValueError) as exc:
        raise EvaluationDataError(f"invalid difficulty judgments: {path}: {exc}") from exc
    return LoadedDifficultyJudgments(
        judgments=judgments,
        content_hash=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index, _ in indexed[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(
    human_ranks: list[float],
    system_scores: list[float],
) -> float | None:
    """Return Spearman's rho with average ranks, or null for a constant ordering."""

    if len(human_ranks) != len(system_scores) or len(human_ranks) < 2:
        raise EvaluationDataError("Spearman correlation requires equal inputs of length at least 2")
    human = _average_ranks(human_ranks)
    system = _average_ranks(system_scores)
    human_mean = sum(human) / len(human)
    system_mean = sum(system) / len(system)
    numerator = sum(
        (human_value - human_mean) * (system_value - system_mean)
        for human_value, system_value in zip(human, system, strict=True)
    )
    human_variance = sum((value - human_mean) ** 2 for value in human)
    system_variance = sum((value - system_mean) ** 2 for value in system)
    denominator = math.sqrt(human_variance * system_variance)
    if denominator == 0:
        return None
    return max(-1.0, min(1.0, numerator / denominator))


def calculate_pairwise_agreement(
    human_ranks: list[float],
    system_scores: list[float],
) -> tuple[float, int, int, int]:
    """Return strict ordered-pair agreement; system ties count as non-agreements."""

    if len(human_ranks) != len(system_scores) or len(human_ranks) < 2:
        raise EvaluationDataError("pairwise agreement requires equal inputs of length at least 2")
    pair_count = 0
    agreed_count = 0
    system_tie_count = 0
    for left in range(len(human_ranks) - 1):
        for right in range(left + 1, len(human_ranks)):
            pair_count += 1
            if system_scores[left] == system_scores[right]:
                system_tie_count += 1
                continue
            human_order = human_ranks[left] < human_ranks[right]
            system_order = system_scores[left] < system_scores[right]
            if human_order == system_order:
                agreed_count += 1
    return agreed_count / pair_count, pair_count, agreed_count, system_tie_count


def _difficulty_item(
    book: BookProfile,
    human_rank: int,
    configured_weights: dict[str, float],
) -> DifficultyEvaluationItem | None:
    difficulty = book.difficulty_profile
    components: dict[DifficultyComponentName, float | None] = {
        "lexical_difficulty": difficulty.lexical_difficulty,
        "syntactic_complexity": difficulty.syntactic_complexity,
        "concept_density": difficulty.concept_density,
        "prerequisite_demand": difficulty.prerequisite_demand,
    }
    available = {name: value for name, value in components.items() if value is not None}
    active_total = sum(configured_weights[name] for name in available)
    if not available or active_total == 0:
        return None
    active_weights: dict[DifficultyComponentName, float] = {
        name: configured_weights[name] / active_total for name in components if name in available
    }
    score = sum(available[name] * weight for name, weight in active_weights.items())
    return DifficultyEvaluationItem(
        book_id=book.book_id,
        human_rank=human_rank,
        system_difficulty=score,
        components=components,
        active_weights=active_weights,
    )


def evaluate_book_difficulty(
    loaded_judgments: LoadedDifficultyJudgments,
    books: list[BookProfile],
    loaded_config: LoadedEvaluationConfig,
) -> DifficultyEvaluationReport:
    """Compare human relative order with reproducible aggregate difficulty scores."""

    judgments = loaded_judgments.judgments
    config = loaded_config.config.difficulty
    books_by_id = {book.book_id: book for book in books}
    missing = sorted(item.book_id for item in judgments.items if item.book_id not in books_by_id)
    if missing:
        raise EvaluationDataError(
            "difficulty judgments reference missing book profiles: " + ", ".join(missing)
        )

    items: list[DifficultyEvaluationItem] = []
    excluded: list[str] = []
    for judgment in sorted(judgments.items, key=lambda item: item.human_rank):
        book = books_by_id[judgment.book_id]
        if book.concept_profile.topic_distribution.get(judgments.topic_id, 0.0) <= 0:
            raise EvaluationDataError(
                f"difficulty judgment topic {judgments.topic_id} does not match book "
                f"{judgment.book_id}"
            )
        item = _difficulty_item(book, judgment.human_rank, config.component_weights)
        if item is None:
            excluded.append(judgment.book_id)
        else:
            items.append(item)
    if len(items) < config.minimum_comparable_books:
        raise EvaluationDataError(
            "insufficient comparable difficulty profiles: "
            f"need {config.minimum_comparable_books}, found {len(items)}"
        )

    human_ranks = [float(item.human_rank) for item in items]
    system_scores = [item.system_difficulty for item in items]
    spearman = spearman_rank_correlation(human_ranks, system_scores)
    pairwise, pair_count, agreed_count, tie_count = calculate_pairwise_agreement(
        human_ranks, system_scores
    )
    warnings: list[str] = []
    if judgments.is_synthetic:
        warnings.append("Difficulty judgments are synthetic and do not support validity claims.")
    if len(items) < config.small_sample_warning_threshold:
        warnings.append(
            f"Only {len(items)} books were comparable; do not over-interpret these metrics."
        )
    if excluded:
        warnings.append(
            f"Excluded {len(excluded)} labeled books because prose difficulty was unavailable."
        )
    if spearman is None:
        warnings.append("Spearman correlation is unavailable because one ordering is constant.")

    return DifficultyEvaluationReport(
        topic_id=judgments.topic_id,
        label_version=judgments.label_version,
        label_hash=loaded_judgments.content_hash,
        is_synthetic=judgments.is_synthetic,
        comparable_book_count=len(items),
        excluded_book_ids=sorted(excluded),
        spearman_correlation=spearman,
        pairwise_agreement=pairwise,
        pair_count=pair_count,
        agreed_pair_count=agreed_count,
        system_tie_pair_count=tie_count,
        items=items,
        warnings=warnings,
    )
