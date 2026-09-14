import json
from pathlib import Path

import pytest

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.config import load_evaluation_config, load_feature_config, load_ranking_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.evaluation.ablation import (
    build_evidence_ablation_report,
    summarize_evidence_ablation,
)
from bookmatch_ml.evaluation.difficulty import (
    EvaluationDataError,
    calculate_pairwise_agreement,
    evaluate_book_difficulty,
    load_difficulty_judgments,
    spearman_rank_correlation,
)
from bookmatch_ml.evaluation.recommendation import compare_recommendation_baselines
from bookmatch_ml.ranking.loader import load_reader_profile
from bookmatch_ml.schemas import (
    AblationComparison,
    BookProfile,
    DifficultyEvaluationItem,
    DifficultyEvaluationReport,
)

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"


def _profiles() -> list[BookProfile]:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    return build_book_profiles(evidence, load_feature_config(ROOT / "configs" / "features.yaml"))


def _as_operating_systems(book: BookProfile) -> BookProfile:
    concept = book.concept_profile.model_copy(
        update={"topic_distribution": {"operating-systems": 1.0}}
    )
    return book.model_copy(update={"concept_profile": concept})


def test_spearman_and_pairwise_agreement_are_correct() -> None:
    human_ranks = [1.0, 2.0, 3.0]
    system_scores = [0.1, 0.3, 0.2]

    assert spearman_rank_correlation(human_ranks, system_scores) == pytest.approx(0.5)
    agreement, pair_count, agreed_count, tie_count = calculate_pairwise_agreement(
        human_ranks, system_scores
    )
    assert agreement == pytest.approx(2 / 3)
    assert (pair_count, agreed_count, tie_count) == (3, 2, 0)


def test_pairwise_system_ties_are_recorded_as_non_agreements() -> None:
    agreement, pair_count, agreed_count, tie_count = calculate_pairwise_agreement(
        [1.0, 2.0, 3.0], [0.5, 0.5, 0.8]
    )

    assert agreement == pytest.approx(2 / 3)
    assert (pair_count, agreed_count, tie_count) == (3, 2, 1)


def test_invalid_difficulty_order_fails_visibly(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "label_version": "test-v1",
                "is_synthetic": True,
                "topic_id": "operating-systems",
                "items": [
                    {"book_id": "book-a", "human_rank": 1},
                    {"book_id": "book-b", "human_rank": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDataError, match="duplicate human_rank"):
        load_difficulty_judgments(path)


def test_missing_prose_is_excluded_then_insufficient_data_fails(tmp_path: Path) -> None:
    books = _profiles()
    books[1] = _as_operating_systems(books[1])
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "label_version": "test-v1",
                "is_synthetic": True,
                "topic_id": "operating-systems",
                "items": [
                    {"book_id": books[0].book_id, "human_rank": 1},
                    {"book_id": books[1].book_id, "human_rank": 2},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationDataError, match="need 2, found 1"):
        evaluate_book_difficulty(
            load_difficulty_judgments(path),
            books,
            load_evaluation_config(ROOT / "configs" / "evaluation.yaml"),
        )


def test_topic_only_and_readiness_rankings_are_compared() -> None:
    books = _profiles()
    books[1] = _as_operating_systems(books[1])
    reader = load_reader_profile(ROOT / "examples" / "reader_profile.json")

    report = compare_recommendation_baselines(
        reader,
        books,
        load_ranking_config(ROOT / "configs" / "ranking.yaml"),
        load_evaluation_config(ROOT / "configs" / "evaluation.yaml"),
    )

    assert report.candidate_count == 2
    assert report.comparison_k == 2
    assert report.top_k_overlap_rate == 1.0
    assert report.changed_position_count == 2
    assert report.items[0].book_id == books[1].book_id
    assert report.items[0].component_weight_coverage == pytest.approx(0.35)


def test_ablation_summary_records_profile_changes() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))[0]
    report = build_evidence_ablation_report(
        evidence, load_feature_config(ROOT / "configs" / "features.yaml")
    )

    comparisons = summarize_evidence_ablation(report)

    assert len(comparisons) == 2
    assert comparisons[0].from_mode == "toc_only"
    assert "process" in comparisons[1].changed_concept_weights
    assert comparisons[1].newly_available_difficulty_components == [
        "lexical_difficulty",
        "syntactic_complexity",
        "concept_density",
        "prerequisite_demand",
    ]


def test_difficulty_evaluation_item_rejects_inconsistent_components() -> None:
    components = {
        "lexical_difficulty": 0.2,
        "syntactic_complexity": 0.3,
        "concept_density": None,
        "prerequisite_demand": None,
    }
    with pytest.raises(ValueError, match="match non-null component keys"):
        DifficultyEvaluationItem(
            book_id="book-a",
            human_rank=1,
            system_difficulty=0.25,
            components=components,
            active_weights={"lexical_difficulty": 1.0},
        )

    components["lexical_difficulty"] = 1.1
    with pytest.raises(ValueError, match=r"component values must be in \[0, 1\]"):
        DifficultyEvaluationItem(
            book_id="book-a",
            human_rank=1,
            system_difficulty=0.25,
            components=components,
            active_weights={
                "lexical_difficulty": 0.5,
                "syntactic_complexity": 0.5,
            },
        )

    components["lexical_difficulty"] = 0.2
    with pytest.raises(ValueError, match="match weighted difficulty components"):
        DifficultyEvaluationItem(
            book_id="book-a",
            human_rank=1,
            system_difficulty=0.9,
            components=components,
            active_weights={
                "lexical_difficulty": 0.5,
                "syntactic_complexity": 0.5,
            },
        )


def _difficulty_evaluation_item(
    book_id: str,
    human_rank: int,
    score: float,
) -> DifficultyEvaluationItem:
    return DifficultyEvaluationItem(
        book_id=book_id,
        human_rank=human_rank,
        system_difficulty=score,
        components={
            "lexical_difficulty": score,
            "syntactic_complexity": score,
            "concept_density": score,
            "prerequisite_demand": score,
        },
        active_weights={
            "lexical_difficulty": 0.25,
            "syntactic_complexity": 0.25,
            "concept_density": 0.25,
            "prerequisite_demand": 0.25,
        },
    )


def test_difficulty_report_derives_pair_summaries_from_items() -> None:
    items = [
        _difficulty_evaluation_item("book-a", 1, 0.5),
        _difficulty_evaluation_item("book-b", 2, 0.5),
    ]
    with pytest.raises(ValueError, match="agreed_pair_count must match difficulty items"):
        DifficultyEvaluationReport(
            topic_id="operating-systems",
            label_version="test-v1",
            label_hash="sha256:" + "0" * 64,
            is_synthetic=True,
            comparable_book_count=2,
            excluded_book_ids=[],
            spearman_correlation=None,
            pairwise_agreement=1.0,
            pair_count=1,
            agreed_pair_count=1,
            system_tie_pair_count=0,
            items=items,
            warnings=[],
        )

    items[1] = _difficulty_evaluation_item("book-b", 1, 0.6)
    with pytest.raises(ValueError, match="duplicate human_rank"):
        DifficultyEvaluationReport(
            topic_id="operating-systems",
            label_version="test-v1",
            label_hash="sha256:" + "0" * 64,
            is_synthetic=True,
            comparable_book_count=2,
            excluded_book_ids=[],
            spearman_correlation=None,
            pairwise_agreement=1.0,
            pair_count=1,
            agreed_pair_count=1,
            system_tie_pair_count=0,
            items=items,
            warnings=[],
        )


def test_ablation_comparison_rejects_non_adjacent_modes() -> None:
    with pytest.raises(ValueError, match="adjacent evidence modes"):
        AblationComparison(
            from_mode="toc_only",
            to_mode="all_available",
            added_concepts=[],
            removed_concepts=[],
            changed_concept_weights=[],
            added_prerequisites=[],
            removed_prerequisites=[],
            newly_available_difficulty_components=[],
        )
