"""Reproducible baseline evaluation helpers."""

from bookmatch_ml.evaluation.ablation import (
    build_evidence_ablation_report,
    summarize_evidence_ablation,
)
from bookmatch_ml.evaluation.difficulty import (
    calculate_pairwise_agreement,
    evaluate_book_difficulty,
    spearman_rank_correlation,
)
from bookmatch_ml.evaluation.recommendation import compare_recommendation_baselines
from bookmatch_ml.evaluation.report import build_evaluation_report

__all__ = [
    "build_evaluation_report",
    "build_evidence_ablation_report",
    "calculate_pairwise_agreement",
    "compare_recommendation_baselines",
    "evaluate_book_difficulty",
    "spearman_rank_correlation",
    "summarize_evidence_ablation",
]
