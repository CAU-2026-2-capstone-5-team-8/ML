"""Experimental curriculum rubric over existing TOC coverage/prerequisite projections."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bookmatch_ml.concept_v2.profile import BookConceptProfileV2
from bookmatch_ml.config import _load_unique_key_yaml
from bookmatch_ml.schemas import ReaderProfile


class DifficultyPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)
    model_version: str
    config_version: str
    prerequisite_weight: float = Field(gt=0, lt=1)
    easy_max: float = Field(ge=0, le=1)
    manageable_max: float = Field(ge=0, le=1)
    target_burden: float = Field(gt=0, lt=1)
    minimum_assessment_coverage: float = Field(gt=0, le=1)
    levels: dict[str, dict[str, Literal[1, 2, 3]]]

    @model_validator(mode="after")
    def ordered_thresholds(self):
        if not self.easy_max < self.target_burden < self.manageable_max:
            raise ValueError("thresholds must satisfy easy < target < manageable")
        if not self.levels or any(not levels for levels in self.levels.values()):
            raise ValueError("concept levels must not be empty")
        return self


def load_difficulty_policy(path: Path) -> tuple[DifficultyPolicy, str]:
    content = path.read_bytes()
    return (
        DifficultyPolicy.model_validate(_load_unique_key_yaml(content)),
        "sha256:" + hashlib.sha256(content).hexdigest(),
    )


def score_difficulty(
    reader: ReaderProfile,
    book: BookConceptProfileV2,
    loaded: tuple[DifficultyPolicy, str],
) -> dict:
    """Keep intrinsic concept level separate from this reader's learning burden.

    Unknown mastery ranges over [0,1]; bounds are logical bounds, not confidence intervals.
    Taught-before-use candidates are already removed by the existing profile builder.
    """
    policy, policy_hash = loaded
    if reader.topic_id != book.topic_id:
        raise ValueError("reader and book topic differ")
    levels = policy.levels[book.topic_id]
    covered = sorted({item.concept_id for item in book.covered_concepts})
    prerequisites = {item.concept_id: item.weight for item in book.prerequisite_requirements}
    missing = (set(covered) | set(prerequisites)) - levels.keys()
    if missing:
        raise ValueError(f"rubric missing concept levels: {sorted(missing)}")
    mastery = {item.concept_id: item.score for item in reader.concept_readiness}
    if len(mastery) != len(reader.concept_readiness):
        raise ValueError("duplicate concept readiness")

    # Each unique taught concept gets equal mass; repeated headings cannot raise its level.
    coefficients = {
        concept: (1 - policy.prerequisite_weight) * levels[concept] / 3 / len(covered)
        for concept in covered
    }
    total_prerequisite_weight = sum(prerequisites.values())
    prereq_coefficients = {
        concept: weight / total_prerequisite_weight
        for concept, weight in prerequisites.items()
        if total_prerequisite_weight > 0 and weight > 0
    }
    for concept, weight in prereq_coefficients.items():
        coefficients[concept] = coefficients.get(concept, 0) + policy.prerequisite_weight * weight

    def bounds(weights):
        lower = sum(w * (1 - mastery[c]) for c, w in weights.items() if c in mastery)
        upper = lower + sum(w for c, w in weights.items() if c not in mastery)
        return lower, upper

    def band(value):
        if value <= policy.easy_max:
            return "easy"
        if value <= policy.manageable_max:
            return "manageable"
        return "challenging"

    lower, upper = bounds(coefficients)
    gap_lower, gap_upper = bounds(prereq_coefficients)
    total = sum(coefficients.values())
    coverage = sum(w for c, w in coefficients.items() if c in mastery) / total if total else 0
    status = band(lower) if covered and band(lower) == band(upper) else "assessment_needed"
    if coverage < policy.minimum_assessment_coverage:
        status = "assessment_needed"
    if not covered:
        status = "no_concept_evidence"

    def target_fit(burden):
        target = policy.target_burden
        return max(0.0, 1 - abs(burden - target) / (target if burden < target else 1 - target))

    return {
        "book_id": book.book_id,
        "title": book.title,
        "topic_id": book.topic_id,
        "book_level_score": sum(levels[c] for c in covered) / len(covered) if covered else None,
        "concept_levels": {c: levels[c] for c in covered},
        "burden_lower": lower if covered else None,
        "burden_upper": upper if covered else None,
        "prerequisite_gap_lower": gap_lower,
        "prerequisite_gap_upper": gap_upper,
        "assessment_coverage": coverage,
        "difficulty_band": status,
        "recommendation_score": (
            min(target_fit(lower), target_fit(upper))
            if status in {"easy", "manageable", "challenging"}
            else None
        ),
        "next_assessment_concepts": sorted(
            (c for c in coefficients if c not in mastery), key=lambda c: (-coefficients[c], c)
        ),
        "external_prerequisites": sorted(prereq_coefficients),
        "taught_before_use": sorted({c.concept_id for c in book.taught_before_use_candidates}),
        "burden_contributions": {
            c: {"weight": w, "mastery": mastery.get(c)} for c, w in sorted(coefficients.items())
        },
        "model_version": policy.model_version,
        "policy_config_version": policy.config_version,
        "policy_hash": policy_hash,
        "graph_hash": book.graph_hash,
        "mapping_hash": book.matching_config_hash,
        "feature_hash": book.feature_config_hash,
        "toc_hash": book.toc_file_hash,
        "reader_config_hash": reader.config_hash,
    }
