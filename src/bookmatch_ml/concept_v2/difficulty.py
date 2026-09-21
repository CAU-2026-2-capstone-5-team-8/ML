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
    book_prerequisite_weight: float = Field(ge=0, lt=1)
    introductory_max: float = Field(ge=0, le=1)
    intermediate_max: float = Field(ge=0, le=1)
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
        if not self.introductory_max < self.intermediate_max:
            raise ValueError("book thresholds must satisfy introductory < intermediate")
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
    learning_coefficients = {concept: levels[concept] / 3 / len(covered) for concept in covered}
    total_prerequisite_weight = sum(prerequisites.values())
    prereq_coefficients = {
        concept: weight / total_prerequisite_weight
        for concept, weight in prerequisites.items()
        if total_prerequisite_weight > 0 and weight > 0
    }
    burden_configured_weights = {
        "covered_learning": 1 - policy.prerequisite_weight,
        "external_prerequisite_gap": policy.prerequisite_weight,
    }
    burden_active_weights = {"covered_learning": burden_configured_weights["covered_learning"]}
    if prereq_coefficients:
        burden_active_weights["external_prerequisite_gap"] = burden_configured_weights[
            "external_prerequisite_gap"
        ]
    active_total = sum(burden_active_weights.values())
    burden_active_weights = {
        name: weight / active_total for name, weight in burden_active_weights.items()
    }
    coefficients = {
        concept: burden_active_weights["covered_learning"] * weight
        for concept, weight in learning_coefficients.items()
    }
    prerequisite_active_weight = burden_active_weights.get("external_prerequisite_gap", 0)
    for concept, weight in prereq_coefficients.items():
        coefficients[concept] = coefficients.get(concept, 0) + prerequisite_active_weight * weight

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

    def book_band(value):
        if value <= policy.introductory_max:
            return "introductory"
        if value <= policy.intermediate_max:
            return "intermediate"
        return "advanced"

    content_level_score = (
        sum((levels[concept] - 1) / 2 for concept in covered) / len(covered) if covered else None
    )
    external_prerequisite_demand = None
    if prereq_coefficients:
        external_prerequisite_demand = sum(
            prereq_coefficients[concept] * levels[concept] / 3 for concept in prereq_coefficients
        )
    book_active_weights = {"covered_content_level": 1 - policy.book_prerequisite_weight}
    if external_prerequisite_demand is not None:
        book_active_weights["external_prerequisite_demand"] = policy.book_prerequisite_weight
    book_weight_total = sum(book_active_weights.values())
    book_active_weights = {
        name: weight / book_weight_total for name, weight in book_active_weights.items()
    }
    book_difficulty_score = None
    if content_level_score is not None:
        book_difficulty_score = book_active_weights["covered_content_level"] * content_level_score
        if external_prerequisite_demand is not None:
            book_difficulty_score += (
                book_active_weights["external_prerequisite_demand"] * external_prerequisite_demand
            )

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
        "book_difficulty_score": book_difficulty_score,
        "book_difficulty_band": (
            book_band(book_difficulty_score) if book_difficulty_score is not None else None
        ),
        "book_difficulty_components": {
            "covered_content_level": content_level_score,
            "external_prerequisite_demand": external_prerequisite_demand,
        },
        "book_difficulty_active_weights": book_active_weights if covered else {},
        "concept_levels": {c: levels[c] for c in covered},
        "burden_lower": lower if covered else None,
        "burden_upper": upper if covered else None,
        "prerequisite_gap_lower": gap_lower,
        "prerequisite_gap_upper": gap_upper,
        "assessment_coverage": coverage,
        "burden_active_weights": burden_active_weights if covered else {},
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
        "concept_evidence": {
            concept: [
                {
                    "toc_entry_id": mapping.toc_entry_id,
                    "toc_title": mapping.toc_title,
                    "toc_path": mapping.toc_path,
                    "matching_alias": mapping.matching_alias,
                    "match_method": mapping.match_method,
                }
                for mapping in book.toc_mappings
                if mapping.concept_id == concept
            ]
            for concept in covered
        },
        "prerequisite_evidence": {
            item.concept_id: {
                "weight": item.weight,
                "related_target_concepts": item.related_target_concepts,
                "graph_paths": item.graph_paths,
                "evidence_toc_entry_ids": item.evidence_toc_entry_ids,
                "relation_source_types": item.relation_source_types,
            }
            for item in book.prerequisite_requirements
        },
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
