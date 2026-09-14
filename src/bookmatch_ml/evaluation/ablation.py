"""Evidence-richness ablation for one book profile."""

from itertools import pairwise
from typing import Literal

from bookmatch_ml.book.profile import build_book_profile
from bookmatch_ml.config import LoadedFeatureConfig
from bookmatch_ml.data.evidence import calculate_evidence_coverage
from bookmatch_ml.schemas import (
    AblationComparison,
    AblationConcept,
    AblationPrerequisite,
    AblationVariant,
    BookEvidence,
    EvidenceAblationReport,
)

EvidenceMode = Literal["toc_only", "toc_description", "all_available"]
_DESCRIPTION_TYPES = {"description", "publisher_summary"}


def _variant_evidence(evidence: BookEvidence, mode: EvidenceMode) -> BookEvidence:
    if mode == "toc_only":
        documents = []
    elif mode == "toc_description":
        documents = [
            document
            for document in evidence.documents
            if document.document_type in _DESCRIPTION_TYPES
        ]
    else:
        documents = evidence.documents
    return evidence.model_copy(
        update={
            "documents": documents,
            "coverage": calculate_evidence_coverage(documents, evidence.toc),
        }
    )


def build_evidence_ablation_report(
    evidence: BookEvidence,
    loaded_config: LoadedFeatureConfig,
) -> EvidenceAblationReport:
    variants: list[AblationVariant] = []
    modes: tuple[EvidenceMode, ...] = ("toc_only", "toc_description", "all_available")
    for mode in modes:
        profile = build_book_profile(_variant_evidence(evidence, mode), loaded_config)
        concept_profile = profile.concept_profile
        difficulty = profile.difficulty_profile
        variants.append(
            AblationVariant(
                evidence_mode=mode,
                concept_count=len(concept_profile.covered_concepts),
                covered_concepts=[
                    AblationConcept(concept=item.concept, weight=item.weight)
                    for item in concept_profile.covered_concepts
                ],
                prerequisite_concept_count=len(concept_profile.prerequisite_concepts),
                prerequisite_concepts=[
                    AblationPrerequisite(
                        concept=item.concept,
                        weight=item.weight,
                        method=item.method,
                    )
                    for item in concept_profile.prerequisite_concepts
                ],
                lexical_difficulty=difficulty.lexical_difficulty,
                syntactic_complexity=difficulty.syntactic_complexity,
                concept_density=difficulty.concept_density,
                prerequisite_demand=difficulty.prerequisite_demand,
                evidence_coverage=profile.evidence_coverage,
            )
        )
    config = loaded_config.config
    return EvidenceAblationReport(
        book_id=evidence.book_id,
        title=evidence.metadata.title,
        variants=variants,
        feature_version=config.book_profile_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
    )


def summarize_evidence_ablation(
    report: EvidenceAblationReport,
) -> list[AblationComparison]:
    """Describe deterministic changes between adjacent evidence-richness variants."""

    comparisons: list[AblationComparison] = []
    for before, after in pairwise(report.variants):
        before_concepts = {item.concept: item.weight for item in before.covered_concepts}
        after_concepts = {item.concept: item.weight for item in after.covered_concepts}
        before_prerequisites = {item.concept for item in before.prerequisite_concepts}
        after_prerequisites = {item.concept for item in after.prerequisite_concepts}
        difficulty_names = (
            "lexical_difficulty",
            "syntactic_complexity",
            "concept_density",
            "prerequisite_demand",
        )
        comparisons.append(
            AblationComparison(
                from_mode=before.evidence_mode,
                to_mode=after.evidence_mode,
                added_concepts=sorted(set(after_concepts) - set(before_concepts)),
                removed_concepts=sorted(set(before_concepts) - set(after_concepts)),
                changed_concept_weights=sorted(
                    concept
                    for concept in set(before_concepts) & set(after_concepts)
                    if abs(before_concepts[concept] - after_concepts[concept]) > 1e-12
                ),
                added_prerequisites=sorted(after_prerequisites - before_prerequisites),
                removed_prerequisites=sorted(before_prerequisites - after_prerequisites),
                newly_available_difficulty_components=[
                    name
                    for name in difficulty_names
                    if getattr(before, name) is None and getattr(after, name) is not None
                ],
            )
        )
    return comparisons
