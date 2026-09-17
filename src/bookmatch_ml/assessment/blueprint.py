"""Build inspectable, deterministic question targets without question text."""

import hashlib
import json

from bookmatch_ml.assessment.difficulty import assign_target_difficulty
from bookmatch_ml.assessment.pool import (
    AssessmentBlueprintError,
    build_topic_concept_pool,
    validate_profile_handoff,
)
from bookmatch_ml.assessment.schemas import (
    AssessmentBlueprint,
    AssessmentEvidenceRef,
    CognitiveOperation,
    QuestionSpec,
    QuestionSpecShortage,
    SelectedAssessmentConcept,
    TopicConcept,
    TopicConceptPool,
)
from bookmatch_ml.config import LoadedAssessmentConfig, LoadedFeatureConfig
from bookmatch_ml.schemas import BookProfile, CanonicalDataset, QuestionType


def _references(concept: TopicConcept) -> list[AssessmentEvidenceRef]:
    return [reference for support in concept.book_support for reference in support.evidence]


def _compact_references(
    concepts: list[TopicConcept],
    supporting_book_ids: list[str],
    required: list[AssessmentEvidenceRef],
    limit: int,
) -> list[AssessmentEvidenceRef]:
    """Keep source anchors and one reference per concept before filling the audit sample."""

    all_refs = sorted(
        (
            reference
            for concept in concepts
            for reference in _references(concept)
            if reference.book_id in supporting_book_ids
        ),
        key=lambda item: (item.concept_id, item.book_id, item.evidence_type, item.evidence_id),
    )
    selected: list[AssessmentEvidenceRef] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(reference: AssessmentEvidenceRef) -> None:
        key = (
            reference.concept_id,
            reference.book_id,
            reference.evidence_type,
            reference.evidence_id,
        )
        if key not in seen and len(selected) < limit:
            selected.append(reference)
            seen.add(key)

    for reference in required:
        add(reference)
    for concept in concepts:
        if not any(item.concept_id == concept.concept_id for item in selected):
            reference = next(
                (item for item in all_refs if item.concept_id == concept.concept_id), None
            )
            if reference is not None:
                add(reference)
    for reference in all_refs:
        add(reference)
    if {item.concept_id for item in selected} != {item.concept_id for item in concepts}:
        raise AssessmentBlueprintError("evidence reference limit hides a question concept")
    return selected


def _question_id(
    topic_id: str,
    question_type: QuestionType,
    operation: CognitiveOperation,
    primary: str,
    related: list[str],
    evidence: list[AssessmentEvidenceRef],
    config_hash: str,
) -> str:
    identity = {
        "topic_id": topic_id,
        "question_type": question_type,
        "cognitive_operation": operation,
        "primary_concept": primary,
        "related_concepts": related,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "config_hash": config_hash,
    }
    content = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "q_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:20]


def _source_options(
    concept: TopicConcept,
    profiles_by_id: dict[str, BookProfile],
    preference: list[str],
) -> list[tuple[int, str, str, float, AssessmentEvidenceRef]]:
    options = []
    for reference in _references(concept):
        profile = profiles_by_id[reference.book_id]
        analyzed = next(
            (
                item
                for item in profile.difficulty_profile.documents
                if item.document_id == reference.evidence_id
                and item.document_type == reference.evidence_type
            ),
            None,
        )
        if analyzed is not None:
            options.append(
                (
                    preference.index(analyzed.document_type),
                    reference.book_id,
                    reference.evidence_id,
                    analyzed.scores.syntactic_complexity,
                    reference,
                )
            )
    return sorted(options, key=lambda item: item[:3])


def _make_spec(
    *,
    topic_id: str,
    question_type: QuestionType,
    operation: CognitiveOperation,
    primary: TopicConcept,
    related: list[TopicConcept],
    supporting_book_ids: list[str],
    required_evidence: list[AssessmentEvidenceRef],
    source_document_id: str | None,
    source_text_complexity: float | None,
    loaded_config: LoadedAssessmentConfig,
) -> QuestionSpec:
    config = loaded_config.config
    level, relationship, multi_step, rationale = assign_target_difficulty(
        operation, len(related), loaded_config
    )
    evidence = _compact_references(
        [primary, *related],
        supporting_book_ids,
        required_evidence,
        config.max_evidence_refs_per_spec,
    )
    methods = ", ".join(primary.prerequisite_methods)
    topic_book_count = round(primary.book_coverage_count / primary.book_coverage_rate)
    summary = (
        f"{primary.role} concept in {primary.book_coverage_count}/{topic_book_count} "
        f"topic books; evidence types: {', '.join(primary.evidence_types)}"
    )
    if methods:
        summary += f"; inferred prerequisite methods: {methods}"
    if related:
        summary += "; configured relation candidate requires author verification"
    if source_document_id is not None:
        summary += "; shared analyzed prose" if related else "; analyzed prose"
    return QuestionSpec(
        question_id=_question_id(
            topic_id,
            question_type,
            operation,
            primary.concept_id,
            [item.concept_id for item in related],
            evidence,
            loaded_config.content_hash,
        ),
        topic_id=topic_id,
        question_type=question_type,
        cognitive_operation=operation,
        concept_role=primary.role,
        primary_concept=primary.concept_id,
        related_concepts=[item.concept_id for item in related],
        prerequisite_concepts=([primary.concept_id] if primary.role == "prerequisite" else []),
        target_difficulty=level,
        difficulty_version=config.difficulty_version,
        difficulty_rationale=rationale,
        prerequisite_depth=None,
        primary_concept_count=1,
        related_concept_count=len(related),
        relationship_reasoning_required=relationship,
        multi_step_required=multi_step,
        abstraction_level=None,
        source_text_complexity=source_text_complexity,
        assessment_priority=primary.assessment_priority,
        relation_candidate_source="configured_pair" if related else None,
        evidence_summary=summary,
        supporting_book_ids=supporting_book_ids,
        supporting_evidence=evidence,
        source_document_ids=[source_document_id] if source_document_id is not None else [],
        question_spec_version=config.question_spec_version,
        config_version=config.config_version,
        config_hash=loaded_config.content_hash,
    )


def _selected_by_role(
    pool: TopicConceptPool, loaded_config: LoadedAssessmentConfig
) -> tuple[list[TopicConcept], list[TopicConcept], list[SelectedAssessmentConcept]]:
    selected: dict[str, list[TopicConcept]] = {}
    for role in ("covered", "prerequisite"):
        candidates = [item for item in pool.concepts if item.role == role]
        selected[role] = candidates[: loaded_config.config.self_assessment_limits[role]]
    summary = [
        SelectedAssessmentConcept(
            concept_id=item.concept_id,
            role=item.role,
            assessment_priority=item.assessment_priority,
        )
        for role in ("covered", "prerequisite")
        for item in selected[role]
    ]
    return selected["covered"], selected["prerequisite"], summary


def _common_books(left: TopicConcept, right: TopicConcept) -> list[str]:
    return sorted(
        {item.book_id for item in left.book_support} & {item.book_id for item in right.book_support}
    )


def _shared_prose_option(
    left: TopicConcept,
    right: TopicConcept,
    profiles_by_id: dict[str, BookProfile],
    preference: list[str],
) -> tuple[str, str, float, list[AssessmentEvidenceRef]] | None:
    left_options = _source_options(left, profiles_by_id, preference)
    right_options = {
        (book_id, document_id): (complexity, reference)
        for _, book_id, document_id, complexity, reference in _source_options(
            right, profiles_by_id, preference
        )
    }
    for _, book_id, document_id, complexity, left_ref in left_options:
        right_match = right_options.get((book_id, document_id))
        if right_match is not None:
            return book_id, document_id, complexity, [left_ref, right_match[1]]
    return None


def build_assessment_blueprint(
    topic_id: str,
    dataset: CanonicalDataset,
    profiles: list[BookProfile],
    feature_config: LoadedFeatureConfig,
    assessment_config: LoadedAssessmentConfig,
    *,
    canonical_file_hashes: dict[str, str],
    book_profiles_hash: str,
) -> AssessmentBlueprint:
    """Select concept targets and emit versioned specifications from validated evidence."""

    validate_profile_handoff(dataset, profiles, feature_config)
    config = assessment_config.config
    pool = build_topic_concept_pool(topic_id, profiles, feature_config, assessment_config)
    if topic_id not in config.relation_pairs:
        raise AssessmentBlueprintError(f"no configured relation candidates for {topic_id}")
    covered_names = set(feature_config.config.concept.topics[topic_id].root)
    for left, right in config.relation_pairs[topic_id]:
        if left not in covered_names or right not in covered_names:
            raise AssessmentBlueprintError(
                f"configured relation pair is outside {topic_id} covered lexicon: {left}, {right}"
            )

    covered, prerequisites, selected = _selected_by_role(pool, assessment_config)
    selected_covered = {item.concept_id: item for item in covered}
    profiles_by_id = {item.book_id: item for item in profiles}
    preference = config.prose_document_preference
    specs: list[QuestionSpec] = []
    shortages: list[QuestionSpecShortage] = []

    def emit(
        question_type: QuestionType,
        operation: CognitiveOperation,
        candidates: list[QuestionSpec],
        reason: str,
    ) -> None:
        requested = config.question_quotas[question_type][operation]
        chosen = candidates[:requested]
        specs.extend(chosen)
        if len(chosen) < requested:
            shortages.append(
                QuestionSpecShortage(
                    question_type=question_type,
                    cognitive_operation=operation,
                    requested=requested,
                    produced=len(chosen),
                    reason=reason,
                )
            )

    emit(
        "vocabulary",
        "recognize",
        [
            _make_spec(
                topic_id=topic_id,
                question_type="vocabulary",
                operation="recognize",
                primary=item,
                related=[],
                supporting_book_ids=[support.book_id for support in item.book_support],
                required_evidence=[],
                source_document_id=None,
                source_text_complexity=None,
                loaded_config=assessment_config,
            )
            for item in covered
        ],
        "fewer selected covered concepts than requested recognition targets",
    )

    compare_candidates = []
    for left_name, right_name in config.relation_pairs[topic_id]:
        left = selected_covered.get(left_name)
        right = selected_covered.get(right_name)
        if left is None or right is None:
            continue
        common_books = _common_books(left, right)
        if common_books:
            compare_candidates.append(
                _make_spec(
                    topic_id=topic_id,
                    question_type="vocabulary",
                    operation="compare",
                    primary=left,
                    related=[right],
                    supporting_book_ids=common_books,
                    required_evidence=[],
                    source_document_id=None,
                    source_text_complexity=None,
                    loaded_config=assessment_config,
                )
            )
    emit(
        "vocabulary",
        "compare",
        compare_candidates,
        "too few selected configured relation pairs share book evidence",
    )

    emit(
        "background_knowledge",
        "recall",
        [
            _make_spec(
                topic_id=topic_id,
                question_type="background_knowledge",
                operation="recall",
                primary=item,
                related=[],
                supporting_book_ids=[support.book_id for support in item.book_support],
                required_evidence=[],
                source_document_id=None,
                source_text_complexity=None,
                loaded_config=assessment_config,
            )
            for item in prerequisites
        ],
        "fewer inferred prerequisite concepts than requested background targets",
    )

    apply_candidates = []
    for item in covered:
        options = _source_options(item, profiles_by_id, preference)
        if options:
            _, book_id, document_id, complexity, reference = options[0]
            apply_candidates.append(
                _make_spec(
                    topic_id=topic_id,
                    question_type="comprehension",
                    operation="apply",
                    primary=item,
                    related=[],
                    supporting_book_ids=[book_id],
                    required_evidence=[reference],
                    source_document_id=document_id,
                    source_text_complexity=complexity,
                    loaded_config=assessment_config,
                )
            )
    emit(
        "comprehension",
        "apply",
        apply_candidates,
        "too few selected covered concepts occur in analyzed textbook prose",
    )

    integration_candidates = []
    for left_name, right_name in config.relation_pairs[topic_id]:
        left = selected_covered.get(left_name)
        right = selected_covered.get(right_name)
        if left is None or right is None:
            continue
        shared = _shared_prose_option(left, right, profiles_by_id, preference)
        if shared is not None:
            book_id, document_id, complexity, references = shared
            integration_candidates.append(
                _make_spec(
                    topic_id=topic_id,
                    question_type="comprehension",
                    operation="integrate",
                    primary=left,
                    related=[right],
                    supporting_book_ids=[book_id],
                    required_evidence=references,
                    source_document_id=document_id,
                    source_text_complexity=complexity,
                    loaded_config=assessment_config,
                )
            )
    emit(
        "comprehension",
        "integrate",
        integration_candidates,
        "too few selected configured pairs co-occur in one analyzed prose document",
    )

    return AssessmentBlueprint(
        topic_id=topic_id,
        concept_pool=pool,
        selected_assessment_concepts=selected,
        question_specs=specs,
        shortages=shortages,
        prose_grounded_comprehension_count=sum(
            item.question_type == "comprehension" for item in specs
        ),
        blueprint_version=config.blueprint_version,
        config_version=config.config_version,
        config_hash=assessment_config.content_hash,
        canonical_file_hashes=canonical_file_hashes,
        book_profiles_hash=book_profiles_hash,
    )
