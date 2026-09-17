"""Concept-assessment contracts kept separate from current reader and Spring DTOs."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from bookmatch_ml.schemas import ConceptEvidenceType, QuestionType, StrictModel

ConceptRole = Literal["covered", "prerequisite"]
CognitiveOperation = Literal[
    "recognize", "recall", "compare", "relate", "apply", "integrate", "infer"
]
TargetDifficulty = Literal[1, 2, 3]
PrerequisiteMethod = Literal["explicit", "early_prose_proxy", "explicit_and_early_prose_proxy"]


def _canonical_identifier(value: str) -> str:
    if not value or value != value.strip():
        raise ValueError("concept and topic identifiers must be nonblank and trimmed")
    return value


class ConceptSelfAssessmentResponse(StrictModel):
    concept_id: str
    topic_id: str
    knows_concept: bool

    _validate_concept = field_validator("concept_id", "topic_id")(_canonical_identifier)


class ConceptSelfAssessment(StrictModel):
    assessment_id: str = Field(min_length=1)
    topic_id: str
    responses: list[ConceptSelfAssessmentResponse] = Field(min_length=1)
    self_assessment_version: str = Field(min_length=1)

    _validate_topic = field_validator("topic_id")(_canonical_identifier)

    @model_validator(mode="after")
    def responses_must_be_unique_and_on_topic(self) -> "ConceptSelfAssessment":
        names = [response.concept_id.casefold() for response in self.responses]
        if len(names) != len(set(names)):
            raise ValueError("self-assessment contains duplicate concept_id values")
        if any(response.topic_id != self.topic_id for response in self.responses):
            raise ValueError("self-assessment responses must match topic_id")
        return self


class ConceptMastery(StrictModel):
    """Future fusion boundary; v1 intentionally supplies no mastery formula."""

    concept_id: str
    topic_id: str
    self_report_knows: bool | None = None
    quiz_score: float | None = Field(default=None, ge=0, le=1)
    mastery: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_sources: list[Literal["self_report", "quiz"]] = Field(default_factory=list)

    _validate_identity = field_validator("concept_id", "topic_id")(_canonical_identifier)

    @field_validator("evidence_sources")
    @classmethod
    def sources_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("concept mastery evidence sources must be unique")
        return value


class AssessmentEvidenceRef(StrictModel):
    concept_id: str
    book_id: str = Field(min_length=1)
    evidence_type: ConceptEvidenceType
    evidence_id: str = Field(min_length=1)
    mention_count: int = Field(ge=1)

    _validate_concept = field_validator("concept_id")(_canonical_identifier)


class ConceptBookSupport(StrictModel):
    book_id: str = Field(min_length=1)
    weight: float = Field(ge=0, le=1)
    prerequisite_method: PrerequisiteMethod | None = None
    evidence: list[AssessmentEvidenceRef] = Field(min_length=1)


class TopicConcept(StrictModel):
    concept_id: str
    topic_id: str
    role: ConceptRole
    book_coverage_count: int = Field(ge=1)
    book_coverage_rate: float = Field(gt=0, le=1)
    mean_book_weight: float = Field(ge=0, le=1)
    assessment_priority: float = Field(ge=0, le=1)
    evidence_types: list[ConceptEvidenceType] = Field(min_length=1)
    prerequisite_methods: list[PrerequisiteMethod]
    book_support: list[ConceptBookSupport] = Field(min_length=1)

    _validate_identity = field_validator("concept_id", "topic_id")(_canonical_identifier)

    @model_validator(mode="after")
    def coverage_must_match_support(self) -> "TopicConcept":
        ids = [item.book_id for item in self.book_support]
        if len(ids) != self.book_coverage_count or len(ids) != len(set(ids)):
            raise ValueError("book coverage count must match unique supporting books")
        if self.role == "covered" and self.prerequisite_methods:
            raise ValueError("covered concepts must not have prerequisite methods")
        if self.role == "prerequisite" and not self.prerequisite_methods:
            raise ValueError("prerequisite concepts require inference methods")
        return self


class TopicConceptPool(StrictModel):
    topic_id: str
    topic_book_count: int = Field(ge=1)
    covered_concept_count: int = Field(ge=0)
    prerequisite_concept_count: int = Field(ge=0)
    concepts: list[TopicConcept]
    pool_version: str = Field(min_length=1)
    feature_config_version: str = Field(min_length=1)
    feature_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    assessment_config_version: str = Field(min_length=1)
    assessment_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def concepts_must_be_unique_and_on_topic(self) -> "TopicConceptPool":
        keys = [(item.concept_id, item.role) for item in self.concepts]
        if len(keys) != len(set(keys)):
            raise ValueError("topic concept pool contains duplicate concept/role pairs")
        if any(item.topic_id != self.topic_id for item in self.concepts):
            raise ValueError("topic concept pool contains another topic")
        if sum(item.role == "covered" for item in self.concepts) != self.covered_concept_count:
            raise ValueError("covered concept count does not match pool")
        if (
            sum(item.role == "prerequisite" for item in self.concepts)
            != self.prerequisite_concept_count
        ):
            raise ValueError("prerequisite concept count does not match pool")
        return self


class SelectedAssessmentConcept(StrictModel):
    concept_id: str
    role: ConceptRole
    assessment_priority: float = Field(ge=0, le=1)


class QuestionSpec(StrictModel):
    question_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    question_type: QuestionType
    cognitive_operation: CognitiveOperation
    concept_role: ConceptRole
    primary_concept: str
    related_concepts: list[str]
    prerequisite_concepts: list[str]
    target_difficulty: TargetDifficulty
    difficulty_version: str = Field(min_length=1)
    difficulty_rationale: str = Field(min_length=1)
    prerequisite_depth: int | None = Field(default=None, ge=0)
    primary_concept_count: int = Field(ge=1)
    related_concept_count: int = Field(ge=0)
    relationship_reasoning_required: bool
    multi_step_required: bool
    abstraction_level: Literal["concrete", "relational", "abstract"] | None = None
    source_text_complexity: float | None = Field(default=None, ge=0, le=1)
    assessment_priority: float = Field(ge=0, le=1)
    relation_candidate_source: Literal["configured_pair"] | None = None
    evidence_summary: str = Field(min_length=1)
    supporting_book_ids: list[str] = Field(min_length=1)
    supporting_evidence: list[AssessmentEvidenceRef] = Field(min_length=1)
    source_document_ids: list[str]
    question_spec_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    _validate_identity = field_validator("topic_id", "primary_concept")(_canonical_identifier)

    @model_validator(mode="after")
    def specification_must_match_type_and_evidence(self) -> "QuestionSpec":
        if self.primary_concept_count != 1 or self.related_concept_count != len(
            self.related_concepts
        ):
            raise ValueError("concept counts must match the question specification")
        if self.primary_concept in self.related_concepts or len(self.related_concepts) != len(
            set(self.related_concepts)
        ):
            raise ValueError("related concepts must be unique and differ from primary")
        if len(self.supporting_book_ids) != len(set(self.supporting_book_ids)):
            raise ValueError("supporting book IDs must be unique")
        if any(
            item.book_id not in self.supporting_book_ids
            or item.concept_id not in {self.primary_concept, *self.related_concepts}
            for item in self.supporting_evidence
        ):
            raise ValueError("supporting evidence must match question concepts and books")
        expected_role = (
            "prerequisite" if self.question_type == "background_knowledge" else "covered"
        )
        if self.concept_role != expected_role:
            raise ValueError("question type does not match concept role")
        if self.question_type == "comprehension":
            if not self.source_document_ids or self.source_text_complexity is None:
                raise ValueError("comprehension requires analyzed prose document evidence")
            prose_types = {"preface", "introduction", "preview", "sample_chapter", "other"}
            anchored = {
                item.evidence_id
                for item in self.supporting_evidence
                if item.evidence_type in prose_types
            }
            if not set(self.source_document_ids) <= anchored:
                raise ValueError("comprehension source documents require matching prose references")
        elif self.source_document_ids or self.source_text_complexity is not None:
            raise ValueError("non-comprehension specs must not claim prose grounding")
        if self.target_difficulty == 1 and (
            self.related_concepts
            or self.relationship_reasoning_required
            or self.multi_step_required
        ):
            raise ValueError("Level 1 cannot require relation or multi-step reasoning")
        if self.target_difficulty == 3 and not self.multi_step_required:
            raise ValueError("Level 3 requires a multi-step target")
        if self.target_difficulty == 2 and self.multi_step_required:
            raise ValueError("Level 2 must not require multiple steps")
        if bool(self.related_concepts) != (self.relation_candidate_source is not None):
            raise ValueError("related concepts require an explicit candidate source")
        return self


class QuestionSpecShortage(StrictModel):
    question_type: QuestionType
    cognitive_operation: CognitiveOperation
    requested: int = Field(ge=0)
    produced: int = Field(ge=0)
    reason: str = Field(min_length=1)


class AssessmentBlueprint(StrictModel):
    topic_id: str
    concept_pool: TopicConceptPool
    selected_assessment_concepts: list[SelectedAssessmentConcept]
    question_specs: list[QuestionSpec]
    shortages: list[QuestionSpecShortage]
    prose_grounded_comprehension_count: int = Field(ge=0)
    blueprint_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    canonical_file_hashes: dict[str, str]
    book_profiles_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def selected_concepts_and_questions_must_match_pool(self) -> "AssessmentBlueprint":
        pool_keys = {(item.concept_id, item.role) for item in self.concept_pool.concepts}
        selected_keys = {(item.concept_id, item.role) for item in self.selected_assessment_concepts}
        if (
            len(selected_keys) != len(self.selected_assessment_concepts)
            or not selected_keys <= pool_keys
        ):
            raise ValueError("selected concepts must be unique members of the pool")
        ids = [item.question_id for item in self.question_specs]
        if len(ids) != len(set(ids)):
            raise ValueError("question specifications contain duplicate question_id values")
        if self.topic_id != self.concept_pool.topic_id or any(
            item.topic_id != self.topic_id for item in self.question_specs
        ):
            raise ValueError("blueprint topic must match pool and questions")
        if any(
            (item.primary_concept, item.concept_role) not in selected_keys
            for item in self.question_specs
        ):
            raise ValueError("question primary concepts must be selected for assessment")
        if self.prose_grounded_comprehension_count != sum(
            item.question_type == "comprehension" for item in self.question_specs
        ):
            raise ValueError("prose-grounded comprehension count must match questions")
        return self
