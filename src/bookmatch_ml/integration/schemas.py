"""Camel-case DTOs kept separate from internal ML schemas."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from bookmatch_ml.concept_v2.profile import BookConceptProfileV2
from bookmatch_ml.ranking.prerequisite_first_v2 import (
    RankingV2Response as InternalRankingV2Response,
)
from bookmatch_ml.schemas import (
    Assessment,
    AssessmentResponse,
    MatchingBookProfile,
    MatchingConcept,
    MatchingConceptReadiness,
    MatchingReaderProfile,
    QuestionType,
    RankedBook,
    RankingResponse,
    ReaderProfile,
)


class ApiModel(BaseModel):
    """Strict JSON contract using Spring-style camelCase field names."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        allow_inf_nan=False,
        extra="forbid",
        populate_by_name=True,
        serialize_by_alias=True,
        strict=True,
    )


class AssessmentResponseDto(ApiModel):
    question_id: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    concept_id: str | None = None
    concept_tags: list[str] = Field(default_factory=list)
    question_type: QuestionType
    difficulty: str = Field(min_length=1)
    correct: bool | None = None
    score: float | None = Field(default=None, ge=0, le=1)

    def to_internal(self) -> AssessmentResponse:
        return AssessmentResponse.model_validate(self.model_dump(by_alias=False))


class ReaderProfileRequest(ApiModel):
    user_id: int | None = Field(default=None, ge=1)
    assessment_id: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    responses: list[AssessmentResponseDto] = Field(min_length=1)

    def to_internal(self) -> Assessment:
        return Assessment(
            assessment_id=self.assessment_id,
            topic_id=self.topic_id,
            responses=[response.to_internal() for response in self.responses],
        )


class ReaderDimensionDetailDto(ApiModel):
    question_type: QuestionType
    score: float = Field(ge=0, le=1)
    response_count: int = Field(ge=1)
    earned_weight: float = Field(ge=0)
    available_weight: float = Field(gt=0)


class ConceptReadinessDto(ApiModel):
    concept_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)
    response_count: int = Field(ge=1)
    earned_weight: float = Field(ge=0)
    available_weight: float = Field(gt=0)


class ReaderProfileResponse(ApiModel):
    user_id: int | None = Field(default=None, ge=1)
    assessment_id: str
    topic_id: str
    vocabulary: float = Field(ge=0, le=1)
    background_knowledge: float = Field(ge=0, le=1)
    comprehension: float = Field(ge=0, le=1)
    dimension_details: list[ReaderDimensionDetailDto]
    concept_readiness: list[ConceptReadinessDto]
    response_count: int = Field(ge=1)
    profile_version: str
    config_version: str
    config_hash: str

    @classmethod
    def from_internal(cls, profile: ReaderProfile, *, user_id: int | None) -> Self:
        return cls(user_id=user_id, **profile.model_dump())


class MatchingConceptReadinessDto(ApiModel):
    concept_id: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)


class MatchingReaderProfileDto(ApiModel):
    user_id: int | None = Field(default=None, ge=1)
    topic_id: str = Field(min_length=1)
    vocabulary: float = Field(ge=0, le=1)
    background_knowledge: float = Field(ge=0, le=1)
    comprehension: float = Field(ge=0, le=1)
    concept_readiness: list[MatchingConceptReadinessDto] = Field(default_factory=list)
    profile_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    def to_internal(self) -> MatchingReaderProfile:
        return MatchingReaderProfile(
            topic_id=self.topic_id,
            vocabulary=self.vocabulary,
            background_knowledge=self.background_knowledge,
            comprehension=self.comprehension,
            concept_readiness=[
                MatchingConceptReadiness(**item.model_dump(by_alias=False))
                for item in self.concept_readiness
            ],
            profile_version=self.profile_version,
            config_version=self.config_version,
            config_hash=self.config_hash,
        )


class MatchingConceptDto(ApiModel):
    concept: str = Field(min_length=1)
    weight: float = Field(ge=0, le=1)


class BookCandidateDto(ApiModel):
    book_id: str = Field(min_length=1)
    topic_distribution: dict[str, float]
    covered_concepts: list[MatchingConceptDto] = Field(default_factory=list)
    prerequisite_concepts: list[MatchingConceptDto] = Field(default_factory=list)
    lexical_difficulty: float | None = Field(default=None, ge=0, le=1)
    syntactic_complexity: float | None = Field(default=None, ge=0, le=1)
    concept_density: float | None = Field(default=None, ge=0, le=1)
    prerequisite_demand: float | None = Field(default=None, ge=0, le=1)
    feature_version: str = Field(min_length=1)
    config_version: str = Field(min_length=1)
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    concept_profile: BookConceptProfileV2 | None = None

    def to_internal(self) -> MatchingBookProfile:
        return MatchingBookProfile(
            book_id=self.book_id,
            topic_distribution=self.topic_distribution,
            covered_concepts=[
                MatchingConcept(**item.model_dump(by_alias=False)) for item in self.covered_concepts
            ],
            prerequisite_concepts=[
                MatchingConcept(**item.model_dump(by_alias=False))
                for item in self.prerequisite_concepts
            ],
            lexical_difficulty=self.lexical_difficulty,
            syntactic_complexity=self.syntactic_complexity,
            concept_density=self.concept_density,
            prerequisite_demand=self.prerequisite_demand,
            feature_version=self.feature_version,
            config_version=self.config_version,
            config_hash=self.config_hash,
        )


class RankRequest(ApiModel):
    ranking_model: Literal["rank-v1", "rank-prerequisite-first-v2"] = "rank-v1"
    reader_profile: MatchingReaderProfileDto
    candidate_books: list[BookCandidateDto] = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=100)
    book_id: str | None = Field(default=None, min_length=1)
    ranking_strategy: Literal["baseline_v1", "concept_difficulty_v2_experimental"] = "baseline_v1"

    @field_validator("candidate_books")
    @classmethod
    def candidate_books_must_be_unique(
        cls, value: list[BookCandidateDto]
    ) -> list[BookCandidateDto]:
        book_ids = [book.book_id for book in value]
        if len(book_ids) != len(set(book_ids)):
            raise ValueError("candidateBooks contains duplicate bookId values")
        return value

    @model_validator(mode="after")
    def requested_book_must_be_a_candidate(self) -> Self:
        if self.book_id is not None and self.book_id not in {
            book.book_id for book in self.candidate_books
        }:
            raise ValueError("bookId must identify one of candidateBooks")
        return self


class KnowledgeComponentsDto(ApiModel):
    concept_density_fit: float | None = Field(default=None, ge=0, le=1)
    prerequisite_demand_fit: float | None = Field(default=None, ge=0, le=1)
    prerequisite_concept_fit: float | None = Field(default=None, ge=0, le=1)


class MatchingDiagnosticsDto(ApiModel):
    lexical_demand: float | None = Field(default=None, ge=0, le=1)
    knowledge_demand: float | None = Field(default=None, ge=0, le=1)
    syntactic_demand: float | None = Field(default=None, ge=0, le=1)
    assessed_prerequisite_count: int = Field(ge=0)
    inferred_prerequisite_count: int = Field(ge=0)


class RankedBookDto(ApiModel):
    book_id: str
    score: float = Field(ge=0, le=1)
    topic_fit: float | None = Field(default=None, ge=0, le=1)
    vocabulary_fit: float | None = Field(default=None, ge=0, le=1)
    knowledge_fit: float | None = Field(default=None, ge=0, le=1)
    comprehension_fit: float | None = Field(default=None, ge=0, le=1)
    knowledge_components: KnowledgeComponentsDto
    knowledge_active_weights: dict[str, float]
    knowledge_weight_coverage: float = Field(ge=0, le=1)
    active_weights: dict[str, float]
    component_weight_coverage: float = Field(ge=0, le=1)
    unavailable_components: list[str]
    diagnostics: MatchingDiagnosticsDto
    reasons: list[str]
    model_version: str
    config_version: str
    config_hash: str
    reader_profile_version: str
    reader_config_version: str
    reader_config_hash: str
    book_feature_version: str
    book_config_version: str
    book_config_hash: str

    @classmethod
    def from_internal(cls, item: RankedBook) -> Self:
        return cls(
            book_id=item.book_id,
            score=item.score,
            topic_fit=item.components.topic_fit,
            vocabulary_fit=item.components.vocabulary_fit,
            knowledge_fit=item.components.knowledge_fit,
            comprehension_fit=item.components.comprehension_fit,
            knowledge_components=KnowledgeComponentsDto(**item.knowledge_components.model_dump()),
            knowledge_active_weights=item.knowledge_active_weights,
            knowledge_weight_coverage=item.knowledge_weight_coverage,
            active_weights=item.active_weights,
            component_weight_coverage=item.component_weight_coverage,
            unavailable_components=item.unavailable_components,
            diagnostics=MatchingDiagnosticsDto(**item.diagnostics.model_dump()),
            reasons=item.reasons,
            model_version=item.model_version,
            config_version=item.config_version,
            config_hash=item.config_hash,
            reader_profile_version=item.reader_profile_version,
            reader_config_version=item.reader_config_version,
            reader_config_hash=item.reader_config_hash,
            book_feature_version=item.book_feature_version,
            book_config_version=item.book_config_version,
            book_config_hash=item.book_config_hash,
        )


class ExperimentalRankedBookDto(RankedBookDto):
    concept_difficulty: dict[str, object]


# Maps keyed by concept ID. Their keys are data, so they must stay joinable with the
# raw concept IDs used elsewhere in the response and in the concept graph.
_CONCEPT_KEYED_MAPS = frozenset(
    {"concept_levels", "concept_evidence", "prerequisite_evidence", "burden_contributions"}
)


def camelize_payload(value):
    """Recursively convert experimental evidence field names without changing their values."""

    if isinstance(value, dict):
        return {
            to_camel(str(key)): (
                {concept: camelize_payload(entry) for concept, entry in item.items()}
                if key in _CONCEPT_KEYED_MAPS and isinstance(item, dict)
                else camelize_payload(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [camelize_payload(item) for item in value]
    return value


class RankResponse(ApiModel):
    user_id: int | None = Field(default=None, ge=1)
    topic_id: str
    items: list[ExperimentalRankedBookDto | RankedBookDto]
    model_version: str
    config_version: str
    config_hash: str
    reader_profile_version: str
    reader_config_version: str
    reader_config_hash: str

    @classmethod
    def from_internal(cls, response: RankingResponse, *, user_id: int | None) -> Self:
        return cls(
            user_id=user_id,
            topic_id=response.topic_id,
            items=[RankedBookDto.from_internal(item) for item in response.items],
            model_version=response.model_version,
            config_version=response.config_version,
            config_hash=response.config_hash,
            reader_profile_version=response.reader_profile_version,
            reader_config_version=response.reader_config_version,
            reader_config_hash=response.reader_config_hash,
        )


class RankV2DiagnosticsDto(ApiModel):
    requested_limit: int = Field(ge=1)
    returned_count: int = Field(ge=0)
    topic_candidate_count: int = Field(ge=0)
    personalizable_count: int = Field(ge=0)
    concept_only_count: int = Field(ge=0)
    evidence_unavailable_count: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    personalized_candidate_shortage: int = Field(ge=0)


class RankedBookV2Dto(ApiModel):
    book_id: str
    rank: int = Field(ge=1)
    availability_status: Literal["personalizable"]
    prerequisite_readiness: float = Field(ge=0, le=1)
    prerequisite_assessed_count: int = Field(ge=1)
    prerequisite_total_count: int = Field(ge=1)
    prerequisite_coverage: float = Field(gt=0, le=1)
    direct_learning_opportunity: float | None = Field(default=None, ge=0, le=1)
    direct_assessed_count: int = Field(ge=0)
    direct_total_count: int = Field(ge=1)
    direct_coverage: float = Field(ge=0, le=1)
    covered_concepts: list[str]
    inferred_prerequisites: list[str]
    reasons: list[str] = Field(min_length=2)
    model_version: Literal["rank-prerequisite-first-v2"]
    book_feature_version: str
    book_config_version: str
    book_config_hash: str


class RankV2Response(ApiModel):
    user_id: int | None = Field(default=None, ge=1)
    topic_id: str
    items: list[RankedBookV2Dto]
    diagnostics: RankV2DiagnosticsDto
    model_version: Literal["rank-prerequisite-first-v2"]
    config_version: Literal["ranking-v2-config-v1"]
    config_hash: str
    concept_graph_version: str
    concept_graph_hash: str
    graph_review_version: str
    graph_review_hash: str
    reader_profile_version: str
    reader_config_version: str
    reader_config_hash: str

    @classmethod
    def from_internal(cls, response: InternalRankingV2Response, *, user_id: int | None) -> Self:
        """Map the non-scalar internal response to the camel-case API contract."""

        payload = response.model_dump()
        return cls(
            user_id=user_id,
            **payload,
        )
