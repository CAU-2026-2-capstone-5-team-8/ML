"""Provider-independent schemas at the Data-Pipeline/ML boundary."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DocumentType = Literal[
    "description",
    "publisher_summary",
    "preface",
    "introduction",
    "preview",
    "sample_chapter",
    "index",
    "other",
]
ConceptEvidenceType = Literal[
    "toc",
    "description",
    "publisher_summary",
    "preface",
    "introduction",
    "preview",
    "sample_chapter",
    "index",
    "other",
]
SourceType = Literal[
    "metadata_api",
    "publisher_page",
    "author_page",
    "open_textbook",
    "preview_page",
    "sample_page",
    "other",
]
QuestionType = Literal["vocabulary", "background_knowledge", "comprehension"]
RankingComponentName = Literal[
    "topic_fit",
    "vocabulary_fit",
    "knowledge_fit",
    "comprehension_fit",
]
KnowledgeComponentName = Literal[
    "concept_density_fit",
    "prerequisite_demand_fit",
    "prerequisite_concept_fit",
]
DifficultyComponentName = Literal[
    "lexical_difficulty",
    "syntactic_complexity",
    "concept_density",
    "prerequisite_demand",
]


class StrictModel(BaseModel):
    """Reject accidental schema drift at the canonical boundary."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Book(StrictModel):
    book_id: str = Field(pattern=r"^(isbn13:[0-9]{13}|isbn10:[0-9]{9}[0-9X]|book_[0-9a-f]{20})$")
    isbn_10: str | None = None
    isbn_13: str | None = None
    title: str = Field(min_length=1)
    subtitle: str | None = None
    authors: list[str]
    publisher: str | None = None
    published_year: int | None = Field(default=None, ge=1000, le=9999)
    language: str = Field(pattern=r"^[a-z]{2,3}$")
    topics: list[str] = Field(min_length=1)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value

    @field_validator("isbn_10")
    @classmethod
    def isbn_10_must_be_valid(cls, value: str | None) -> str | None:
        if value is not None:
            valid_shape = (
                len(value) == 10
                and value[:9].isdigit()
                and (value[-1].isdigit() or value[-1] == "X")
            )
            digits = [int(character) for character in value[:9]] if valid_shape else []
            if valid_shape:
                digits.append(10 if value[-1] == "X" else int(value[-1]))
            if (
                not valid_shape
                or sum((10 - index) * digit for index, digit in enumerate(digits)) % 11
            ):
                raise ValueError("invalid ISBN-10 check digit")
        return value

    @field_validator("isbn_13")
    @classmethod
    def isbn_13_must_be_valid(cls, value: str | None) -> str | None:
        if value is not None:
            valid_shape = len(value) == 13 and value.isdigit()
            total = (
                sum((1 if index % 2 == 0 else 3) * int(value[index]) for index in range(12))
                if valid_shape
                else 0
            )
            if not valid_shape or (10 - total % 10) % 10 != int(value[-1]):
                raise ValueError("invalid ISBN-13 check digit")
        return value

    @model_validator(mode="after")
    def isbn_identity_must_be_consistent(self) -> "Book":
        if self.book_id.startswith("isbn13:") and self.isbn_13 is None:
            raise ValueError("ISBN-13 book_id requires isbn_13")
        if self.book_id.startswith("isbn10:") and self.isbn_10 is None:
            raise ValueError("ISBN-10 book_id requires isbn_10")
        if self.isbn_13 is not None and self.book_id != f"isbn13:{self.isbn_13}":
            raise ValueError("book_id must use the record's ISBN-13")
        if (
            self.isbn_13 is None
            and self.isbn_10 is not None
            and self.book_id != f"isbn10:{self.isbn_10}"
        ):
            raise ValueError("book_id must use the record's ISBN-10")
        if self.isbn_10 is not None and self.isbn_13 is not None:
            stem = f"978{self.isbn_10[:9]}"
            total = sum(
                (1 if index % 2 == 0 else 3) * int(character)
                for index, character in enumerate(stem)
            )
            corresponding_isbn_13 = f"{stem}{(10 - total % 10) % 10}"
            if self.isbn_13 != corresponding_isbn_13:
                raise ValueError("ISBN-10 and ISBN-13 identify different books")
        return self


class Document(StrictModel):
    document_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    document_type: DocumentType
    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("document text must not be blank")
        return value


class TocEntry(StrictModel):
    toc_entry_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    parent_entry_id: str | None = None
    level: int = Field(ge=1)
    order_index: int = Field(ge=0)
    label: str | None = None
    title: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


class Source(StrictModel):
    source_id: str = Field(min_length=1)
    book_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    source_type: SourceType
    url: str = Field(min_length=1)
    external_id: str | None = None
    retrieved_at: datetime
    license: str | None = None
    rights_note: str | None = None
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("retrieved_at")
    @classmethod
    def retrieved_at_must_include_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a timezone")
        return value


class CanonicalDataset(StrictModel):
    books: list[Book]
    documents: list[Document]
    toc: list[TocEntry]
    sources: list[Source]


class EvidenceCoverage(StrictModel):
    has_metadata: bool
    has_toc: bool
    has_description: bool
    has_preface: bool
    has_introduction: bool
    has_preview: bool
    has_sample_chapter: bool
    has_other_text: bool
    toc_entry_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    prose_document_count: int = Field(ge=0)
    prose_character_count: int = Field(ge=0)


class BookEvidence(StrictModel):
    """Downstream evidence with structure retained and provider details excluded."""

    book_id: str
    metadata: Book
    toc: list[TocEntry]
    documents: list[Document]
    coverage: EvidenceCoverage


class BookCoverageSummary(StrictModel):
    book_id: str
    title: str
    coverage: EvidenceCoverage


class CoverageReport(StrictModel):
    book_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    toc_entry_count: int = Field(ge=0)
    source_count: int = Field(ge=0)
    books: list[BookCoverageSummary]


class ConceptEvidenceRef(StrictModel):
    evidence_type: ConceptEvidenceType
    evidence_id: str
    mention_count: int = Field(ge=1)


class CoveredConcept(StrictModel):
    concept: str
    weight: float = Field(ge=0, le=1)
    evidence_types: list[ConceptEvidenceType]
    evidence: list[ConceptEvidenceRef]


class PrerequisiteConcept(StrictModel):
    concept: str
    weight: float = Field(ge=0, le=1)
    method: Literal["explicit", "early_prose_proxy", "explicit_and_early_prose_proxy"]
    evidence: list[ConceptEvidenceRef]


class ConceptProfile(StrictModel):
    book_id: str
    covered_concepts: list[CoveredConcept]
    prerequisite_concepts: list[PrerequisiteConcept]
    topic_distribution: dict[str, float]
    analyzed_toc_entry_ids: list[str]
    analyzed_document_ids: list[str]
    profile_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("topic_distribution")
    @classmethod
    def topic_distribution_must_be_normalized(cls, value: dict[str, float]) -> dict[str, float]:
        if any(score < 0 or score > 1 for score in value.values()):
            raise ValueError("topic distribution values must be in [0, 1]")
        if value and abs(sum(value.values()) - 1.0) > 1e-9:
            raise ValueError("topic distribution must sum to 1.0")
        return value


class DifficultyRawFeatures(StrictModel):
    mean_token_length: float = Field(ge=0)
    vocabulary_diversity: float = Field(ge=0, le=1)
    long_word_ratio: float = Field(ge=0, le=1)
    domain_term_ratio: float = Field(ge=0)
    mean_sentence_tokens: float = Field(ge=0)
    median_sentence_tokens: float = Field(ge=0)
    long_sentence_ratio: float = Field(ge=0, le=1)
    clause_markers_per_sentence: float = Field(ge=0)
    concept_mention_count: int = Field(ge=0)
    unique_concept_count: int = Field(ge=0)
    concept_mentions_per_1000_tokens: float = Field(ge=0)
    unique_concepts_per_1000_tokens: float = Field(ge=0)
    prerequisite_mention_count: int = Field(ge=0)
    prerequisite_mentions_per_1000_tokens: float = Field(ge=0)
    prerequisite_cue_sentence_ratio: float = Field(ge=0, le=1)


class DifficultyScores(StrictModel):
    lexical_difficulty: float = Field(ge=0, le=1)
    syntactic_complexity: float = Field(ge=0, le=1)
    concept_density: float = Field(ge=0, le=1)
    prerequisite_demand: float = Field(ge=0, le=1)


class DocumentDifficulty(StrictModel):
    document_id: str
    document_type: DocumentType
    character_count: int = Field(ge=0)
    token_count: int = Field(ge=1)
    sentence_count: int = Field(ge=1)
    raw: DifficultyRawFeatures
    scores: DifficultyScores


class ExcludedProseDocument(StrictModel):
    document_id: str
    document_type: DocumentType
    character_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    reason: Literal["below_minimum_tokens", "no_tokens"]


class DifficultyProfile(StrictModel):
    book_id: str
    lexical_difficulty: float | None = Field(default=None, ge=0, le=1)
    syntactic_complexity: float | None = Field(default=None, ge=0, le=1)
    concept_density: float | None = Field(default=None, ge=0, le=1)
    prerequisite_demand: float | None = Field(default=None, ge=0, le=1)
    analyzed_document_count: int = Field(ge=0)
    analyzed_character_count: int = Field(ge=0)
    analyzed_token_count: int = Field(ge=0)
    documents: list[DocumentDifficulty]
    excluded_documents: list[ExcludedProseDocument]
    aggregation_rule: str
    feature_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class BookProfile(StrictModel):
    book_id: str
    concept_profile: ConceptProfile
    difficulty_profile: DifficultyProfile
    evidence_coverage: EvidenceCoverage
    feature_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def subprofiles_must_match_book_and_config(self) -> "BookProfile":
        if self.concept_profile.book_id != self.book_id:
            raise ValueError("concept profile book_id does not match book profile")
        if self.difficulty_profile.book_id != self.book_id:
            raise ValueError("difficulty profile book_id does not match book profile")
        if self.concept_profile.config_hash != self.config_hash:
            raise ValueError("concept profile config hash does not match book profile")
        if self.difficulty_profile.config_hash != self.config_hash:
            raise ValueError("difficulty profile config hash does not match book profile")
        return self


class AblationConcept(StrictModel):
    concept: str
    weight: float = Field(ge=0, le=1)


class AblationPrerequisite(StrictModel):
    concept: str
    weight: float = Field(ge=0, le=1)
    method: Literal["explicit", "early_prose_proxy", "explicit_and_early_prose_proxy"]


class AblationVariant(StrictModel):
    evidence_mode: Literal["toc_only", "toc_description", "all_available"]
    concept_count: int = Field(ge=0)
    covered_concepts: list[AblationConcept]
    prerequisite_concept_count: int = Field(ge=0)
    prerequisite_concepts: list[AblationPrerequisite]
    lexical_difficulty: float | None = Field(default=None, ge=0, le=1)
    syntactic_complexity: float | None = Field(default=None, ge=0, le=1)
    concept_density: float | None = Field(default=None, ge=0, le=1)
    prerequisite_demand: float | None = Field(default=None, ge=0, le=1)
    evidence_coverage: EvidenceCoverage


class EvidenceAblationReport(StrictModel):
    book_id: str
    title: str
    variants: list[AblationVariant]
    feature_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class AssessmentResponse(StrictModel):
    question_id: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    concept_id: str | None = None
    concept_tags: list[str] = Field(default_factory=list)
    question_type: QuestionType
    difficulty: str = Field(min_length=1)
    correct: bool | None = None
    score: float | None = Field(default=None, ge=0, le=1)

    @field_validator("question_id", "topic_id", "difficulty")
    @classmethod
    def identifiers_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("concept_id")
    @classmethod
    def concept_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("concept_id must not be blank")
        return value.strip() if value is not None else None

    @field_validator("concept_tags")
    @classmethod
    def concept_tags_must_be_unique_and_non_blank(cls, value: list[str]) -> list[str]:
        normalized = [tag.strip() for tag in value]
        if any(not tag for tag in normalized):
            raise ValueError("concept_tags must not contain blanks")
        if len(normalized) != len(set(normalized)):
            raise ValueError("concept_tags must not contain duplicates")
        return normalized

    @model_validator(mode="after")
    def exactly_one_result_value_is_required(self) -> "AssessmentResponse":
        if (self.correct is None) == (self.score is None):
            raise ValueError("exactly one of correct or score is required")
        return self


class Assessment(StrictModel):
    assessment_id: str = Field(min_length=1)
    topic_id: str = Field(min_length=1)
    responses: list[AssessmentResponse] = Field(min_length=1)

    @field_validator("assessment_id", "topic_id")
    @classmethod
    def assessment_identifiers_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def responses_must_match_assessment(self) -> "Assessment":
        question_ids = [response.question_id for response in self.responses]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("assessment contains duplicate question_id values")
        mismatched = sorted(
            response.question_id
            for response in self.responses
            if response.topic_id != self.topic_id
        )
        if mismatched:
            raise ValueError(
                f"responses do not match assessment topic {self.topic_id}: {', '.join(mismatched)}"
            )
        return self


class ReaderDimensionDetail(StrictModel):
    question_type: QuestionType
    score: float = Field(ge=0, le=1)
    response_count: int = Field(ge=1)
    earned_weight: float = Field(ge=0)
    available_weight: float = Field(gt=0)


class ConceptReadiness(StrictModel):
    concept_id: str
    score: float = Field(ge=0, le=1)
    response_count: int = Field(ge=1)
    earned_weight: float = Field(ge=0)
    available_weight: float = Field(gt=0)


class ReaderProfile(StrictModel):
    assessment_id: str
    topic_id: str
    vocabulary: float = Field(ge=0, le=1)
    background_knowledge: float = Field(ge=0, le=1)
    comprehension: float = Field(ge=0, le=1)
    dimension_details: list[ReaderDimensionDetail]
    concept_readiness: list[ConceptReadiness]
    response_count: int = Field(ge=1)
    profile_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def dimension_details_must_match_scores(self) -> "ReaderProfile":
        details = {detail.question_type: detail for detail in self.dimension_details}
        expected = {
            "vocabulary": self.vocabulary,
            "background_knowledge": self.background_knowledge,
            "comprehension": self.comprehension,
        }
        if set(details) != set(expected) or len(details) != len(self.dimension_details):
            raise ValueError("dimension_details must contain each question type exactly once")
        if any(abs(details[name].score - score) > 1e-9 for name, score in expected.items()):
            raise ValueError("dimension detail score does not match reader profile")
        if sum(detail.response_count for detail in details.values()) != self.response_count:
            raise ValueError("dimension response counts do not match reader profile")
        return self


class KnowledgeFitComponents(StrictModel):
    concept_density_fit: float | None = Field(default=None, ge=0, le=1)
    prerequisite_demand_fit: float | None = Field(default=None, ge=0, le=1)
    prerequisite_concept_fit: float | None = Field(default=None, ge=0, le=1)


class RankingComponents(StrictModel):
    topic_fit: float | None = Field(default=None, ge=0, le=1)
    vocabulary_fit: float | None = Field(default=None, ge=0, le=1)
    knowledge_fit: float | None = Field(default=None, ge=0, le=1)
    comprehension_fit: float | None = Field(default=None, ge=0, le=1)


class MatchingDiagnostics(StrictModel):
    lexical_demand: float | None = Field(default=None, ge=0, le=1)
    knowledge_demand: float | None = Field(default=None, ge=0, le=1)
    syntactic_demand: float | None = Field(default=None, ge=0, le=1)
    assessed_prerequisite_count: int = Field(ge=0)
    inferred_prerequisite_count: int = Field(ge=0)


class RankedBook(StrictModel):
    book_id: str
    score: float = Field(ge=0, le=1)
    components: RankingComponents
    knowledge_components: KnowledgeFitComponents
    knowledge_active_weights: dict[KnowledgeComponentName, float]
    knowledge_weight_coverage: float = Field(ge=0, le=1)
    active_weights: dict[RankingComponentName, float]
    component_weight_coverage: float = Field(ge=0, le=1)
    unavailable_components: list[RankingComponentName]
    diagnostics: MatchingDiagnostics
    reasons: list[str]
    model_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reader_profile_version: str
    reader_config_version: str
    reader_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    book_feature_version: str
    book_config_version: str
    book_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("active_weights")
    @classmethod
    def active_weights_must_be_normalized(
        cls, value: dict[RankingComponentName, float]
    ) -> dict[RankingComponentName, float]:
        if any(weight < 0 or weight > 1 for weight in value.values()):
            raise ValueError("active ranking weights must be in [0, 1]")
        if value and abs(sum(value.values()) - 1.0) > 1e-9:
            raise ValueError("active ranking weights must sum to 1.0")
        return value

    @field_validator("knowledge_active_weights")
    @classmethod
    def knowledge_active_weights_must_be_normalized(
        cls, value: dict[KnowledgeComponentName, float]
    ) -> dict[KnowledgeComponentName, float]:
        if any(weight < 0 or weight > 1 for weight in value.values()):
            raise ValueError("active knowledge weights must be in [0, 1]")
        if value and abs(sum(value.values()) - 1.0) > 1e-9:
            raise ValueError("active knowledge weights must sum to 1.0")
        return value


class RankingResponse(StrictModel):
    topic_id: str
    items: list[RankedBook]
    model_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    reader_profile_version: str
    reader_config_version: str
    reader_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class DifficultyJudgment(StrictModel):
    book_id: str = Field(min_length=1)
    human_rank: int = Field(ge=1)


class DifficultyJudgments(StrictModel):
    label_version: str = Field(min_length=1)
    is_synthetic: bool
    topic_id: str = Field(min_length=1)
    items: list[DifficultyJudgment] = Field(min_length=2)

    @model_validator(mode="after")
    def items_must_define_one_complete_order(self) -> "DifficultyJudgments":
        book_ids = [item.book_id for item in self.items]
        if len(book_ids) != len(set(book_ids)):
            raise ValueError("difficulty judgments contain duplicate book_id values")
        ranks = [item.human_rank for item in self.items]
        if len(ranks) != len(set(ranks)):
            raise ValueError("difficulty judgments contain duplicate human_rank values")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("human_rank values must form a contiguous order starting at 1")
        return self


class LoadedDifficultyJudgments(StrictModel):
    judgments: DifficultyJudgments
    content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class DifficultyEvaluationItem(StrictModel):
    book_id: str
    human_rank: int = Field(ge=1)
    system_difficulty: float = Field(ge=0, le=1)
    components: dict[DifficultyComponentName, float | None]
    active_weights: dict[DifficultyComponentName, float]

    @field_validator("components")
    @classmethod
    def components_must_be_complete_and_bounded(
        cls, value: dict[DifficultyComponentName, float | None]
    ) -> dict[DifficultyComponentName, float | None]:
        expected = {
            "lexical_difficulty",
            "syntactic_complexity",
            "concept_density",
            "prerequisite_demand",
        }
        if set(value) != expected:
            raise ValueError(f"difficulty components keys must be exactly: {sorted(expected)}")
        if any(
            component < 0 or component > 1 for component in value.values() if component is not None
        ):
            raise ValueError("difficulty component values must be in [0, 1]")
        return value

    @field_validator("active_weights")
    @classmethod
    def active_weights_must_sum_to_one(
        cls, value: dict[DifficultyComponentName, float]
    ) -> dict[DifficultyComponentName, float]:
        if not value or any(weight < 0 or weight > 1 for weight in value.values()):
            raise ValueError("difficulty active weights must be non-empty and in [0, 1]")
        if abs(sum(value.values()) - 1.0) > 1e-9:
            raise ValueError("difficulty active weights must sum to 1.0")
        return value

    @model_validator(mode="after")
    def active_weights_must_match_available_components(self) -> "DifficultyEvaluationItem":
        available = {name for name, component in self.components.items() if component is not None}
        if set(self.active_weights) != available:
            raise ValueError("difficulty active weights must match non-null component keys")
        expected_score = sum(
            self.components[name] * weight
            for name, weight in self.active_weights.items()
            if self.components[name] is not None
        )
        if abs(self.system_difficulty - expected_score) > 1e-9:
            raise ValueError("system_difficulty must match weighted difficulty components")
        return self


class DifficultyEvaluationReport(StrictModel):
    topic_id: str
    label_version: str
    label_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    is_synthetic: bool
    comparable_book_count: int = Field(ge=2)
    excluded_book_ids: list[str]
    spearman_correlation: float | None = Field(default=None, ge=-1, le=1)
    pairwise_agreement: float = Field(ge=0, le=1)
    pair_count: int = Field(ge=1)
    agreed_pair_count: int = Field(ge=0)
    system_tie_pair_count: int = Field(ge=0)
    items: list[DifficultyEvaluationItem]
    warnings: list[str]

    @model_validator(mode="after")
    def summary_must_match_items(self) -> "DifficultyEvaluationReport":
        if len(self.items) != self.comparable_book_count:
            raise ValueError("comparable_book_count must match difficulty items")
        book_ids = [item.book_id for item in self.items]
        if len(book_ids) != len(set(book_ids)):
            raise ValueError("difficulty evaluation items contain duplicate book_id values")
        human_ranks = [item.human_rank for item in self.items]
        if len(human_ranks) != len(set(human_ranks)):
            raise ValueError("difficulty evaluation items contain duplicate human_rank values")
        if set(book_ids) & set(self.excluded_book_ids):
            raise ValueError("comparable and excluded difficulty book IDs must be disjoint")
        expected_pairs = len(self.items) * (len(self.items) - 1) // 2
        if self.pair_count != expected_pairs:
            raise ValueError("pair_count must match the number of difficulty item pairs")
        expected_agreements = 0
        expected_ties = 0
        for left in range(len(self.items) - 1):
            for right in range(left + 1, len(self.items)):
                left_item = self.items[left]
                right_item = self.items[right]
                if left_item.system_difficulty == right_item.system_difficulty:
                    expected_ties += 1
                    continue
                human_order = left_item.human_rank < right_item.human_rank
                system_order = left_item.system_difficulty < right_item.system_difficulty
                if human_order == system_order:
                    expected_agreements += 1
        if self.agreed_pair_count != expected_agreements:
            raise ValueError("agreed_pair_count must match difficulty items")
        if self.system_tie_pair_count != expected_ties:
            raise ValueError("system_tie_pair_count must match difficulty items")
        expected_agreement = expected_agreements / expected_pairs
        if abs(self.pairwise_agreement - expected_agreement) > 1e-9:
            raise ValueError("pairwise_agreement must match difficulty items")
        return self


class RecommendationComparisonItem(StrictModel):
    book_id: str
    topic_only_rank: int = Field(ge=1)
    topic_only_score: float = Field(ge=0, le=1)
    readiness_rank: int = Field(ge=1)
    readiness_score: float = Field(ge=0, le=1)
    readiness_rank_change: int
    component_weight_coverage: float = Field(ge=0, le=1)
    unavailable_components: list[RankingComponentName]


class RecommendationComparisonReport(StrictModel):
    topic_id: str
    candidate_count: int = Field(ge=1)
    comparison_k: int = Field(ge=1)
    top_k_overlap_count: int = Field(ge=0)
    top_k_overlap_rate: float = Field(ge=0, le=1)
    changed_position_count: int = Field(ge=0)
    topic_only_version: str
    readiness_model_version: str
    ranking_config_version: str
    ranking_config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    items: list[RecommendationComparisonItem]


class AblationComparison(StrictModel):
    from_mode: Literal["toc_only", "toc_description"]
    to_mode: Literal["toc_description", "all_available"]
    added_concepts: list[str]
    removed_concepts: list[str]
    changed_concept_weights: list[str]
    added_prerequisites: list[str]
    removed_prerequisites: list[str]
    newly_available_difficulty_components: list[DifficultyComponentName]

    @model_validator(mode="after")
    def evidence_modes_must_be_adjacent(self) -> "AblationComparison":
        allowed = {
            ("toc_only", "toc_description"),
            ("toc_description", "all_available"),
        }
        if (self.from_mode, self.to_mode) not in allowed:
            raise ValueError("ablation comparison must use adjacent evidence modes")
        return self


class EvaluationFailureCase(StrictModel):
    category: Literal["difficulty_unavailable", "ranking_evidence_limited"]
    book_id: str
    detail: str


class EvaluationReport(StrictModel):
    evaluation_version: str
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    difficulty: DifficultyEvaluationReport
    recommendation: RecommendationComparisonReport
    evidence_ablation: EvidenceAblationReport
    ablation_comparisons: list[AblationComparison]
    failure_cases: list[EvaluationFailureCase]
