"""Versioned evidence breakdown; observed success is not a calibrated ability level."""

from collections import defaultdict
from typing import Literal

from pydantic import Field

from bookmatch_ml.config import LoadedReaderConfig
from bookmatch_ml.reader.profile import AssessmentError, build_reader_profile
from bookmatch_ml.schemas import Assessment, AssessmentResponse, QuestionType, StrictModel

# Explicit order, independent of the reader score weights. Changes require a version bump.
DIFFICULTIES = ("easy", "medium", "hard")
QUESTION_TYPES: tuple[QuestionType, ...] = (
    "vocabulary",
    "background_knowledge",
    "comprehension",
)


class DepthEvidence(StrictModel):
    question_type: QuestionType
    difficulty: Literal["easy", "medium", "hard"]
    response_count: int = Field(ge=0)
    score: float | None = Field(default=None, ge=0, le=1)
    full_credit_count: int = Field(ge=0)
    question_ids: list[str]


class NextDepthCheck(StrictModel):
    question_type: QuestionType
    difficulty: Literal["easy", "medium", "hard"]
    reason: Literal["review_observed_gap", "unassessed"]


class ConceptDepthDiagnostic(StrictModel):
    concept_id: str
    observed_score: float = Field(ge=0, le=1)
    response_count: int = Field(ge=1)
    evidence: list[DepthEvidence]
    fully_observed: bool
    next_check: NextDepthCheck | None


class ReaderDiagnostics(StrictModel):
    assessment_id: str
    topic_id: str
    response_count: int = Field(ge=1)
    untagged_response_count: int = Field(ge=0)
    concepts: list[ConceptDepthDiagnostic]
    diagnostic_version: Literal["reader-depth-evidence-v1"] = "reader-depth-evidence-v1"
    profile_version: str
    config_version: str
    config_hash: str
    limitations: list[str]


def _next_check(cells: list[DepthEvidence]) -> NextDepthCheck | None:
    """Review observed non-full-credit evidence before probing unassessed cells."""
    review = next((c for c in cells if c.full_credit_count < c.response_count), None)
    missing = next((c for c in cells if c.response_count == 0), None)
    selected = review or missing
    if selected is None:
        return None
    return NextDepthCheck(
        question_type=selected.question_type,
        difficulty=selected.difficulty,
        reason="review_observed_gap" if review is not None else "unassessed",
    )


def build_reader_diagnostics(
    assessment: Assessment,
    loaded_config: LoadedReaderConfig,
) -> ReaderDiagnostics:
    """Retain the concept × type × declared difficulty evidence hidden by an average.

    Full credit is an observed response result, never an expertise threshold. Scores
    and tags have exactly the same semantics as the existing reader-profile endpoint.
    """
    if set(loaded_config.config.difficulty_weights) != set(DIFFICULTIES):
        raise AssessmentError("reader-depth-evidence-v1 requires easy, medium, hard difficulties")
    ordered = assessment.model_copy(
        update={"responses": sorted(assessment.responses, key=lambda r: r.question_id)}
    )
    profile = build_reader_profile(ordered, loaded_config)
    grouped: dict[str, list[AssessmentResponse]] = defaultdict(list)
    untagged = 0
    for response in ordered.responses:
        concepts = set(response.concept_tags)
        if response.concept_id is not None:
            concepts.add(response.concept_id)
        if not concepts:
            untagged += 1
        for concept in concepts:
            grouped[concept].append(response)
    diagnostics = []
    for concept in profile.concept_readiness:
        cells = []
        for difficulty in DIFFICULTIES:
            for kind in QUESTION_TYPES:
                responses = [
                    r
                    for r in grouped[concept.concept_id]
                    if r.difficulty == difficulty and r.question_type == kind
                ]
                scores = [r.score if r.score is not None else float(r.correct) for r in responses]
                cells.append(
                    DepthEvidence(
                        question_type=kind,
                        difficulty=difficulty,
                        response_count=len(responses),
                        score=sum(scores) / len(scores) if scores else None,
                        full_credit_count=sum(score == 1.0 for score in scores),
                        question_ids=[r.question_id for r in responses],
                    )
                )
        diagnostics.append(
            ConceptDepthDiagnostic(
                concept_id=concept.concept_id,
                observed_score=concept.score,
                response_count=concept.response_count,
                evidence=cells,
                fully_observed=all(c.response_count > 0 for c in cells),
                next_check=_next_check(cells),
            )
        )
    return ReaderDiagnostics(
        assessment_id=assessment.assessment_id,
        topic_id=assessment.topic_id,
        response_count=profile.response_count,
        untagged_response_count=untagged,
        concepts=diagnostics,
        profile_version=profile.profile_version,
        config_version=profile.config_version,
        config_hash=profile.config_hash,
        limitations=[
            "문항에 지정된 난이도별 응답 근거이며 검증된 개인 능력 등급이나 책 난이도가 아닙니다.",
            "어려운 문항의 정답으로 미응시한 쉬운 문항의 이해도를 추정하지 않습니다.",
            "입력 계약에 자기평가와 객관식의 구분이 없어 두 응답의 신뢰도를 구분하지 않습니다.",
            "다음 확인 항목은 진단 제안이며 문항의 존재나 출제를 보장하지 않습니다.",
        ],
    )
