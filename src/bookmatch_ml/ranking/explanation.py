"""Deterministic Korean explanation templates grounded in matching values."""

from bookmatch_ml.config import ExplanationConfig
from bookmatch_ml.schemas import (
    MatchingBookProfile,
    MatchingDiagnostics,
    MatchingReaderProfile,
    RankingComponents,
)


def _gap_reason(
    label: str,
    readiness: float,
    demand: float,
    config: ExplanationConfig,
) -> str:
    gap = demand - readiness
    if abs(gap) <= config.close_gap:
        return f"{label}는 현재 준비도와 대체로 맞습니다."
    if gap >= config.moderate_gap:
        return f"{label}는 현재 준비도보다 높습니다."
    if gap > 0:
        return f"{label}는 현재 준비도보다 다소 높습니다."
    if gap <= -config.moderate_gap:
        return f"{label}는 현재 준비도보다 낮습니다."
    return f"{label}는 현재 준비도보다 다소 낮습니다."


def build_reasons(
    reader: MatchingReaderProfile,
    book: MatchingBookProfile,
    components: RankingComponents,
    diagnostics: MatchingDiagnostics,
    config: ExplanationConfig,
) -> list[str]:
    reasons: list[str] = []

    if components.topic_fit is None:
        reasons.append("주제 적합도를 계산할 수 있는 근거가 없습니다.")
    elif components.topic_fit >= config.topic_match:
        reasons.append(f"선택한 {reader.topic_id} 주제와 관련성이 높습니다.")
    elif components.topic_fit == 0:
        reasons.append(f"선택한 {reader.topic_id} 주제와 일치하지 않습니다.")
    else:
        reasons.append(f"선택한 {reader.topic_id} 주제와 일부 관련이 있습니다.")

    lexical_missing = components.vocabulary_fit is None
    syntax_missing = components.comprehension_fit is None
    if lexical_missing and syntax_missing:
        reasons.append("충분한 실제 본문이 없어 어휘 및 문장 난이도 평가는 제한적입니다.")
    elif lexical_missing:
        reasons.append("충분한 실제 본문이 없어 어휘 난이도 평가는 제한적입니다.")
    elif syntax_missing:
        reasons.append("충분한 실제 본문이 없어 문장 난이도 평가는 제한적입니다.")

    if diagnostics.lexical_demand is not None:
        reasons.append(
            _gap_reason("어휘 난이도", reader.vocabulary, diagnostics.lexical_demand, config)
        )
    if diagnostics.knowledge_demand is not None:
        reasons.append(
            _gap_reason(
                "개념·선행지식 요구도",
                reader.background_knowledge,
                diagnostics.knowledge_demand,
                config,
            )
        )
    elif components.knowledge_fit is None:
        reasons.append("본문 또는 평가된 선행 개념이 부족해 선행지식 적합도 평가는 제한적입니다.")
    if diagnostics.syntactic_demand is not None:
        reasons.append(
            _gap_reason(
                "문장 복잡도",
                reader.comprehension,
                diagnostics.syntactic_demand,
                config,
            )
        )

    if diagnostics.inferred_prerequisite_count:
        reasons.append(
            "추정 선행 개념 "
            f"{diagnostics.inferred_prerequisite_count}개 중 평가와 겹치는 항목은 "
            f"{diagnostics.assessed_prerequisite_count}개입니다."
        )

    maximum = config.maximum_covered_concepts
    concepts = [item.concept for item in book.covered_concepts[:maximum]]
    if concepts:
        reasons.append("주요 개념: " + ", ".join(concepts) + ".")
    return reasons
