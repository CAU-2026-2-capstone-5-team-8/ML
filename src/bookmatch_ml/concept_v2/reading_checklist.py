"""Deterministic graph-ordered checks, not a diagnosis of mastery or book depth."""


def build_checklist(request, book, covered, prerequisites, parents, observations):
    remaining = covered | prerequisites
    dependencies = {c: sorted(parents[c] & remaining) for c in remaining}
    rows = []
    while remaining:
        ready = sorted(c for c in remaining if not (set(dependencies[c]) & remaining))
        if not ready:
            raise ValueError("cyclic prerequisite graph")
        for concept in ready:
            observed = observations.get((concept, request.ability))
            state = (
                "unmeasured"
                if observed is None
                else (
                    "correct"
                    if observed.correct_count == observed.response_count
                    else "needs-practice"
                )
            )
            rows.append(
                {
                    "conceptId": concept,
                    "isCovered": concept in covered,
                    "isPrerequisite": concept in prerequisites,
                    "state": state,
                    "responseCount": None if observed is None else observed.response_count,
                    "correctCount": None if observed is None else observed.correct_count,
                    "nextAction": {
                        "unmeasured": "assess-concept",
                        "needs-practice": "review-concept",
                        "correct": "continue-learning",
                    }[state],
                    "dependsOn": dependencies[concept],
                    "requiredFor": sorted(c for c, deps in dependencies.items() if concept in deps),
                    "evidence": [
                        e.model_dump(mode="json", by_alias=True)
                        for e in book.concept_evidence
                        if e.concept_id == concept
                    ],
                    "teachingSufficiency": "unverified",
                }
            )
        remaining = remaining - set(ready)
    return {
        "version": "reading-checklist-v2",
        "interpretation": "observed_answers_not_calibrated_mastery",
        "orderPolicy": "prerequisites-first-stable-concept-id",
        "concepts": rows,
        "limitations": [
            "정답 관찰은 숙달 확정이 아니며 문항 수와 평가 범위를 함께 확인해야 합니다.",
            "선수관계는 검토된 그래프의 후보이며 책의 실제 요구사항을 확정하지 않습니다.",
            "연결 없는 개념은 난이도 순서가 아닙니다. 교육과정이 아닌 확인 목록입니다.",
            "목차에 나오는 선수개념도 충분히 설명한다고 가정하지 않습니다.",
            "부분 목차와 다른 판본의 근거로 책 전체의 설명 깊이·본문 난이도를 판단하지 않습니다.",
        ],
    }
