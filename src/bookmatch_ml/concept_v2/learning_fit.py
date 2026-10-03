"""Observation-based learning recommendations; no calibrated difficulty or mastery."""

from typing import Literal

from pydantic import Field, model_validator

from bookmatch_ml.concept_v2.presentation import concept_graph
from bookmatch_ml.config import LoadedRankingV2Config
from bookmatch_ml.integration.schemas import ApiModel

Ability = Literal["meaning", "application", "reasoning"]


class Observation(ApiModel):
    concept_id: str
    ability: Ability
    response_count: int = Field(gt=0)
    correct_count: int = Field(ge=0)

    @model_validator(mode="after")
    def valid_count(self):
        if self.correct_count > self.response_count:
            raise ValueError("correctCount cannot exceed responseCount")
        return self


class LearningBook(ApiModel):
    book_id: str = Field(min_length=1)
    covered_concepts: list[str]
    source_artifact_version: str
    source_artifact_hash: str


class LearningFitRequest(ApiModel):
    topic_id: str
    ability: Ability = "application"
    observations: list[Observation]
    candidate_books: list[LearningBook] = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)


def recommend_learning(request: LearningFitRequest, config: LoadedRankingV2Config) -> dict:
    graph = concept_graph(config, request.topic_id)
    node_ids = {node["id"] for node in graph["nodes"]}
    parents = {node: set() for node in node_ids}
    for edge in graph["edges"]:
        parents[edge["target"]].add(edge["source"])
    observations = {}
    for item in request.observations:
        key = (item.concept_id, item.ability)
        if item.concept_id not in node_ids or key in observations:
            raise ValueError("unknown or duplicate concept ability observation")
        observations[key] = item
    ids = [book.book_id for book in request.candidate_books]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate book")

    def state(concept):
        observed = observations.get((concept, request.ability))
        if observed is None:
            return "unmeasured"
        return "correct" if observed.correct_count == observed.response_count else "needs-practice"

    def ancestors(target):
        found = set()
        pending = list(parents[target])
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(parents[current])
        return found

    plans = []
    unavailable = 0
    for book in request.candidate_books:
        covered = set(book.covered_concepts)
        if len(covered) != len(book.covered_concepts) or covered - node_ids:
            raise ValueError("unknown or duplicate covered concept")
        if not covered:
            unavailable += 1
            continue
        prerequisite = set().union(*(ancestors(c) for c in covered)) - covered
        foundation = [{"conceptId": c, "state": state(c)} for c in sorted(prerequisite)]
        targets = [{"conceptId": c, "state": state(c)} for c in sorted(covered)]
        gaps = sum(row["state"] == "needs-practice" for row in foundation)
        unchecked = sum(row["state"] == "unmeasured" for row in foundation)
        practice = sum(row["state"] == "needs-practice" for row in targets)
        new = sum(row["state"] == "unmeasured" for row in targets)
        status = (
            "foundation-gap"
            if gaps
            else "check-first"
            if unchecked or not foundation
            else "ready-to-explore"
        )
        reasons = []
        if gaps:
            reasons.append(f"선수개념 후보 {gaps}개에서 오답이 있어 먼저 연습하면 좋아요.")
        elif unchecked:
            reasons.append(f"선수개념 후보 {unchecked}개는 아직 문제로 확인하지 않았어요.")
        elif foundation:
            reasons.append("평가한 선수개념 후보에서는 모두 정답을 확인했어요.")
        else:
            reasons.append(
                "현재 지도에서 이 책의 선수개념 후보가 확인되지 않아 준비도 판단을 보류해요."
            )
        reasons.append(f"책 내용 중 연습할 개념 {practice}개, 미평가 개념 {new}개가 있어요.")
        plans.append(
            {
                "bookId": book.book_id,
                "status": status,
                "reviewOnly": practice + new == 0,
                "coveredConcepts": sorted(covered),
                "inferredPrerequisites": sorted(prerequisite),
                "foundation": foundation,
                "foundationStatus": "observed-graph-candidates"
                if foundation
                else "not-established",
                "targets": targets,
                "practiceConceptCount": practice,
                "unmeasuredConceptCount": new,
                "reasons": reasons,
                "sourceArtifactVersion": book.source_artifact_version,
                "sourceArtifactHash": book.source_artifact_hash,
            }
        )
    # Explicit baseline: new learning before review, then foundation category.
    # More TOC headings alone do not improve rank; ties use stable book identity.
    priority = {"ready-to-explore": 0, "check-first": 1, "foundation-gap": 2}
    plans.sort(
        key=lambda p: (
            p["reviewOnly"],
            priority[p["status"]],
            p["foundationStatus"] == "not-established",
            p["practiceConceptCount"] == 0,
            p["bookId"],
        )
    )
    chosen = plans[: request.limit]
    for rank, item in enumerate(chosen, 1):
        item["rank"] = rank
    return {
        "topicId": request.topic_id,
        "ability": request.ability,
        "modelVersion": "concept-learning-v1",
        "interpretation": "observed_answers_not_calibrated_mastery",
        "depthStatus": "unverified",
        "graphVersion": graph["version"],
        "graphHash": graph["graphHash"],
        "configHash": graph["configHash"],
        "graphReviewVersion": graph["reviewVersion"],
        "graphReviewHash": graph["reviewHash"],
        "candidateCount": len(ids),
        "mappedCandidateCount": len(plans),
        "unmappedCandidateCount": unavailable,
        "items": chosen,
    }
