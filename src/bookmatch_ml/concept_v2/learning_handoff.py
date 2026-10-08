"""Explicit opt-in evidence enrichment; the frozen matching adapter stays unchanged."""

from bookmatch_ml.concept_v2.book_evidence_mapping import BookEvidenceConceptMappingReport
from bookmatch_ml.concept_v2.learning_evidence import LearningEvidence, normalize_evidence
from bookmatch_ml.schemas import MatchingBookProfile


def enrich_candidates(
    mapping: BookEvidenceConceptMappingReport, candidates: list[MatchingBookProfile]
) -> list[dict]:
    books = {book.book_id: book for book in mapping.books}
    ids = [book.book_id for book in candidates]
    if len(books) != len(mapping.books) or len(ids) != len(set(ids)) or set(ids) != set(books):
        raise ValueError("mapping and candidate identities must match exactly once")
    result = []
    for candidate in sorted(candidates, key=lambda b: b.book_id):
        mapped = books[candidate.book_id]
        covered = {c.concept for c in candidate.covered_concepts}
        if covered != {c.concept_id for c in mapped.concepts}:
            raise ValueError("mapping and candidate concepts differ")
        if set(mapped.topics) != set(candidate.topic_distribution):
            raise ValueError("mapping and candidate topics differ")
        supports = []
        for concept in mapped.concepts:
            if concept.topic_id not in candidate.topic_distribution:
                raise ValueError("mapping concept topic differs")
            for source in concept.supporting_evidence:
                supports.append(
                    LearningEvidence(
                        concept_id=concept.concept_id,
                        **source.model_dump(
                            include=set(LearningEvidence.model_fields) - {"concept_id"}
                        ),
                    )
                )
        evidence = normalize_evidence(supports, covered)
        row = candidate.model_dump(mode="json")
        row["feature_version"] += "-evidence-v1"
        if len(row["feature_version"]) > 80:
            raise ValueError("enriched feature version exceeds backend limit")
        for concept in row["covered_concepts"]:
            concept["evidence"] = [
                e.model_dump(mode="json", by_alias=False)
                for e in evidence
                if e.concept_id == concept["concept"]
            ]
        result.append(row)
    return result
