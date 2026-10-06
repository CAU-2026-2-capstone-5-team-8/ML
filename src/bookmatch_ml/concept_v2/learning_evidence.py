"""Minimal source references used by the reading checklist; no copied book prose."""

from urllib.parse import urlsplit

from pydantic import Field, field_validator

from bookmatch_ml.data.book_evidence import EditionRelation, EvidenceType
from bookmatch_ml.integration.schemas import ApiModel


class LearningEvidence(ApiModel):
    concept_id: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_url: str | None = None
    evidence_type: EvidenceType
    edition_relation: EditionRelation
    toc_path: list[str] | None = None
    matching_alias: str = Field(min_length=1)
    match_method: str = Field(min_length=1)
    provenance_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("source_url")
    @classmethod
    def safe_source_url(cls, value):
        if value is not None:
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
                raise ValueError("source URL must be an HTTP(S) reference without credentials")
        return value


def normalize_evidence(rows: list[LearningEvidence], covered: set[str]) -> list[LearningEvidence]:
    unique = {}
    identities = {}
    for row in rows:
        if row.concept_id not in covered:
            raise ValueError("evidence concept must belong to this book's covered concepts")
        identity = row.model_dump(exclude={"concept_id", "matching_alias", "match_method"})
        if row.evidence_id in identities and identities[row.evidence_id] != identity:
            raise ValueError("conflicting evidence identity")
        identities[row.evidence_id] = identity
        unique[row.model_dump_json(by_alias=True)] = row
    return [unique[key] for key in sorted(unique)]
