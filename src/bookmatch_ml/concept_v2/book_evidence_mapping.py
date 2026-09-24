"""Map source-aware book evidence to deduplicated production concept presence."""

from collections import defaultdict

from pydantic import Field

from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    match_concept_text_production,
)
from bookmatch_ml.config import LoadedFeatureConfig
from bookmatch_ml.data.book_evidence import (
    EditionRelation,
    EvidenceTier,
    EvidenceType,
    ImportedBookEvidence,
)
from bookmatch_ml.schemas import SourceType, StrictModel

MAPPING_REPORT_VERSION = "book-evidence-concept-presence-v1"


class ConceptEvidenceSupport(StrictModel):
    """One source-aware evidence row supporting a canonical concept."""

    evidence_id: str
    evidence_type: EvidenceType
    source_id: str
    provider: str
    source_type: SourceType
    source_evidence_tier: EvidenceTier | None = None
    edition_relation: EditionRelation
    toc_path: list[str] | None = None
    matching_alias: str
    match_method: str
    provenance_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


class BookConceptPresence(StrictModel):
    """Set-valued book concept presence with all retained supporting rows."""

    topic_id: str
    concept_id: str
    supporting_evidence: list[ConceptEvidenceSupport]
    support_count: int = Field(ge=1)


class BookEvidenceConceptMapping(StrictModel):
    """Production concept mapping for one imported book-evidence record."""

    book_id: str
    title: str
    topics: list[str]
    matcher_version: str
    model_version: str
    matching_config_version: str
    matching_config_hash: str
    evidence_row_count: int = Field(ge=1)
    matched_evidence_row_count: int = Field(ge=0)
    ambiguous_evidence_row_count: int = Field(ge=0)
    concept_presence_count: int = Field(ge=0)
    raw_match_occurrence_count: int = Field(ge=0)
    concepts: list[BookConceptPresence]


class BookEvidenceConceptMappingReport(StrictModel):
    """Deterministic Scale-50-capable mapping report without ranking weights."""

    report_version: str
    book_evidence_contract_version: str
    book_evidence_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    matcher_version: str
    model_version: str
    matching_config_version: str
    matching_config_hash: str
    total_books: int = Field(ge=1)
    books_with_matched_concepts: int = Field(ge=0)
    books_without_matched_concepts: int = Field(ge=0)
    unique_book_concept_presence_count: int = Field(ge=0)
    raw_match_occurrence_count: int = Field(ge=0)
    books: list[BookEvidenceConceptMapping]


def build_book_evidence_concept_mapping_report(
    records: list[ImportedBookEvidence],
    book_evidence_hash: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
) -> BookEvidenceConceptMappingReport:
    """Match every evidence row, then deduplicate concept presence within each book/topic."""

    books: list[BookEvidenceConceptMapping] = []
    for record in sorted(records, key=lambda item: item.book.book_id):
        topics = sorted(set(record.book.topics) & set(graph.graph.nodes))
        supports: dict[tuple[str, str], list[ConceptEvidenceSupport]] = defaultdict(list)
        matched_evidence_ids: set[str] = set()
        ambiguous_evidence_ids: set[str] = set()
        raw_match_count = 0
        for evidence in sorted(record.evidence, key=lambda item: item.evidence_id):
            for topic in topics:
                matches, ambiguous = match_concept_text_production(
                    evidence.text,
                    topic,
                    features,
                    graph,
                    matching,
                )
                if ambiguous:
                    ambiguous_evidence_ids.add(evidence.evidence_id)
                    continue
                if matches:
                    matched_evidence_ids.add(evidence.evidence_id)
                raw_match_count += len(matches)
                for match in matches:
                    supports[(topic, match.concept_id)].append(
                        ConceptEvidenceSupport(
                            evidence_id=evidence.evidence_id,
                            evidence_type=evidence.evidence_type,
                            source_id=evidence.source_id,
                            provider=evidence.provider,
                            source_type=evidence.source_type,
                            source_evidence_tier=evidence.source_evidence_tier,
                            edition_relation=evidence.edition_relation,
                            toc_path=evidence.toc_path,
                            matching_alias=match.matching_alias,
                            match_method=match.match_method,
                            provenance_hash=evidence.provenance_hash,
                        )
                    )
        concepts = [
            BookConceptPresence(
                topic_id=topic,
                concept_id=concept,
                supporting_evidence=sorted(
                    items,
                    key=lambda item: (
                        item.evidence_id,
                        item.matching_alias,
                        item.evidence_type,
                    ),
                ),
                support_count=len(items),
            )
            for (topic, concept), items in sorted(supports.items())
        ]
        books.append(
            BookEvidenceConceptMapping(
                book_id=record.book.book_id,
                title=record.book.title,
                topics=topics,
                matcher_version=matching.config.matcher_version,
                model_version=matching.config.model_version,
                matching_config_version=matching.config.config_version,
                matching_config_hash=matching.content_hash,
                evidence_row_count=len(record.evidence),
                matched_evidence_row_count=len(matched_evidence_ids),
                ambiguous_evidence_row_count=len(ambiguous_evidence_ids),
                concept_presence_count=len(concepts),
                raw_match_occurrence_count=raw_match_count,
                concepts=concepts,
            )
        )
    return BookEvidenceConceptMappingReport(
        report_version=MAPPING_REPORT_VERSION,
        book_evidence_contract_version="book-evidence-v1",
        book_evidence_hash=book_evidence_hash,
        matcher_version=matching.config.matcher_version,
        model_version=matching.config.model_version,
        matching_config_version=matching.config.config_version,
        matching_config_hash=matching.content_hash,
        total_books=len(books),
        books_with_matched_concepts=sum(bool(book.concepts) for book in books),
        books_without_matched_concepts=sum(not book.concepts for book in books),
        unique_book_concept_presence_count=sum(len(book.concepts) for book in books),
        raw_match_occurrence_count=sum(book.raw_match_occurrence_count for book in books),
        books=books,
    )
