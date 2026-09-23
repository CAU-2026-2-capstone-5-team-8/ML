"""Compatibility tests for source-aware concepts and the existing ranking contract."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from bookmatch_ml.api import create_app
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.cli import app
from bookmatch_ml.concept_v2.book_evidence_mapping import (
    BookConceptPresence,
    BookEvidenceConceptMapping,
    BookEvidenceConceptMappingReport,
    ConceptEvidenceSupport,
)
from bookmatch_ml.config import load_feature_config, load_ranking_config, load_reader_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.integration.schemas import BookCandidateDto, MatchingReaderProfileDto
from bookmatch_ml.io import write_json, write_jsonl
from bookmatch_ml.ranking.matching import rank_matching_books, to_matching_reader_profile
from bookmatch_ml.ranking.source_aware_adapter import (
    ADAPTER_VERSION,
    SourceAwareAdapterError,
    build_source_aware_matching_book_profiles,
)
from bookmatch_ml.reader.profile import build_reader_profile, load_assessment

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures/canonical"
FEATURES = load_feature_config(ROOT / "configs/features.yaml")
RANKING = load_ranking_config(ROOT / "configs/ranking.yaml")
READER = load_reader_config(ROOT / "configs/reader.yaml")
HASH = "sha256:" + "a" * 64
APP = create_app()


def _profiles():
    return build_book_profiles(
        assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR)),
        FEATURES,
    )


def _support(identifier: str) -> ConceptEvidenceSupport:
    return ConceptEvidenceSupport(
        evidence_id=identifier,
        evidence_type="toc_exact",
        source_id=f"source_{identifier[-4:]}",
        provider="fixture",
        source_type="other",
        source_evidence_tier="exact_edition_toc",
        edition_relation="exact",
        toc_path=["Fixture", identifier],
        matching_alias="fixture alias",
        match_method="normalized_alias_span_v2",
        provenance_hash=HASH,
    )


def _book_mapping(profile, concepts: list[BookConceptPresence]):
    return BookEvidenceConceptMapping(
        book_id=profile.book_id,
        title=f"Title for {profile.book_id}",
        topics=sorted(profile.concept_profile.topic_distribution),
        matcher_version="normalized_alias_span_v2",
        model_version="concept-matching-v2-overlap",
        matching_config_version="concept-matching-config-v3",
        matching_config_hash=HASH,
        evidence_row_count=2,
        matched_evidence_row_count=1 if concepts else 0,
        ambiguous_evidence_row_count=0,
        concept_presence_count=len(concepts),
        raw_match_occurrence_count=sum(item.support_count for item in concepts),
        concepts=concepts,
    )


def _presence(topic: str, concept: str, suffixes: list[str]) -> BookConceptPresence:
    supports = [_support(f"evidence_{suffix * 20}"[:29]) for suffix in suffixes]
    return BookConceptPresence(
        topic_id=topic,
        concept_id=concept,
        supporting_evidence=supports,
        support_count=len(supports),
    )


def _mapping() -> BookEvidenceConceptMappingReport:
    profiles = _profiles()
    by_topic = {
        next(iter(profile.concept_profile.topic_distribution)): profile for profile in profiles
    }
    os_presence = _presence("operating-systems", "process", ["1", "2"])
    duplicate_process = _presence("operating-systems", "process", ["3"])
    scheduling = _presence("operating-systems", "scheduling", ["4"])
    matrix = _presence("linear-algebra", "matrix", ["5"])
    vector_space = _presence("linear-algebra", "vector space", ["6"])
    books = [
        _book_mapping(by_topic["operating-systems"], [os_presence, duplicate_process, scheduling]),
        _book_mapping(by_topic["linear-algebra"], [matrix, vector_space]),
    ]
    return BookEvidenceConceptMappingReport(
        report_version="book-evidence-concept-presence-v1",
        book_evidence_contract_version="book-evidence-v1",
        book_evidence_hash=HASH,
        matcher_version="normalized_alias_span_v2",
        model_version="concept-matching-v2-overlap",
        matching_config_version="concept-matching-config-v3",
        matching_config_hash=HASH,
        total_books=2,
        books_with_matched_concepts=2,
        books_without_matched_concepts=0,
        unique_book_concept_presence_count=4,
        raw_match_occurrence_count=6,
        books=sorted(books, key=lambda item: item.book_id),
    )


def test_exact_join_uses_binary_presence_and_preserves_existing_fields() -> None:
    profiles = _profiles()
    candidates = build_source_aware_matching_book_profiles(_mapping(), profiles)
    profile_by_id = {profile.book_id: profile for profile in profiles}

    assert len(candidates) == 2
    for candidate in candidates:
        source = profile_by_id[candidate.book_id]
        difficulty = source.difficulty_profile
        assert candidate.topic_distribution == source.concept_profile.topic_distribution
        assert [item.model_dump() for item in candidate.prerequisite_concepts] == [
            {"concept": item.concept, "weight": item.weight}
            for item in source.concept_profile.prerequisite_concepts
        ]
        assert candidate.lexical_difficulty == difficulty.lexical_difficulty
        assert candidate.syntactic_complexity == difficulty.syntactic_complexity
        assert candidate.concept_density == difficulty.concept_density
        assert candidate.prerequisite_demand == difficulty.prerequisite_demand
        assert all(item.weight == 1.0 for item in candidate.covered_concepts)

    os_candidate = next(
        item for item in candidates if "operating-systems" in item.topic_distribution
    )
    assert [item.concept for item in os_candidate.covered_concepts] == ["process", "scheduling"]
    linear_algebra_profile = next(
        profile
        for profile in profiles
        if "linear-algebra" in profile.concept_profile.topic_distribution
    )
    assert linear_algebra_profile.difficulty_profile.lexical_difficulty is None


def test_support_count_does_not_change_binary_weight() -> None:
    candidate = next(
        item
        for item in build_source_aware_matching_book_profiles(_mapping(), _profiles())
        if "operating-systems" in item.topic_distribution
    )

    assert {item.concept: item.weight for item in candidate.covered_concepts} == {
        "process": 1.0,
        "scheduling": 1.0,
    }


def test_join_rejects_missing_and_duplicate_book_ids() -> None:
    profiles = _profiles()
    mapping = _mapping()

    with pytest.raises(SourceAwareAdapterError, match="book IDs differ"):
        build_source_aware_matching_book_profiles(
            mapping.model_copy(update={"books": mapping.books[:1]}), profiles
        )
    with pytest.raises(SourceAwareAdapterError, match="duplicate adapter book IDs"):
        build_source_aware_matching_book_profiles(
            mapping.model_copy(update={"books": [*mapping.books, mapping.books[0]]}), profiles
        )
    with pytest.raises(SourceAwareAdapterError, match="duplicate adapter book IDs"):
        build_source_aware_matching_book_profiles(mapping, [*profiles, profiles[0]])


def test_candidates_validate_through_camel_case_api_dto() -> None:
    candidate = build_source_aware_matching_book_profiles(_mapping(), _profiles())[0]
    dto = BookCandidateDto(**candidate.model_dump())
    payload = dto.model_dump_json(by_alias=True)

    assert "topicDistribution" in payload
    assert BookCandidateDto.model_validate_json(payload).to_internal() == candidate


@pytest.mark.parametrize(
    ("assessment_path", "topic", "expected_concept"),
    [
        ("examples/assessment.json", "operating-systems", "process"),
        ("examples/concept_matching_la_assessment.json", "linear-algebra", "matrix"),
    ],
)
def test_source_aware_candidates_run_existing_ranking_end_to_end(
    assessment_path: str,
    topic: str,
    expected_concept: str,
) -> None:
    reader = to_matching_reader_profile(
        build_reader_profile(load_assessment(ROOT / assessment_path), READER)
    )
    candidates = build_source_aware_matching_book_profiles(_mapping(), _profiles())

    first = rank_matching_books(reader, candidates, RANKING, limit=10)
    second = rank_matching_books(reader, list(reversed(candidates)), RANKING, limit=10)

    assert first == second
    assert first.topic_id == topic
    assert len(first.items) == 1
    assert any(
        reason.startswith("주요 개념:") and expected_concept in reason
        for reason in first.items[0].reasons
    )


async def _post_rank(payload: dict[str, object]) -> httpx.Response:
    transport = httpx.ASGITransport(app=APP)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.post("/ml/rank", json=payload)


@pytest.mark.parametrize(
    "assessment_path",
    ["examples/assessment.json", "examples/concept_matching_la_assessment.json"],
)
def test_adapter_candidates_are_accepted_by_existing_rank_endpoint(
    assessment_path: str,
) -> None:
    reader = to_matching_reader_profile(
        build_reader_profile(load_assessment(ROOT / assessment_path), READER)
    )
    candidates = build_source_aware_matching_book_profiles(_mapping(), _profiles())
    payload = {
        "readerProfile": MatchingReaderProfileDto(user_id=7, **reader.model_dump()).model_dump(
            mode="json", by_alias=True
        ),
        "candidateBooks": [
            BookCandidateDto(**candidate.model_dump()).model_dump(mode="json", by_alias=True)
            for candidate in candidates
        ],
        "limit": 10,
    }

    response = asyncio.run(_post_rank(payload))

    assert response.status_code == 200
    assert response.json()["topicId"] == reader.topic_id
    assert len(response.json()["items"]) == 1
    assert any(reason.startswith("주요 개념:") for reason in response.json()["items"][0]["reasons"])


def test_cli_writes_deterministic_candidates_and_provenance_report(tmp_path: Path) -> None:
    mapping_path = tmp_path / "mapping.json"
    profiles_path = tmp_path / "profiles.jsonl"
    write_json(_mapping(), mapping_path)
    write_jsonl(_profiles(), profiles_path)
    outputs = []
    for prefix in ("first", "second"):
        candidates = tmp_path / f"{prefix}.jsonl"
        report = tmp_path / f"{prefix}.json"
        result = CliRunner().invoke(
            app,
            [
                "build-matching-book-candidates",
                "--concept-mapping",
                str(mapping_path),
                "--book-profiles",
                str(profiles_path),
                "--output",
                str(candidates),
                "--report",
                str(report),
            ],
        )
        assert result.exit_code == 0, result.output
        outputs.append((candidates.read_bytes(), report.read_bytes()))

    assert outputs[0] == outputs[1]
    report = json.loads(outputs[0][1])
    assert report["adapter_version"] == ADAPTER_VERSION
    assert report["joined_book_count"] == 2
    assert report["raw_occurrence_count_used_for_ranking"] is False
    assert report["binary_covered_concept_weight"] == 1.0
