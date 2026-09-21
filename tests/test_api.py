import asyncio
import json
import shutil
from importlib.resources import files
from pathlib import Path

import httpx

from bookmatch_ml.api import create_app
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.concept_v2.graph import load_concept_graph
from bookmatch_ml.concept_v2.profile import (
    build_book_concept_profile_v2,
    load_concept_matching_config,
)
from bookmatch_ml.config import load_feature_config, load_reader_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.integration.schemas import (
    BookCandidateDto,
    MatchingReaderProfileDto,
    ReaderProfileRequest,
)
from bookmatch_ml.ranking.matching import to_matching_book_profile, to_matching_reader_profile
from bookmatch_ml.reader.profile import build_reader_profile, load_assessment

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"
FEATURE_CONFIG = load_feature_config(ROOT / "configs" / "features.yaml")
READER_CONFIG = load_reader_config(ROOT / "configs" / "reader.yaml")
APP = create_app()


async def _request(method: str, path: str, *, json_body: object | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=APP)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method, path, json=json_body)


def _post(path: str, json_body: object) -> httpx.Response:
    return asyncio.run(_request("POST", path, json_body=json_body))


def _get(path: str) -> httpx.Response:
    return asyncio.run(_request("GET", path))


def _reader_request() -> dict[str, object]:
    payload = json.loads((ROOT / "examples" / "assessment.json").read_text(encoding="utf-8"))
    request = ReaderProfileRequest(user_id=42, **payload)
    return request.model_dump(mode="json", by_alias=True)


def _matching_reader_payload() -> dict[str, object]:
    profile = build_reader_profile(
        load_assessment(ROOT / "examples" / "assessment.json"), READER_CONFIG
    )
    matching = to_matching_reader_profile(profile)
    dto = MatchingReaderProfileDto(user_id=42, **matching.model_dump())
    return dto.model_dump(mode="json", by_alias=True)


def _candidate_payloads() -> list[dict[str, object]]:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    profiles = build_book_profiles(evidence, FEATURE_CONFIG)
    return [
        BookCandidateDto(**to_matching_book_profile(profile).model_dump()).model_dump(
            mode="json", by_alias=True
        )
        for profile in profiles
    ]


def _experimental_request(mastery: float) -> dict[str, object]:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))
    os_evidence = next(book for book in evidence if "operating-systems" in book.metadata.topics)
    graph = load_concept_graph(ROOT / "configs" / "concept_graph.yaml", FEATURE_CONFIG)
    matching = load_concept_matching_config(ROOT / "configs" / "concept_matching.yaml")
    profile = build_book_concept_profile_v2(
        os_evidence,
        "operating-systems",
        FEATURE_CONFIG,
        graph,
        matching,
        "sha256:" + "0" * 64,
    )
    candidate = next(
        item for item in _candidate_payloads() if item["bookId"] == os_evidence.book_id
    )
    candidate["conceptProfileV2"] = profile.model_dump(mode="json")
    reader = _matching_reader_payload()
    reader["conceptReadiness"] = [
        {"conceptId": concept, "score": mastery}
        for concept in graph.graph.nodes["operating-systems"]
    ]
    return {
        "readerProfile": reader,
        "candidateBooks": [candidate],
        "rankingStrategy": "concept_difficulty_v2_experimental",
    }


def test_reader_profile_endpoint_uses_camel_case_and_echoes_correlation_id() -> None:
    first = _post("/ml/reader-profile", _reader_request())
    second = _post("/ml/reader-profile", _reader_request())

    assert first.status_code == 200
    assert first.json() == second.json()
    payload = first.json()
    assert payload["userId"] == 42
    assert payload["topicId"] == "operating-systems"
    assert payload["responseCount"] == 6
    assert payload["profileVersion"] == "reader-v1"
    assert "backgroundKnowledge" in payload
    assert "background_knowledge" not in payload
    assert payload["configHash"].startswith("sha256:")


def test_rank_endpoint_returns_flat_components_and_evidence_diagnostics() -> None:
    request = {
        "readerProfile": _matching_reader_payload(),
        "candidateBooks": _candidate_payloads(),
        "limit": 5,
    }

    response = _post("/ml/rank", request)

    assert response.status_code == 200
    payload = response.json()
    assert payload["userId"] == 42
    assert payload["topicId"] == "operating-systems"
    assert payload["modelVersion"] == "rank-v1"
    assert len(payload["items"]) == 1
    item = payload["items"][0]
    assert item["topicFit"] == 1.0
    assert item["vocabularyFit"] is not None
    assert item["knowledgeFit"] is not None
    assert item["comprehensionFit"] is not None
    assert item["reasons"]
    assert item["componentWeightCoverage"] == 1.0
    assert item["diagnostics"]["inferredPrerequisiteCount"] >= 0
    assert item["bookFeatureVersion"] == "book-v1"
    assert "conceptDifficulty" not in item


def test_rank_endpoint_opt_in_concept_difficulty_keeps_book_score_reader_independent() -> None:
    novice = _post("/ml/rank", _experimental_request(0.0))
    expert = _post("/ml/rank", _experimental_request(1.0))

    assert novice.status_code == 200, novice.text
    assert expert.status_code == 200, expert.text
    novice_item = novice.json()["items"][0]
    expert_item = expert.json()["items"][0]
    assert novice.json()["modelVersion"] == "concept-difficulty-v2-experimental"
    assert (
        novice_item["conceptDifficulty"]["bookDifficultyScore"]
        == expert_item["conceptDifficulty"]["bookDifficultyScore"]
    )
    assert (
        novice_item["conceptDifficulty"]["burdenLower"]
        >= expert_item["conceptDifficulty"]["burdenUpper"]
    )
    assert novice_item["score"] == novice_item["conceptDifficulty"]["recommendationScore"]
    assert novice_item["conceptDifficulty"]["conceptEvidence"]


def test_experimental_rank_filters_unrelated_candidates_before_profile_validation() -> None:
    request = _experimental_request(0.5)
    unrelated = next(
        candidate
        for candidate in _candidate_payloads()
        if candidate["topicDistribution"].get("linear-algebra") == 1.0
    )
    request["candidateBooks"].append(unrelated)

    response = _post("/ml/rank", request)

    assert response.status_code == 200, response.text
    assert len(response.json()["items"]) == 1


def test_baseline_app_starts_with_legacy_external_config_directory(monkeypatch, tmp_path: Path):
    for name in ("reader.yaml", "ranking.yaml"):
        shutil.copyfile(ROOT / "configs" / name, tmp_path / name)
    monkeypatch.setenv("BOOKMATCH_ML_CONFIG_DIR", str(tmp_path))

    application = create_app()

    assert {route.path for route in application.routes if route.path.startswith("/ml/")} == {
        "/ml/reader-profile",
        "/ml/rank",
    }


def test_rank_endpoint_supports_specific_cross_topic_fit() -> None:
    candidates = _candidate_payloads()
    linear_algebra_id = candidates[1]["bookId"]
    request = {
        "readerProfile": _matching_reader_payload(),
        "candidateBooks": candidates,
        "bookId": linear_algebra_id,
    }

    response = _post("/ml/rank", request)

    assert response.status_code == 200
    assert [item["bookId"] for item in response.json()["items"]] == [linear_algebra_id]
    assert response.json()["items"][0]["topicFit"] == 0.0


def test_rank_endpoint_exposes_missing_prose_without_zero_substitution() -> None:
    candidate = _candidate_payloads()[0]
    candidate.update(
        {
            "lexicalDifficulty": None,
            "syntacticComplexity": None,
            "conceptDensity": None,
            "prerequisiteDemand": None,
            "prerequisiteConcepts": [],
        }
    )
    request = {
        "readerProfile": _matching_reader_payload(),
        "candidateBooks": [candidate],
    }

    response = _post("/ml/rank", request)

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["score"] == 1.0
    assert item["vocabularyFit"] is None
    assert item["knowledgeFit"] is None
    assert item["comprehensionFit"] is None
    assert item["activeWeights"] == {"topic_fit": 1.0}
    assert item["componentWeightCoverage"] == 0.35
    assert item["unavailableComponents"] == [
        "vocabulary_fit",
        "knowledge_fit",
        "comprehension_fit",
    ]


def test_api_rejects_domain_invalid_assessment() -> None:
    request = _reader_request()
    request["responses"] = request["responses"][:1]

    response = _post("/ml/reader-profile", request)

    assert response.status_code == 422
    assert "insufficient responses" in response.json()["detail"]


def test_api_rejects_duplicate_candidates_and_unknown_specific_book() -> None:
    candidates = _candidate_payloads()
    base = {"readerProfile": _matching_reader_payload()}

    duplicate = _post("/ml/rank", {**base, "candidateBooks": [candidates[0], candidates[0]]})
    unknown = _post(
        "/ml/rank",
        {**base, "candidateBooks": candidates, "bookId": "book-not-supplied"},
    )

    assert duplicate.status_code == 422
    assert unknown.status_code == 422


def test_openapi_contract_exposes_only_the_two_calculation_routes() -> None:
    schema = _get("/openapi.json").json()

    assert set(schema["paths"]) == {"/ml/reader-profile", "/ml/rank"}
    reader_properties = schema["components"]["schemas"]["ReaderProfileRequest"]["properties"]
    rank_properties = schema["components"]["schemas"]["RankRequest"]["properties"]
    assert "assessmentId" in reader_properties
    assert "assessment_id" not in reader_properties
    assert "candidateBooks" in rank_properties


def test_app_factory_uses_packaged_configs_outside_repository_working_directory(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    application = create_app()

    assert {route.path for route in application.routes if route.path.startswith("/ml/")} == {
        "/ml/reader-profile",
        "/ml/rank",
    }


def test_api_module_does_not_create_an_app_or_load_configs_at_import() -> None:
    import bookmatch_ml.api as api_module

    assert not hasattr(api_module, "app")


def test_packaged_api_configs_match_versioned_project_defaults() -> None:
    packaged = files("bookmatch_ml.default_configs")

    for name in ("reader.yaml", "ranking.yaml", "concept_difficulty.yaml"):
        assert packaged.joinpath(name).read_bytes() == (ROOT / "configs" / name).read_bytes()
