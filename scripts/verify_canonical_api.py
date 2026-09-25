"""Verify canonical data -> profiles -> local HTTP API; never claim model accuracy."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.config import (
    load_feature_config,
    load_ranking_config,
    load_ranking_v2_config,
    load_reader_config,
)
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.integration.schemas import (
    BookCandidateDto,
    MatchingReaderProfileDto,
    RankRequest,
    ReaderProfileRequest,
)
from bookmatch_ml.integration.service import IntegrationService
from bookmatch_ml.ranking.matching import to_matching_book_profile, to_matching_reader_profile

ROOT = Path(__file__).resolve().parents[1]


def verify_equal(expected: object, actual: object, label: str) -> None:
    if expected != actual:
        raise ValueError(f"{label}: HTTP response differs from local calculation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        parser.error("base-url must be a local HTTP origin")
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new run directory")

    def post(path: str, body: dict) -> dict:
        request = Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"{path}: unexpected status {response.status}")
            return json.load(response)

    evidence = assemble_book_evidence(load_canonical_dataset(args.data_dir))
    config = load_feature_config(args.config_dir / "features.yaml")
    profiles = build_book_profiles(evidence, config)
    verify_equal(profiles, build_book_profiles(evidence, config), "book profile determinism")
    service = IntegrationService(
        load_reader_config(args.config_dir / "reader.yaml"),
        load_ranking_config(args.config_dir / "ranking.yaml"),
        load_ranking_v2_config(args.config_dir / "ranking_v2.yaml"),
    )
    assessment = ReaderProfileRequest.model_validate_json(args.assessment.read_bytes())
    reader_request = assessment.model_dump(mode="json", by_alias=True)
    reader = service.build_reader_profile(assessment)
    reader_response = post("/ml/reader-profile", reader_request)
    verify_equal(reader.model_dump(mode="json", by_alias=True), reader_response, "reader profile")

    # Project the public response to the matching contract, retaining concept readiness.
    from bookmatch_ml.schemas import ReaderProfile

    internal_reader = ReaderProfile.model_validate(
        reader.model_dump(exclude={"user_id"}, by_alias=False)
    )
    matching_reader = MatchingReaderProfileDto(
        user_id=reader.user_id, **to_matching_reader_profile(internal_reader).model_dump()
    )
    request = RankRequest(
        reader_profile=matching_reader,
        candidate_books=[
            BookCandidateDto(**to_matching_book_profile(book).model_dump()) for book in profiles
        ],
        limit=5,
    )
    rank_request = request.model_dump(mode="json", by_alias=True)
    expected = service.rank(request).model_dump(mode="json", by_alias=True)
    result = post("/ml/rank", rank_request)
    verify_equal(expected, result, "ranking")
    verify_equal(result, post("/ml/rank", rank_request), "ranking determinism")
    summary = {
        "verification_version": "canonical-http-v1",
        "scope": "integration correctness, not difficulty or recommendation accuracy",
        "input_sha256": {
            name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
            for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        },
        "assessment_sha256": hashlib.sha256(args.assessment.read_bytes()).hexdigest(),
        "book_count": len(profiles),
        "books_with_toc": sum(book.evidence_coverage.has_toc for book in profiles),
        "toc_entries": sum(book.evidence_coverage.toc_entry_count for book in profiles),
        "books_with_analyzed_prose": sum(
            book.difficulty_profile.analyzed_document_count > 0 for book in profiles
        ),
        "books_with_covered_concepts": sum(
            bool(book.concept_profile.covered_concepts) for book in profiles
        ),
        "http_matches_local_calculation": True,
        "repeat_rank_identical": True,
        "feature_config_hash": config.content_hash,
        "ranking": result,
    }
    args.output_dir.mkdir(parents=True)
    for name, value in {
        "reader-request.json": reader_request,
        "reader-response.json": reader_response,
        "rank-request.json": rank_request,
        "report.json": summary,
    }.items():
        (args.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "ranking"}, indent=2))


if __name__ == "__main__":
    main()
