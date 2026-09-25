"""Human-reviewed assessment eligibility without changing recommendation knowledge."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from bookmatch_ml.assessment.blueprint import build_assessment_blueprint
from bookmatch_ml.assessment.pool import build_topic_concept_pool
from bookmatch_ml.assessment.review import (
    AssessmentConceptReviewError,
    build_assessment_concept_review_packet,
    eligible_concepts,
    load_assessment_concept_reviews,
    reviewed_assessment_config,
    validate_assessment_concept_reviews,
)
from bookmatch_ml.assessment.schemas import (
    AssessmentConceptReviewArtifact,
    AssessmentConceptReviewRow,
    LoadedAssessmentConceptReviews,
    TopicConcept,
    TopicConceptPool,
)
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.cli import app
from bookmatch_ml.config import load_assessment_config, load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.io import write_jsonl

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "canonical"
FEATURES = load_feature_config(ROOT / "configs" / "features.yaml")
LEGACY = load_assessment_config(ROOT / "configs" / "assessment.yaml")
REVIEWED = load_assessment_config(ROOT / "configs" / "assessment_reviewed.yaml")
HASH = "sha256:" + "a" * 64


def _inputs():
    dataset = load_canonical_dataset(FIXTURE)
    profiles = build_book_profiles(assemble_book_evidence(dataset), FEATURES)
    return dataset, profiles


def _reviews(rows: list[dict[str, object]]) -> LoadedAssessmentConceptReviews:
    artifact = AssessmentConceptReviewArtifact.model_validate(
        {"review_version": "assessment-concept-review-v1", "reviews": rows}
    )
    identity = json.dumps(artifact.model_dump(mode="json"), sort_keys=True).encode()
    return LoadedAssessmentConceptReviews(
        artifact=artifact,
        content_hash="sha256:" + hashlib.sha256(identity).hexdigest(),
    )


def _synthetic_la_pool(os_pool: TopicConceptPool) -> TopicConceptPool:
    concept_payload = os_pool.concepts[0].model_dump(mode="json")
    concept_payload.update({"topic_id": "linear-algebra", "concept_id": "matrix"})
    for support in concept_payload["book_support"]:
        for evidence in support["evidence"]:
            evidence["concept_id"] = "matrix"
    concept = TopicConcept.model_validate(concept_payload)
    return TopicConceptPool.model_validate(
        os_pool.model_dump(mode="json")
        | {
            "topic_id": "linear-algebra",
            "covered_concept_count": 1,
            "prerequisite_concept_count": 0,
            "concepts": [concept.model_dump(mode="json")],
        }
    )


def _legacy_blueprint(topic_id: str = "operating-systems"):
    dataset, profiles = _inputs()
    return build_assessment_blueprint(
        topic_id,
        dataset,
        profiles,
        FEATURES,
        LEGACY,
        canonical_file_hashes={
            name: HASH for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        },
        book_profiles_hash=HASH,
    )


def _decision(
    concept_id: str,
    concept_role: str,
    status: str,
    *,
    topic_id: str = "operating-systems",
    reason_code: str | None = None,
    review_note: str = "",
) -> dict[str, object]:
    return {
        "topic_id": topic_id,
        "concept_id": concept_id,
        "concept_role": concept_role,
        "status": status,
        "reason_code": reason_code,
        "review_note": review_note,
    }


def test_review_schema_accepts_explicit_decisions_and_unreviewed_rows() -> None:
    artifact = AssessmentConceptReviewArtifact.model_validate(
        {
            "review_version": "assessment-concept-review-v1",
            "reviews": [
                _decision("process", "covered", "eligible", review_note="Diagnostic OS concept."),
                _decision(
                    "computer architecture",
                    "prerequisite",
                    "ineligible",
                    reason_code="too_general",
                    review_note="Too broad for this assessment.",
                ),
                _decision("thread", "covered", "unreviewed"),
            ],
        }
    )

    assert [item.status for item in artifact.reviews] == [
        "eligible",
        "ineligible",
        "unreviewed",
    ]


def test_sample_review_fixture_is_explicitly_non_production() -> None:
    loaded = load_assessment_concept_reviews(
        ROOT / "tests" / "fixtures" / "assessment_concept_reviews.yaml"
    )

    assert [item.status for item in loaded.artifact.reviews] == [
        "eligible",
        "ineligible",
        "eligible",
        "unreviewed",
    ]


def test_review_schema_rejects_duplicate_keys() -> None:
    row = _decision("process", "covered", "unreviewed")
    with pytest.raises(ValidationError, match="duplicate review keys"):
        AssessmentConceptReviewArtifact.model_validate(
            {"review_version": "assessment-concept-review-v1", "reviews": [row, row]}
        )


@pytest.mark.parametrize(
    "row",
    [
        _decision("process", "covered", "eligible"),
        _decision(
            "process",
            "covered",
            "eligible",
            reason_code="too_general",
            review_note="Conflicting reason.",
        ),
        _decision("process", "covered", "ineligible", review_note="Missing reason."),
        _decision(
            "process",
            "covered",
            "unreviewed",
            review_note="Not actually reviewed.",
        ),
    ],
)
def test_review_schema_rejects_invalid_status_reason_combinations(
    row: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AssessmentConceptReviewRow.model_validate(row)


def test_review_validation_rejects_stale_concept_and_unknown_topic() -> None:
    _, profiles = _inputs()
    empty = _reviews([])
    reviewed = reviewed_assessment_config(REVIEWED, empty)
    pool = build_topic_concept_pool("operating-systems", profiles, FEATURES, reviewed)

    stale = _reviews([_decision("not-in-current-pool", "covered", "unreviewed")])
    with pytest.raises(AssessmentConceptReviewError, match="stale or unknown"):
        validate_assessment_concept_reviews(stale, [pool], reviewed)

    unknown_topic = _reviews(
        [_decision("process", "covered", "unreviewed") | {"topic_id": "not-a-topic"}]
    )
    with pytest.raises(AssessmentConceptReviewError, match="unknown review topics"):
        validate_assessment_concept_reviews(unknown_topic, [pool], reviewed)


def test_review_validation_accepts_multiple_topics() -> None:
    _, profiles = _inputs()
    reviews = _reviews(
        [
            _decision(
                "process",
                "covered",
                "eligible",
                review_note="Fixture OS decision.",
            ),
            _decision(
                "matrix",
                "covered",
                "unreviewed",
                topic_id="linear-algebra",
            ),
        ]
    )
    reviewed = reviewed_assessment_config(REVIEWED, reviews)
    os_pool = build_topic_concept_pool("operating-systems", profiles, FEATURES, reviewed)
    la_pool = _synthetic_la_pool(os_pool)
    pools = [os_pool, la_pool]

    validate_assessment_concept_reviews(reviews, pools, reviewed)

    os_pool, la_pool = pools
    assert [item.concept_id for item in eligible_concepts(os_pool, "covered", reviews)] == [
        "process"
    ]
    assert eligible_concepts(la_pool, "covered", reviews) == []


def test_reviewed_selection_keeps_only_eligible_without_implicit_fallback() -> None:
    dataset, profiles = _inputs()
    original_profiles = [item.model_copy(deep=True) for item in profiles]
    reviews = _reviews(
        [
            _decision(
                "process",
                "covered",
                "eligible",
                review_note="Core OS diagnostic concept.",
            ),
            _decision(
                "scheduling",
                "covered",
                "ineligible",
                reason_code="other",
                review_note="Fixture-only exclusion.",
            ),
            _decision(
                "data structures",
                "prerequisite",
                "eligible",
                review_note="Fixture prerequisite retained.",
            ),
            _decision(
                "computer architecture",
                "prerequisite",
                "ineligible",
                reason_code="low_diagnostic_value",
                review_note="Fixture-only exclusion.",
            ),
        ]
    )

    blueprint = build_assessment_blueprint(
        "operating-systems",
        dataset,
        profiles,
        FEATURES,
        REVIEWED,
        canonical_file_hashes={
            name: HASH for name in ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        },
        book_profiles_hash=HASH,
        concept_reviews=reviews,
    )

    assert [(item.role, item.concept_id) for item in blueprint.selected_assessment_concepts] == [
        ("covered", "process"),
        ("prerequisite", "data structures"),
    ]
    assert {item.primary_concept for item in blueprint.question_specs} <= {
        "process",
        "data structures",
    }
    assert all(
        item.question_spec_version == "question-spec-v1" for item in blueprint.question_specs
    )
    assert blueprint.blueprint_version == "assessment-blueprint-v1"
    assert blueprint.config_version == "assessment-config-v2-reviewed"
    assert any("review-eligible" in item.reason for item in blueprint.shortages)
    assert profiles == original_profiles


def test_reviewed_selection_preserves_priority_order() -> None:
    _, profiles = _inputs()
    reviews = _reviews(
        [
            _decision("thread", "covered", "eligible", review_note="Eligible fixture row."),
            _decision("process", "covered", "eligible", review_note="Eligible fixture row."),
            _decision(
                "virtual memory",
                "covered",
                "eligible",
                review_note="Eligible fixture row.",
            ),
        ]
    )
    reviewed = reviewed_assessment_config(REVIEWED, reviews)
    pool = build_topic_concept_pool("operating-systems", profiles, FEATURES, reviewed)

    assert [item.concept_id for item in eligible_concepts(pool, "covered", reviews)] == [
        "process",
        "thread",
        "virtual memory",
    ]


def test_covered_and_prerequisite_review_keys_are_independent() -> None:
    _, profiles = _inputs()
    reviews = _reviews(
        [
            _decision("process", "covered", "eligible", review_note="Covered is eligible."),
            _decision(
                "process",
                "prerequisite",
                "ineligible",
                reason_code="misclassified_prerequisite",
                review_note="Prerequisite role is not eligible.",
            ),
        ]
    )
    reviewed = reviewed_assessment_config(REVIEWED, reviews)
    pool = build_topic_concept_pool("operating-systems", profiles, FEATURES, reviewed)
    covered = next(item for item in pool.concepts if item.concept_id == "process")
    prerequisite = covered.model_copy(
        update={
            "role": "prerequisite",
            "prerequisite_methods": ["explicit"],
            "book_support": [
                item.model_copy(update={"prerequisite_method": "explicit"})
                for item in covered.book_support
            ],
        }
    )
    augmented = pool.model_copy(
        update={
            "concepts": [*pool.concepts, prerequisite],
            "prerequisite_concept_count": pool.prerequisite_concept_count + 1,
        }
    )
    validate_assessment_concept_reviews(reviews, [augmented], reviewed)

    assert [item.role for item in eligible_concepts(augmented, "covered", reviews)] == ["covered"]
    assert eligible_concepts(augmented, "prerequisite", reviews) == []


def test_review_and_config_hash_is_deterministic_and_sensitive(tmp_path: Path) -> None:
    content = """review_version: assessment-concept-review-v1
reviews: []
"""
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text(content, encoding="utf-8")
    second_path.write_text(content, encoding="utf-8")
    first = load_assessment_concept_reviews(first_path)
    second = load_assessment_concept_reviews(second_path)

    assert first.content_hash == second.content_hash
    assert (
        reviewed_assessment_config(REVIEWED, first).content_hash
        == reviewed_assessment_config(REVIEWED, second).content_hash
    )

    second_path.write_text(content + "# changed bytes\n", encoding="utf-8")
    changed = load_assessment_concept_reviews(second_path)
    assert (
        reviewed_assessment_config(REVIEWED, first).content_hash
        != reviewed_assessment_config(REVIEWED, changed).content_hash
    )


def test_legacy_assessment_v1_snapshot_is_unchanged() -> None:
    blueprint = _legacy_blueprint()
    digest = hashlib.sha256(blueprint.model_dump_json().encode()).hexdigest()

    assert digest == "70f120ad4adc7fb579b8107f8285ac4b38fd1f9b87fcd01d30f8beef7385ecb9"
    assert blueprint.config_version == "assessment-config-v1"


def test_legacy_linear_algebra_assessment_v1_snapshot_is_unchanged() -> None:
    blueprint = _legacy_blueprint("linear-algebra")
    digest = hashlib.sha256(blueprint.model_dump_json().encode()).hexdigest()

    assert digest == "6fa7b4b8478c9c98043fb9d25eaa93876b3750db879c9f694a0d948c449cb252"
    assert blueprint.config_version == "assessment-config-v1"


def test_os_review_packet_is_deterministic_and_bounded() -> None:
    _, profiles = _inputs()
    reviews = _reviews([])
    reviewed = reviewed_assessment_config(REVIEWED, reviews)
    pool = build_topic_concept_pool("operating-systems", profiles, FEATURES, reviewed)
    legacy = _legacy_blueprint()
    reserve = reviewed.config.review_reserve

    first = build_assessment_concept_review_packet(pool, legacy, reviewed, reviews, reserve=reserve)
    second = build_assessment_concept_review_packet(
        pool, legacy, reviewed, reviews, reserve=reserve
    )

    assert first == second
    assert first.candidate_count <= 18
    assert all(item.status == "unreviewed" for item in first.candidates)
    process = next(item for item in first.candidates if item.concept_id == "process")
    assert process.concept_role == "covered"
    assert process.current_question_spec_target is True


def test_la_review_packet_is_deterministic_and_topic_specific() -> None:
    _, profiles = _inputs()
    reviews = _reviews([])
    reviewed = reviewed_assessment_config(REVIEWED, reviews)
    pool = build_topic_concept_pool("linear-algebra", profiles, FEATURES, reviewed)
    legacy = _legacy_blueprint("linear-algebra")
    reserve = reviewed.config.review_reserve

    first = build_assessment_concept_review_packet(pool, legacy, reviewed, reviews, reserve=reserve)
    second = build_assessment_concept_review_packet(
        pool, legacy, reviewed, reviews, reserve=reserve
    )

    assert first == second
    assert first.topic_id == "linear-algebra"
    assert all(item.topic_id == "linear-algebra" for item in first.candidates)
    assert all(item.status == "unreviewed" for item in first.candidates)


def test_la_unreviewed_rows_do_not_change_os_reviewed_selection() -> None:
    _, profiles = _inputs()
    os_only = _reviews(
        [
            _decision(
                "process",
                "covered",
                "eligible",
                review_note="Fixture OS decision.",
            )
        ]
    )
    with_la_queue = _reviews(
        [
            _decision(
                "process",
                "covered",
                "eligible",
                review_note="Fixture OS decision.",
            ),
            _decision(
                "matrix",
                "covered",
                "unreviewed",
                topic_id="linear-algebra",
            ),
        ]
    )

    os_config = reviewed_assessment_config(REVIEWED, os_only)
    multi_config = reviewed_assessment_config(REVIEWED, with_la_queue)
    baseline = build_topic_concept_pool("operating-systems", profiles, FEATURES, os_config)
    augmented = build_topic_concept_pool("operating-systems", profiles, FEATURES, multi_config)
    validate_assessment_concept_reviews(os_only, [baseline], os_config)
    validate_assessment_concept_reviews(
        with_la_queue,
        [augmented, _synthetic_la_pool(augmented)],
        multi_config,
    )

    assert [item.concept_id for item in eligible_concepts(baseline, "covered", os_only)] == [
        item.concept_id for item in eligible_concepts(augmented, "covered", with_la_queue)
    ]
    assert [item.concept_id for item in eligible_concepts(baseline, "prerequisite", os_only)] == [
        item.concept_id for item in eligible_concepts(augmented, "prerequisite", with_la_queue)
    ]


def test_review_packet_cli_writes_identical_outputs(tmp_path: Path) -> None:
    _, profiles = _inputs()
    books = tmp_path / "profiles.jsonl"
    write_jsonl(profiles, books)
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    args = [
        "prepare-assessment-concept-review",
        "--data-dir",
        str(FIXTURE),
        "--books",
        str(books),
        "--topic",
        "operating-systems",
        "--feature-config",
        str(ROOT / "configs" / "features.yaml"),
        "--legacy-assessment-config",
        str(ROOT / "configs" / "assessment.yaml"),
        "--reviewed-assessment-config",
        str(ROOT / "configs" / "assessment_reviewed.yaml"),
        "--concept-reviews",
        str(ROOT / "tests" / "fixtures" / "assessment_concept_reviews.yaml"),
    ]
    for output in outputs:
        result = CliRunner().invoke(app, [*args, "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert "process" in result.output
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_current_os_human_review_is_complete_and_exact() -> None:
    reviews = load_assessment_concept_reviews(ROOT / "configs" / "assessment_concept_reviews.yaml")
    os_reviews = [item for item in reviews.artifact.reviews if item.topic_id == "operating-systems"]
    decisions = {
        (item.concept_role, item.concept_id): (item.status, item.reason_code) for item in os_reviews
    }

    assert len(os_reviews) == 16
    assert all(item.status != "unreviewed" for item in os_reviews)
    assert sum(item.status == "eligible" for item in os_reviews) == 11
    assert sum(item.status == "ineligible" for item in os_reviews) == 5
    assert decisions == {
        ("covered", "process"): ("eligible", None),
        ("covered", "thread"): ("eligible", None),
        ("covered", "scheduling"): ("eligible", None),
        ("covered", "synchronization"): ("eligible", None),
        ("covered", "concurrency"): ("eligible", None),
        ("covered", "virtual memory"): ("eligible", None),
        ("covered", "file system"): ("eligible", None),
        ("covered", "deadlock"): ("eligible", None),
        ("covered", "virtualization"): ("eligible", None),
        ("covered", "security"): ("ineligible", "low_diagnostic_value"),
        ("covered", "protection"): ("ineligible", "too_ambiguous"),
        ("prerequisite", "computer architecture"): ("eligible", None),
        ("prerequisite", "assembly language"): ("eligible", None),
        ("prerequisite", "programming"): ("ineligible", "too_general"),
        ("prerequisite", "algorithms"): ("ineligible", "low_diagnostic_value"),
        ("prerequisite", "data structures"): ("ineligible", "low_diagnostic_value"),
    }


def test_current_la_review_queue_is_explicitly_unreviewed() -> None:
    reviews = load_assessment_concept_reviews(ROOT / "configs" / "assessment_concept_reviews.yaml")
    la_reviews = [item for item in reviews.artifact.reviews if item.topic_id == "linear-algebra"]

    assert [(item.concept_role, item.concept_id) for item in la_reviews] == [
        ("covered", "matrix"),
        ("covered", "vector"),
        ("covered", "linear system"),
        ("covered", "orthogonality"),
        ("covered", "dimension"),
        ("covered", "determinant"),
        ("covered", "gaussian elimination"),
        ("covered", "basis"),
        ("covered", "eigenvalue"),
        ("covered", "vector space"),
        ("covered", "diagonalization"),
        ("prerequisite", "high school algebra"),
        ("prerequisite", "systems of equations"),
    ]
    assert all(item.status == "unreviewed" for item in la_reviews)
    assert all(item.reason_code is None and item.review_note == "" for item in la_reviews)


def test_current_os_reviewed_config_hash_is_deterministic() -> None:
    path = ROOT / "configs" / "assessment_concept_reviews.yaml"
    first = reviewed_assessment_config(REVIEWED, load_assessment_concept_reviews(path))
    second = reviewed_assessment_config(REVIEWED, load_assessment_concept_reviews(path))

    assert first.content_hash == second.content_hash
