"""Deterministic concept-assessment and intended-difficulty contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from bookmatch_ml.assessment.blueprint import build_assessment_blueprint
from bookmatch_ml.assessment.difficulty import assign_target_difficulty
from bookmatch_ml.assessment.pool import AssessmentBlueprintError, build_topic_concept_pool
from bookmatch_ml.assessment.schemas import (
    AssessmentBlueprint,
    ConceptMastery,
    ConceptSelfAssessment,
    QuestionSpec,
)
from bookmatch_ml.assessment.self_assessment import validate_self_assessment
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.cli import app
from bookmatch_ml.config import (
    AssessmentPriorityConfig,
    ConfigError,
    TopicLexicon,
    load_assessment_config,
    load_feature_config,
)
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.io import write_jsonl
from bookmatch_ml.schemas import PrerequisiteConcept

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "canonical"
FEATURES = load_feature_config(ROOT / "configs" / "features.yaml")
ASSESSMENT = load_assessment_config(ROOT / "configs" / "assessment.yaml")
HASH = "sha256:" + "a" * 64


def _inputs():
    dataset = load_canonical_dataset(FIXTURE)
    profiles = build_book_profiles(assemble_book_evidence(dataset), FEATURES)
    return dataset, profiles


def _blueprint(topic: str = "operating-systems") -> AssessmentBlueprint:
    dataset, profiles = _inputs()
    return build_assessment_blueprint(
        topic,
        dataset,
        profiles,
        FEATURES,
        ASSESSMENT,
        canonical_file_hashes={name: HASH for name in ("books", "documents", "toc", "sources")},
        book_profiles_hash=HASH,
    )


def test_pool_merges_duplicate_concepts_by_role_and_keeps_single_book_concepts() -> None:
    _, profiles = _inputs()
    original = profiles[0]
    duplicate = original.model_copy(update={"book_id": "book_33333333333333333333"})

    pool = build_topic_concept_pool(
        "operating-systems", [duplicate, original], FEATURES, ASSESSMENT
    )
    reverse = build_topic_concept_pool(
        "operating-systems", [original, duplicate], FEATURES, ASSESSMENT
    )

    assert pool == reverse
    assert pool.covered_concept_count > 0
    assert pool.prerequisite_concept_count > 0
    process = next(item for item in pool.concepts if item.concept_id == "process")
    assert process.role == "covered"
    assert process.book_coverage_count == 2
    assert [support.book_id for support in process.book_support] == sorted(
        support.book_id for support in process.book_support
    )
    assert all("difficulty" not in type(item).model_fields for item in pool.concepts)
    single = build_topic_concept_pool("operating-systems", [original], FEATURES, ASSESSMENT)
    assert all(item.book_coverage_count == 1 for item in single.concepts)
    assert any(item.role == "prerequisite" for item in single.concepts)


def test_same_concept_name_can_keep_separate_covered_and_prerequisite_roles() -> None:
    _, profiles = _inputs()
    original = profiles[0]
    covered = next(
        item for item in original.concept_profile.covered_concepts if item.concept == "process"
    )
    inferred = PrerequisiteConcept(
        concept="process",
        weight=0.2,
        method="explicit",
        evidence=covered.evidence,
    )
    concept_profile = original.concept_profile.model_copy(
        update={
            "prerequisite_concepts": [
                *original.concept_profile.prerequisite_concepts,
                inferred,
            ]
        }
    )
    profile = original.model_copy(update={"concept_profile": concept_profile})
    prerequisite = FEATURES.config.prerequisite
    topics = dict(prerequisite.topics)
    topics["operating-systems"] = TopicLexicon(
        root={**topics["operating-systems"].root, "process": ["process"]}
    )
    features = FEATURES.model_copy(
        update={
            "config": FEATURES.config.model_copy(
                update={"prerequisite": prerequisite.model_copy(update={"topics": topics})}
            )
        }
    )

    pool = build_topic_concept_pool("operating-systems", [profile], features, ASSESSMENT)

    assert {
        (item.concept_id, item.role) for item in pool.concepts if item.concept_id == "process"
    } == {
        ("process", "covered"),
        ("process", "prerequisite"),
    }


def test_assessment_priority_is_configured_and_not_difficulty() -> None:
    _, profiles = _inputs()
    modified = ASSESSMENT.model_copy(
        update={
            "config": ASSESSMENT.config.model_copy(
                update={
                    "priority": AssessmentPriorityConfig(mean_book_weight=0, book_coverage_rate=1)
                }
            )
        }
    )
    first = build_topic_concept_pool("operating-systems", profiles, FEATURES, ASSESSMENT)
    second = build_topic_concept_pool("operating-systems", profiles, FEATURES, modified)
    assert any(
        a.assessment_priority != b.assessment_priority
        for a, b in zip(first.concepts, second.concepts, strict=True)
    )
    assert all(item.assessment_priority == 1 for item in second.concepts)


def test_self_assessment_validates_boolean_duplicates_and_topic_membership() -> None:
    pool = _blueprint().concept_pool
    valid = ConceptSelfAssessment.model_validate(
        {
            "assessment_id": "self-1",
            "topic_id": "operating-systems",
            "responses": [
                {"concept_id": "process", "topic_id": "operating-systems", "knows_concept": True},
                {"concept_id": "thread", "topic_id": "operating-systems", "knows_concept": False},
            ],
            "self_assessment_version": ASSESSMENT.config.self_assessment_version,
        }
    )
    validate_self_assessment(valid, pool)
    assert [item.knows_concept for item in valid.responses] == [True, False]
    with pytest.raises(ValidationError, match="duplicate concept_id"):
        ConceptSelfAssessment.model_validate(
            valid.model_dump()
            | {"responses": [valid.responses[0].model_dump(), valid.responses[0].model_dump()]}
        )
    with pytest.raises(ValidationError, match="topic_id"):
        ConceptSelfAssessment.model_validate(
            valid.model_dump()
            | {"responses": [valid.responses[0].model_dump() | {"topic_id": "linear-algebra"}]}
        )
    with pytest.raises(ValidationError):
        ConceptSelfAssessment.model_validate(
            valid.model_dump()
            | {"responses": [valid.responses[0].model_dump() | {"knows_concept": 1}]}
        )
    with pytest.raises(AssessmentBlueprintError, match="unknown"):
        validate_self_assessment(
            valid.model_copy(
                update={"responses": [valid.responses[0].model_copy(update={"concept_id": "fake"})]}
            ),
            pool,
        )
    with pytest.raises(ValidationError, match="identifiers"):
        ConceptSelfAssessment.model_validate(
            valid.model_dump()
            | {"responses": [valid.responses[0].model_dump() | {"concept_id": " process "}]}
        )


def test_mastery_schema_reserves_separate_evidence_without_fusion_formula() -> None:
    mastery = ConceptMastery(
        concept_id="process", topic_id="operating-systems", self_report_knows=True
    )
    assert mastery.quiz_score is None
    assert mastery.mastery is None
    assert mastery.confidence is None
    with pytest.raises(ValidationError):
        ConceptMastery(concept_id="process", topic_id="operating-systems", confidence=1.2)


def test_difficulty_rules_are_operation_first_and_concept_count_is_not_level() -> None:
    assert assign_target_difficulty("recognize", 0, ASSESSMENT)[:3] == (1, False, False)
    assert assign_target_difficulty("recall", 0, ASSESSMENT)[0] == 1
    assert assign_target_difficulty("compare", 1, ASSESSMENT)[:3] == (2, True, False)
    assert assign_target_difficulty("apply", 0, ASSESSMENT)[0] == 2
    assert assign_target_difficulty("integrate", 1, ASSESSMENT)[:3] == (3, True, True)
    assert assign_target_difficulty("infer", 0, ASSESSMENT)[0] == 3
    with pytest.raises(AssessmentBlueprintError, match="requires at least"):
        assign_target_difficulty("compare", 0, ASSESSMENT)
    with pytest.raises(AssessmentBlueprintError, match="Level 1"):
        assign_target_difficulty("recognize", 1, ASSESSMENT)


def test_blueprint_ids_metadata_and_evidence_are_deterministic() -> None:
    first = _blueprint()
    second = _blueprint()
    assert first == second
    assert len({item.question_id for item in first.question_specs}) == len(first.question_specs)
    assert first.config_hash == ASSESSMENT.content_hash
    assert first.concept_pool.feature_config_hash == FEATURES.content_hash
    assert all(item.config_hash == ASSESSMENT.content_hash for item in first.question_specs)
    assert all(item.difficulty_version == "question-difficulty-v1" for item in first.question_specs)
    assert all(item.difficulty_rationale for item in first.question_specs)
    assert all(item.prerequisite_depth is None for item in first.question_specs)
    assert all(item.abstraction_level is None for item in first.question_specs)
    assert all(
        item.source_text_complexity is None or item.question_type == "comprehension"
        for item in first.question_specs
    )
    assert all(
        item.question_type in {"vocabulary", "background_knowledge", "comprehension"}
        for item in first.question_specs
    )
    assert all(item.supporting_evidence for item in first.question_specs)
    assert "This synthetic preface" not in first.model_dump_json()


def test_question_spec_rejects_invalid_level_and_unanchored_comprehension() -> None:
    blueprint = _blueprint()
    comparison = next(
        item for item in blueprint.question_specs if item.cognitive_operation == "compare"
    )
    with pytest.raises(ValidationError, match="Level 1"):
        QuestionSpec.model_validate(comparison.model_dump() | {"target_difficulty": 1})
    comprehension = next(
        item for item in blueprint.question_specs if item.question_type == "comprehension"
    )
    with pytest.raises(ValidationError, match="matching prose references"):
        QuestionSpec.model_validate(
            comprehension.model_dump() | {"source_document_ids": ["missing-prose"]}
        )
    with pytest.raises(ValidationError):
        QuestionSpec.model_validate(comprehension.model_dump() | {"target_difficulty": 4})
    with pytest.raises(ValidationError, match="supporting evidence must match"):
        bad_evidence = comprehension.model_dump()["supporting_evidence"]
        bad_evidence[0]["book_id"] = "another-book"
        QuestionSpec.model_validate(
            comprehension.model_dump() | {"supporting_evidence": bad_evidence}
        )


def test_missing_prose_reports_shortage_and_never_creates_comprehension() -> None:
    blueprint = _blueprint("linear-algebra")
    assert blueprint.prose_grounded_comprehension_count == 0
    assert all(item.question_type != "comprehension" for item in blueprint.question_specs)
    assert any(item.question_type == "comprehension" for item in blueprint.shortages)


def test_config_validation_rejects_bad_rules(tmp_path: Path) -> None:
    content = (ROOT / "configs" / "assessment.yaml").read_text(encoding="utf-8")
    for replacement in (
        content.replace("mean_book_weight: 0.7", "mean_book_weight: 0.8"),
        content.replace("  compare: 2\n  relate: 2", "  compare: 3\n  relate: 2"),
        content.replace("  recognize: 1", "  recognize: 3", 1),
        content + "\nconfig_version: duplicate\n",
    ):
        path = tmp_path / "invalid.yaml"
        path.write_text(replacement, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_assessment_config(path)


def test_cli_writes_deterministic_artifact_and_rejects_stale_profiles(tmp_path: Path) -> None:
    _, profiles = _inputs()
    books = tmp_path / "profiles.jsonl"
    write_jsonl(profiles, books)
    outputs = [tmp_path / "first.json", tmp_path / "second.json"]
    args = [
        "build-assessment-blueprint",
        "--data-dir",
        str(FIXTURE),
        "--books",
        str(books),
        "--topic",
        "operating-systems",
        "--feature-config",
        str(ROOT / "configs" / "features.yaml"),
        "--assessment-config",
        str(ROOT / "configs" / "assessment.yaml"),
    ]
    for output in outputs:
        result = CliRunner().invoke(app, [*args, "--output", str(output)])
        assert result.exit_code == 0, result.output
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert payload["question_specs"]
    assert payload["canonical_file_hashes"]["books.jsonl"].startswith("sha256:")
    coverage = profiles[0].evidence_coverage.model_copy(
        update={"toc_entry_count": profiles[0].evidence_coverage.toc_entry_count + 1}
    )
    altered = profiles[0].model_copy(update={"evidence_coverage": coverage})
    write_jsonl([altered, profiles[1]], books)
    result = CliRunner().invoke(app, [*args, "--output", str(tmp_path / "invalid.json")])
    assert result.exit_code == 1
    assert "book profiles are stale" in result.output
