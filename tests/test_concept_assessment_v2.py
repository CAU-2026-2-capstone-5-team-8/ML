import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bookmatch_ml.assessment.concept_blueprint import build_concept_blueprint
from bookmatch_ml.assessment.concept_contract import ConceptBlueprint
from bookmatch_ml.config import load_reader_config
from bookmatch_ml.reader.abilities import build_concept_ability_profile
from bookmatch_ml.schemas import Assessment, AssessmentResponse

ROOT = Path(__file__).parents[1]


def test_toc_only_blueprint_is_reproducible_and_covers_distinct_targets(tmp_path):
    source = ROOT / "tests/fixtures/canonical"
    for path in source.glob("*.jsonl"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    books = [json.loads(line) for line in (tmp_path / "books.jsonl").read_text().splitlines()]
    books[0]["topics"] = ["linear-algebra"]
    (tmp_path / "books.jsonl").write_text("\n".join(json.dumps(b) for b in books) + "\n")
    targets = ROOT / "configs/concept_assessment_targets.json"
    concepts = json.loads(targets.read_text())["concepts"]
    entries = [
        {
            "toc_entry_id": f"concept-{i}",
            "book_id": books[0]["book_id"],
            "parent_entry_id": None,
            "level": 1,
            "order_index": i,
            "label": str(i + 1),
            "title": c,
            "source_id": "source_111",
        }
        for i, c in enumerate(concepts)
    ]
    (tmp_path / "toc.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    (tmp_path / "documents.jsonl").write_text("")
    first = build_concept_blueprint(tmp_path, targets, ROOT / "configs")
    assert first == build_concept_blueprint(tmp_path, targets, ROOT / "configs")
    assert len(first.question_specs) == 18
    assert len({s.question_id for s in first.question_specs}) == 18
    assert {s.measurement_context for s in first.question_specs} == {"prior-knowledge"}
    data = first.model_dump()
    data["question_specs"][0]["assessment_objective"] = "Changed goal"
    with pytest.raises(ValidationError, match="ID does not match"):
        ConceptBlueprint.model_validate(data)
    (tmp_path / "toc.jsonl").write_text("")
    with pytest.raises(ValueError, match="no TOC evidence"):
        build_concept_blueprint(tmp_path, targets, ROOT / "configs")


def test_passage_and_unknown_context_do_not_prove_prior_knowledge():
    common = dict(
        topic_id="linear-algebra",
        concept_id="matrix",
        difficulty="easy",
        question_type="comprehension",
        cognitive_operation="apply",
        answer_mode="MULTIPLE_CHOICE",
    )
    assessment = Assessment(
        assessment_id="context",
        topic_id="linear-algebra",
        responses=[
            AssessmentResponse(
                **common, question_id="prior", correct=False, measurement_context="prior-knowledge"
            ),
            AssessmentResponse(
                **common,
                question_id="provided",
                correct=True,
                measurement_context="provided-information",
            ),
            AssessmentResponse(**common, question_id="old", correct=True),
        ],
    )
    profile = build_concept_ability_profile(
        assessment, load_reader_config(ROOT / "configs/reader.yaml")
    )
    assert profile["abilities"][0]["correctCount"] == 0
    assert profile["abilities"][0]["responseCount"] == 1
    assert profile["providedInformationAbilities"][0]["correctCount"] == 1
    assert profile["legacyContextAbilities"][0]["questionIds"] == ["old"]
