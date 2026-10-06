import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bookmatch_ml.assessment.concept_contract import ConceptBlueprint
from bookmatch_ml.topic_preparation import TopicOutline, prepare_topic_content

ROOT = Path(__file__).parents[1]
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]


def outline_data():
    return {
        "outline_version": "topic-outline-v1",
        "topic_id": "synthetic-topic",
        "concepts": [
            {
                "id": name.lower(),
                "name": name,
                "aliases": [name, f"{name} technique"],
                "abilities": {
                    ability: {
                        "objective": f"Assess {name} with {ability} independently.",
                        "misconceptions": [
                            f"Confuse {name} with its inverse",
                            f"Overgeneralize {name}",
                        ],
                        "difficulty": difficulty,
                    }
                    for ability, difficulty in (
                        ("meaning", 1),
                        ("application", 2),
                        ("reasoning", 3),
                    )
                },
            }
            for name in NAMES
        ],
        "edges": [
            {
                "prerequisite": "alpha",
                "dependent": "bravo",
                "reason": "Synthetic candidate relationship",
            }
        ],
    }


def canonical(directory, toc=True, second_book=True):
    for path in (ROOT / "tests/fixtures/canonical").glob("*.jsonl"):
        (directory / path.name).write_bytes(path.read_bytes())
    books = [json.loads(line) for line in (directory / "books.jsonl").read_text().splitlines()]
    for book in books:
        book["topics"] = ["synthetic-topic"]
    (directory / "books.jsonl").write_text("".join(json.dumps(b) + "\n" for b in books))
    entries = (
        [
            {
                "toc_entry_id": f"toc-{i}-{name}",
                "book_id": book["book_id"],
                "parent_entry_id": None,
                "level": 1,
                "order_index": j,
                "label": None,
                "title": name,
                "source_id": "source_111" if i == 0 else "source_222",
            }
            for i, book in enumerate(books if second_book else books[:1])
            for j, name in enumerate(NAMES)
        ]
        if toc
        else []
    )
    (directory / "toc.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (directory / "documents.jsonl").write_text("")
    return directory


def test_grounded_blueprint_is_deterministic_and_never_claims_readiness(tmp_path):
    data = canonical(tmp_path)
    outline = TopicOutline.model_validate(outline_data())
    output = tmp_path / "result"
    report = prepare_topic_content(
        data, outline, output, ROOT / "configs", proposal_source="ai_proposed"
    )
    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert (
        prepare_topic_content(
            data, outline, output, ROOT / "configs", proposal_source="ai_proposed"
        )
        == report
    )
    assert {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    } == before
    assert report["conceptCount"] == 6 and report["mappedBookCount"] == 2
    assert report["questionSpecCount"] == 18 and report["status"] == "CONCEPTS_READY"
    assert report["diagnosisReady"] is False and report["generatedQuestionCount"] == 0
    assert report["humanReview"] == "unreviewed"
    blueprint = ConceptBlueprint.model_validate_json((output / "blueprint.json").read_text())
    assert all(len(spec.evidence_references) == 2 for spec in blueprint.question_specs)
    assert "ai_proposed" in (output / "configs/concept_graph.yaml").read_text()
    assert all("prose" not in p.name for p in output.iterdir())


@pytest.mark.parametrize("toc,second_book", [(False, True), (True, False)])
def test_missing_or_single_book_evidence_cannot_create_blueprint(tmp_path, toc, second_book):
    report = prepare_topic_content(
        canonical(tmp_path, toc, second_book),
        TopicOutline.model_validate(outline_data()),
        tmp_path / "result",
        ROOT / "configs",
    )
    assert report["status"] == "NEEDS_EVIDENCE"
    assert report["questionSpecCount"] == 0
    assert not (tmp_path / "result/blueprint.json").exists()


@pytest.mark.parametrize("kind", ["cycle", "unknown", "shared_alias", "duplicate_id"])
def test_invalid_outlines_fail_before_matching(kind):
    data = outline_data()
    if kind == "cycle":
        data["edges"].append(
            {
                "prerequisite": "bravo",
                "dependent": "alpha",
                "reason": "This creates a forbidden cycle",
            }
        )
    elif kind == "unknown":
        data["edges"][0]["dependent"] = "missing"
    elif kind == "shared_alias":
        data["concepts"][1]["aliases"][0] = " alpha "
    else:
        data["concepts"][1]["id"] = "alpha"
    with pytest.raises(ValidationError):
        TopicOutline.model_validate(data)


def test_unknown_topic_is_rejected_and_manual_outline_stays_a_proposal(tmp_path):
    data = canonical(tmp_path)
    outline = TopicOutline.model_validate(outline_data())
    report = prepare_topic_content(data, outline, tmp_path / "manual", ROOT / "configs")
    assert report["proposalSource"] == "proposed_seed" and report["humanReview"] == "unreviewed"
    wrong = outline.model_copy(update={"topic_id": "another-topic"})
    with pytest.raises(ValueError, match="no books"):
        prepare_topic_content(data, wrong, tmp_path / "wrong", ROOT / "configs")


def test_bilingual_preparation_reads_original_toc_without_fabricating_translation(tmp_path):
    data = canonical(tmp_path)
    names = ["계층 구조", "전송 제어", "주소 체계", "라우팅", "흐름 제어", "이름 해석"]
    outline = outline_data()
    for concept, name in zip(outline["concepts"], names, strict=True):
        concept["name"] = name
        concept["aliases"].append(name)
    entries = [json.loads(line) for line in (data / "toc.jsonl").read_text().splitlines()]
    for entry in entries:
        entry["title"] = names[NAMES.index(entry["title"])]
    (data / "toc.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)
    )
    before = (data / "toc.jsonl").read_bytes()
    output = tmp_path / "result"
    report = prepare_topic_content(
        data, TopicOutline.model_validate(outline), output, ROOT / "configs"
    )
    assert report["questionSpecCount"] == 18
    assert (data / "toc.jsonl").read_bytes() == before
    assert all("en_title" not in entry for entry in entries)
    profiles = [
        json.loads(line) for line in (output / "book-profiles.jsonl").read_text().splitlines()
    ]
    assert all(row["toc_title"] in names for profile in profiles for row in profile["toc_mappings"])
