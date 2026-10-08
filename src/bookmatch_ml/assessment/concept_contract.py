"""Standalone concept assessment v2 contract; copied at repository boundaries."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Ability = Literal["meaning", "application", "reasoning"]
OPERATIONS = {"meaning": "recognize", "application": "apply", "reasoning": "infer"}


def content_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("*", mode="after")
    @classmethod
    def trimmed_text(cls, value):
        strings = [value] if isinstance(value, str) else value if isinstance(value, list) else []
        for item in strings:
            if isinstance(item, str) and (not item.strip() or item != item.strip()):
                raise ValueError("text must be nonblank and trimmed")
        return value


class TocReference(StrictModel):
    book_id: str
    toc_entry_id: str


class ConceptQuestionSpec(StrictModel):
    question_spec_version: Literal["concept-question-spec-v2"]
    question_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    primary_concept: str
    ability: Ability
    cognitive_operation: Literal["recognize", "apply", "infer"]
    measurement_context: Literal["prior-knowledge"]
    target_difficulty: Literal[1, 2, 3]
    assessment_objective: str
    misconception_targets: list[str] = Field(min_length=2)
    evidence_references: list[TocReference] = Field(min_length=1)
    config_version: str
    config_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    canonical_file_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_target(self):
        if self.cognitive_operation != OPERATIONS[self.ability]:
            raise ValueError("ability and cognitive operation disagree")
        if set(self.canonical_file_hashes) != {
            "books.jsonl",
            "documents.jsonl",
            "toc.jsonl",
            "sources.jsonl",
        }:
            raise ValueError("all four canonical hashes are required")
        import re

        if any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", h) for h in self.canonical_file_hashes.values()
        ):
            raise ValueError("invalid canonical hash")
        refs = [(r.book_id, r.toc_entry_id) for r in self.evidence_references]
        if len(set(refs)) != len(refs) or len(set(self.misconception_targets)) != len(
            self.misconception_targets
        ):
            raise ValueError("duplicate evidence or misconception")
        data = self.model_dump(exclude={"question_id"})
        if self.question_id != "q_" + content_hash(data).split(":")[1][:20]:
            raise ValueError("question ID does not match specification")
        return self


class ConceptBlueprint(StrictModel):
    blueprint_version: Literal["concept-assessment-blueprint-v2"]
    topic_id: str
    question_specs: list[ConceptQuestionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_specs(self):
        if any(s.topic_id != self.topic_id for s in self.question_specs):
            raise ValueError("blueprint topic mismatch")
        cells = [(s.primary_concept, s.ability) for s in self.question_specs]
        if len(set(cells)) != len(cells):
            raise ValueError("duplicate concept ability target")
        return self


class ConceptProviderQuestion(StrictModel):
    stem: str
    choices: list[str] = Field(min_length=4, max_length=4)
    correct_choice_index: int = Field(ge=0, le=3)
    explanation: str

    @model_validator(mode="after")
    def unique_choices(self):
        import unicodedata

        normalized = [
            " ".join(unicodedata.normalize("NFKC", c).casefold().split()) for c in self.choices
        ]
        if len(set(normalized)) != 4:
            raise ValueError("duplicate choices")
        return self


class ConceptGeneratedQuestion(ConceptProviderQuestion):
    generated_question_id: str = Field(pattern=r"^gq_[0-9a-f]{32}$")
    generated_question_version: Literal["generated-question-v5"]
    question_spec_id: str = Field(pattern=r"^q_[0-9a-f]{20}$")
    topic_id: str
    primary_concept: str
    ability: Ability
    cognitive_operation: Literal["recognize", "apply", "infer"]
    measurement_context: Literal["prior-knowledge"]
    target_difficulty: Literal[1, 2, 3]
    assessment_objective: str
    evidence_references: list[TocReference] = Field(min_length=1)
    question_spec_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    input_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    generation_model: str
    output_language: str = Field(pattern=r"^en(?:-[A-Za-z]{2,8})?$")
    prompt_version: Literal[
        "concept-question-generation-prompt-v1", "concept-question-generation-prompt-v2"
    ]
    generation_config_version: Literal["concept-question-generation-config-v1"]
    usage: dict[str, int | None]

    @model_validator(mode="after")
    def validate_identity(self):
        if self.cognitive_operation != OPERATIONS[self.ability]:
            raise ValueError("ability and operation mismatch")
        if set(self.usage) != {"input_tokens", "output_tokens", "total_tokens"} or any(
            v is not None and v < 0 for v in self.usage.values()
        ):
            raise ValueError("invalid token usage")
        data = self.model_dump(exclude={"generated_question_id", "usage"})
        if self.generated_question_id != "gq_" + content_hash(data).split(":")[1][:32]:
            raise ValueError("generated ID does not match content")
        return self
