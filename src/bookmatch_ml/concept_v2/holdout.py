"""Fresh, prediction-blind holdouts for matcher generalization evaluation."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from bookmatch_ml.book.text import normalize_text
from bookmatch_ml.concept_v2.evidence_evaluation import (
    GENERIC_TOC_TITLES,
    SUPPORTED_EVIDENCE_TYPES,
    TOC_EVIDENCE_TYPES,
    EvidenceConceptGoldReview,
)
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph
from bookmatch_ml.concept_v2.matcher_experiment_evaluation import MatcherVariant, _predict_variant
from bookmatch_ml.concept_v2.matcher_experiments import LoadedMatcherV2ExperimentConfig
from bookmatch_ml.concept_v2.profile import (
    LoadedConceptMatchingConfig,
    match_concept_text,
)
from bookmatch_ml.config import ConfigError, LoadedFeatureConfig
from bookmatch_ml.data.book_evidence import EvidenceType, ImportedBookEvidence

MANIFEST_VERSION = "evidence-concept-holdout-manifest-v1"
PREDICTION_VERSION = "evidence-concept-holdout-predictions-v1"
REVIEW_VERSION = "evidence-concept-holdout-review-v1"
GENERAL_SAMPLE_VERSION = "evidence-concept-general-holdout-v1"
CHALLENGE_SAMPLE_VERSION = "evidence-concept-challenge-holdout-v1"
DEFAULT_GENERAL_SAMPLE_SIZES: dict[str, int] = {
    "toc_exact": 16,
    "toc_public_web_exact": 16,
    "toc_same_work": 6,
    "description": 11,
    "subject": 20,
    "metadata_minimal": 20,
}
DEFAULT_PER_BOOK_CAP = 8
DEFAULT_GENERIC_TOC_CAP = 2
CHALLENGE_CATEGORY_TARGETS: dict[str, int] = {
    "nested_overlap": 8,
    "independent_cooccurrence": 4,
    "morphological_lexical": 8,
    "modifier_insertion": 8,
    "virtual_machine_context": 4,
    "path_context_candidate": 12,
    "parent_overinheritance_counterexample": 8,
}
CHALLENGE_MIN_SIZE = 40
CHALLENGE_MAX_SIZE = 60
MATCHER_VARIANTS: tuple[MatcherVariant, ...] = (
    "v1",
    "v2_a_overlap",
    "v2_b_forms",
    "v2_c1_naive_path_union",
    "v2_c2_context_assisted",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HoldoutEvidenceRow(_Strict):
    topic: str
    book_id: str
    book_title: str
    evidence_id: str
    evidence_type: EvidenceType
    evidence_text: str
    toc_path: list[str] | None = None
    source_id: str
    provider: str
    edition_relation: str
    source_evidence_tier: str | None = None
    challenge_categories: list[str] = Field(default_factory=list)

    @field_validator("challenge_categories")
    @classmethod
    def categories_are_sorted_and_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("challenge categories must be sorted and unique")
        return value

    @model_validator(mode="after")
    def evidence_shape_is_consistent(self) -> "HoldoutEvidenceRow":
        if self.evidence_type in TOC_EVIDENCE_TYPES and not self.toc_path:
            raise ValueError("TOC holdout rows require a TOC path")
        if self.evidence_type not in TOC_EVIDENCE_TYPES and self.toc_path is not None:
            raise ValueError("metadata holdout rows cannot contain a TOC path")
        return self


class EvidenceConceptHoldoutManifest(_Strict):
    manifest_version: Literal["evidence-concept-holdout-manifest-v1"]
    sample_version: Literal[
        "evidence-concept-general-holdout-v1",
        "evidence-concept-challenge-holdout-v1",
    ]
    holdout_kind: Literal["general", "challenge"]
    book_evidence_contract_version: Literal["book-evidence-v1"]
    book_evidence_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    frozen_gold_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    excluded_evidence_ids_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    excluded_evidence_count: int = Field(ge=1)
    requested_sample_size_by_evidence_type: dict[str, int] | None = None
    challenge_category_targets: dict[str, int] | None = None
    per_book_cap: int = Field(gt=0)
    generic_toc_cap_by_evidence_type: int = Field(ge=0)
    entries: list[HoldoutEvidenceRow]

    @model_validator(mode="after")
    def manifest_is_consistent(self) -> "EvidenceConceptHoldoutManifest":
        ids = [row.evidence_id for row in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate holdout evidence ID")
        counts = Counter(row.book_id for row in self.entries)
        if counts and max(counts.values()) > self.per_book_cap:
            raise ValueError("holdout exceeds per-book sampling cap")
        if self.holdout_kind == "general":
            if self.requested_sample_size_by_evidence_type is None:
                raise ValueError("general holdout requires source-group sample sizes")
            if self.challenge_category_targets is not None:
                raise ValueError("general holdout cannot have challenge targets")
        else:
            if self.challenge_category_targets is None:
                raise ValueError("challenge holdout requires category targets")
            if self.requested_sample_size_by_evidence_type is not None:
                raise ValueError("challenge holdout cannot have source-group sample sizes")
            if any(not row.challenge_categories for row in self.entries):
                raise ValueError("challenge rows require at least one category")
        return self


class HoldoutVariantPrediction(_Strict):
    variant: MatcherVariant
    predicted_concept_ids: list[str]

    @field_validator("predicted_concept_ids")
    @classmethod
    def concepts_are_sorted_and_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("predicted concepts must be sorted and unique")
        return value


class HoldoutPredictionRow(_Strict):
    evidence_id: str
    predictions: list[HoldoutVariantPrediction]

    @model_validator(mode="after")
    def variants_are_unique(self) -> "HoldoutPredictionRow":
        variants = [prediction.variant for prediction in self.predictions]
        if variants != list(MATCHER_VARIANTS):
            raise ValueError("prediction variants are incomplete or out of order")
        return self


class EvidenceConceptHoldoutPredictions(_Strict):
    prediction_version: Literal["evidence-concept-holdout-predictions-v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    feature_config_hash: str
    graph_hash: str
    matching_v1_config_hash: str
    matcher_v2_experiment_config_hash: str
    entries: list[HoldoutPredictionRow]


class HoldoutReviewRow(HoldoutEvidenceRow):
    human_gold_concept_ids: list[str]
    review_status: Literal["unreviewed", "reviewed"]
    review_outcome: Literal["pending", "labeled", "no_concept", "not_judgable"]
    review_note: str | None = None

    @field_validator("human_gold_concept_ids")
    @classmethod
    def gold_is_sorted_and_unique(cls, value: list[str]) -> list[str]:
        if value != sorted(set(value)):
            raise ValueError("human gold concepts must be sorted and unique")
        return value

    @model_validator(mode="after")
    def review_fields_are_consistent(self) -> "HoldoutReviewRow":
        if self.review_status == "unreviewed":
            if self.review_outcome != "pending" or self.human_gold_concept_ids:
                raise ValueError("unreviewed rows must be pending without gold")
        elif self.review_outcome == "pending":
            raise ValueError("reviewed rows cannot remain pending")
        if self.review_outcome == "labeled" and not self.human_gold_concept_ids:
            raise ValueError("labeled rows require human gold")
        if self.review_outcome in {"no_concept", "not_judgable"} and self.human_gold_concept_ids:
            raise ValueError(f"{self.review_outcome} rows cannot contain human gold")
        if self.review_note is not None and not self.review_note.strip():
            raise ValueError("review note must be null or non-blank")
        return self


class EvidenceConceptHoldoutReview(_Strict):
    review_version: Literal["evidence-concept-holdout-review-v1"]
    manifest_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prediction_artifact_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prediction_visibility: Literal["hidden_during_human_review"]
    graph_hash: str
    graph_version: str
    entries: list[HoldoutReviewRow]


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _ids_hash(values: set[str]) -> str:
    payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _selection_key(version: str, row: HoldoutEvidenceRow) -> tuple[str, str]:
    payload = "\0".join(
        (version, row.evidence_type, row.topic, row.book_id, row.evidence_id)
    ).encode()
    return hashlib.sha256(payload).hexdigest(), row.evidence_id


def _is_generic_toc(row: HoldoutEvidenceRow) -> bool:
    if row.evidence_type not in TOC_EVIDENCE_TYPES:
        return False
    normalized = normalize_text(row.evidence_text)
    return normalized in GENERIC_TOC_TITLES or any(
        normalized.startswith(f"{title} ") for title in GENERIC_TOC_TITLES
    )


def _population(
    records: list[ImportedBookEvidence],
    graph: LoadedConceptGraph,
    excluded_ids: set[str],
) -> list[HoldoutEvidenceRow]:
    rows: list[HoldoutEvidenceRow] = []
    for record in sorted(records, key=lambda item: item.book.book_id):
        topics = sorted(set(record.book.topics) & set(graph.graph.nodes))
        for topic in topics:
            for item in sorted(record.evidence, key=lambda evidence: evidence.evidence_id):
                if item.evidence_type not in SUPPORTED_EVIDENCE_TYPES:
                    continue
                if item.evidence_id in excluded_ids:
                    continue
                rows.append(
                    HoldoutEvidenceRow(
                        topic=topic,
                        book_id=record.book.book_id,
                        book_title=record.book.title,
                        evidence_id=item.evidence_id,
                        evidence_type=item.evidence_type,
                        evidence_text=item.text,
                        toc_path=item.toc_path,
                        source_id=item.source_id,
                        provider=item.provider,
                        edition_relation=item.edition_relation,
                        source_evidence_tier=item.source_evidence_tier,
                    )
                )
    return rows


def _deduplicate(rows: list[HoldoutEvidenceRow], version: str) -> list[HoldoutEvidenceRow]:
    chosen: dict[tuple[str, str, str, str], HoldoutEvidenceRow] = {}
    for row in sorted(rows, key=lambda value: _selection_key(version, value)):
        key = (row.topic, row.book_id, row.evidence_type, normalize_text(row.evidence_text))
        chosen.setdefault(key, row)
    return list(chosen.values())


def _sample_group(
    rows: list[HoldoutEvidenceRow],
    size: int,
    version: str,
    book_counts: Counter[str],
    per_book_cap: int,
    generic_cap: int,
) -> list[HoldoutEvidenceRow]:
    by_topic_book: dict[str, dict[str, list[HoldoutEvidenceRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in _deduplicate(rows, version):
        by_topic_book[row.topic][row.book_id].append(row)
    for books in by_topic_book.values():
        for book_rows in books.values():
            book_rows.sort(key=lambda value: _selection_key(version, value), reverse=True)
    selected: list[HoldoutEvidenceRow] = []
    generic_count = 0
    while len(selected) < size:
        progressed = False
        for topic in sorted(by_topic_book):
            books = by_topic_book[topic]
            for book_id in sorted(
                books,
                key=lambda value: (
                    book_counts[value],
                    hashlib.sha256(f"{version}\0{topic}\0{value}".encode()).hexdigest(),
                    value,
                ),
            ):
                if book_counts[book_id] >= per_book_cap:
                    continue
                while books[book_id]:
                    candidate = books[book_id].pop()
                    if _is_generic_toc(candidate) and generic_count >= generic_cap:
                        continue
                    selected.append(candidate)
                    book_counts[book_id] += 1
                    generic_count += _is_generic_toc(candidate)
                    progressed = True
                    break
                if progressed:
                    break
            if len(selected) >= size:
                break
        if not progressed:
            break
    return selected


_NESTED_PHRASES = (
    "vector space",
    "linear transformation",
    "linear independence",
    "memory management",
    "virtual memory",
    "file system",
    "distributed systems",
)
_CONCEPT_PHRASES = (
    "computer architecture",
    "programming",
    "process",
    "thread",
    "scheduling",
    "concurrency",
    "synchronization",
    "deadlock",
    "memory management",
    "virtual memory",
    "storage",
    "file system",
    "input output",
    "protection",
    "security",
    "virtualization",
    "distributed systems",
    "systems of equations",
    "linear system",
    "gaussian elimination",
    "matrix",
    "vector",
    "vector space",
    "linear independence",
    "basis",
    "dimension",
    "rank",
    "determinant",
    "eigenvalue",
    "eigenvector",
    "orthogonality",
    "inner product",
    "linear transformation",
    "diagonalization",
    "least squares",
    "singular value decomposition",
)


def _present_phrases(value: str) -> set[str]:
    normalized = normalize_text(value)
    return {
        phrase
        for phrase in _CONCEPT_PHRASES
        if re.search(rf"(?<!\w){re.escape(phrase)}s?(?!\w)", normalized)
    }


def _challenge_categories(row: HoldoutEvidenceRow) -> list[str]:
    text = normalize_text(row.evidence_text)
    path = [normalize_text(value) for value in (row.toc_path or [])]
    parents = path[:-1]
    categories: set[str] = set()
    if any(re.search(rf"(?<!\w){re.escape(phrase)}s?(?!\w)", text) for phrase in _NESTED_PHRASES):
        categories.add("nested_overlap")
    stripped = text
    for phrase in _NESTED_PHRASES:
        stripped = re.sub(rf"(?<!\w){re.escape(phrase)}s?(?!\w)", " ", stripped)
    if re.search(r"(?<!\w)vector(?!\w)", stripped) and re.search(
        r"(?<!\w)vector spaces?(?!\w)", text
    ):
        categories.add("independent_cooccurrence")
    if re.search(
        r"(?<!\w)(?:multithread\w*|threads?|programming exercises?|linear maps?)(?!\w)",
        text,
    ):
        categories.add("morphological_lexical")
    if re.search(r"(?<!\w)distributed\s+\w+\s+systems?(?!\w)", text) or re.search(
        r"(?<!\w)systems?\s+of\s+(?:\w+\s+){1,4}(?:equations?|variables?)(?!\w)",
        text,
    ):
        categories.add("modifier_insertion")
    if re.search(r"(?<!\w)virtual machines?(?!\w)", text):
        categories.add("virtual_machine_context")
    leaf_phrases = _present_phrases(text)
    parent_phrases = (
        set().union(*(_present_phrases(parent) for parent in parents)) if parents else set()
    )
    if row.evidence_type in TOC_EVIDENCE_TYPES and parent_phrases - leaf_phrases:
        categories.add("path_context_candidate")
        if leaf_phrases or _is_generic_toc(row):
            categories.add("parent_overinheritance_counterexample")
    return sorted(categories)


def build_holdout_manifests(
    records: list[ImportedBookEvidence],
    graph: LoadedConceptGraph,
    frozen_review: EvidenceConceptGoldReview,
    book_evidence_hash: str,
    frozen_gold_hash: str,
    general_sizes: dict[str, int] | None = None,
    per_book_cap: int = DEFAULT_PER_BOOK_CAP,
    generic_toc_cap: int = DEFAULT_GENERIC_TOC_CAP,
) -> tuple[EvidenceConceptHoldoutManifest, EvidenceConceptHoldoutManifest]:
    """Select prediction-blind general and challenge memberships."""

    excluded_ids = {row.evidence_id for row in frozen_review.entries}
    population = _population(records, graph, excluded_ids)
    sizes = dict(general_sizes or DEFAULT_GENERAL_SAMPLE_SIZES)
    by_type: dict[str, list[HoldoutEvidenceRow]] = defaultdict(list)
    for row in population:
        by_type[row.evidence_type].append(row)
    general_counts: Counter[str] = Counter()
    general_rows: list[HoldoutEvidenceRow] = []
    for evidence_type, size in sizes.items():
        general_rows.extend(
            _sample_group(
                by_type[evidence_type],
                size,
                GENERAL_SAMPLE_VERSION,
                general_counts,
                per_book_cap,
                generic_toc_cap,
            )
        )
    general_rows.sort(key=lambda row: (row.topic, row.book_id, row.evidence_type, row.evidence_id))
    general_ids = {row.evidence_id for row in general_rows}

    categorized: list[HoldoutEvidenceRow] = []
    for row in _deduplicate(
        [candidate for candidate in population if candidate.evidence_id not in general_ids],
        CHALLENGE_SAMPLE_VERSION,
    ):
        categories = _challenge_categories(row)
        if categories:
            categorized.append(row.model_copy(update={"challenge_categories": categories}))
    selected: dict[str, HoldoutEvidenceRow] = {}
    challenge_counts: Counter[str] = Counter()
    for category, target in CHALLENGE_CATEGORY_TARGETS.items():
        while sum(category in row.challenge_categories for row in selected.values()) < target:
            candidates = [
                row
                for row in categorized
                if category in row.challenge_categories
                and row.evidence_id not in selected
                and challenge_counts[row.book_id] < per_book_cap
            ]
            if not candidates:
                break
            candidate = min(
                candidates,
                key=lambda row: (
                    challenge_counts[row.book_id],
                    _selection_key(f"{CHALLENGE_SAMPLE_VERSION}:{category}", row),
                ),
            )
            selected[candidate.evidence_id] = candidate
            challenge_counts[candidate.book_id] += 1
            if len(selected) >= CHALLENGE_MAX_SIZE:
                break
        if len(selected) >= CHALLENGE_MAX_SIZE:
            break
    if len(selected) < CHALLENGE_MIN_SIZE:
        for candidate in sorted(
            categorized,
            key=lambda row: (
                challenge_counts[row.book_id],
                _selection_key(CHALLENGE_SAMPLE_VERSION, row),
            ),
        ):
            if (
                candidate.evidence_id in selected
                or challenge_counts[candidate.book_id] >= per_book_cap
            ):
                continue
            selected[candidate.evidence_id] = candidate
            challenge_counts[candidate.book_id] += 1
            if len(selected) >= CHALLENGE_MIN_SIZE:
                break
    challenge_rows = sorted(
        selected.values(),
        key=lambda row: (row.topic, row.book_id, row.evidence_type, row.evidence_id),
    )
    common = {
        "manifest_version": MANIFEST_VERSION,
        "book_evidence_contract_version": "book-evidence-v1",
        "book_evidence_hash": book_evidence_hash,
        "frozen_gold_hash": frozen_gold_hash,
        "excluded_evidence_ids_hash": _ids_hash(excluded_ids),
        "excluded_evidence_count": len(excluded_ids),
        "per_book_cap": per_book_cap,
        "generic_toc_cap_by_evidence_type": generic_toc_cap,
    }
    return (
        EvidenceConceptHoldoutManifest(
            **common,
            sample_version=GENERAL_SAMPLE_VERSION,
            holdout_kind="general",
            requested_sample_size_by_evidence_type=sizes,
            entries=general_rows,
        ),
        EvidenceConceptHoldoutManifest(
            **common,
            sample_version=CHALLENGE_SAMPLE_VERSION,
            holdout_kind="challenge",
            challenge_category_targets=CHALLENGE_CATEGORY_TARGETS,
            entries=challenge_rows,
        ),
    )


def load_holdout_manifest(path: Path) -> EvidenceConceptHoldoutManifest:
    try:
        return EvidenceConceptHoldoutManifest.model_validate_json(path.read_bytes())
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid evidence concept holdout manifest: {path}: {exc}") from exc


def validate_manifest_against_source(
    manifest: EvidenceConceptHoldoutManifest,
    records: list[ImportedBookEvidence],
    graph: LoadedConceptGraph,
    excluded_ids: set[str],
) -> None:
    population = {row.evidence_id: row for row in _population(records, graph, excluded_ids)}
    for row in manifest.entries:
        source = population.get(row.evidence_id)
        if source is None:
            raise ValueError(f"holdout evidence is absent or excluded: {row.evidence_id}")
        if source.model_dump(exclude={"challenge_categories"}) != row.model_dump(
            exclude={"challenge_categories"}
        ):
            raise ValueError(f"holdout evidence changed: {row.evidence_id}")


def build_holdout_predictions(
    manifest: EvidenceConceptHoldoutManifest,
    manifest_hash: str,
    features: LoadedFeatureConfig,
    graph: LoadedConceptGraph,
    matching: LoadedConceptMatchingConfig,
    experiment: LoadedMatcherV2ExperimentConfig,
) -> EvidenceConceptHoldoutPredictions:
    entries: list[HoldoutPredictionRow] = []
    for row in manifest.entries:
        v1_matches, _ = match_concept_text(row.evidence_text, row.topic, features, graph, matching)
        proxy = row.model_copy(
            update={"predicted_concept_ids": sorted(m.concept_id for m in v1_matches)}
        )
        predictions = [
            HoldoutVariantPrediction(
                variant=variant,
                predicted_concept_ids=_predict_variant(
                    variant, proxy, features, graph, matching, experiment
                ),
            )
            for variant in MATCHER_VARIANTS
        ]
        entries.append(HoldoutPredictionRow(evidence_id=row.evidence_id, predictions=predictions))
    return EvidenceConceptHoldoutPredictions(
        prediction_version=PREDICTION_VERSION,
        manifest_hash=manifest_hash,
        feature_config_hash=features.content_hash,
        graph_hash=graph.content_hash,
        matching_v1_config_hash=matching.content_hash,
        matcher_v2_experiment_config_hash=experiment.content_hash,
        entries=entries,
    )


def build_holdout_review(
    manifest: EvidenceConceptHoldoutManifest,
    manifest_hash: str,
    prediction_artifact_hash: str,
    graph: LoadedConceptGraph,
) -> EvidenceConceptHoldoutReview:
    return EvidenceConceptHoldoutReview(
        review_version=REVIEW_VERSION,
        manifest_hash=manifest_hash,
        prediction_artifact_hash=prediction_artifact_hash,
        prediction_visibility="hidden_during_human_review",
        graph_hash=graph.content_hash,
        graph_version=graph.graph.graph_version,
        entries=[
            HoldoutReviewRow(
                **row.model_dump(),
                human_gold_concept_ids=[],
                review_status="unreviewed",
                review_outcome="pending",
            )
            for row in manifest.entries
        ],
    )


def load_holdout_review(
    path: Path,
    manifest: EvidenceConceptHoldoutManifest,
    manifest_hash: str,
    prediction_hash: str,
    graph: LoadedConceptGraph,
) -> EvidenceConceptHoldoutReview:
    try:
        review = EvidenceConceptHoldoutReview.model_validate_json(path.read_bytes())
        if (
            review.manifest_hash != manifest_hash
            or review.prediction_artifact_hash != prediction_hash
        ):
            raise ValueError("holdout review provenance differs from fixed artifacts")
        expected = {row.evidence_id: row for row in manifest.entries}
        actual = {row.evidence_id: row for row in review.entries}
        if set(expected) != set(actual):
            raise ValueError("holdout review membership differs from manifest")
        editable = {"human_gold_concept_ids", "review_status", "review_outcome", "review_note"}
        for evidence_id, row in actual.items():
            if row.model_dump(exclude=editable) != expected[evidence_id].model_dump():
                raise ValueError(f"holdout review evidence changed: {evidence_id}")
            unknown = set(row.human_gold_concept_ids) - set(graph.graph.nodes[row.topic])
            if unknown:
                raise ValueError(f"unknown holdout gold concepts: {sorted(unknown)}")
    except (OSError, ValidationError, ValueError) as exc:
        raise ConfigError(f"invalid evidence concept holdout review: {path}: {exc}") from exc
    return review


def update_holdout_review_entry(
    review: EvidenceConceptHoldoutReview,
    graph: LoadedConceptGraph,
    evidence_id: str,
    outcome: Literal["labeled", "no_concept", "not_judgable"],
    gold_ids: list[str],
    note: str | None,
) -> EvidenceConceptHoldoutReview:
    matches = [index for index, row in enumerate(review.entries) if row.evidence_id == evidence_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one holdout row for {evidence_id}")
    index = matches[0]
    row = review.entries[index]
    normalized_gold = sorted(set(gold_ids))
    unknown = set(normalized_gold) - set(graph.graph.nodes[row.topic])
    if unknown:
        raise ValueError(f"unknown {row.topic} concepts: {sorted(unknown)}")
    entries = list(review.entries)
    entries[index] = HoldoutReviewRow.model_validate(
        {
            **row.model_dump(),
            "human_gold_concept_ids": normalized_gold,
            "review_status": "reviewed",
            "review_outcome": outcome,
            "review_note": note,
        }
    )
    return review.model_copy(update={"entries": entries})


def review_summary(review: EvidenceConceptHoldoutReview) -> dict[str, int]:
    reviewed = sum(row.review_status == "reviewed" for row in review.entries)
    return {
        "total": len(review.entries),
        "reviewed": reviewed,
        "remaining": len(review.entries) - reviewed,
    }


def review_packet(
    review: EvidenceConceptHoldoutReview,
    graph: LoadedConceptGraph,
    limit: int,
) -> dict[str, object]:
    rows = [row for row in review.entries if row.review_status == "unreviewed"][:limit]
    return {
        "prediction_visibility": review.prediction_visibility,
        "entries": [
            {
                "number": index,
                "evidence_id": row.evidence_id,
                "book_title": row.book_title,
                "topic": row.topic,
                "evidence_type": row.evidence_type,
                "evidence_text": row.evidence_text,
                "toc_path": row.toc_path,
                "provider": row.provider,
                "edition_relation": row.edition_relation,
                "challenge_categories": row.challenge_categories,
                "canonical_concept_ids": sorted(graph.graph.nodes[row.topic]),
            }
            for index, row in enumerate(rows, start=1)
        ],
    }
