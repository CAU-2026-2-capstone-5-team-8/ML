"""Command-line entry points for reproducible batch workflows."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import ValidationError

from bookmatch_ml.assessment.blueprint import build_assessment_blueprint
from bookmatch_ml.assessment.pool import AssessmentBlueprintError
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.concept_v2.book_evidence_mapping import (
    build_book_evidence_concept_mapping_report,
)
from bookmatch_ml.concept_v2.evidence_evaluation import (
    EvidenceConceptGoldReview,
    build_evidence_concept_gold_review,
    evaluate_evidence_concept_gold,
    load_evidence_concept_gold_review,
    review_summary,
    update_review_entry,
)
from bookmatch_ml.concept_v2.gold_evaluation import (
    DEFAULT_SAMPLE_SIZES,
    build_toc_concept_gold_review,
    evaluate_toc_concept_gold,
    load_toc_concept_gold_review,
)
from bookmatch_ml.concept_v2.graph import LoadedConceptGraph, load_concept_graph
from bookmatch_ml.concept_v2.holdout import (
    EvidenceConceptHoldoutPredictions,
    EvidenceConceptHoldoutReview,
    build_holdout_manifests,
    build_holdout_predictions,
    build_holdout_review,
    load_holdout_manifest,
    load_holdout_review,
    review_packet,
    update_holdout_review_entry,
    validate_manifest_against_source,
)
from bookmatch_ml.concept_v2.holdout import (
    review_summary as holdout_review_summary,
)
from bookmatch_ml.concept_v2.holdout import (
    sha256_file as holdout_sha256_file,
)
from bookmatch_ml.concept_v2.holdout_evaluation import (
    build_holdout_evaluation_report,
    load_holdout_predictions,
)
from bookmatch_ml.concept_v2.matcher_experiment_evaluation import (
    build_matcher_v2_experiment_report,
)
from bookmatch_ml.concept_v2.matcher_experiments import load_matcher_v2_experiment_config
from bookmatch_ml.concept_v2.matching import (
    ConceptMatchingReport,
    match_book_concepts,
    order_matching_items,
)
from bookmatch_ml.concept_v2.profile import (
    build_book_concept_profile_v2,
    load_concept_matching_config,
)
from bookmatch_ml.concept_v2.toc import TocTreeError
from bookmatch_ml.concept_v2.validation import (
    build_concept_validation_report,
    build_review_template,
    load_concept_graph_reviews,
)
from bookmatch_ml.config import (
    ConfigError,
    load_assessment_config,
    load_evaluation_config,
    load_feature_config,
    load_ranking_config,
    load_ranking_policy_config,
    load_reader_config,
)
from bookmatch_ml.data.book_evidence import (
    BookEvidenceImportError,
    load_book_evidence,
    summarize_book_evidence,
)
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import CanonicalDataError, load_canonical_dataset
from bookmatch_ml.evaluation.ablation import build_evidence_ablation_report
from bookmatch_ml.evaluation.difficulty import EvaluationDataError, load_difficulty_judgments
from bookmatch_ml.evaluation.ranking_policies import evaluate_ranking_policies
from bookmatch_ml.evaluation.report import build_evaluation_report
from bookmatch_ml.io import write_json, write_jsonl
from bookmatch_ml.ranking.concept_readiness_experiments import (
    ConceptReadinessExperimentError,
    evaluate_concept_readiness_ranking,
)
from bookmatch_ml.ranking.loader import (
    RankingInputError,
    load_book_profiles,
    load_reader_profile,
)
from bookmatch_ml.ranking.matching import RankingError, rank_books, score_book_fit
from bookmatch_ml.ranking.source_aware_adapter import (
    SourceAwareAdapterError,
    build_source_aware_adapter_report,
    build_source_aware_matching_book_profiles,
    load_concept_mapping_report,
)
from bookmatch_ml.reader.profile import AssessmentError, build_reader_profile, load_assessment
from bookmatch_ml.schemas import BookCoverageSummary, CoverageReport, RankingResponse

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
DEFAULT_FEATURE_CONFIG = Path("configs/features.yaml")
DEFAULT_READER_CONFIG = Path("configs/reader.yaml")
DEFAULT_RANKING_CONFIG = Path("configs/ranking.yaml")
DEFAULT_EVALUATION_CONFIG = Path("configs/evaluation.yaml")
DEFAULT_RANKING_POLICY_CONFIG = Path("configs/ranking_policies.yaml")
DEFAULT_ASSESSMENT_CONFIG = Path("configs/assessment.yaml")
DEFAULT_CONCEPT_GRAPH_CONFIG = Path("configs/concept_graph.yaml")
DEFAULT_CONCEPT_MATCHING_CONFIG = Path("configs/concept_matching.yaml")
DEFAULT_CONCEPT_MATCHING_V2_CONFIG = Path("configs/concept_matching_v2.yaml")
DEFAULT_CONCEPT_GRAPH_REVIEWS_CONFIG = Path("configs/concept_graph_reviews.yaml")
DEFAULT_MATCHER_V2_EXPERIMENT_CONFIG = Path("configs/matcher_v2_experiments.yaml")


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


@app.callback()
def main() -> None:
    """Inspect and transform canonical book evidence."""


@app.command("inspect-data")
def inspect_data(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=False,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Directory containing the four canonical JSONL files.",
        ),
    ],
) -> None:
    """Validate canonical input and print deterministic evidence coverage JSON."""

    try:
        dataset = load_canonical_dataset(data_dir)
        evidence = assemble_book_evidence(dataset)
    except CanonicalDataError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report = CoverageReport(
        book_count=len(dataset.books),
        document_count=len(dataset.documents),
        toc_entry_count=len(dataset.toc),
        source_count=len(dataset.sources),
        books=[
            BookCoverageSummary(
                book_id=item.book_id,
                title=item.metadata.title,
                coverage=item.coverage,
            )
            for item in evidence
        ],
    )
    typer.echo(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
    )


@app.command("inspect-book-evidence")
def inspect_book_evidence(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=False,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Data-Pipeline book-evidence-v1 JSONL artifact.",
        ),
    ],
) -> None:
    """Validate source-aware evidence and print availability counts."""

    try:
        records = load_book_evidence(input_path)
    except BookEvidenceImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            summarize_book_evidence(records),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("map-book-evidence-concepts")
def map_book_evidence_concepts_command(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_V2_CONFIG,
) -> None:
    """Map book-evidence-v1 rows to deduplicated production concept presence."""

    try:
        input_hash = _sha256_file(input_path)
        records = load_book_evidence(input_path)
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        report = build_book_evidence_concept_mapping_report(
            records,
            input_hash,
            features,
            graph,
            matching,
        )
        if _sha256_file(input_path) != input_hash:
            raise ValueError("book evidence changed during concept mapping")
        write_json(report, output)
    except (BookEvidenceImportError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "total_books": report.total_books,
                "books_with_matched_concepts": report.books_with_matched_concepts,
                "books_without_matched_concepts": report.books_without_matched_concepts,
                "unique_book_concept_presence_count": report.unique_book_concept_presence_count,
                "raw_match_occurrence_count": report.raw_match_occurrence_count,
                "matcher_version": report.matcher_version,
            },
            sort_keys=True,
        )
    )


@app.command("build-matching-book-candidates")
def build_matching_book_candidates_command(
    concept_mapping: Annotated[
        Path, typer.Option("--concept-mapping", exists=True, dir_okay=False, resolve_path=True)
    ],
    book_profiles: Annotated[
        Path, typer.Option("--book-profiles", exists=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    report_output: Annotated[Path, typer.Option("--report", dir_okay=False, resolve_path=True)],
) -> None:
    """Build API-compatible ranking candidates from source-aware concept presence."""

    paths = (concept_mapping, book_profiles, output, report_output)
    if len(set(paths)) != len(paths):
        typer.echo("Error: adapter input and output paths must be distinct", err=True)
        raise typer.Exit(code=1)
    try:
        mapping_hash = _sha256_file(concept_mapping)
        profiles_hash = _sha256_file(book_profiles)
        mapping = load_concept_mapping_report(concept_mapping)
        profiles = load_book_profiles(book_profiles)
        candidates = build_source_aware_matching_book_profiles(mapping, profiles)
        if (
            _sha256_file(concept_mapping) != mapping_hash
            or _sha256_file(book_profiles) != profiles_hash
        ):
            raise ValueError("adapter input changed during candidate generation")
        write_jsonl(candidates, output)
        candidate_hash = _sha256_file(output)
        report = build_source_aware_adapter_report(
            mapping,
            profiles,
            candidates,
            mapping_hash,
            profiles_hash,
            candidate_hash,
        )
        write_json(report, report_output)
    except (SourceAwareAdapterError, RankingInputError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "adapter_version": report.adapter_version,
                "candidate_count": report.joined_book_count,
                "with_covered_concepts": report.candidates_with_covered_concepts,
                "with_prerequisite_concepts": report.candidates_with_prerequisite_concepts,
                "output": str(output),
                "report": str(report_output),
            },
            sort_keys=True,
        )
    )


@app.command("evaluate-concept-readiness-ranking")
def evaluate_concept_readiness_ranking_command(
    concept_mapping: Annotated[
        Path, typer.Option("--concept-mapping", exists=True, dir_okay=False, resolve_path=True)
    ],
    readers: Annotated[
        list[Path] | None,
        typer.Option("--reader", exists=True, dir_okay=False, resolve_path=True),
    ] = None,
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)] = Path(
        "data/reports/concept-readiness-ranking-v1.json"
    ),
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    review_config: Annotated[
        Path, typer.Option("--review-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_REVIEWS_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_V2_CONFIG,
) -> None:
    """Compare offline concept-readiness variants without changing production rank-v1."""

    if not readers:
        typer.echo("Error: at least one --reader is required", err=True)
        raise typer.Exit(code=1)
    input_paths = {
        "concept_mapping": concept_mapping,
        "feature_config": feature_config,
        "graph_config": graph_config,
        "review_config": review_config,
        "matching_config": matching_config,
        **{f"reader:{index}": path for index, path in enumerate(readers)},
    }
    try:
        if output.resolve() in {path.resolve() for path in input_paths.values()}:
            raise ConceptReadinessExperimentError("--output must not overwrite an experiment input")
        input_hashes = {name: _sha256_file(path) for name, path in input_paths.items()}
        mapping = load_concept_mapping_report(concept_mapping)
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        reviews = load_concept_graph_reviews(review_config, graph)
        matching = load_concept_matching_config(matching_config)
        reader_profiles = [load_reader_profile(path) for path in readers]
        if len({reader.topic_id for reader in reader_profiles}) != len(reader_profiles):
            raise ConceptReadinessExperimentError("duplicate reader topic")
        reader_hashes = {
            reader.topic_id: _sha256_file(path)
            for reader, path in zip(reader_profiles, readers, strict=True)
        }
        report = evaluate_concept_readiness_ranking(
            mapping,
            reader_profiles,
            graph,
            reviews,
            matching,
            mapping_report_hash=_sha256_file(concept_mapping),
            reader_profile_hashes=reader_hashes,
            input_hashes=input_hashes,
        )
        if input_hashes != {name: _sha256_file(path) for name, path in input_paths.items()}:
            raise ConceptReadinessExperimentError("experiment input changed during evaluation")
        write_json(report, output)
    except (
        ConceptReadinessExperimentError,
        ConfigError,
        RankingInputError,
        SourceAwareAdapterError,
        ValueError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "accepted_edge_count": len(report.accepted_graph.edges),
                "experiment_version": report.experiment_version,
                "output": str(output),
                "topics": [topic.topic_id for topic in report.topics],
            },
            sort_keys=True,
        )
    )


def _build_expected_evidence_review(
    input_path: Path,
    feature_config: Path,
    graph_config: Path,
    matching_config: Path,
) -> tuple[EvidenceConceptGoldReview, LoadedConceptGraph]:
    records = load_book_evidence(input_path)
    features = load_feature_config(feature_config)
    graph = load_concept_graph(graph_config, features)
    matching = load_concept_matching_config(matching_config)
    return (
        build_evidence_concept_gold_review(
            records=records,
            graph=graph,
            features=features,
            matching=matching,
            book_evidence_hash=_sha256_file(input_path),
        ),
        graph,
    )


@app.command("build-evidence-concept-gold-review")
def build_evidence_concept_gold_review_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
) -> None:
    """Build a deterministic source-stratified evidence review template."""

    try:
        input_hash = _sha256_file(input_path)
        review, graph = _build_expected_evidence_review(
            input_path, feature_config, graph_config, matching_config
        )
        if output.exists():
            existing = load_evidence_concept_gold_review(output, review, graph).review
            if existing != review:
                raise ValueError("refusing to overwrite a changed evidence concept review")
        if _sha256_file(input_path) != input_hash:
            raise ValueError("book evidence changed during review generation")
        write_json(review, output)
    except (BookEvidenceImportError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"output": str(output), **review_summary(review)}, sort_keys=True))


@app.command("review-evidence-concepts")
def review_evidence_concepts_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    evidence_id: Annotated[str, typer.Option("--evidence-id")],
    outcome: Annotated[Literal["labeled", "no_concept", "not_judgable"], typer.Option("--outcome")],
    gold: Annotated[
        str,
        typer.Option(
            "--gold",
            help="Comma-separated canonical concept IDs; required only for labeled outcomes.",
        ),
    ] = "",
    note: Annotated[str | None, typer.Option("--note")] = None,
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
) -> None:
    """Apply one explicit human decision to a pinned review row."""

    try:
        expected, graph = _build_expected_evidence_review(
            input_path, feature_config, graph_config, matching_config
        )
        loaded = load_evidence_concept_gold_review(review_file, expected, graph)
        gold_ids = [value.strip() for value in gold.split(",") if value.strip()]
        updated = update_review_entry(loaded.review, graph, evidence_id, outcome, gold_ids, note)
        if _sha256_file(review_file) != loaded.content_hash:
            raise ValueError("review file changed while applying the human decision")
        write_json(updated, review_file)
    except (BookEvidenceImportError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"review": str(review_file), **review_summary(updated)}, sort_keys=True))


@app.command("evaluate-evidence-concept-gold")
def evaluate_evidence_concept_gold_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
) -> None:
    """Report review progress and metrics for human-reviewed evidence rows only."""

    try:
        expected, graph = _build_expected_evidence_review(
            input_path, feature_config, graph_config, matching_config
        )
        loaded = load_evidence_concept_gold_review(review_file, expected, graph)
        report = evaluate_evidence_concept_gold(loaded)
        if _sha256_file(review_file) != loaded.content_hash:
            raise ValueError("review file changed during evaluation")
        write_json(report, output)
    except (BookEvidenceImportError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "metric_status": report.metric_status,
                "reviewed": report.reviewed_entry_count,
                "remaining": report.remaining_entry_count,
            },
            sort_keys=True,
        )
    )


@app.command("evaluate-matcher-v2-experiments")
def evaluate_matcher_v2_experiments_command(
    input_path: Annotated[
        Path,
        typer.Option("--input", exists=True, dir_okay=False, resolve_path=True),
    ],
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
    experiment_config: Annotated[
        Path, typer.Option("--experiment-config", exists=True, dir_okay=False)
    ] = DEFAULT_MATCHER_V2_EXPERIMENT_CONFIG,
) -> None:
    """Evaluate isolated matcher variants against the frozen human gold."""

    try:
        expected, graph = _build_expected_evidence_review(
            input_path, feature_config, graph_config, matching_config
        )
        loaded = load_evidence_concept_gold_review(review_file, expected, graph)
        features = load_feature_config(feature_config)
        matching = load_concept_matching_config(matching_config)
        experiment = load_matcher_v2_experiment_config(experiment_config)
        report = build_matcher_v2_experiment_report(
            loaded,
            features,
            graph,
            matching,
            experiment,
        )
        if _sha256_file(review_file) != loaded.content_hash:
            raise ValueError("frozen review changed during matcher-v2 evaluation")
        write_json(report, output)
    except (BookEvidenceImportError, ConfigError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "evaluable": report.evaluable_entry_count,
                "variants": [variant.variant for variant in report.variants],
            },
            sort_keys=True,
        )
    )


@app.command("build-evidence-concept-holdout-manifests")
def build_evidence_concept_holdout_manifests_command(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, dir_okay=False, resolve_path=True)
    ],
    frozen_review: Annotated[
        Path, typer.Option("--frozen-review", exists=True, dir_okay=False, resolve_path=True)
    ],
    general_output: Annotated[
        Path, typer.Option("--general-output", dir_okay=False, resolve_path=True)
    ],
    challenge_output: Annotated[
        Path, typer.Option("--challenge-output", dir_okay=False, resolve_path=True)
    ],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
) -> None:
    """Fix prediction-blind fresh holdout memberships and their hashes."""

    try:
        input_hash = _sha256_file(input_path)
        frozen_hash = _sha256_file(frozen_review)
        records = load_book_evidence(input_path)
        frozen = EvidenceConceptGoldReview.model_validate_json(frozen_review.read_bytes())
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        general, challenge = build_holdout_manifests(
            records,
            graph,
            frozen,
            input_hash,
            frozen_hash,
        )
        general_ids = {row.evidence_id for row in general.entries}
        challenge_ids = {row.evidence_id for row in challenge.entries}
        frozen_ids = {row.evidence_id for row in frozen.entries}
        if general_ids & challenge_ids or (general_ids | challenge_ids) & frozen_ids:
            raise ValueError("fresh holdout memberships overlap each other or frozen gold")
        for path, manifest in ((general_output, general), (challenge_output, challenge)):
            if path.exists() and load_holdout_manifest(path) != manifest:
                raise ValueError(f"refusing to overwrite changed holdout manifest: {path}")
            write_json(manifest, path)
        if _sha256_file(input_path) != input_hash or _sha256_file(frozen_review) != frozen_hash:
            raise ValueError("source evidence or frozen gold changed during manifest generation")
    except (BookEvidenceImportError, ConfigError, ValidationError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "general_count": len(general.entries),
                "general_hash": _sha256_file(general_output),
                "challenge_count": len(challenge.entries),
                "challenge_hash": _sha256_file(challenge_output),
                "frozen_overlap": 0,
                "cross_holdout_overlap": 0,
            },
            sort_keys=True,
        )
    )


@app.command("build-evidence-concept-holdout-reviews")
def build_evidence_concept_holdout_reviews_command(
    input_path: Annotated[
        Path, typer.Option("--input", exists=True, dir_okay=False, resolve_path=True)
    ],
    frozen_review: Annotated[
        Path, typer.Option("--frozen-review", exists=True, dir_okay=False, resolve_path=True)
    ],
    general_manifest: Annotated[
        Path, typer.Option("--general-manifest", exists=True, dir_okay=False, resolve_path=True)
    ],
    challenge_manifest: Annotated[
        Path, typer.Option("--challenge-manifest", exists=True, dir_okay=False, resolve_path=True)
    ],
    general_predictions: Annotated[
        Path, typer.Option("--general-predictions", dir_okay=False, resolve_path=True)
    ],
    challenge_predictions: Annotated[
        Path, typer.Option("--challenge-predictions", dir_okay=False, resolve_path=True)
    ],
    general_review: Annotated[
        Path, typer.Option("--general-review", dir_okay=False, resolve_path=True)
    ],
    challenge_review: Annotated[
        Path, typer.Option("--challenge-review", dir_okay=False, resolve_path=True)
    ],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
    experiment_config: Annotated[
        Path, typer.Option("--experiment-config", exists=True, dir_okay=False)
    ] = DEFAULT_MATCHER_V2_EXPERIMENT_CONFIG,
) -> None:
    """Generate fixed variant predictions, then blinded human-review templates."""

    try:
        source_hash = _sha256_file(input_path)
        frozen_hash = _sha256_file(frozen_review)
        records = load_book_evidence(input_path)
        frozen = EvidenceConceptGoldReview.model_validate_json(frozen_review.read_bytes())
        excluded_ids = {row.evidence_id for row in frozen.entries}
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        experiment = load_matcher_v2_experiment_config(experiment_config)
        paths = (
            (general_manifest, general_predictions, general_review),
            (challenge_manifest, challenge_predictions, challenge_review),
        )
        outputs: list[dict[str, object]] = []
        for manifest_path, predictions_path, review_path in paths:
            manifest_hash = _sha256_file(manifest_path)
            manifest = load_holdout_manifest(manifest_path)
            if (
                manifest.book_evidence_hash != source_hash
                or manifest.frozen_gold_hash != frozen_hash
            ):
                raise ValueError(f"holdout manifest source provenance differs: {manifest_path}")
            validate_manifest_against_source(manifest, records, graph, excluded_ids)
            predictions = build_holdout_predictions(
                manifest,
                manifest_hash,
                features,
                graph,
                matching,
                experiment,
            )
            if predictions_path.exists():
                existing = EvidenceConceptHoldoutPredictions.model_validate_json(
                    predictions_path.read_bytes()
                )
                if existing != predictions:
                    raise ValueError(
                        f"refusing to overwrite changed holdout predictions: {predictions_path}"
                    )
            write_json(predictions, predictions_path)
            prediction_hash = _sha256_file(predictions_path)
            blank_review = build_holdout_review(
                manifest,
                manifest_hash,
                prediction_hash,
                graph,
            )
            if review_path.exists():
                review = load_holdout_review(
                    review_path, manifest, manifest_hash, prediction_hash, graph
                )
            else:
                review = blank_review
                write_json(review, review_path)
            outputs.append(
                {
                    "kind": manifest.holdout_kind,
                    "manifest_hash": manifest_hash,
                    "prediction_hash": prediction_hash,
                    "review": str(review_path),
                    **holdout_review_summary(review),
                }
            )
        if _sha256_file(input_path) != source_hash or _sha256_file(frozen_review) != frozen_hash:
            raise ValueError("source evidence or frozen gold changed during prediction generation")
    except (BookEvidenceImportError, ConfigError, ValidationError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"holdouts": outputs, "predictions_hidden": True}, sort_keys=True))


@app.command("show-evidence-concept-holdout")
def show_evidence_concept_holdout_command(
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 10,
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
) -> None:
    """Print a blinded packet: evidence and concept choices, never predictions."""

    try:
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        review = json.loads(review_file.read_text(encoding="utf-8"))
        parsed = EvidenceConceptHoldoutReview.model_validate(review)
        packet = review_packet(parsed, graph, limit)
    except (ConfigError, ValidationError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True))


@app.command("review-evidence-concept-holdout")
def review_evidence_concept_holdout_command(
    manifest_file: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, resolve_path=True)
    ],
    predictions_file: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False, resolve_path=True)
    ],
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    evidence_id: Annotated[str, typer.Option("--evidence-id")],
    outcome: Annotated[Literal["labeled", "no_concept", "not_judgable"], typer.Option("--outcome")],
    gold: Annotated[str, typer.Option("--gold")] = "",
    note: Annotated[str | None, typer.Option("--note")] = None,
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
) -> None:
    """Apply one explicit human label without exposing fixed predictions."""

    try:
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        manifest = load_holdout_manifest(manifest_file)
        manifest_hash = holdout_sha256_file(manifest_file)
        prediction_hash = holdout_sha256_file(predictions_file)
        review = load_holdout_review(review_file, manifest, manifest_hash, prediction_hash, graph)
        gold_ids = [value.strip() for value in gold.split(",") if value.strip()]
        updated = update_holdout_review_entry(review, graph, evidence_id, outcome, gold_ids, note)
        write_json(updated, review_file)
    except (ConfigError, ValidationError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(holdout_review_summary(updated), sort_keys=True))


@app.command("evaluate-evidence-concept-holdout")
def evaluate_evidence_concept_holdout_command(
    manifest_file: Annotated[
        Path, typer.Option("--manifest", exists=True, dir_okay=False, resolve_path=True)
    ],
    predictions_file: Annotated[
        Path, typer.Option("--predictions", exists=True, dir_okay=False, resolve_path=True)
    ],
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
    experiment_config: Annotated[
        Path, typer.Option("--experiment-config", exists=True, dir_okay=False)
    ] = DEFAULT_MATCHER_V2_EXPERIMENT_CONFIG,
) -> None:
    """Evaluate fixed predictions after a fresh holdout is fully reviewed."""

    try:
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        experiment = load_matcher_v2_experiment_config(experiment_config)
        manifest_hash = holdout_sha256_file(manifest_file)
        prediction_hash = holdout_sha256_file(predictions_file)
        review_hash = holdout_sha256_file(review_file)
        manifest = load_holdout_manifest(manifest_file)
        predictions = load_holdout_predictions(predictions_file, manifest, manifest_hash)
        review = load_holdout_review(
            review_file,
            manifest,
            manifest_hash,
            prediction_hash,
            graph,
        )
        expected_hashes = (
            features.content_hash,
            graph.content_hash,
            matching.content_hash,
            experiment.content_hash,
        )
        prediction_hashes = (
            predictions.feature_config_hash,
            predictions.graph_hash,
            predictions.matching_v1_config_hash,
            predictions.matcher_v2_experiment_config_hash,
        )
        if prediction_hashes != expected_hashes:
            raise ValueError("current matcher/config hashes differ from fixed predictions")
        report = build_holdout_evaluation_report(
            manifest,
            predictions,
            review,
            manifest_hash,
            prediction_hash,
            review_hash,
        )
        if (
            holdout_sha256_file(manifest_file) != manifest_hash
            or holdout_sha256_file(predictions_file) != prediction_hash
            or holdout_sha256_file(review_file) != review_hash
        ):
            raise ValueError("holdout artifacts changed during evaluation")
        write_json(report, output)
    except (ConfigError, ValidationError, ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "holdout_kind": report.holdout_kind,
                "evaluable": report.evaluable_entry_count,
                "variants": [variant.variant for variant in report.variants],
            },
            sort_keys=True,
        )
    )


@app.command("build-book-profiles")
def build_profiles(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=False,
            file_okay=False,
            resolve_path=True,
            help="Directory containing the four canonical JSONL files.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Destination book_profiles.jsonl path.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned feature configuration YAML.",
        ),
    ] = DEFAULT_FEATURE_CONFIG,
) -> None:
    """Build deterministic concept and prose-difficulty profiles."""

    try:
        dataset = load_canonical_dataset(data_dir)
        evidence = assemble_book_evidence(dataset)
        loaded_config = load_feature_config(config)
        profiles = build_book_profiles(evidence, loaded_config)
        write_jsonl(profiles, output)
    except (CanonicalDataError, ConfigError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    difficulty_count = sum(
        profile.difficulty_profile.analyzed_document_count > 0 for profile in profiles
    )
    typer.echo(
        json.dumps(
            {
                "book_count": len(profiles),
                "books_with_difficulty": difficulty_count,
                "config_hash": loaded_config.content_hash,
                "config_version": loaded_config.config.config_version,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("build-assessment-blueprint")
def build_assessment_blueprint_command(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", file_okay=False, resolve_path=True),
    ],
    books: Annotated[
        Path,
        typer.Option("--books", exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    topic: Annotated[str, typer.Option("--topic", help="Canonical topic ID.")],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ],
    feature_config: Annotated[
        Path,
        typer.Option("--feature-config", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_FEATURE_CONFIG,
    assessment_config: Annotated[
        Path,
        typer.Option("--assessment-config", exists=True, dir_okay=False, readable=True),
    ] = DEFAULT_ASSESSMENT_CONFIG,
) -> None:
    """Build an auditable concept pool and question targets from canonical evidence."""

    try:
        canonical_files = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        canonical_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        book_profiles_hash = _sha256_file(books)
        dataset = load_canonical_dataset(data_dir)
        profiles = load_book_profiles(books)
        features = load_feature_config(feature_config)
        assessment = load_assessment_config(assessment_config)
        if canonical_hashes != {
            name: _sha256_file(data_dir / name) for name in canonical_files
        } or book_profiles_hash != _sha256_file(books):
            raise AssessmentBlueprintError(
                "canonical input or book profiles changed during loading"
            )
        blueprint = build_assessment_blueprint(
            topic,
            dataset,
            profiles,
            features,
            assessment,
            canonical_file_hashes=canonical_hashes,
            book_profiles_hash=book_profiles_hash,
        )
        write_json(blueprint, output)
    except (
        CanonicalDataError,
        RankingInputError,
        ConfigError,
        AssessmentBlueprintError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "topic_id": blueprint.topic_id,
                "pool_concepts": len(blueprint.concept_pool.concepts),
                "selected_concepts": len(blueprint.selected_assessment_concepts),
                "question_specs": len(blueprint.question_specs),
                "shortages": len(blueprint.shortages),
                "config_version": blueprint.config_version,
                "config_hash": blueprint.config_hash,
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("ablate-book-profile")
def ablate_profile(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=False,
            file_okay=False,
            resolve_path=True,
            help="Directory containing the four canonical JSONL files.",
        ),
    ],
    book_id: Annotated[str, typer.Option("--book-id", help="Canonical book identifier.")],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Destination ablation report JSON path.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned feature configuration YAML.",
        ),
    ] = DEFAULT_FEATURE_CONFIG,
) -> None:
    """Compare TOC-only, TOC+description, and all-evidence profiles."""

    try:
        dataset = load_canonical_dataset(data_dir)
        evidence = assemble_book_evidence(dataset)
        selected = next((item for item in evidence if item.book_id == book_id), None)
        if selected is None:
            raise CanonicalDataError(f"book not found for ablation: {book_id}")
        loaded_config = load_feature_config(config)
        report = build_evidence_ablation_report(selected, loaded_config)
        write_json(report, output)
    except (CanonicalDataError, ConfigError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "book_id": report.book_id,
                "config_hash": report.config_hash,
                "output": str(output),
                "variant_count": len(report.variants),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("reader-profile")
def reader_profile(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Structured assessment JSON path.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned reader-scoring configuration YAML.",
        ),
    ] = DEFAULT_READER_CONFIG,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional destination JSON path; otherwise print the profile.",
        ),
    ] = None,
) -> None:
    """Convert one topic assessment into an explainable reader profile."""

    try:
        assessment = load_assessment(input_path)
        loaded_config = load_reader_config(config)
        profile = build_reader_profile(assessment, loaded_config)
        if output is not None:
            write_json(profile, output)
    except (AssessmentError, ConfigError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output is None:
        typer.echo(
            json.dumps(
                profile.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo(
            json.dumps(
                {
                    "assessment_id": profile.assessment_id,
                    "config_hash": profile.config_hash,
                    "output": str(output),
                    "profile_version": profile.profile_version,
                    "topic_id": profile.topic_id,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


@app.command("rank")
def rank(
    reader: Annotated[
        Path,
        typer.Option(
            "--reader",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="ReaderProfile JSON path.",
        ),
    ],
    books: Annotated[
        Path,
        typer.Option(
            "--books",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="BookProfile JSONL path.",
        ),
    ],
    limit: Annotated[int, typer.Option("--limit", min=1, help="Maximum ranked items.")] = 5,
    book_id: Annotated[
        str | None,
        typer.Option("--book-id", help="Score only this canonical book ID."),
    ] = None,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned ranking configuration YAML.",
        ),
    ] = DEFAULT_RANKING_CONFIG,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Optional destination JSON path; otherwise print rankings.",
        ),
    ] = None,
) -> None:
    """Rank topic candidates or calculate one specific-book fit score."""

    try:
        reader_profile_value = load_reader_profile(reader)
        book_profiles = load_book_profiles(books)
        loaded_config = load_ranking_config(config)
        if book_id is None:
            response = rank_books(reader_profile_value, book_profiles, loaded_config, limit)
        else:
            selected = next((book for book in book_profiles if book.book_id == book_id), None)
            if selected is None:
                raise RankingInputError(f"book profile not found: {book_id}")
            item = score_book_fit(reader_profile_value, selected, loaded_config)
            response = RankingResponse(
                topic_id=reader_profile_value.topic_id,
                items=[item],
                model_version=loaded_config.config.model_version,
                config_version=loaded_config.config.config_version,
                config_hash=loaded_config.content_hash,
                reader_profile_version=reader_profile_value.profile_version,
                reader_config_version=reader_profile_value.config_version,
                reader_config_hash=reader_profile_value.config_hash,
            )
        if output is not None:
            write_json(response, output)
    except (ConfigError, RankingError, RankingInputError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if output is None:
        typer.echo(
            json.dumps(
                response.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        typer.echo(
            json.dumps(
                {
                    "config_hash": response.config_hash,
                    "item_count": len(response.items),
                    "model_version": response.model_version,
                    "output": str(output),
                    "topic_id": response.topic_id,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )


@app.command("evaluate-ranking-policies")
def evaluate_ranking_policies_command(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", file_okay=False, resolve_path=True),
    ],
    books: Annotated[
        Path,
        typer.Option("--books", exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    reader: Annotated[
        Path,
        typer.Option("--reader", exists=True, dir_okay=False, readable=True, resolve_path=True),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, resolve_path=True),
    ],
    policy_config: Annotated[
        Path,
        typer.Option("--policy-config", exists=True, dir_okay=False, resolve_path=True),
    ] = DEFAULT_RANKING_POLICY_CONFIG,
    ranking_config: Annotated[
        Path,
        typer.Option("--ranking-config", exists=True, dir_okay=False, resolve_path=True),
    ] = DEFAULT_RANKING_CONFIG,
) -> None:
    """Compare evidence policies on one validated canonical book snapshot."""

    try:
        dataset = load_canonical_dataset(data_dir)
        profiles = load_book_profiles(books)
        canonical_ids = {book.book_id for book in dataset.books}
        profile_ids = {book.book_id for book in profiles}
        if canonical_ids != profile_ids:
            raise EvaluationDataError(
                "book profiles do not match canonical dataset: "
                f"missing {sorted(canonical_ids - profile_ids)}, "
                f"extra {sorted(profile_ids - canonical_ids)}"
            )
        report = evaluate_ranking_policies(
            load_reader_profile(reader),
            profiles,
            load_ranking_config(ranking_config),
            load_ranking_policy_config(policy_config),
            canonical_book_count=len(dataset.books),
        )
        write_json(report, output)
    except (
        CanonicalDataError,
        ConfigError,
        EvaluationDataError,
        RankingInputError,
        RankingError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "topic_id": report.topic_id,
                "candidate_count": report.candidate_count,
                "policy_config_hash": report.policy_config_hash,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@app.command("evaluate-concept-matching")
def evaluate_concept_matching_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", file_okay=False, resolve_path=True)],
    reader: Annotated[
        Path, typer.Option("--reader", exists=True, dir_okay=False, resolve_path=True)
    ],
    books: Annotated[Path, typer.Option("--books", exists=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
    ranking_config: Annotated[
        Path, typer.Option("--ranking-config", exists=True, dir_okay=False)
    ] = DEFAULT_RANKING_CONFIG,
) -> None:
    """Compare unchanged rank-v1 with TOC concept matching for one topic."""

    try:
        canonical_files = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        input_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        input_hashes["reader"] = _sha256_file(reader)
        input_hashes["books_v1"] = _sha256_file(books)
        dataset = load_canonical_dataset(data_dir)
        evidence = assemble_book_evidence(dataset)
        reader_profile_value = load_reader_profile(reader)
        v1_profiles = load_book_profiles(books)
        if {item.book_id for item in v1_profiles} != {item.book_id for item in dataset.books}:
            raise ValueError("v1 books do not match canonical book IDs")
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        v1_ranking = load_ranking_config(ranking_config)
        rebuilt_v1 = build_book_profiles(evidence, features)
        if {item.book_id: item.model_dump(mode="json") for item in v1_profiles} != {
            item.book_id: item.model_dump(mode="json") for item in rebuilt_v1
        }:
            raise ValueError("v1 book profiles are stale for the canonical input or feature config")
        v1_by_id = {item.book_id: item for item in v1_profiles}
        topic = reader_profile_value.topic_id
        if topic not in graph.graph.nodes:
            raise ValueError(f"unsupported concept graph topic: {topic}")
        items = []
        for book in evidence:
            if topic not in book.metadata.topics:
                continue
            profile = build_book_concept_profile_v2(
                book, topic, features, graph, matching, input_hashes["toc.jsonl"]
            )
            items.append(
                match_book_concepts(
                    reader_profile_value, profile, matching, v1_by_id[book.book_id], v1_ranking
                )
            )
        if input_hashes != {
            **{name: _sha256_file(data_dir / name) for name in canonical_files},
            "reader": _sha256_file(reader),
            "books_v1": _sha256_file(books),
        }:
            raise ValueError("concept matching inputs changed during evaluation")
        report = ConceptMatchingReport(
            topic_id=topic,
            candidate_count=len(items),
            items=order_matching_items(items),
            model_version=matching.config.model_version,
            matching_config_version=matching.config.config_version,
            matching_config_hash=matching.content_hash,
            graph_version=graph.graph.graph_version,
            graph_hash=graph.content_hash,
            feature_config_hash=features.content_hash,
            reader_profile_version=reader_profile_value.profile_version,
            reader_config_hash=reader_profile_value.config_hash,
            input_hashes=input_hashes,
        )
        write_json(report, output)
    except (
        CanonicalDataError,
        ConfigError,
        RankingError,
        RankingInputError,
        TocTreeError,
        ValueError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {"topic_id": topic, "candidate_count": len(items), "output": str(output)},
            sort_keys=True,
        )
    )


@app.command("build-concept-review")
def build_concept_review_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", file_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    readers: Annotated[
        list[Path] | None, typer.Option("--reader", exists=True, dir_okay=False, resolve_path=True)
    ] = None,
    review_template: Annotated[
        Path | None, typer.Option("--review-template", dir_okay=False, resolve_path=True)
    ] = None,
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
    review_config: Annotated[
        Path,
        typer.Option(
            "--review-config",
            exists=False,
            dir_okay=False,
            help="Optional version-controlled human edge review YAML.",
        ),
    ] = DEFAULT_CONCEPT_GRAPH_REVIEWS_CONFIG,
) -> None:
    """Build structural evidence and an optional generated review snapshot."""

    try:
        if review_template == output:
            raise ValueError("review template path must differ from the report")
        canonical_files = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        input_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        dataset = load_canonical_dataset(data_dir)
        evidence = assemble_book_evidence(dataset)
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        reviews = (
            load_concept_graph_reviews(review_config, graph) if review_config.exists() else None
        )
        if reviews is not None:
            input_hashes["concept_graph_reviews.yaml"] = reviews.content_hash
        reader_by_topic = {}
        reader_files = {}
        for path in readers or []:
            reader_profile_value = load_reader_profile(path)
            topic = reader_profile_value.topic_id
            if topic in reader_by_topic:
                raise ValueError(f"duplicate reader topic: {topic}")
            reader_by_topic[topic] = reader_profile_value
            reader_files[topic] = path
            input_hashes[f"reader:{topic}"] = _sha256_file(path)
        report = build_concept_validation_report(
            evidence,
            graph,
            features,
            matching,
            reader_by_topic,
            input_hashes["toc.jsonl"],
            input_hashes,
            reviews,
        )
        current_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        current_hashes.update(
            {f"reader:{topic}": _sha256_file(path) for topic, path in reader_files.items()}
        )
        if reviews is not None:
            current_hashes["concept_graph_reviews.yaml"] = _sha256_file(review_config)
        if current_hashes != input_hashes:
            raise ValueError("concept review inputs changed during generation")
        generated_review = (
            build_review_template(graph, reviews) if review_template is not None else None
        )
        write_json(report, output)
        if review_template is not None:
            write_json(generated_review, review_template)
    except (
        CanonicalDataError,
        ConfigError,
        RankingInputError,
        TocTreeError,
        ValueError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "book_count": len(report.books),
                "edge_count": len(report.graph_edges),
                "review_template": str(review_template) if review_template else None,
            },
            sort_keys=True,
        )
    )


@app.command("build-toc-concept-gold-review")
def build_toc_concept_gold_review_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", file_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
) -> None:
    """Build the deterministic 100-entry blank TOC concept gold review."""

    try:
        canonical_files = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        input_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        evidence = assemble_book_evidence(load_canonical_dataset(data_dir))
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        review = build_toc_concept_gold_review(
            evidence=evidence,
            graph=graph,
            features=features,
            matching=matching,
            toc_file_hash=input_hashes["toc.jsonl"],
            canonical_input_hashes=input_hashes,
        )
        if output.exists():
            existing = load_toc_concept_gold_review(output, review, graph).review
            if existing != review:
                raise ValueError("refusing to overwrite a changed TOC concept gold review")
        current_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        if current_hashes != input_hashes:
            raise ValueError("TOC concept gold review inputs changed during generation")
        write_json(review, output)
    except (
        CanonicalDataError,
        ConfigError,
        TocTreeError,
        ValueError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = {
        topic: {
            "sampled": sum(row.topic == topic for row in review.entries),
            "predicted_matched": sum(
                row.topic == topic and bool(row.predicted_concept_ids) for row in review.entries
            ),
            "predicted_unmatched": sum(
                row.topic == topic and not row.predicted_concept_ids for row in review.entries
            ),
        }
        for topic in sorted(DEFAULT_SAMPLE_SIZES)
    }
    typer.echo(json.dumps({"output": str(output), "topics": summary}, sort_keys=True))


@app.command("evaluate-toc-concept-gold")
def evaluate_toc_concept_gold_command(
    data_dir: Annotated[Path, typer.Option("--data-dir", file_okay=False, resolve_path=True)],
    review_file: Annotated[
        Path, typer.Option("--review", exists=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option("--output", dir_okay=False, resolve_path=True)],
    feature_config: Annotated[
        Path, typer.Option("--feature-config", exists=True, dir_okay=False)
    ] = DEFAULT_FEATURE_CONFIG,
    graph_config: Annotated[
        Path, typer.Option("--graph-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_GRAPH_CONFIG,
    matching_config: Annotated[
        Path, typer.Option("--matching-config", exists=True, dir_okay=False)
    ] = DEFAULT_CONCEPT_MATCHING_CONFIG,
) -> None:
    """Evaluate current TOC concept predictions against completed human gold labels."""

    try:
        canonical_files = ("books.jsonl", "documents.jsonl", "toc.jsonl", "sources.jsonl")
        input_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        evidence = assemble_book_evidence(load_canonical_dataset(data_dir))
        features = load_feature_config(feature_config)
        graph = load_concept_graph(graph_config, features)
        matching = load_concept_matching_config(matching_config)
        expected = build_toc_concept_gold_review(
            evidence=evidence,
            graph=graph,
            features=features,
            matching=matching,
            toc_file_hash=input_hashes["toc.jsonl"],
            canonical_input_hashes=input_hashes,
        )
        loaded = load_toc_concept_gold_review(review_file, expected, graph)
        report = evaluate_toc_concept_gold(loaded, graph)
        current_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        if current_hashes != input_hashes or _sha256_file(review_file) != loaded.content_hash:
            raise ValueError("TOC concept gold evaluation inputs changed during evaluation")
        write_json(report, output)
    except (
        CanonicalDataError,
        ConfigError,
        TocTreeError,
        ValueError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "reviewed_entry_count": report.combined.reviewed_entry_count,
                "micro_f1": report.combined.micro_f1,
            },
            sort_keys=True,
        )
    )


@app.command("evaluate")
def evaluate(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=False,
            file_okay=False,
            resolve_path=True,
            help="Canonical data directory used for the evidence ablation.",
        ),
    ],
    books: Annotated[
        Path,
        typer.Option(
            "--books",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="BookProfile JSONL path.",
        ),
    ],
    reader: Annotated[
        Path,
        typer.Option(
            "--reader",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="ReaderProfile JSON path.",
        ),
    ],
    difficulty_labels: Annotated[
        Path,
        typer.Option(
            "--difficulty-labels",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Human relative-difficulty judgment JSON path.",
        ),
    ],
    ablation_book_id: Annotated[
        str,
        typer.Option("--ablation-book-id", help="Rich-evidence canonical book identifier."),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Destination evaluation report JSON path.",
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned evaluation configuration YAML.",
        ),
    ] = DEFAULT_EVALUATION_CONFIG,
    feature_config: Annotated[
        Path,
        typer.Option(
            "--feature-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned feature configuration YAML.",
        ),
    ] = DEFAULT_FEATURE_CONFIG,
    ranking_config: Annotated[
        Path,
        typer.Option(
            "--ranking-config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Versioned ranking configuration YAML.",
        ),
    ] = DEFAULT_RANKING_CONFIG,
) -> None:
    """Run difficulty, recommendation, and evidence-ablation baselines."""

    try:
        dataset = load_canonical_dataset(data_dir)
        evidence = assemble_book_evidence(dataset)
        selected = next((item for item in evidence if item.book_id == ablation_book_id), None)
        if selected is None:
            raise EvaluationDataError(f"book not found for ablation: {ablation_book_id}")
        report = build_evaluation_report(
            judgments=load_difficulty_judgments(difficulty_labels),
            reader=load_reader_profile(reader),
            books=load_book_profiles(books),
            ablation_evidence=selected,
            feature_config=load_feature_config(feature_config),
            ranking_config=load_ranking_config(ranking_config),
            evaluation_config=load_evaluation_config(config),
        )
        write_json(report, output)
    except (
        CanonicalDataError,
        ConfigError,
        EvaluationDataError,
        RankingError,
        RankingInputError,
        OSError,
    ) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        json.dumps(
            {
                "comparable_difficulty_books": report.difficulty.comparable_book_count,
                "config_hash": report.config_hash,
                "evaluation_version": report.evaluation_version,
                "failure_case_count": len(report.failure_cases),
                "output": str(output),
                "recommendation_candidates": report.recommendation.candidate_count,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
