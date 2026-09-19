"""Command-line entry points for reproducible batch workflows."""

import hashlib
import json
from pathlib import Path
from typing import Annotated

import typer

from bookmatch_ml.assessment.blueprint import build_assessment_blueprint
from bookmatch_ml.assessment.pool import AssessmentBlueprintError
from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.concept_v2.graph import load_concept_graph
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
    ConceptGraphReviewFile,
    build_concept_validation_report,
    build_review_template,
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
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import CanonicalDataError, load_canonical_dataset
from bookmatch_ml.evaluation.ablation import build_evidence_ablation_report
from bookmatch_ml.evaluation.difficulty import EvaluationDataError, load_difficulty_judgments
from bookmatch_ml.evaluation.ranking_policies import evaluate_ranking_policies
from bookmatch_ml.evaluation.report import build_evaluation_report
from bookmatch_ml.io import write_json, write_jsonl
from bookmatch_ml.ranking.loader import (
    RankingInputError,
    load_book_profiles,
    load_reader_profile,
)
from bookmatch_ml.ranking.matching import RankingError, rank_books, score_book_fit
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
) -> None:
    """Build structural graph/TOC evidence and an optional unreviewed decision file."""

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
        )
        current_hashes = {name: _sha256_file(data_dir / name) for name in canonical_files}
        current_hashes.update(
            {f"reader:{topic}": _sha256_file(path) for topic, path in reader_files.items()}
        )
        if current_hashes != input_hashes:
            raise ValueError("concept review inputs changed during generation")
        blank_review = build_review_template(graph) if review_template is not None else None
        if review_template is not None and review_template.exists():
            existing_review = ConceptGraphReviewFile.model_validate_json(
                review_template.read_text(encoding="utf-8")
            )
            if existing_review != blank_review:
                raise ValueError("refusing to overwrite a changed concept graph review file")
        write_json(report, output)
        if review_template is not None:
            write_json(blank_review, review_template)
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
