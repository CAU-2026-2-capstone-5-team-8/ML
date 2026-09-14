"""Command-line entry points for reproducible batch workflows."""

import json
from pathlib import Path
from typing import Annotated

import typer

from bookmatch_ml.book.profile import build_book_profiles
from bookmatch_ml.config import (
    ConfigError,
    load_feature_config,
    load_ranking_config,
    load_reader_config,
)
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import CanonicalDataError, load_canonical_dataset
from bookmatch_ml.evaluation.ablation import build_evidence_ablation_report
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


if __name__ == "__main__":
    app()
