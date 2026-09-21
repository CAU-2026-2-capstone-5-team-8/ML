"""Thin HTTP adapter for Spring Boot orchestration."""

import os
from importlib.resources import as_file, files
from pathlib import Path

from fastapi import FastAPI, HTTPException

from bookmatch_ml.concept_v2.difficulty import load_difficulty_policy
from bookmatch_ml.config import (
    LoadedRankingConfig,
    LoadedReaderConfig,
    load_ranking_config,
    load_reader_config,
)
from bookmatch_ml.integration.schemas import (
    RankRequest,
    RankResponse,
    ReaderProfileRequest,
    ReaderProfileResponse,
)
from bookmatch_ml.integration.service import IntegrationService
from bookmatch_ml.ranking.matching import RankingError
from bookmatch_ml.reader.profile import AssessmentError

CONFIG_DIR_ENV = "BOOKMATCH_ML_CONFIG_DIR"


def _load_reader_config(path: Path | None) -> LoadedReaderConfig:
    """Load an explicit, environment-provided, or packaged reader config."""

    if path is not None:
        return load_reader_config(path)
    if config_dir := os.environ.get(CONFIG_DIR_ENV):
        return load_reader_config(Path(config_dir) / "reader.yaml")
    resource = files("bookmatch_ml.default_configs").joinpath("reader.yaml")
    with as_file(resource) as packaged_path:
        return load_reader_config(packaged_path)


def _load_ranking_config(path: Path | None) -> LoadedRankingConfig:
    """Load an explicit, environment-provided, or packaged ranking config."""

    if path is not None:
        return load_ranking_config(path)
    if config_dir := os.environ.get(CONFIG_DIR_ENV):
        return load_ranking_config(Path(config_dir) / "ranking.yaml")
    resource = files("bookmatch_ml.default_configs").joinpath("ranking.yaml")
    with as_file(resource) as packaged_path:
        return load_ranking_config(packaged_path)


def create_app(
    reader_config_path: Path | None = None,
    ranking_config_path: Path | None = None,
    concept_difficulty_config_path: Path | None = None,
) -> FastAPI:
    """Create an offline calculation API with versioned local configuration."""

    reader_config = _load_reader_config(reader_config_path)
    ranking_config = _load_ranking_config(ranking_config_path)
    if concept_difficulty_config_path is not None:
        difficulty_policy = load_difficulty_policy(concept_difficulty_config_path)
    elif config_dir := os.environ.get(CONFIG_DIR_ENV):
        difficulty_policy = load_difficulty_policy(Path(config_dir) / "concept_difficulty.yaml")
    else:
        resource = files("bookmatch_ml.default_configs").joinpath("concept_difficulty.yaml")
        with as_file(resource) as packaged_path:
            difficulty_policy = load_difficulty_policy(packaged_path)
    service = IntegrationService(reader_config, ranking_config, difficulty_policy)
    application = FastAPI(
        title="BookMatch ML Integration API",
        version="0.1.0",
        description="Stateless reader-profile and reader-book matching calculations.",
    )

    @application.post("/ml/reader-profile", response_model=ReaderProfileResponse)
    def reader_profile(request: ReaderProfileRequest) -> ReaderProfileResponse:
        try:
            return service.build_reader_profile(request)
        except (AssessmentError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @application.post("/ml/rank", response_model=RankResponse)
    def rank(request: RankRequest) -> RankResponse:
        try:
            return service.rank(request)
        except (RankingError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application
