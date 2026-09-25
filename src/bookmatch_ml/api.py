"""Thin HTTP adapter for Spring Boot orchestration."""

import os
from importlib.resources import as_file, files
from pathlib import Path

from fastapi import FastAPI, HTTPException

from bookmatch_ml.config import (
    LoadedRankingConfig,
    LoadedRankingV2Config,
    LoadedReaderConfig,
    load_ranking_config,
    load_ranking_v2_config,
    load_reader_config,
)
from bookmatch_ml.integration.schemas import (
    RankRequest,
    RankResponse,
    RankV2Response,
    ReaderProfileRequest,
    ReaderProfileResponse,
)
from bookmatch_ml.integration.service import IntegrationService
from bookmatch_ml.ranking.matching import RankingError
from bookmatch_ml.ranking.prerequisite_first_v2 import RankingV2Error
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


def _load_ranking_v2_config(path: Path | None) -> LoadedRankingV2Config | None:
    """Load v2 policy when explicitly provided or available to the deployment."""

    if path is not None:
        return load_ranking_v2_config(path)
    if config_dir := os.environ.get(CONFIG_DIR_ENV):
        configured_path = Path(config_dir) / "ranking_v2.yaml"
        return load_ranking_v2_config(configured_path) if configured_path.is_file() else None
    resource = files("bookmatch_ml.default_configs").joinpath("ranking_v2.yaml")
    with as_file(resource) as packaged_path:
        return load_ranking_v2_config(packaged_path)


def create_app(
    reader_config_path: Path | None = None,
    ranking_config_path: Path | None = None,
    ranking_v2_config_path: Path | None = None,
) -> FastAPI:
    """Create an offline calculation API with versioned local configuration."""

    reader_config = _load_reader_config(reader_config_path)
    ranking_config = _load_ranking_config(ranking_config_path)
    ranking_v2_config = _load_ranking_v2_config(ranking_v2_config_path)
    service = IntegrationService(reader_config, ranking_config, ranking_v2_config)
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

    @application.post("/ml/rank", response_model=RankResponse | RankV2Response)
    def rank(request: RankRequest) -> RankResponse | RankV2Response:
        try:
            return service.rank(request)
        except (RankingError, RankingV2Error, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application
