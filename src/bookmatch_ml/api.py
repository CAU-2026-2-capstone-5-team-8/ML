"""Thin HTTP adapter for Spring Boot orchestration."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from bookmatch_ml.config import load_ranking_config, load_reader_config
from bookmatch_ml.integration.schemas import (
    RankRequest,
    RankResponse,
    ReaderProfileRequest,
    ReaderProfileResponse,
)
from bookmatch_ml.integration.service import IntegrationService
from bookmatch_ml.ranking.matching import RankingError
from bookmatch_ml.reader.profile import AssessmentError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(
    reader_config_path: Path | None = None,
    ranking_config_path: Path | None = None,
) -> FastAPI:
    """Create an offline calculation API with versioned local configuration."""

    reader_config = load_reader_config(reader_config_path or PROJECT_ROOT / "configs/reader.yaml")
    ranking_config = load_ranking_config(
        ranking_config_path or PROJECT_ROOT / "configs/ranking.yaml"
    )
    service = IntegrationService(reader_config, ranking_config)
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


app = create_app()
