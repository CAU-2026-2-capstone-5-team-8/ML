"""Pure orchestration used by the HTTP adapter."""

from bookmatch_ml.config import LoadedRankingConfig, LoadedReaderConfig
from bookmatch_ml.integration.schemas import (
    RankRequest,
    RankResponse,
    ReaderProfileRequest,
    ReaderProfileResponse,
)
from bookmatch_ml.ranking.matching import rank_matching_books, score_matching_book_fit
from bookmatch_ml.reader.profile import build_reader_profile
from bookmatch_ml.schemas import RankingResponse


class IntegrationService:
    """Run ML calculations without persistence, authentication, or provider access."""

    def __init__(
        self,
        reader_config: LoadedReaderConfig,
        ranking_config: LoadedRankingConfig,
    ) -> None:
        self._reader_config = reader_config
        self._ranking_config = ranking_config

    def build_reader_profile(self, request: ReaderProfileRequest) -> ReaderProfileResponse:
        profile = build_reader_profile(request.to_internal(), self._reader_config)
        return ReaderProfileResponse.from_internal(profile, user_id=request.user_id)

    def rank(self, request: RankRequest) -> RankResponse:
        reader = request.reader_profile.to_internal()
        books = [book.to_internal() for book in request.candidate_books]
        if request.book_id is None:
            response = rank_matching_books(reader, books, self._ranking_config, request.limit)
        else:
            book = next(book for book in books if book.book_id == request.book_id)
            item = score_matching_book_fit(reader, book, self._ranking_config)
            config = self._ranking_config.config
            response = RankingResponse(
                topic_id=reader.topic_id,
                items=[item],
                model_version=config.model_version,
                config_version=config.config_version,
                config_hash=self._ranking_config.content_hash,
                reader_profile_version=reader.profile_version,
                reader_config_version=reader.config_version,
                reader_config_hash=reader.config_hash,
            )
        return RankResponse.from_internal(response, user_id=request.reader_profile.user_id)
