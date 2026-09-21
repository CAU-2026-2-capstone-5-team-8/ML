"""Pure orchestration used by the HTTP adapter."""

from bookmatch_ml.concept_v2.difficulty import DifficultyPolicy, score_difficulty
from bookmatch_ml.config import LoadedRankingConfig, LoadedReaderConfig
from bookmatch_ml.integration.schemas import (
    ExperimentalRankedBookDto,
    RankedBookDto,
    RankRequest,
    RankResponse,
    ReaderProfileRequest,
    ReaderProfileResponse,
    camelize_payload,
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
        difficulty_policy: tuple[DifficultyPolicy, str] | None,
    ) -> None:
        self._reader_config = reader_config
        self._ranking_config = ranking_config
        self._difficulty_policy = difficulty_policy

    def build_reader_profile(self, request: ReaderProfileRequest) -> ReaderProfileResponse:
        profile = build_reader_profile(request.to_internal(), self._reader_config)
        return ReaderProfileResponse.from_internal(profile, user_id=request.user_id)

    def rank(self, request: RankRequest) -> RankResponse:
        reader = request.reader_profile.to_internal()
        books = [book.to_internal() for book in request.candidate_books]
        if request.ranking_strategy == "concept_difficulty_v2_experimental":
            return self._rank_concept_difficulty(request, reader, books)
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

    def _rank_concept_difficulty(self, request, reader, books) -> RankResponse:
        if self._difficulty_policy is None:
            raise ValueError("concept difficulty strategy is not configured")
        selected = list(zip(request.candidate_books, books, strict=True))
        if request.book_id is not None:
            selected = [item for item in selected if item[0].book_id == request.book_id]
        elif self._ranking_config.config.filter_topic_candidates:
            selected = [
                item
                for item in selected
                if item[1].topic_distribution.get(reader.topic_id, 0.0) > 0
            ]
        items: list[RankedBookDto] = []
        for candidate, book in selected:
            profile = candidate.concept_profile
            if profile is None:
                raise ValueError(
                    "conceptProfile is required for concept_difficulty_v2_experimental"
                )
            if profile.book_id != candidate.book_id:
                raise ValueError("conceptProfile bookId does not match candidate bookId")
            if profile.topic_id != reader.topic_id:
                raise ValueError(
                    "experimental concept difficulty requires reader and book topics to match"
                )
            baseline = score_matching_book_fit(reader, book, self._ranking_config)
            result = score_difficulty(reader, profile, self._difficulty_policy)
            if result["recommendation_score"] is None:
                continue
            reasons = [
                *baseline.reasons,
                (
                    f"Intrinsic concept difficulty is {result['book_difficulty_band']} "
                    f"({result['book_difficulty_score']:.3f})."
                ),
                (
                    f"Estimated learning burden is {result['difficulty_band']} "
                    f"({result['burden_lower']:.3f}-{result['burden_upper']:.3f})."
                ),
            ]
            policy, policy_hash = self._difficulty_policy
            payload = RankedBookDto.from_internal(baseline).model_dump(by_alias=False)
            payload.update(
                score=result["recommendation_score"],
                reasons=reasons,
                model_version=policy.model_version,
                config_version=policy.config_version,
                config_hash=policy_hash,
                concept_difficulty=camelize_payload(result),
            )
            item = ExperimentalRankedBookDto.model_validate(payload)
            items.append(item)
        if not items:
            raise ValueError(
                "no candidate has sufficient concept evidence and reader assessment coverage"
            )
        items.sort(key=lambda item: (-item.score, item.book_id))
        items = items[: 1 if request.book_id is not None else request.limit]
        policy, policy_hash = self._difficulty_policy
        return RankResponse(
            user_id=request.reader_profile.user_id,
            topic_id=reader.topic_id,
            items=items,
            model_version=policy.model_version,
            config_version=policy.config_version,
            config_hash=policy_hash,
            reader_profile_version=reader.profile_version,
            reader_config_version=reader.config_version,
            reader_config_hash=reader.config_hash,
        )
