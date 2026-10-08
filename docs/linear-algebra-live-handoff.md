# Linear Algebra live handoff

2026-10-01에 실제 Python ML → Spring Boot → PostgreSQL → 브라우저 전체 흐름을 검증했다. 실행 순서·DB 격리·등록·문항 bootstrap·테스트·한계는 [Backend 재현 문서](../../Backend/docs/linear-algebra-live-handoff.md)에 있다.

## Fixed inputs

[Pin manifest](../configs/experiments/linear-algebra-live-handoff-v1.json)는 canonical 4파일, pilot 3파일, 승인 문항·review·grounding의 SHA-256 및 선정 book_id를 고정한다. 원본은 Git에서 제외되어 있다. 준비 스크립트의 모든 입력 경로를 명시해야 하며 없는 자료는 고정된 offline snapshot을 받아야 한다. 최신 재수집 자료를 같은 스냅샷으로 취급하지 않는다.

`scripts/prepare_linear_algebra_handoff.py`는 canonical을 검증하고 source-aware mapping/profile로 candidates를 다시 구성해 원본 pilot과 대조한다. 목차 근거가 있는 실도서 3권을 선택하고 원본 feature/config 버전·null scalar·빈 prerequisite 배열을 보존한다. 원문 포함 packet은 ignored `data/output/`에만 저장한다. 기존 출력 폴더를 덮어쓰지 않는다.

## Direct REST Client flow

[examples/linear-algebra-live.http](../examples/linear-algebra-live.http)의 `baseUrl`을 맞추고 `laProfile` → `laRank` 순서로 실행한다. 입력 답변 9개는 합성이며 userId=1은 이 stateless API의 correlation 값이다. 응답 점수와 config hash는 실제 계산 결과다.

reader-profile 응답 전체를 rank에 넣지 않는다. dimensionDetails와 readiness의 responseCount/earnedWeight/availableWeight는 rank 요청 계약에 없으므로 제거하고 `{conceptId, score}`만 전달한다. 고정 요청이 생성하는 8개 sorted readiness row를 각각 응답 변수로 연결한다. 사용자·분야·세 영역 점수·profile/config 버전/hash도 첫 응답에서 가져온다. 후보 featureVersion은 등록 스냅샷 이름이 아닌 `book-v1`이다.

## Repeat, comparison, and errors

```sh
.venv/bin/python scripts/verify_linear_algebra_live.py \
  --packet-dir data/output/linear-algebra-live-handoff-v1 \
  --output-dir data/output/la-direct-check-new \
  --ml-url http://127.0.0.1:8011
```

실제 HTTP 응답을 기록하고 같은 입력의 동일성, 다른 답변의 점수·readiness·추천 요청 변경, 5권 요청에 3권만 반환하는 후보 부족, readiness 없음(200·0권), 특정 책 개인화 불가(422), 잘못된 score(422), 중복 candidate(422)를 검증한다. `.http`의 모든 프로필 변수도 실제 응답과 대조한다. 새 결과 폴더를 사용해야 한다. Backend URL과 명시적 bootstrap user/topic ID를 추가하면 신규 진단 2개를 만들어 전체 흐름도 검증한다.

직접 실행한 [프로필](../examples/linear-algebra-live/ml-profile.json), [추천](../examples/linear-algebra-live/ml-rank.json), [readiness 없음](../examples/linear-algebra-live/ml-rank-unassessed.json), [오류](../examples/linear-algebra-live/ml-invalid-score.json)는 요청과 실제 응답을 함께 포함한다. 책 원문이나 목차 본문은 포함하지 않는다. 후보 3권의 모두 true/false 프로필에서 순위 변경은 요구하지 않았으며, 평가 범위에 따라 personalizable 여부는 달라진다.

전체 305 tests 및 Ruff check/format이 통과했다. 새 테스트는 외부 네트워크 없이 실제 API 계산으로 REST Client 응답 연결·null 값·후보 부족·변경 답변·스냅샷 실패를 검증한다. 현재 결과는 연결의 재현성을 확인하며 진단·추천의 정확도를 평가하지 않는다.

## 대량 데이터 후속 연결

`configs/experiments/discovery-catalog-live-v1.json`은 2026-09-29의 6분야 canonical processed24개 파일 hash, 471권/439TOC 수, LA83 원본 실험 hash를 고정한다. `scripts/prepare_discovery_catalog_handoff.py`는 explicit canonical-topics-dir/la-pilot-dir/la-packet-dir/output-dir만 읽으며 provider 호출을 하지 않는다. source-aware LA projection의 모든 값과83개 ID를 대조하고 기존3권 출처 인정을 명시한 bulk manifest를 만든다. 다른5분야는 메타데이터/확보상태만 넘긴다. 원본과 기존 output을 수정하지 않는다.

`prepare_linear_algebra_handoff.py`의 ML REST Client response binding을 공용 `build_ml_http`로 추출해 3권과83권 packet 모두 계산된 readiness/configHash를 사용한다. `verify_discovery_catalog_live.py`가 실제 catalog 페이지/6필터/기존추천 보존/달라진답변/83후보 추천/저장재조회/ML 중복후보422/완료답변409를 기록한다. 사용 명령과 실제 UI·Backend 결과는 [Backend 실행 기록](../../Backend/docs/linear-algebra-live-handoff.md#후속-완료-대량-수집-카탈로그-연결-2026-10-01)에 있다.

ML 308개 테스트 통과. 계산 정책·개념 사전·추천 대상 선별은 변경하지 않았다.83후보 중 개념연결11권, 실제 진단에서는 personalizable6/conceptOnly5/evidenceUnavailable72이며 Top-5에 편입 문제집도 남아있다. 수집량을 추천 품질 향상으로 해석하지 않는다. 로컬 ignored output은 `data/output/discovery-catalog-live-v1`와 `data/output/discovery-catalog-live-verification-v1`이다. HTTP 검증 출력은 synthetic 사용자/답변의 실제 API 응답이다.
