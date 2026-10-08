# 계정·개념 진단과 v3 근거 형식 통합 — 2026-10-03

PR #32의 계정별 진단·학습 추천 API를 main `1be4a1b`의 근거 형식과 통합했다.
기존 기능 브랜치 `a8b1851`에서 출발했으며, 실행 중인 원래 체크아웃과 서비스는
유지하고 별도 작업 공간에서 병합·검증했다.

## 유지한 기능

- 개념·능력별 수행 관찰과 자기평가 분리(`concept-abilities-v2`).
- `/ml/reader-profile`, `/ml/reader-diagnostics`와 공통 개념 지도 API.
- `/ml/learning-fit`의 미평가 상태와 선수관계 근거, 기존 `/ml/rank` 계약.
- 실제 도서 자료를 고정해 Backend에 전달하는 준비·검증 도구.

## 근거 형식과 분석 방식

v1/v2 handoff에 임의의 `en_text`를 추가하던 실험 확장은 제거하고, main의 엄격한
v1/v2/v3 reader를 사용한다. v3의 `en_text`는 값이 null이어도 직렬화에서 유지한다.
이와 달리 canonical book/TOC/document의 선택 영문 필드는 없을 때 생략하는 기존
규약을 그대로 따른다. 원문 공백·해시·발췌 범위·출처 사용 조건을 보존한다.

예전 브랜치의 영문 자동 우선 선택과 미번역 한글 자동 제외는 기본 분석에서 제거했다.
개념 연결, 본문 난이도, 문항 근거 추출은 원문을 사용하며 영문 추가만으로 결과가
바뀌지 않는다. 번역문을 원문 해시의 제시문으로 잘못 연결하지 않도록 검사한다.
영문은 `compare-english-evidence`와 `--review-packet`으로 명시적으로 비교한다.

폐기된 `book/english.py`, `features_english.yaml`, `concept_matching_english.yaml`을
제거했다. 해당 실험을 사용하던 명령은 최신 README의 v3 비교 명령으로 전환한다.
기존 CLI 기본 설정은 `features.yaml`과 `concept_matching_v2.yaml`이다.
PR #37의 내적 용어 후보는 별도 작업이며 이번 통합에 포함하지 않는다.

## 검증

| 검사 | 결과 |
|---|---|
| ML 전체 pytest | 366개 통과 |
| Ruff lint / format | 통과 |
| Backend 전체 clean build | 201개 통과, 실패·오류·제외 0개 |
| Backend 실제 ML 계약/도서 전달/E2E 검사 | 6개 포함, 모두 통과 |
| v1 / v2 기존 실제 입력 | 각각 98권 로드, 입력 해시 유지 |
| v3 실제 입력 | 10권·666개 목차, 입력 해시 유지 |
| 영문 비교 출력 | 두 번 실행 및 기존 main 출력과 바이트 동일 |

Backend는 원래 앱의 8011 포트 대신 통합 코드의 임시 ML 서버 8012를 사용했다.
실제 ML 계약·3권 및 83권 handoff 검사와 테스트용 PostgreSQL을 포함했다.
기존 8011 서비스와 임시 8012 서비스에 같은 합성 요청을 보낸 다음 6개 결과도 동일했다:

- 개념 지도 조회, 읽기 준비도 계산, 진단 근거 계산.
- 관찰이 없는 학습 추천, 객관식 관찰이 있는 학습 추천.
- 자기평가를 객관식 관찰로 보낸 잘못된 요청의 422 응답.

준비도 요청에는 `cognitiveOperation`, `answerMode`, `measurementContext`를 넣어
새 관찰 필드가 포함된 응답을 비교했다. 이 HTTP 비교는 저장을 수행하지 않는다.
비교 집계는 무시된 로컬 경로 `data/reports/account-concept-v3-integration-20261003/`에 있다.

ML 재현은 저장소 루트에서 `PYTHONPATH=src python -m pytest -q`로 실행한다.
실제 Backend 검사는 Java 21과 Docker, 고정된 로컬 handoff 산출물이 필요하며
`ML_CONTRACT_BASE_URL`을 별도 ML 서버로, `LA_HANDOFF_DIR`와 `DISCOVERY_HANDOFF_DIR`를
각각 준비된 산출물 경로로 지정한 후 `./gradlew clean build --no-daemon --console=plain`을 실행한다.

## 다음 통합 단계

이번 검증은 코드·API·테스트용 DB 호환성 확인이다. 운영 배포나 추천 정확도,
번역의 사람 평가 완료를 의미하지 않는다. 현재 실행 중인 앱은 교체하지 않았다.
이 PR 이후 Backend #31과 Frontend #7·#8을 최신 main에 맞춰 통합하고,
로그인부터 진단·추천·서재 저장·독서 기록까지 브라우저에서 다시 확인한다.
