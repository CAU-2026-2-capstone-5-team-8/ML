# v2·v3 추천 순서 사람 평가

## 목적과 범위

기존 비교기는 순위와 빈 평가표만 만들었다. 이제 평가자가 기록한 적합도와 두 알고리즘의 순서를 비교할 수 있다. 알고리즘, API, 앱 기본 모델은 변경하지 않는다.

`compare_learning_order.py`의 출력 계약은 `learning-order-comparison-v2`다. 이전 v1 파일은 수정하지 말고 새 출력 폴더로 다시 생성한다. 새 평가표의 `comparison_hash`는 comparison.json의 정확한 바이트에 연결된다. 원본·설정·그래프 해시는 comparison.json 안에 보존된다. JSON 재정렬만 해도 파일 해시가 바뀌므로 평가 도중 비교 파일을 수정하지 않는다. 이는 자료 혼합 방지이며 디지털 서명이나 평가자 신원 인증이 아니다.

## 실행 순서

저장소 루트에서 `uv sync --locked` 후 실행한다. Windows는 `PYTHONUTF8=1`을 설정한다.

```sh
uv run --no-sync python scripts/compare_learning_order.py --manifest data/output/local-release-20261004/backend-manifest.json --output-dir data/output/review-round-1
```

실제 로컬 handoff가 필요하다. 다른 팀원이 사용하는 경로에는 그 팀원의 검증된 manifest를 넣는다. 산출물의 도서·응답 내용은 자동으로 가짜 사람 평가로 채우지 않는다.

1. 평가 진행자가 `human-review.csv`를 평가자별로 복사한다. comparison.json은 순위 노출을 피하기 위해 평가자가 점수를 확정하기 전에는 보여주지 않는다.
2. 평가자는 같은 snapshot의 목차 원출처·개념 연결과 `observations_json`의 가상 응답을 보고 다음 책으로의 적합도를 평가한다.
3. 수정할 열은 `suitability_1_to_5`, `reason`, `reviewer`, `reviewed_at` 네 개다. 나머지 식별값·제목·응답은 그대로 둔다.
4. 점수는 1(선수 보완이 우선), 2(상당한 보완 필요), 3(조건부 적합), 4(대체로 적합), 5(우선 추천). 숫자를 적은 행은 이유·평가자·YYYY-MM-DD 날짜가 필수다. 판단 근거가 없으면 점수를 비우고 이유를 남긴다.
5. 같은 사람이 여러 파일에 같은 책/시나리오를 평가하면 중복으로 거부한다. 서로 다른 평가자는 일관된 서로 다른 식별자를 사용한다. 최소 두 명의 독립 검토를 권장하지만 프로그램은 ID만 확인하며 독립성을 인증하지 않는다.

```sh
uv run --no-sync python scripts/evaluate_learning_order.py --comparison data/output/review-round-1/comparison.json --reviews data/output/review-round-1/reviewer-a.csv data/output/review-round-1/reviewer-b.csv --output data/output/review-round-1/evaluation.json
```

출력 파일이 있으면 덮어쓰지 않고 실패한다. 평가 파일과 원문 자료는 ignored data 디렉터리에 두며 Git에 올리지 않는다. 평가를 나중에 추가했다면 새 결과 파일명을 사용한다.

## 집계 기준

평가자별·시나리오별로, v2와 v3 양쪽 상위 20권 목록에 공통으로 있고 점수도 있는 책만 비교한다. 한쪽에만 반환된 책은 `ratedOutsideCommonBooks`로 집계한다. 현재 7권은 모두 포함된다. 후보가 많아지면 전수 평가가 아니다.

- `agree` / `disagree`: 사람 점수 차이와 알고리즘의 우선순위가 같은/반대인 책 쌍 수.
- `humanTie`: 사람 점수가 같은 쌍. 순서 정답으로 사용하지 않는다.
- `modelTie`: 사람은 차이를 두었지만 알고리즘은 동순위인 쌍.
- `agreementOnDecidedPairs`: agree / (agree + disagree). 분모가 0이면 null.
- `decidedPairCoverage`: (agree + disagree) / (agree + disagree + modelTie). 분모가 0이면 null.
- `ratingCoverage`: 점수가 있는 공통 도서 수 / 공통 후보 도서 수.

일치율만 비교하면 동순위를 많이 만드는 모델이 유리해질 수 있다. 반드시 결정 범위와 실제 쌍 수를 같이 읽는다. 두 모델은 같은 사람 평가와 공통 후보 집합을 사용하지만 결정한 쌍은 다를 수 있다. 따라서 일치율 차이만으로 승자를 선언하지 않는다.

v2는 현재 정렬 기준인 준비도 상태, reviewOnly, 선수관계 미확립 여부, 연습개념 존재 여부가 같은 책을 묶는다. 마지막 bookId 정렬은 우열로 보지 않는다. v3는 응답의 rankGroup을 사용한다. 여러 사람의 점수를 평균내어 불일치를 지우지 않고 각자 결과를 제공한다.

평가 전에는 `awaiting_reviews`, 점수가 있으면 `descriptive_review_results`다. `accuracyMeasured`는 계속 false다. 결과는 가상 응답에 대한 사람의 선호 일치도이며, 실제 학습 효과나 본문 난이도 정확도가 아니다. v3 활성화·운영 승격은 수행하지 않는다.

## 2026-10-06 검증

- 실제 보유 handoff: 25권, 개념 연결 7권, 가상 시나리오 6개, 빈 평가표 42행.
- 실제 사람 평가 0행 → awaiting_reviews. 정확도 값이나 개선 수치를 만들어내지 않음.
- 동일 입력의 comparison.json·human-review.csv를 두 번 생성해 바이트 일치 확인.
- 전체 테스트 410개 통과. 기존 FastAPI/Starlette deprecated 경고 2개.
- Ruff 검사 및 포맷 검사 통과. lockfile 의존성 사용.
- 회귀 검사: 해시·도서·응답 혼합, 중복 투표, 잘못된 점수/날짜, 동순위, 미평가, 공통 후보 제한, CLI 결과 재현성 및 덮어쓰기 방지.

다음 실질 단계는 독립 평가자가 자료를 보고 점수와 이유를 입력하는 것이다. 그 결과에서 오탐 개념 연결·선수관계·문항 오류를 구분한 뒤 기준이나 데이터를 수정한다.
