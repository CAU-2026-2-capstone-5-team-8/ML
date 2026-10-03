# 내적 용어 후보 비교 — 2026-10-03

실제 영문 목차 검토에서 `Dot Product`와 복수형이 내적 개념에 연결되지 않는 4개 항목을
확인했다. 유클리드 공간의 dot product를 기존 `inner product` 개념에 연결하는 두 표현만
추가한 설정 `configs/concept_matching_dot_product_v1.yaml`을 만들었다.
모든 내적을 dot product와 동일시하는 규칙이나 `product` 단독 표현은 추가하지 않는다.

## 적용 범위

이 설정은 `--matching-config`로 지정하는 후보다. `config_version`, `profile_version`,
`model_version`에 후보를 구분하는 새 값을 사용한다. 알고리즘은 기존
`normalized_alias_span_v2`를 유지하고, 단어 경계·중복 문구 처리·기존 제외 규칙·추천
정책도 그대로 사용한다. 기본 운영 설정과 원문/영문 선택 방식은 바뀌지 않는다.

검토 중 발견한 `Differences` 번역 의미 문제는 이 용어 사전 변경으로 해결되지 않는다.
원래 번역 자료는 수정하지 않았고, 외부 번역 호출은 0건이다.

## 실제 10권 비교

입력은 기존 `english-evidence-comparison-20261003/book-evidence-v3.jsonl`이다.
같은 666개 원문·영문 목차 쌍과 같은 특징/그래프 설정에서 매칭 설정만 바꿨다.

| 항목 | 기본 설정 | 내적 표현 후보 |
|---|---:|---:|
| 원문 책/개념 쌍 | 13 | 13 |
| 저장 영문 책/개념 쌍 | 131 | 132 |
| 영문에서 매칭된 책 | 10 | 10 |
| 원문·영문 비교 대상 목차 | 666 | 666 |

영문 목차 4개에 내적 연결이 추가된다. 그중 한영 혼합 원문에 이미 `Dot Product`가
있던 1개 항목은 원문 쪽 연결도 추가된다. 기존 매칭·모호성 결과는 보존되며 나머지
662개 목차의 결과는 동일하다. 이미 다른 목차에서 내적을 다룬 책이 있어 행 수준의
4개 추가가 책/개념 쌍 4개 증가를 의미하지 않는다.

기본·후보 각각 두 번 실행한 결과는 바이트가 같다. 입력 해시는 실행 전후 같았다.
동명의 JSON에 입력/설정/출력 해시, 변경된 근거 ID와 양쪽 매칭을 보관했다.
원문 전체가 없는 비교 출력은 로컬 `data/reports/dot-product-alias-20261003/`에 있다.

## 회귀 검증과 한계

- 단수/복수, 대문자, 하이픈, 줄바꿈, 괄호 표현을 일반화된 합성 사례로 검사했다.
- 외적·텐서곱, `product` 단독, `production`/`productivity` 및 부분 단어는 새 내적
  연결을 만들지 않는다. 다른 주제인 운영체제에는 이 규칙이 적용되지 않는다.
- 기존 내적·벡터·벡터공간을 함께 언급하면 서로 다른 개념은 유지하고 내적은 중복 계산하지 않는다.
- 기존 고정 자료 94개, General 89개, Challenge 40개 총 223개에서 개념·매칭 표현·방법과
  모호성 여부를 행별로 대조해 기본 설정과 동일함을 확인했다. 고정 정답은 수정하지 않았다.

기존 자료에서 변화가 없다는 것은 기존 동작을 보존했다는 회귀 결과다. 새 표현에 대한
독립적인 사람 정확도 평가가 아니며, 실제 4개 항목은 이 후보를 만들 때 이미 본 자료다.
따라서 새 정확도 수치나 운영 승격 근거로 사용하지 않는다. 다음 단계는 다른 도서에서
해당 표현을 포함한 독립 표본과 혼동 가능한 곱 표현을 확보해 의미 단위로 검토하는 것이다.

## 재현

```bash
uv run bookmatch-ml compare-english-evidence \
  --input <book-evidence-v3.jsonl> \
  --matching-config configs/concept_matching_v2.yaml \
  --output data/reports/dot-product-baseline.json

uv run bookmatch-ml compare-english-evidence \
  --input <book-evidence-v3.jsonl> \
  --matching-config configs/concept_matching_dot_product_v1.yaml \
  --output data/reports/dot-product-candidate.json
```

기존 파일을 덮어쓰지 않으므로 반복 실행은 새 출력 경로를 지정한다.
