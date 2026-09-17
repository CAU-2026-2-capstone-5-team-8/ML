# Ranking evidence policy experiment v1

Run date: 2026-09-17. This is a reproducible presentation and eligibility experiment over the
existing deterministic score. It does not estimate model quality or choose a production policy.

## Why compare policies

The current ranker renormalizes the configured weights over available components. A topic-matched
book with no usable prose can therefore score `1.0` when topic fit is its only component. Its
`component_weight_coverage` is `0.35`, so sorting by total score alone puts an evidence-limited
book ahead of books with measured readiness fit. Coverage records which parts of the configured
formula could be calculated. It is neither book quality nor statistical confidence.

The experiment keeps the score formula, component values, active weights, missing values, and
explanations unchanged. The three policies alter only readiness eligibility and presentation:

| Policy | Readiness group | Evidence-limited candidates |
| --- | --- | --- |
| `renormalized` | All topic candidates, sorted by the existing renormalized score and book ID | Remain in the same list with coverage and unavailable components shown |
| `minimum_coverage` | Candidates whose coverage meets the configured minimum, sorted by the same score | Visible in an `ineligible` section; `readiness_score` is `null`, with reason and diagnostic score retained |
| `two_stage` | Same eligible readiness group | Visible in a separate `topic_only` section, ordered by topic fit and book ID; its topic score has a different meaning from readiness score |

The configured minimum is inclusive: `0.65`. With the default `0.35` topic weight, this requires
at least `0.30` additional available component weight. This threshold is an illustrative policy
parameter, not a validated cutoff. Change it only through a new versioned policy configuration.
Eligible candidates can still have partial readiness evidence; the report labels every candidate
`full` or `partial` according to whether all configured component weight is available.

Each policy result contains every topic candidate, group membership, display position, position
within its group, movement relative to the renormalized display, coverage, unavailable components,
scores, component decomposition, active weights, existing explanation, and configuration metadata.
For excluded candidates, `diagnostic_renormalized_score` preserves the old calculation for audit;
it is not a readiness rank within the selected policy. `display_position_change` describes a
presentation move across groups, not an improvement in book quality or score.

Top-K comparison uses only the readiness group. If either policy has fewer than K eligible books,
the report marks the comparison unavailable instead of implying equal-length lists.

`/ml/rank` and the existing `rank` CLI still use the unmodified renormalized baseline. This
experimental report is a separate batch output; no Spring DTO contract changes are required.

## Reproduce

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl

uv run bookmatch-ml reader-profile \
  --input examples/assessment.json \
  --output data/output/reader_profile.json

uv run bookmatch-ml evaluate-ranking-policies \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --reader data/output/reader_profile.json \
  --output data/reports/ranking_policy_evaluation.json
```

The command validates all four canonical files and requires the profile artifact to contain
exactly their ten book IDs. Use `--ranking-config` and `--policy-config` to run another versioned
configuration. Generated profiles and reports remain ignored under `data/`.

## Input snapshot and reproducibility

The current local canonical handoff has 10 books, 31 documents, 1,163 TOC entries, and 52 sources.
Eight books have prose difficulty. Five books match the Operating Systems reader topic; one of
those five has no usable prose difficulty. The earlier [baseline evaluation](evaluation-baseline-v1.md)
recorded 28 documents, 1,091 TOC entries, 50 sources, six prose profiles, and two limited OS books.
This report reflects the current local snapshot; no upstream repository was changed.

| File or configuration | SHA-256 |
| --- | --- |
| `books.jsonl` | `92c4452516ff7d721220aa98c534da9e532fa8b5b18655c4d0755347eb53d77c` |
| `documents.jsonl` | `18e2144b7ff7249db920889eab39814054e81721c47dd92e7b604a696255109d` |
| `toc.jsonl` | `76f72b975f47effca5554987e4c56ad4e1986b6ddb91d4e4c748be18a0fe6851` |
| `sources.jsonl` | `13b008b86de0077fbe1f143f9f98741c5d95547b1d3237c770814a0d32c5f62e` |
| `ranking.yaml` | `4518b4af413d277e9785c4683050a467e586c35b9943513b014c6596f0d5847c` |
| `ranking_policies.yaml` | `b01a66e538f408f81bcea3435cfe3255389985aff0de57e6fd6d9017bad0316e` |

The book profile, reader profile, and policy report were regenerated and compared byte for byte
with a second run. All three comparisons matched.

## Observed Operating Systems result

The example reader comes from `examples/assessment.json`. The table shows readiness scores under
the unchanged baseline and group/position under either coverage policy. Scores are rounded for
display only. Both coverage policies have the same readiness order on this snapshot; their
evidence-limited sections communicate different meanings.

| Book | Baseline rank / score | Coverage | Coverage-policy group / display position |
| --- | ---: | ---: | --- |
| *Operating Systems* (`9780130319999`) | 1 / 1.0000 | 0.35 | `ineligible` or `topic_only` / 5 |
| *Operating Systems and Middleware* | 2 / 0.7370 | 1.00 | `readiness` / 1 |
| *Operating Systems: Three Easy Pieces* | 3 / 0.7051 | 1.00 | `readiness` / 2 |
| *xv6: a simple, Unix-like teaching operating system* | 4 / 0.6969 | 1.00 | `readiness` / 3 |
| *Think OS* | 5 / 0.6515 | 1.00 | `readiness` / 4 |

The limited *Operating Systems* book has no vocabulary, knowledge, or comprehension fit, so its
readiness score is `null` in the coverage policies. Its baseline `1.0` remains visible as a
diagnostic calculation. It is not interpreted as a low-scoring or low-quality book. The other
four OS books have all configured ranking components available in this snapshot.

At K=3, the renormalized readiness list is *Operating Systems*, *Operating Systems and
Middleware*, and *Three Easy Pieces*. The coverage-policy readiness list is *Operating Systems
and Middleware*, *Three Easy Pieces*, and *xv6*. One book leaves and one enters the readiness Top
3. Both comparisons are complete because four books meet the threshold. The two-stage topic-only
book is shown separately; its score should not be sorted together with readiness scores.

## Limits and next decision gate

The example assessment and historical difficulty labels are synthetic. There is no behavioral or
human judgment evidence that one policy gives better recommendations. This snapshot has only one
limited OS candidate and no eligible OS candidate with partial component coverage, so it does not
exercise a rich mix of intermediate coverage levels. Component availability also says nothing
about the reliability of an available prose sample or the validity of the fit formula.

Before choosing a default UI or API policy, review these group semantics with the product and
Spring teams, test how people interpret the separate topic-only section, and compare policies on
versioned human reader/book judgments. Keep the threshold configurable and retain the existing
renormalized baseline for comparison.
