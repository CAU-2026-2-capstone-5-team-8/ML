# Matcher-v2 fresh holdouts v1

## Purpose

These holdouts test whether the matcher-v2 ablation results generalize beyond the frozen 94-row
gold set. They do not change production matcher-v1, ranking, the concept graph, or the frozen gold.
No holdout metric is available until a person labels the new rows.

## Leakage controls

The build is deliberately split into two commands:

1. `build-evidence-concept-holdout-manifests` selects membership from evidence source, topic,
   book, normalized text, and TOC structure only. It accepts no matcher or matcher-v2 config.
2. The manifest files and hashes are persisted before
   `build-evidence-concept-holdout-reviews` generates predictions.
3. Predictions for v1, v2-A, v2-B, v2-C1, and v2-C2 live in separate artifacts.
4. Human-review JSON and the default packet contain no prediction fields. The review file only
   pins the prediction artifact hash so evaluation can later prove which rules were frozen.
5. Once human labeling begins, matcher/config changes require a new versioned holdout rather than
   rewriting these artifacts.

The exclusion source is the complete frozen artifact
`reviews/evidence_concept_gold_review_v1.json` (SHA-256
`477a83407c270ecbf019d07bb3970f282b8be868460f8df27f06c6a37d9e12d7`). Its 94 evidence IDs are
absent from both holdouts. General and challenge memberships are also disjoint.

## General holdout

The general set is a deterministic, prediction-blind, source-stratified sample. It uses a
per-book cap of 8, normalized within-book duplicate removal, and a generic-TOC-heading cap of 2
per source group.

| Dimension | Count |
| --- | ---: |
| Total | 89 |
| Linear Algebra | 42 |
| Operating Systems | 47 |
| Books | 42 |
| Maximum rows from one book | 8 |
| `toc_exact` | 16 |
| `toc_public_web_exact` | 16 |
| `toc_same_work` | 6 |
| `description` | 11 |
| `subject` | 20 |
| `metadata_minimal` | 20 |

Only 11 eligible descriptions remain after excluding the frozen set, so the general set does not
force an arbitrary 100-row total. Same-Work TOC evidence is naturally concentrated in one source
book; its group is therefore kept small instead of being presented as broad book coverage.

Manifest SHA-256:
`3bc2504787e12ac6da2e66e1c49c23a07153b4aaf59f78a9154a78b49c7f2c02`.

## Challenge holdout

The challenge set contains 40 rows across 20 books, with no book contributing more than 6 rows.
Category membership is assigned from deterministic lexical and TOC-structure patterns, not from
whether any matcher variant succeeds or fails.

| Category | Assignments |
| --- | ---: |
| Path-context candidate | 15 |
| Nested/overlapping phrase | 11 |
| Modifier insertion | 9 |
| Parent over-inheritance counterexample | 8 |
| Morphological/lexical variant | 8 |
| Virtual-machine context | 4 |
| Independent co-occurrence | 0 |

Rows may have more than one category, so assignments do not sum to 40. The remaining fresh
evidence contains no natural row satisfying the conservative independent co-occurrence detector;
the set records zero rather than synthesizing evidence or weakening the definition. The four
fresh virtual-machine rows are instructional TOC contexts. No fresh public evidence provides the
opposite “mere exercise environment” use, so that negative side of the requested contrast remains
a documented limitation.

Evidence types are 5 `toc_exact`, 28 `toc_public_web_exact`, 6 `toc_same_work`, and 1
`metadata_minimal`. Topics are 15 Linear Algebra and 25 Operating Systems.

Manifest SHA-256:
`036b412bed0e78af8595c2acbd385fc0ca3b774be82e47228232aa5a56935488`.

## Prediction and review artifacts

| Holdout | Prediction artifact SHA-256 | Review rows |
| --- | --- | ---: |
| General | `032d4887591fde785297f2b45a360a7d0f14529b8b30942bb55a3a021c0b613e` | 89 |
| Challenge | `b1fa0f289c6d0314f02e9e16e73cdcd379ab2f44bed1a746acfc4a4be1858319` | 40 |

Both review templates start at 0 reviewed and expose
`prediction_visibility: hidden_during_human_review`. Human decisions are applied with
`review-evidence-concept-holdout`; that command validates canonical concept IDs and never reads or
prints predictions.

No precision, recall, F1, exact-match, or policy result is calculated at this stage.
