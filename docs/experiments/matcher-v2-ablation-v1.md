# Matcher-v2 frozen-gold ablation

## Scope and reproducibility

This experiment changes neither production matcher-v1 nor ranking, reader profiles, book-profile
scoring, source weights, the concept graph, upstream evidence, or human gold. It evaluates isolated
deterministic variants against the frozen 94-row review.

- Frozen artifact: `reviews/evidence_concept_gold_review_v1.json`
- Start/end SHA-256: `477a83407c270ecbf019d07bb3970f282b8be868460f8df27f06c6a37d9e12d7`
- Evaluable rows: 94; not judgable: 0
- Experiment config: `configs/matcher_v2_experiments.yaml`
- Experiment version: `matcher-v2-ablation-v1`

Reproduce the machine-readable report with:

```bash
uv run bookmatch-ml evaluate-matcher-v2-experiments \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --review reviews/evidence_concept_gold_review_v1.json \
  --output data/reports/matcher-v2-ablation-v1.json
```

Every policy still unions concept presence into a set per book. Raw row occurrences remain a
diagnostic and are not a score or weight.

## Variants

- **v1:** frozen matcher-v1 predictions.
- **v2-A:** v1 aliases plus span-aware overlap suppression. A shorter nested match is suppressed
  only when its canonical concept is part of the longer canonical concept; independent occurrences
  and distinct concepts such as `thread` plus `programming` remain valid.
- **v2-B:** v2-A plus reviewed synonyms/compound forms, bounded ordered-token patterns, and one
  deterministic description-context exclusion distinguishing a supplied virtual-machine exercise
  environment from virtualization teaching context.
- **v2-C1:** diagnostic union of v2-B matches from the leaf and every TOC path component.
- **v2-C2:** v2-B leaf matches remain primary. Only an unmatched, non-generic leaf may fall back to
  the nearest matching parent. Generic leaves such as `Exercises` do not inherit a parent concept.

No evidence ID, ISBN, or book title is present in a matcher rule.

## Overall results

| Variant | TP | FP | FN | Precision | Recall | F1 | Exact | Fixed v1 assignments | New assignments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 51 | 4 | 13 | 92.73% | 79.69% | 85.71% | 86.17% | 0 | 0 |
| v2-A | 51 | 1 | 13 | 98.08% | 79.69% | 87.93% | 88.30% | 3 | 0 |
| v2-B | 59 | 0 | 5 | 100.00% | 92.19% | 95.93% | 94.68% | 12 | 0 |
| v2-C1 diagnostic | 64 | 4 | 0 | 94.12% | 100.00% | 96.97% | 95.74% | 17 | 4 |
| v2-C2 assisted | 64 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% | 17 | 0 |

Relative to v1, v2-A changes only precision (+5.35 percentage points) and removes the three nested
`vector` false positives without losing a TP. V2-B adds eight TPs, removes the remaining FP, and
introduces no error on this sample. C1 proves that unconditional path union over-inherits parent
concepts. C2 avoids the four observed C1 errors, but its perfect score is in-sample and must not be
treated as proof of out-of-sample perfection.

## Topic results

| Variant | Topic | Rows | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | --- | ---: | ---: | --- | ---: |
| v1 | Linear Algebra | 44 | 27/3/4 | 90.00% / 87.10% / 88.52% | 88.64% |
| v1 | Operating Systems | 50 | 24/1/9 | 96.00% / 72.73% / 82.76% | 84.00% |
| v2-A | Linear Algebra | 44 | 27/0/4 | 100.00% / 87.10% / 93.10% | 93.18% |
| v2-A | Operating Systems | 50 | 24/1/9 | 96.00% / 72.73% / 82.76% | 84.00% |
| v2-B | Linear Algebra | 44 | 30/0/1 | 100.00% / 96.77% / 98.36% | 97.73% |
| v2-B | Operating Systems | 50 | 29/0/4 | 100.00% / 87.88% / 93.55% | 92.00% |
| v2-C1 | Linear Algebra | 44 | 31/2/0 | 93.94% / 100.00% / 96.88% | 95.45% |
| v2-C1 | Operating Systems | 50 | 33/2/0 | 94.29% / 100.00% / 97.06% | 96.00% |
| v2-C2 | Linear Algebra | 44 | 31/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | Operating Systems | 50 | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |

## Evidence-family results

| Variant | Family | Rows/books | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | --- | ---: | ---: | --- | ---: |
| v1 | TOC | 38/19 | 21/2/10 | 91.30% / 67.74% / 77.78% | 73.68% |
| v1 | Metadata | 56/42 | 30/2/3 | 93.75% / 90.91% / 92.31% | 94.64% |
| v2-A | TOC | 38/19 | 21/0/10 | 100.00% / 67.74% / 80.77% | 78.95% |
| v2-A | Metadata | 56/42 | 30/1/3 | 96.77% / 90.91% / 93.75% | 94.64% |
| v2-B | TOC | 38/19 | 26/0/5 | 100.00% / 83.87% / 91.23% | 86.84% |
| v2-B | Metadata | 56/42 | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C1 | TOC | 38/19 | 31/4/0 | 88.57% / 100.00% / 93.94% | 89.47% |
| v2-C1 | Metadata | 56/42 | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | TOC | 38/19 | 31/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | Metadata | 56/42 | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |

## Evidence-type results

Rows/books are invariant across variants: `toc_exact` 16/3, `toc_public_web_exact` 16/15,
`toc_same_work` 6/1, `description` 16/16, `subject` 20/20, and `metadata_minimal` 20/20.
The one-book same-work result is especially unsuitable for generalization.

| Variant | Type | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | --- | ---: | --- | ---: |
| v1 | `toc_exact` | 11/1/5 | 91.67% / 68.75% / 78.57% | 75.00% |
| v1 | `toc_public_web_exact` | 8/1/3 | 88.89% / 72.73% / 80.00% | 75.00% |
| v1 | `toc_same_work` | 2/0/2 | 100.00% / 50.00% / 66.67% | 66.67% |
| v1 | `description` | 30/2/3 | 93.75% / 90.91% / 92.31% | 81.25% |
| v2-A | `toc_exact` | 11/0/5 | 100.00% / 68.75% / 81.48% | 81.25% |
| v2-A | `toc_public_web_exact` | 8/0/3 | 100.00% / 72.73% / 84.21% | 81.25% |
| v2-A | `toc_same_work` | 2/0/2 | 100.00% / 50.00% / 66.67% | 66.67% |
| v2-A | `description` | 30/1/3 | 96.77% / 90.91% / 93.75% | 81.25% |
| v2-B | `toc_exact` | 15/0/1 | 100.00% / 93.75% / 96.77% | 93.75% |
| v2-B | `toc_public_web_exact` | 9/0/2 | 100.00% / 81.82% / 90.00% | 87.50% |
| v2-B | `toc_same_work` | 2/0/2 | 100.00% / 50.00% / 66.67% | 66.67% |
| v2-B | `description` | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C1 | `toc_exact` | 16/2/0 | 88.89% / 100.00% / 94.12% | 87.50% |
| v2-C1 | `toc_public_web_exact` | 11/1/0 | 91.67% / 100.00% / 95.65% | 93.75% |
| v2-C1 | `toc_same_work` | 4/1/0 | 80.00% / 100.00% / 88.89% | 83.33% |
| v2-C1 | `description` | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | `toc_exact` | 16/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | `toc_public_web_exact` | 11/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | `toc_same_work` | 4/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| v2-C2 | `description` | 33/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |

`subject` and `metadata_minimal` remain 0/0/0 with 100% exact-set match for every variant because
both gold and prediction are empty on all sampled rows. Their zero PRF is the defined empty-
denominator behavior, not an error.

## Book-level policy results

### TOC only

| Variant | Books | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | ---: | ---: | --- | ---: |
| v1 | 19 | 19/2/33 | 90.48% / 36.54% / 52.05% | 31.58% |
| v2-A | 19 | 19/0/33 | 100.00% / 36.54% / 53.52% | 36.84% |
| v2-B | 19 | 24/0/28 | 100.00% / 46.15% / 63.16% | 47.37% |
| v2-C1 | 19 | 28/3/24 | 90.32% / 53.85% / 67.47% | 57.89% |
| v2-C2 | 19 | 28/0/24 | 100.00% / 53.85% / 70.00% | 68.42% |

### Metadata only

| Variant | Books | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | ---: | ---: | --- | ---: |
| v1 | 42 | 30/2/7 | 93.75% / 81.08% / 86.96% | 85.71% |
| v2-A | 42 | 30/1/7 | 96.77% / 81.08% / 88.24% | 85.71% |
| v2-B | 42 | 33/0/4 | 100.00% / 89.19% / 94.29% | 90.48% |
| v2-C1 | 42 | 33/0/4 | 100.00% / 89.19% / 94.29% | 90.48% |
| v2-C2 | 42 | 33/0/4 | 100.00% / 89.19% / 94.29% | 90.48% |

### Combined unweighted

| Variant | Books | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | ---: | ---: | --- | ---: |
| v1 | 50 | 47/4/12 | 92.16% / 79.66% / 85.45% | 78.00% |
| v2-A | 50 | 47/1/12 | 97.92% / 79.66% / 87.85% | 80.00% |
| v2-B | 50 | 55/0/4 | 100.00% / 93.22% / 96.49% | 92.00% |
| v2-C1 | 50 | 59/3/0 | 95.16% / 100.00% / 97.52% | 94.00% |
| v2-C2 | 50 | 59/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |

TOC-only policy recall remains much lower than row-level TOC recall because policy gold is the
union of all reviewed evidence for each TOC-eligible book, including concepts supported only by
metadata. This is intentional and unchanged from v1.

## Fixed, remaining, and new assignments

### v2-A

Fixed three FP assignments, all `vector` nested inside a matched `vector space` span:

- `evidence_c8c7b682c2eed436001f`
- `evidence_5592aeb1467ecf822bf1`
- `evidence_182a7e404a4f12fa1ac5`

The other fourteen v1 errors remain. No new error was introduced.

### v2-B

In addition to v2-A, fixed these eight FN assignments:

- `evidence_c8c7b682c2eed436001f`: `linear transformation`
- `evidence_4b9a2f0defd52d547285`: `thread`, `programming`
- `evidence_f766cef8915333882364`: `programming`
- `evidence_1e1cc0b4b9ce520c7cf5`: `distributed systems`
- `evidence_3c1565a9d77c9dd3daa6`: `storage`
- `evidence_12221ead4d7ec7b863d2`: `systems of equations`, `linear system`

It also fixed the `virtualization` FP on `evidence_f766cef8915333882364`. Five path-context FN
assignments remain: `evidence_b9952c679f8c56aab6d6`, `evidence_d5c382bdb71af9646639`,
`evidence_0f0eccedaa371bd45f2e`, `evidence_63f7bae02fa267641db0`, and
`evidence_bd8f6c857f52d62edc60`. No new error was introduced.

### v2-C1 diagnostic

C1 fixes all seventeen v1 errors but introduces four parent-over-inheritance false positives:

- `evidence_312c93199bd1a1d6678c`: `thread`
- `evidence_5176c52fbf6563aff697`: `vector space`
- `evidence_91d97f6829244e0cfb4d`: `vector space`
- `evidence_ed34dab97c4524d0c95f`: `storage`

This variant is rejected as a production policy.

### v2-C2 assisted

C2 fixes the five remaining path FN assignments without an observed new error. No v1 assignment
remains wrong in this 94-row sample. The result is promising but high-risk for over-interpretation:
the path policy was designed after inspecting the same small gold set, `toc_same_work` represents
one book, and there is no independent holdout annotation yet.

## Generalization and adoption assessment

The span-overlap rule is the strongest production candidate: it is structural, fixes exactly the
three repeated nested matches, preserves independently positioned concepts, and introduces no
error. Reviewed direct synonyms and bounded morphology are also explainable, but they need a fresh
holdout with negative uses of `programming`, `storage`, and modifier-separated phrases.

The virtual-machine context exclusion is deterministic and passes both exercise-environment and
teaching-context tests, but it is supported by one gold error and should stay experimental until
more descriptions are labeled. Broad stemming, fuzzy matching, embeddings, learned thresholds,
and evidence-specific exceptions were not implemented.

C1 is rejected because it trades recall for four known false positives. C2 is a better structural
candidate than C1, but its nearest-matching-parent rule should remain behind an experiment flag
until evaluated on a larger path-focused holdout containing generic children, unrelated subsection
titles, and deeper mixed-concept paths.

## Minimum next production-candidate set

1. Promote the v2-A canonical-aware span-overlap policy.
2. Separately ablate and validate the low-risk v2-B direct synonym/morphology rules on a fresh
   holdout before promotion; keep the virtual-machine exclusion independently switchable.
3. Do not promote path matching yet. Build a path-focused human-reviewed holdout, then compare C2
   against leaf-only v2-B without changing this frozen gold.
