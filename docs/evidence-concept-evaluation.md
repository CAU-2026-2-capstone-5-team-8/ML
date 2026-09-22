# Evidence concept human evaluation

## Purpose

This framework measures how well the existing deterministic concept matcher maps each
`book-evidence-v1` source category. It does not tune aliases, assign source weights, change the
concept graph, or alter ranking v1.

The previous `reviews/toc_concept_gold_review.json` is a 100-row blank review template: all 100
rows are `unreviewed`, no row has a human gold concept, and no row has a review note. It remains
the pinned evaluation for the older canonical TOC input. The source-aware evaluation uses the
separate, versioned `reviews/evidence_concept_gold_review_v1.json` artifact.

## Deterministic sample

The v1 sample requests 94 entries:

| Evidence type | Requested rows |
| --- | ---: |
| `toc_same_work` | 6 |
| `toc_exact` | 16 |
| `toc_public_web_exact` | 16 |
| `description` | 16 |
| `subject` | 20 |
| `metadata_minimal` | 20 |

Sampling is stratified rather than proportional to source row counts. It alternates topics and
books, caps the complete sample at eight rows per book, deduplicates identical normalized text
within a book/source group, and retains at most two generic TOC titles per TOC source group.
Generic titles are controlled rather than removed because false positives on headings such as
`Introduction`, `Summary`, and `Exercises` are useful evaluation cases.

The same input artifact and configs produce byte-identical JSON. The review pins the exact
`book-evidence-v1` hash, graph hash, feature hash, matching-config hash, generated predictions,
and evidence provenance fields.

## Prediction semantics

Each evidence row is matched independently with the existing `normalized_alias_phrase_v1`
matcher. TOC predictions use the evidence text, exactly as the existing matcher uses a TOC leaf
title. The full TOC path is retained for human context but is not silently added to model input.
Descriptions, subjects, and titles use the same aliases and exclusions without new thresholds,
embeddings, or LLM classification.

`prediction_matches` records the canonical concept ID, matching alias, and match method. An
ambiguous alias produces no prediction and is recorded as `ambiguous_alias`.

## Human decisions

An entry starts with:

```json
{
  "human_gold_concept_ids": [],
  "review_status": "unreviewed",
  "review_outcome": "pending"
}
```

Only canonical graph concept IDs are valid. A reviewer chooses one of:

- `labeled`: one or more canonical concepts are supported by the evidence.
- `no_concept`: the evidence supports none of the configured concepts.
- `not_judgable`: the evidence is too ambiguous to label; it is reviewed but excluded from
  metrics.

One decision can be applied without editing the whole file:

```bash
uv run bookmatch-ml review-evidence-concepts \
  --input PATH/TO/book-evidence.jsonl \
  --review reviews/evidence_concept_gold_review_v1.json \
  --evidence-id evidence_0123456789abcdef0123 \
  --outcome labeled \
  --gold "virtual memory,memory management" \
  --note "The heading explicitly names both concepts."
```

Use `--outcome no_concept` with no `--gold` for a reviewed empty set. Use
`--outcome not_judgable` and a note when context is insufficient. The command validates the
current evidence artifact and generated prediction before atomically updating one row.

## Metrics

Pending entries are never interpreted as empty gold. With no judgeable human labels, evaluation
returns `metric_status: unavailable`, progress counts, and empty metric lists. A partly reviewed
file returns `partial`; a fully reviewed file returns `complete`.

For judgeable rows the report contains exact-set match rate and micro, row-macro, and book-macro
precision/recall/F1 for:

- all evidence;
- each topic;
- each evidence type;
- TOC and metadata families.

Book-macro metrics give each represented book equal weight. Empty predicted or gold denominators
produce zero precision or recall; exact-set match separately credits two empty sets.

After the full sample is reviewed, policy diagnostics compare `toc_only`, `metadata_only`, and
`combined_unweighted`. They remain unavailable during partial review so missing labels cannot
distort a source comparison. Within a book, concept IDs are set-unioned before scoring, so twelve
matching TOC rows still mean only `book covers concept = true`. Raw occurrence count is diagnostic
only and is never a score or weight. Until reviews are complete, policy metrics are unavailable
rather than fabricated.

## Frozen v1 human-review baseline

The 94-row review was completed on 2026-09-22 and is frozen as
`evidence-concept-gold-review-v1` before any matcher-v2 changes:

- total/reviewed/evaluable: 94/94/94;
- remaining/not judgable: 0/0;
- review file hash:
  `sha256:477a83407c270ecbf019d07bb3970f282b8be868460f8df27f06c6a37d9e12d7`;
- prediction/evidence projection hash:
  `edcbe7e9365c4e7b8eb329ca8a074f29d0ea24ddca4a1c9ce1ff83e9b1f8918f`.

The labels used AI-assisted presentation and labeling guidance, but every stored decision was
explicitly approved by the human reviewer. This is not an independently annotated, multi-expert
dataset. The gold labels were fixed before matcher improvement. Later matcher, alias, or input
changes must be evaluated against this same artifact; the gold must not be rewritten to agree with
a changed matcher.

Reproduce the complete evaluation with:

```bash
uv run bookmatch-ml evaluate-evidence-concept-gold \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --review reviews/evidence_concept_gold_review_v1.json \
  --output data/reports/evidence-concept-gold-evaluation-v1.json
```

### Matcher-v1 results

All values below use 94 evaluable rows. TP/FP/FN are assignment counts, not row counts.

| Scope | Rows | Books | TP | FP | FN | Micro P | Micro R | Micro F1 | Exact set | Row macro P/R/F1 | Book macro P/R/F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| All | 94 | 50 | 51 | 4 | 13 | 92.73% | 79.69% | 85.71% | 86.17% | 25.38% / 26.29% / 25.65% | 27.05% / 25.10% / 25.67% |
| Linear Algebra | 44 | 25 | 27 | 3 | 4 | 90.00% | 87.10% | 88.52% | 88.64% | 26.95% / 29.22% / 27.71% | 28.10% / 28.43% / 27.81% |
| Operating Systems | 50 | 25 | 24 | 1 | 9 | 96.00% | 72.73% | 82.76% | 84.00% | 24.00% / 23.71% / 23.85% | 26.00% / 21.76% / 23.52% |
| `toc_exact` | 16 | 3 | 11 | 1 | 5 | 91.67% | 68.75% | 78.57% | 75.00% | 53.12% / 56.25% / 54.17% | 88.89% / 63.89% / 74.29% |
| `toc_public_web_exact` | 16 | 15 | 8 | 1 | 3 | 88.89% | 72.73% | 80.00% | 75.00% | 46.88% / 50.00% / 47.92% | 43.33% / 46.67% / 44.44% |
| `toc_same_work` | 6 | 1 | 2 | 0 | 2 | 100.00% | 50.00% | 66.67% | 66.67% | 33.33% / 33.33% / 33.33% | 100.00% / 66.67% / 80.00% |
| `description` | 16 | 16 | 30 | 2 | 3 | 93.75% | 90.91% | 92.31% | 81.25% | 36.61% / 35.71% / 36.13% | 36.61% / 35.71% / 36.13% |
| `subject` | 20 | 20 | 0 | 0 | 0 | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| `metadata_minimal` | 20 | 20 | 0 | 0 | 0 | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| TOC family | 38 | 19 | 21 | 2 | 10 | 91.30% | 67.74% | 77.78% | 73.68% | 47.37% / 50.00% / 48.25% | 53.51% / 50.44% / 51.03% |
| Metadata family | 56 | 42 | 30 | 2 | 3 | 93.75% | 90.91% | 92.31% | 94.64% | 10.46% / 10.20% / 10.32% | 13.95% / 13.61% / 13.76% |

The zero precision/recall/F1 for `subject` and `metadata_minimal` does not indicate incorrect
predictions: every sampled row had both an empty prediction and empty gold set, producing 100%
exact-set match. Macro PRF defines empty denominators as zero. `toc_same_work` has only six rows
from one book, and `toc_exact` has only three represented books, so those values must not be
generalized beyond this sample.

### Book-level source policies

Each policy unions prediction and gold concept IDs into sets per book. Gold is aggregated from all
reviewed evidence for each eligible book, while predictions are restricted to the policy's allowed
source types. Raw prediction occurrences are reported only to verify deduplication.

| Policy | Books | TP | FP | FN | Micro P/R/F1 | Exact set | Book macro P/R/F1 | Raw / unique predicted assignments |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: |
| TOC only | 19 | 19 | 2 | 33 | 90.48% / 36.54% / 52.05% | 31.58% | 53.51% / 39.79% / 42.69% | 23 / 21 |
| Metadata only | 42 | 30 | 2 | 7 | 93.75% / 81.08% / 86.96% | 85.71% | 13.95% / 13.01% / 13.42% | 32 / 32 |
| Combined unweighted | 50 | 47 | 4 | 12 | 92.16% / 79.66% / 85.45% | 78.00% | 27.05% / 25.10% / 25.67% | 55 / 51 |

The policy comparison measures more than source-local row accuracy. In particular, TOC-only gold
still represents all reviewed concepts for each TOC-eligible book, which exposes concepts supported
only by metadata as false negatives. Metadata-only has higher recall in this sample because the
16 descriptions contain dense multi-concept evidence; field-level subjects and titles contribute
only correct empty sets. Combined covers all 50 books and recovers more assignments than either
policy alone, but it also unions their false positives. No source weights or occurrence scores are
used.

### Matcher-v1 failure taxonomy

The complete review contains four false-positive and thirteen false-negative concept assignments.

| Error | Assignments | Evidence IDs | Representative case | Likely remedy |
| --- | ---: | --- | --- | --- |
| FP: nested alias substring | 3 | `evidence_c8c7b682c2eed436001f`, `evidence_5592aeb1467ecf822bf1`, `evidence_182a7e404a4f12fa1ac5` | `vector spaces` also emits `vector` | Not a new alias. Prefer overlap/longest-phrase handling; context is needed to avoid suppressing legitimate separate mentions. |
| FP: phrase used as environment, not taught concept | 1 | `evidence_f766cef8915333882364` | `Linux virtual machine` emits `virtualization` although it describes the exercise environment | Alias or normalization alone is insufficient; local semantic context or a carefully tested exclusion is required. |
| FN: direct synonym missing | 1 | `evidence_c8c7b682c2eed436001f` | `linear maps` should support `linear transformation` | A reviewed alias can solve this case; no TOC path is needed. |
| FN: lexical/morphological alias coverage | 3 | `evidence_4b9a2f0defd52d547285` (2), `evidence_f766cef8915333882364` | `Multithreaded programming`; `programming exercises` | Alias additions and/or conservative morphological normalization can solve these direct mentions; verify resulting precision. |
| FN: non-contiguous or rephrased concept | 2 | `evidence_12221ead4d7ec7b863d2` (2) | `Systems of Two Linear Equations...` should support `systems of equations` and `linear system` | Structured token-pattern normalization may help; a literal alias alone would be brittle. The TOC parent also supplies useful confirmation. |
| FN: domain modifier inserted inside alias | 1 | `evidence_1e1cc0b4b9ce520c7cf5` | `distributed UNIX systems` should support `distributed systems` | A constrained modifier-tolerant pattern or reviewed alias can solve it; full semantic classification is not required for this example. |
| FN: semantic/form variation | 1 | `evidence_3c1565a9d77c9dd3daa6` | `Multimedia Storage` should support `storage` | A reviewed alias/pattern may solve it, but bare `storage` is broader and requires precision checks. |
| FN: parent/full TOC path omitted from matcher input | 5 | `evidence_b9952c679f8c56aab6d6`, `evidence_d5c382bdb71af9646639`, `evidence_0f0eccedaa371bd45f2e`, `evidence_63f7bae02fa267641db0`, `evidence_bd8f6c857f52d62edc60` | `Permutations and Cofactors` under `Determinants`; `Hardware Issues` under `Input/Output` | Alias/leaf normalization is insufficient. A matcher-v2 path-context experiment is required, with safeguards against blindly copying every parent concept. |

### Data-quality observations

Two sampled records are explicitly companion material rather than ordinary textbook editions:

| ISBN-13 | Canonical title | Canonical subtitle | Provider/source | Evidence type |
| --- | --- | --- | --- | --- |
| `9780132819824` | *Introductory Linear Algebra With Applications* | *Students Solutions Manual* | Open Library `OL7340360M`, metadata API | `metadata_minimal` |
| `9780716721772` | *Linear algebra* | *a study guide and solutions manual for students and instructors* | Open Library `OL21681769M`, metadata API | `metadata_minimal` |

The subtitles are strong evidence that these are separate solution/study-guide companion editions,
not merely malformed title strings. They should not silently contribute evidence as if they were
the primary textbook. For a shelf-photo product they may still be valid identified physical books,
so a later Data-Pipeline rule should classify or flag publication role (`solution_manual`,
`study_guide`, and similar) rather than destructively dropping the record. This is an observation
and validation-rule candidate only; the canonical data was not changed during this review.
