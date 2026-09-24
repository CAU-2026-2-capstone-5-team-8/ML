# Matcher-v2 fresh holdout results v1

## Evaluation integrity

The 89-row General and 40-row Challenge holdouts were selected and prediction artifacts were
frozen before human labeling. Predictions remained hidden until both review files reached complete
status. This report evaluates the fixed artifacts without changing matcher-v1, matcher-v2 rules,
aliases, normalization, TOC-path logic, ranking, or the concept graph.

- Original frozen gold SHA-256: `477a83407c270ecbf019d07bb3970f282b8be868460f8df27f06c6a37d9e12d7`
- General manifest SHA-256: `3bc2504787e12ac6da2e66e1c49c23a07153b4aaf59f78a9154a78b49c7f2c02`
- Challenge manifest SHA-256: `036b412bed0e78af8595c2acbd385fc0ca3b774be82e47228232aa5a56935488`
- General prediction SHA-256: `032d4887591fde785297f2b45a360a7d0f14529b8b30942bb55a3a021c0b613e`
- Challenge prediction SHA-256: `b1fa0f289c6d0314f02e9e16e73cdcd379ab2f44bed1a746acfc4a4be1858319`
- General review SHA-256: `10879c31d896bdf8f9de2ca8a29bb3f4ad1cf75467c9adc1b89cc64c7fe35033`
- Challenge review SHA-256: `004d636241ae6487a7edd3414f97ba82aca2419c85d8538625e90157a0dc2bfc`
- Not judgable: 0 in both holdouts

Reproduce the machine-readable reports with:

```bash
uv run bookmatch-ml evaluate-evidence-concept-holdout \
  --manifest reviews/evidence_concept_holdout_general_v1_manifest.json \
  --predictions reviews/evidence_concept_holdout_general_v1_predictions.json \
  --review reviews/evidence_concept_holdout_general_v1.json \
  --output data/reports/matcher-v2-general-holdout-v1.json

uv run bookmatch-ml evaluate-evidence-concept-holdout \
  --manifest reviews/evidence_concept_holdout_challenge_v1_manifest.json \
  --predictions reviews/evidence_concept_holdout_challenge_v1_predictions.json \
  --review reviews/evidence_concept_holdout_challenge_v1.json \
  --output data/reports/matcher-v2-challenge-holdout-v1.json
```

The command rejects incomplete reviews, mismatched membership/provenance, and matcher/config hashes
that differ from the fixed predictions. Repeated runs are byte-identical.

## Overall results

### Original frozen gold, 94 rows

| Variant | TP | FP | FN | Precision | Recall | F1 | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 51 | 4 | 13 | 92.73% | 79.69% | 85.71% | 86.17% |
| v2-A | 51 | 1 | 13 | 98.08% | 79.69% | 87.93% | 88.30% |
| v2-B | 59 | 0 | 5 | 100.00% | 92.19% | 95.93% | 94.68% |
| v2-C1 | 64 | 4 | 0 | 94.12% | 100.00% | 96.97% | 95.74% |
| v2-C2 | 64 | 0 | 0 | 100.00% | 100.00% | 100.00% | 100.00% |

### Fresh General holdout, 89 rows

| Variant | TP | FP | FN | Precision | Recall | F1 | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 54 | 3 | 17 | 94.74% | 76.06% | 84.37% | 78.65% |
| v2-A | 54 | 0 | 17 | 100.00% | 76.06% | 86.40% | 82.02% |
| v2-B | 57 | 0 | 14 | 100.00% | 80.28% | 89.06% | 85.39% |
| v2-C1 | 63 | 6 | 8 | 91.30% | 88.73% | 90.00% | 88.76% |
| v2-C2 | 62 | 0 | 9 | 100.00% | 87.32% | 93.23% | 89.89% |

### Fresh Challenge holdout, 40 rows

| Variant | TP | FP | FN | Precision | Recall | F1 | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 34 | 3 | 17 | 91.89% | 66.67% | 77.27% | 60.00% |
| v2-A | 34 | 1 | 17 | 97.14% | 66.67% | 79.07% | 65.00% |
| v2-B | 42 | 2 | 9 | 95.45% | 82.35% | 88.42% | 85.00% |
| v2-C1 | 48 | 14 | 3 | 77.42% | 94.12% | 84.96% | 65.00% |
| v2-C2 | 45 | 2 | 6 | 95.74% | 88.24% | 91.84% | 90.00% |

## General holdout breakdown

### Topic

| Variant | Topic | Rows | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | --- | ---: | ---: | --- | ---: |
| v1 | Linear Algebra | 42 | 22/3/5 | 88.00% / 81.48% / 84.62% | 80.95% |
| v1 | Operating Systems | 47 | 32/0/12 | 100.00% / 72.73% / 84.21% | 76.60% |
| v2-A | Linear Algebra | 42 | 22/0/5 | 100.00% / 81.48% / 89.80% | 88.10% |
| v2-A | Operating Systems | 47 | 32/0/12 | 100.00% / 72.73% / 84.21% | 76.60% |
| v2-B | Linear Algebra | 42 | 23/0/4 | 100.00% / 85.19% / 92.00% | 90.48% |
| v2-B | Operating Systems | 47 | 34/0/10 | 100.00% / 77.27% / 87.18% | 80.85% |
| v2-C1 | Linear Algebra | 42 | 23/5/4 | 82.14% / 85.19% / 83.64% | 88.10% |
| v2-C1 | Operating Systems | 47 | 40/1/4 | 97.56% / 90.91% / 94.12% | 89.36% |
| v2-C2 | Linear Algebra | 42 | 23/0/4 | 100.00% / 85.19% / 92.00% | 90.48% |
| v2-C2 | Operating Systems | 47 | 39/0/5 | 100.00% / 88.64% / 93.98% | 89.36% |

### Evidence family

| Variant | Family | Rows | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | --- | ---: | ---: | --- | ---: |
| v1 | TOC | 38 | 32/1/14 | 96.97% / 69.57% / 81.01% | 63.16% |
| v1 | Metadata | 51 | 22/2/3 | 91.67% / 88.00% / 89.80% | 90.20% |
| v2-A | TOC | 38 | 32/0/14 | 100.00% / 69.57% / 82.05% | 65.79% |
| v2-A | Metadata | 51 | 22/0/3 | 100.00% / 88.00% / 93.62% | 94.12% |
| v2-B | TOC | 38 | 34/0/12 | 100.00% / 73.91% / 85.00% | 71.05% |
| v2-B | Metadata | 51 | 23/0/2 | 100.00% / 92.00% / 95.83% | 96.08% |
| v2-C1 | TOC | 38 | 40/6/6 | 86.96% / 86.96% / 86.96% | 78.95% |
| v2-C1 | Metadata | 51 | 23/0/2 | 100.00% / 92.00% / 95.83% | 96.08% |
| v2-C2 | TOC | 38 | 39/0/7 | 100.00% / 84.78% / 91.76% | 81.58% |
| v2-C2 | Metadata | 51 | 23/0/2 | 100.00% / 92.00% / 95.83% | 96.08% |

### Evidence type

Cells show `TP/FP/FN; P/R/F1; exact`. `subject` has empty gold and prediction on all 20 rows,
so its PRF is the defined zero-denominator value while exact match is 100%.

| Type | v1 | v2-A | v2-B | v2-C1 | v2-C2 |
| --- | --- | --- | --- | --- | --- |
| `toc_exact` (16) | 14/1/4; 93.33/77.78/84.85; 68.75 | 14/0/4; 100/77.78/87.50; 75.00 | 15/0/3; 100/83.33/90.91; 81.25 | 15/4/3; 78.95/83.33/81.08; 75.00 | 15/0/3; 100/83.33/90.91; 81.25 |
| `toc_public_web_exact` (16) | 17/0/6; 100/73.91/85.00; 62.50 | 17/0/6; 100/73.91/85.00; 62.50 | 18/0/5; 100/78.26/87.80; 68.75 | 20/2/3; 90.91/86.96/88.89; 75.00 | 20/0/3; 100/86.96/93.02; 81.25 |
| `toc_same_work` (6) | 1/0/4; 100/20.00/33.33; 50.00 | 1/0/4; 100/20.00/33.33; 50.00 | 1/0/4; 100/20.00/33.33; 50.00 | 5/0/0; 100/100/100; 100 | 4/0/1; 100/80.00/88.89; 83.33 |
| `description` (11) | 21/2/1; 91.30/95.45/93.33; 72.73 | 21/0/1; 100/95.45/97.67; 90.91 | 21/0/1; 100/95.45/97.67; 90.91 | 21/0/1; 100/95.45/97.67; 90.91 | 21/0/1; 100/95.45/97.67; 90.91 |
| `subject` (20) | 0/0/0; 0/0/0; 100 | 0/0/0; 0/0/0; 100 | 0/0/0; 0/0/0; 100 | 0/0/0; 0/0/0; 100 | 0/0/0; 0/0/0; 100 |
| `metadata_minimal` (20) | 1/0/2; 100/33.33/50.00; 90.00 | 1/0/2; 100/33.33/50.00; 90.00 | 2/0/1; 100/66.67/80.00; 95.00 | 2/0/1; 100/66.67/80.00; 95.00 | 2/0/1; 100/66.67/80.00; 95.00 |

## Challenge category breakdown

A row may belong to multiple categories and is evaluated in every assigned category. Category row
counts therefore overlap and must not be summed as if they were disjoint samples.

| Category | Rows | Variant | TP/FP/FN | Precision / Recall / F1 | Exact |
| --- | ---: | --- | ---: | --- | ---: |
| Nested overlap | 11 | v1 | 11/2/1 | 84.62% / 91.67% / 88.00% | 72.73% |
|  |  | v2-A | 11/0/1 | 100.00% / 91.67% / 95.65% | 90.91% |
|  |  | v2-B | 12/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
|  |  | v2-C1 | 12/1/0 | 92.31% / 100.00% / 96.00% | 90.91% |
|  |  | v2-C2 | 12/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| Morphological/lexical | 8 | v1 | 8/0/1 | 100.00% / 88.89% / 94.12% | 87.50% |
|  |  | v2-A | 8/0/1 | 100.00% / 88.89% / 94.12% | 87.50% |
|  |  | v2-B | 9/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
|  |  | v2-C1 | 9/5/0 | 64.29% / 100.00% / 78.26% | 50.00% |
|  |  | v2-C2 | 9/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
| Modifier insertion | 9 | v1 | 6/0/10 | 100.00% / 37.50% / 54.55% | 0.00% |
|  |  | v2-A | 6/0/10 | 100.00% / 37.50% / 54.55% | 0.00% |
|  |  | v2-B | 13/1/3 | 92.86% / 81.25% / 86.67% | 77.78% |
|  |  | v2-C1 | 15/1/1 | 93.75% / 93.75% / 93.75% | 77.78% |
|  |  | v2-C2 | 13/1/3 | 92.86% / 81.25% / 86.67% | 77.78% |
| Parent-overinheritance counterexample | 8 | v1 | 8/1/1 | 88.89% / 88.89% / 88.89% | 87.50% |
|  |  | v2-A | 8/1/1 | 88.89% / 88.89% / 88.89% | 87.50% |
|  |  | v2-B | 8/1/1 | 88.89% / 88.89% / 88.89% | 87.50% |
|  |  | v2-C1 | 9/10/0 | 47.37% / 100.00% / 64.29% | 0.00% |
|  |  | v2-C2 | 8/1/1 | 88.89% / 88.89% / 88.89% | 87.50% |
| Path-context candidate | 15 | v1 | 11/1/8 | 91.67% / 57.89% / 70.97% | 66.67% |
|  |  | v2-A | 11/1/8 | 91.67% / 57.89% / 70.97% | 66.67% |
|  |  | v2-B | 11/2/8 | 84.62% / 57.89% / 68.75% | 66.67% |
|  |  | v2-C1 | 17/12/2 | 58.62% / 89.47% / 70.83% | 26.67% |
|  |  | v2-C2 | 14/2/5 | 87.50% / 73.68% / 80.00% | 80.00% |
| Virtual-machine context | 4 | v1 | 4/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
|  |  | v2-A | 4/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
|  |  | v2-B | 4/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
|  |  | v2-C1 | 4/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |
|  |  | v2-C2 | 4/0/0 | 100.00% / 100.00% / 100.00% | 100.00% |

## Generalization findings

### v2-A

The overlap rule generalizes cleanly. General FP falls from 3 to 0 with TP and FN unchanged; all
three fresh General errors were `vector` nested inside `vector space`. Challenge FP falls from 3 to
1, with both exact `Vector Spaces` nested errors removed. The remaining Challenge FP is
`evidence_bcd3d5ad9077f2cc8077`, where `Spaces of Vectors` does not present the canonical phrase as
one containing span. No new v2-A error appears.

### v2-B

V2-B improves General recall from 76.06% to 80.28% while keeping 100% precision, so the low-risk
form rules do generalize beyond the original 94 rows. On Challenge, the morphological/lexical group
reaches 100% F1 and modifier insertion rises from 54.55% to 86.67% F1. It is not uniformly safe:
`evidence_8de49c7514eba5e8a1bb` introduces a `systems of equations` FP from the phrase `Systems of
Differential Equations`. The full v2-B bundle therefore needs rule-level separation before
production promotion.

### v2-C1 and v2-C2

C1's parent over-inheritance reproduces decisively. In the Challenge counterexample category it
creates 10 FP assignments, drops precision to 47.37%, and has 0% exact match. It is rejected.

C2 is substantially safer and improves on v2-B: General TP rises 57 to 62 without an FP, and
Challenge TP rises 42 to 45 with the same two errors classified as FP. In the Challenge
path-context category, recall rises from 57.89% to 73.68% and exact match from 66.67% to 80.00%.
However C2 is no longer perfect: overall it leaves 9 General FN and 6 Challenge FN. It preserves
the counterexamples as well as v2-B rather than introducing C1's parent FP, but the category still
contains one inherited FP and one FN. C2 therefore needs another broader path holdout before
production promotion.

All four fresh virtual-machine rows are genuine virtualization teaching contexts. Every variant,
including v2-B and C2, scores 4 TP, 0 FP, and 0 FN, so the exercise-environment exclusion did not
remove these positive cases. This holdout still lacks a fresh exercise-environment negative.

## Fresh error observations

- V1 General exposes 3 nested-span FP and 17 FN. The FN include path-dependent security,
  deadlock, virtual-memory, memory-management, and input/output cases plus unseen lexical forms.
- V2-B General has 0 FP and 14 FN. Remaining misses include `linear mapping`, `orthonormal`,
  `linear dependence`, broad distributed-system forms, and path-only concepts.
- V2-C2 General has 0 FP and 9 FN; its nearest-parent fallback cannot recover every deeper or
  semantically indirect path relation.
- V2-B Challenge has 2 FP and 9 FN. One FP is the remaining `Spaces of Vectors` overlap case; the
  other is the differential-equation modifier false match described above.
- V2-C1 Challenge has 14 FP, including inappropriate inheritance of `process`, `vector space`,
  `storage`, `security`, and `distributed systems` from ancestors.
- V2-C2 Challenge has 2 FP and 6 FN. It fixes path-supported `linear transformation`, `vector`,
  and one `vector space` assignment relative to v2-B but does not recover all deep-path cases.

The machine-readable reports retain every `(kind, evidence_id, concept_id)` error assignment.

## Production assessment

- **Sufficient evidence to promote:** v2-A overlap suppression. It reproduced on both fresh sets,
  removed FP without losing TP, and introduced no new error.
- **Needs rule-level validation:** v2-B. Its direct morphology and synonym behavior generalizes,
  but the modifier/non-contiguous bundle introduces a differential-equation FP. Promote only after
  splitting and independently testing the risky rule.
- **Needs more validation:** v2-C2. It clearly improves path recall without C1's broad inheritance,
  but its original 100% result falls to 93.23% General F1 and 91.84% Challenge F1.
- **Reject:** v2-C1 naive path union. Fresh counterexamples confirm severe over-inheritance.

No matcher variant is changed or promoted by this report.
