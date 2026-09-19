# Concept graph and TOC mapping validation v1

## Purpose and boundary

Concept matching v2 remains unchanged. This validation slice exposes the evidence behind its
two most uncertain inputs: proposed prerequisite edges and exact TOC alias mappings. It does not
produce a recommendation score, modify `rank-v1`, change v2 readiness or opportunity formulas,
or alter `/ml/rank`.

`configs/concept_graph.yaml` is a proposed seed graph rather than prerequisite ground truth.
Every current edge retains `source_type: proposed_seed`. A TOC observation can show that two
headings occur in an order; it cannot establish that the prerequisite relation is educationally
valid. The graph config, generated evidence, and future human decisions remain separate.

## Reproduce the artifact

First regenerate the existing v2 inputs as documented in
[Concept matching v2](concept-matching-v2.md), including the synthetic LA reader profile. Then:

```bash
uv run bookmatch-ml build-concept-review \
  --data-dir ../Data-Pipeline/data/processed \
  --reader examples/reader_profile.json \
  --reader data/output/concept_matching_la_reader.json \
  --output data/reports/concept_validation.json \
  --review-template data/reviews/concept_graph_review.json
```

The report records hashes of all four canonical files and any reader profiles. It rebuilds the
same v2 Book Concept Profiles from canonical evidence. Generated artifacts under `data/` are
ignored. The committed `examples/concept_graph_review.json` demonstrates the strict review
schema and contains only `unreviewed` rows.

## Reading graph edge evidence

Each of the 24 graph edges shows topic, prerequisite, dependent, relation and source types, edge
version, books containing the dependent, and all five topic books. Each book observation is one
of:

- `observed_before_in_toc`: both concepts mapped and prerequisite first position is lower;
- `observed_at_or_after_in_toc`: both mapped at the same or reversed position;
- `prerequisite_not_mapped`: dependent mapped but prerequisite did not map;
- `dependent_not_mapped`: dependent did not map, regardless of prerequisite presence.

For every mapped side, `prerequisite_first` or `dependent_first` includes the TOC entry ID,
title, full path, level, depth-first position, matching alias, and occurrence count. These are
observations for review. No `valid`, `invalid`, `correct`, or `wrong` field is generated.

The per-book projected candidate queue also expands v2's unchanged first-occurrence heuristic.
It shows the prerequisite's first occurrence and every related target's first occurrence rather
than only the `taught_before_use_candidate` or `external_or_late_candidate` label. A heading in
an overview or a combined heading may trigger an early occurrence without teaching the concept.

## Reading the mapping queues

Every mapped row retains book identity, topic, TOC entry and source IDs, title, full path,
canonical concept, matching alias and method, level, traversal position, and the concept's total
occurrence count in that book. One TOC entry may have multiple rows when it explicitly contains
multiple configured concepts.

Every non-mapped TOC entry remains in `unmatched_entries` with its full path, position, and either
`unmatched` or `ambiguous_alias`. The builder does not guess a likely concept. The book summary
contains total, matched, unmatched, ambiguous, distinct concept, and mapping-rate values plus up
to twelve unmatched titles sampled across root branches. The full queue remains the source for
review.

Mapping rate is **not mapping quality**. A detailed TOC contains summaries, exercises, examples,
proper names, and implementation sections outside the deliberately small lexicon. A coarse TOC
can have a high rate while offering little evidence. Human reviewers must separately label false
positives among matches and false negatives among unmatched rows before changing aliases.

## Human review schema and workflow

The optional template has one row for every configured edge and starts with `status:
unreviewed`. Allowed statuses are `unreviewed`, `accepted`, `rejected`, and `uncertain`.
Non-unreviewed decisions require a reviewer ID, timezone-aware review timestamp, and rationale.
Duplicate edge decisions are rejected.

Generated templates belong in `data/reviews/`. A completed human review should be copied in a
dedicated later change to `reviews/concept_graph_review.json`. Matching code does not load that
file. An accepted decision therefore does not change recommendation output or silently promote
an edge. Changing `source_type` to `human_reviewed_seed` requires an explicit later graph config
and version change.

## Structural results on the current ten books

These are reproducible structural measurements, not recommendation accuracy or educational
validity.

| Topic | TOC matched / total | Mapping rate | Mapped graph concepts | Edges jointly mapped in ≥1 book | Edges not jointly mapped | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Operating Systems | 182 / 606 | 0.300 | 16 / 17 | 9 / 10 | 1 | 0 |
| Linear Algebra | 205 / 557 | 0.368 | 19 / 20 | 13 / 14 | 1 | 0 |

Across the five books per topic, direct edge/book observations were:

| Topic | Before | Same or after | Prerequisite missing | Dependent missing |
| --- | ---: | ---: | ---: | ---: |
| Operating Systems | 17 | 11 | 15 | 7 |
| Linear Algebra | 30 | 7 | 12 | 21 |

Mean reader-assessment coverage is reported separately for each axis. With the existing OS
fixture, mean readiness coverage is 0.490 and mean opportunity coverage is 0.577. With the
synthetic LA fixture, the means are 1.000 and 0.786. These describe which weighted concepts have
mastery measurements. Synthetic assessment values cannot calibrate the v2 policy thresholds.

| Book | Matched / total | Rate | Distinct concepts | Representative unmatched titles |
| --- | ---: | ---: | ---: | --- |
| *xv6* | 16 / 99 | .162 | 8 | Operating system interfaces; Page tables; Page faults; Locking |
| *OS and Middleware* | 59 / 179 | .330 | 11 | Preface; Introduction; Atomic Transactions; Messaging, RPC, and Web Services |
| *Think OS* | 23 / 65 | .354 | 9 | Compilation; More bits and bytes; Caching; Multitasking |
| Stallings, *Operating Systems* | 60 / 201 | .299 | 15 | Preface; PART ONE BACKGROUND; PART THREE MEMORY; Review Questions |
| *Three Easy Pieces* | 24 / 62 | .387 | 11 | Intro; Dialogue; Persistence; Summary |
| *Understanding Linear Algebra* | 61 / 222 | .275 | 15 | Systems of linear equations; Reduced row echelon form; Linear dependence; Exercises |
| *Linear Algebra* | 29 / 96 | .302 | 11 | Gauss's Method; Definition and Examples; Maps Between Spaces; Similarity |
| *Linear Algebra with Applications* | 72 / 167 | .431 | 17 | Systems of Linear Equations; Projections and Planes; Subspaces and Spanning |
| *Elementary linear algebra* | 8 / 9 | .889 | 8 | NUMERICAL METHODS |
| Strang, *Introduction to Linear Algebra* | 35 / 63 | .556 | 15 | Lengths and Dot Products; Solving Linear Equations; Projections |

The low xv6 rate includes plausible false negatives such as `Page tables`, `Page faults`,
`Locking`, and `Races`, along with intentionally out-of-lexicon implementation headings. The
low *Understanding Linear Algebra* rate includes repeated `Introduction`, `Summary`, and
`Exercises`, but also likely lexicon gaps such as plural `Systems of linear equations` and
`Reduced row echelon form`. Aliases should only change after row-level human review.

Likely false-positive risks among current matched rows include `The compilation process →
process` in *Think OS*, `Process Locking → process` in *xv6*, and the broad `scheduling` match in
`Disk Scheduling`. These examples show why increasing coverage alone is an unsafe goal.

## Edges and order cases needing review

- `programming → process` is never jointly mapped: process appears in all five OS books while
  programming is not a configured TOC mapping node in any of them. Current TOCs cannot validate
  the proposed relation.
- `high school algebra → linear system` is also never jointly mapped. High-school background is
  unlikely to appear as a chapter heading, so absence is weak evidence either way.
- `storage → file system` is jointly mapped only in Stallings and depends on `disk scheduling →
  storage`, which itself needs semantic review.
- `inner product → orthogonality` is jointly mapped in only one LA book, where `Orthogonality`
  appears at position 74 and `Inner Product Spaces` at 141. This may reflect teaching order,
  missing earlier aliases, or a questionable edge direction.
- `eigenvalue → eigenvector` maps both concepts to the same combined heading in four books. The
  heuristic labels same-position occurrences as external/late; it cannot recover an order from
  that heading.
- `concurrency → synchronization` and `synchronization → deadlock` also have combined-heading
  cases. Same-entry matches do not prove external prerequisite demand.
- *Think OS* places `Virtual memory` at position 12 and `Memory management` at 30, so `memory
  management → virtual memory` becomes external/late. The TOC alone cannot say whether earlier
  chapters supply enough background.
- *Think OS* maps `The compilation process` to `process` at position 3. That false-positive risk
  can make later thread, scheduling, or concurrency targets appear to have an early prerequisite.

These cases are why first occurrence remains a reviewable proxy rather than teaching-order
truth.

## Opportunity descriptors

The v2 formula remains:

```text
Opportunity = Σ_assessed B(c)(1-U(c)) / Σ_assessed B(c)
```

It is an average novelty signal over measured concepts. A book with two unknown concepts and a
book with ten can have the same opportunity score. Validation therefore adds only descriptive
fields: `learning_candidate_count`, `known_concept_count`, `unknown_mastery_count`,
`measured_learning_weight`, and `total_book_concept_weight`. They are not combined into a score
or used for ordering. Concept occurrence weights are structural proxies, so even these totals do
not directly measure learning quantity.

## Next human decisions

Review false-positive matches first because they can distort order for several edges. Then label
plausible false negatives without adding aliases in bulk. Review edge direction and scope using
the book-level paths, prioritizing edges with no jointly mapped books, combined headings, or
mixed order across books. Only after these decisions should a later change update aliases or
promote accepted graph edges.
