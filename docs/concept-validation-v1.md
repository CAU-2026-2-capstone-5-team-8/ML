# Concept graph and TOC mapping validation v1

## Purpose and boundary

This validation slice exposes the evidence behind concept matching v2's two most uncertain
inputs: proposed prerequisite edges and exact TOC alias mappings. It does not produce a
recommendation score, modify `rank-v1`, change v2 readiness or opportunity formulas, or alter
`/ml/rank`. The matching config contains only three reviewed mechanical corrections: two phrase
exclusions and one plural alias described below.

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
  --review-config configs/concept_graph_reviews.yaml \
  --output data/reports/concept_validation.json \
  --review-template data/reviews/concept_graph_review.json
```

The report records hashes of all four canonical files and any reader profiles. It rebuilds the
same v2 Book Concept Profiles from canonical evidence. Generated artifacts under `data/` are
ignored. `configs/concept_graph_reviews.yaml` is the version-controlled source of human review
decisions. The command remains usable without that file and then emits `unreviewed` decisions.
The report schema is `concept-validation-v2` because it adds explicit ordering states and joined
human decisions; the v1 first-occurrence fields and legacy observation remain present.

## Reading graph edge evidence

Each of the 24 graph edges shows topic, prerequisite, dependent, relation and source types, edge
version, books containing the dependent, and all five topic books. `state` separates:

- `strict_before`: both concepts are observed in different entries and prerequisite is earlier;
- `same_entry`: both first occur in the same TOC entry;
- `strict_after`: both occur in different entries and prerequisite is later;
- `one_or_both_unobserved`: TOC mapping cannot compare both sides.

The older `observed_before_in_toc`, `observed_at_or_after_in_toc`,
`prerequisite_not_mapped`, and `dependent_not_mapped` classification remains in `legacy_state`
for artifact readers. An edge also reports the four new counts and its jointly observable book
count. `same_entry` never contributes to `strict_before_count`.

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

Reviewers edit `configs/concept_graph_reviews.yaml`, not generated JSON. It contains exactly one
row for every configured graph edge. Allowed `review_status` values are `unreviewed`, `accepted`,
`rejected`, and `needs_revision`. A decided edge requires a non-blank `review_note`; initial rows
remain `unreviewed` with a null note. Duplicate, missing, and unknown edge references are rejected.

`build-concept-review` joins these decisions to each edge's automatic evidence in both the main
report and `data/reviews/concept_graph_review.json`. The latter is a replaceable generated
snapshot. Human review config is never read by profile matching or ranking, so a decision cannot
change a score or promote `source_type`. Such promotion requires a later explicit graph change.

## Narrow deterministic matcher corrections

`configs/concept_matching.yaml` keeps the changes scoped to v2 TOC mapping:

- `compilation process` excludes only the `process` concept for that phrase;
- `disk scheduling` excludes only canonical process/CPU `scheduling` while leaving any other
  concept match alone;
- `systems of linear equations` is an alias addition for `linear system`.

No other unmatched review candidate was added as an alias.

## Structural results on the current ten books

These are reproducible structural measurements, not recommendation accuracy or educational
validity.

| Topic | TOC matched / total | Mapping rate | Mapped graph concepts | Edges jointly mapped in ≥1 book | Edges not jointly mapped | Ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Operating Systems | 181 / 606 | 0.299 | 16 / 17 | 9 / 10 | 1 | 0 |
| Linear Algebra | 208 / 557 | 0.373 | 19 / 20 | 13 / 14 | 1 | 0 |

Across the five books per topic, direct edge/book observations were:

| Topic | Before | Same or after | Prerequisite missing | Dependent missing |
| --- | ---: | ---: | ---: | ---: |
| Operating Systems | 17 | 11 | 15 | 7 |
| Linear Algebra | 31 | 7 | 15 | 17 |

The explicit ordering-evidence distribution is:

| Topic | Strict before | Same entry | Strict after | One or both unobserved |
| --- | ---: | ---: | ---: | ---: |
| Operating Systems | 17 | 2 | 9 | 22 |
| Linear Algebra | 31 | 6 | 1 | 32 |

Mean reader-assessment coverage is reported separately for each axis. With the existing OS
fixture, mean readiness coverage is 0.490 and mean opportunity coverage is 0.577. With the
synthetic LA fixture, the means are 1.000 and 0.793. These describe which weighted concepts have
mastery measurements. Synthetic assessment values cannot calibrate the v2 policy thresholds.

| Book | Matched / total | Rate | Distinct concepts | Representative unmatched titles |
| --- | ---: | ---: | ---: | --- |
| *xv6* | 16 / 99 | .162 | 8 | Operating system interfaces; Page tables; Page faults; Locking |
| *OS and Middleware* | 59 / 179 | .330 | 11 | Preface; Introduction; Atomic Transactions; Messaging, RPC, and Web Services |
| *Think OS* | 22 / 65 | .338 | 9 | Compilation; More bits and bytes; Caching; Multitasking |
| Stallings, *Operating Systems* | 60 / 201 | .299 | 15 | Preface; PART ONE BACKGROUND; PART THREE MEMORY; Review Questions |
| *Three Easy Pieces* | 24 / 62 | .387 | 11 | Intro; Dialogue; Persistence; Summary |
| *Understanding Linear Algebra* | 62 / 222 | .279 | 15 | Reduced row echelon form; Linear dependence; Exercises |
| *Linear Algebra* | 29 / 96 | .302 | 11 | Gauss's Method; Definition and Examples; Maps Between Spaces; Similarity |
| *Linear Algebra with Applications* | 74 / 167 | .443 | 18 | Projections and Planes; Subspaces and Spanning |
| *Elementary linear algebra* | 8 / 9 | .889 | 9 | NUMERICAL METHODS |
| Strang, *Introduction to Linear Algebra* | 35 / 63 | .556 | 15 | Lengths and Dot Products; Solving Linear Equations; Projections |

The low xv6 rate includes plausible false negatives such as `Page tables`, `Page faults`,
`Locking`, and `Races`, along with intentionally out-of-lexicon implementation headings. The
low *Understanding Linear Algebra* rate includes repeated `Introduction`, `Summary`, and
`Exercises`, but also likely lexicon gaps such as plural `Systems of linear equations` and
`Reduced row echelon form`. Aliases should only change after row-level human review.

The confirmed `The compilation process → process` and `Disk Scheduling → scheduling` false
positives are now excluded. Risks such as `Process Locking → process` remain review items.

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
- *Think OS* now excludes `The compilation process`; its first actual process evidence is the
  `Processes` chapter at position 8. A chapter heading is still only structural evidence and
  does not prove how much prerequisite material was taught there.

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
