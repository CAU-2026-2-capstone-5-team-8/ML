# Concept-readiness ranking experiment v1

## Question and boundary

Scale-50 currently has no prose-derived lexical, syntactic, concept-density, or
prerequisite-demand features. Production rank-v1 therefore activates only
`topic_fit`, assigning all 25 books in each topic the same score. This experiment asks
whether reviewed concept evidence can distinguish books within a topic using:

- the existing deterministic `ReaderProfile.concept_readiness` fixtures;
- source-aware, book-level concept presence from `normalized_alias_span_v2`; and
- only prerequisite edges marked `accepted` by the completed human review.

This is an offline diagnostic. It does not modify `/ml/rank`, ranking weights, ranking
configuration, `MatchingBookProfile`, the production concept graph, the matcher, the
source-aware adapter, or the API contract. Evidence occurrence counts and source types
are not weights, unknown readiness is not zero, and the four unavailable prose
difficulty fields remain unavailable.

## Inputs and graph projection

The accepted projection contains 18 of the 24 candidate edges. Rejected and
`needs_revision` edges are excluded. It is a deterministic, topic-local DAG. The
primary prerequisite diagnostic uses transitive ancestors with set deduplication; a
direct-edge-only result is retained as a sensitivity check.

The experiment reuses the policy thresholds already defined by
`configs/concept_matching_v2.yaml` rather than introducing production thresholds:

| Parameter | Value |
| --- | ---: |
| Minimum prerequisite assessment coverage | 0.50 |
| Minimum opportunity assessment coverage | 0.50 |
| Minimum readiness score | 0.60 |

These thresholds only categorize the D1 diagnostic. They are not promoted into the
production ranker by this experiment.

## Variant definitions

- **A — topic only:** the current production behavior for Scale-50.
- **B — direct concepts:** for assessed covered concepts, report mastery mean,
  `learning_opportunity = mean(1 - mastery)`, and assessment coverage.
- **C — accepted prerequisites:** report mastery mean and coverage for accepted direct
  prerequisites and, separately, transitive prerequisite ancestors.
- **D1 — two stage:** use the existing coverage/readiness policy to categorize a book
  as `eligible`, `challenge_candidate`, or `insufficient_evidence`, then order by direct
  learning opportunity. This is a diagnostic grouping, not a production eligibility
  rule.
- **D2 — unweighted:** only when both axes are available, calculate
  `(prerequisite_readiness + direct_learning_opportunity) / 2`. Missing axes are not
  renormalized.

All concepts are sets at book level. Repeated evidence and `support_count` cannot alter
the result.

## Scale-50 results

### Linear Algebra (25 books)

| Variant | Books with usable value | Distinct value/group count | Largest tie | Singleton books |
| --- | ---: | ---: | ---: | ---: |
| A topic only | 25 | 1 | 25 | 0 |
| B direct opportunity | 10 | 10 including unavailable | 15 | 8 |
| C direct prerequisites | 10 | 8 including unavailable | 15 | 5 |
| C transitive prerequisites | 10 | 7 including unavailable | 15 | 4 |
| D1 two stage | 25 grouped; 10 evidence-qualified | 10 | 15 | 8 |
| D2 unweighted | 10 | 10 including unavailable | 15 | 8 |

D1 status counts are 10 `eligible` and 15 `insufficient_evidence`. One book gains
additional prerequisites through transitive expansion. All 10 prerequisite-bearing
books have complete prerequisite assessment coverage. Direct concept coverage among
the 10 books ranges from 0.625 to 0.917.

### Operating Systems (25 books)

| Variant | Books with usable value | Distinct value/group count | Largest tie | Singleton books |
| --- | ---: | ---: | ---: | ---: |
| A topic only | 25 | 1 | 25 | 0 |
| B direct opportunity | 11 | 7 including unavailable | 14 | 3 |
| C direct prerequisites | 11 | 7 including unavailable | 14 | 4 |
| C transitive prerequisites | 11 | 6 including unavailable | 14 | 3 |
| D1 two stage | 25 grouped; 3 evidence-qualified | 10 | 14 | 7 |
| D2 unweighted | 11 | 9 including unavailable | 14 | 5 |

D1 status counts are 2 `eligible`, 1 `challenge_candidate`, and 22
`insufficient_evidence`. Six books gain additional prerequisites through transitive
expansion. Transitive prerequisite assessment coverage among applicable books ranges
from 0.50 to 0.714. Direct concept coverage is lower and more variable: one
concept-bearing book has no assessed covered concept, and the remaining applicable
coverage ranges from 0.333 to 0.833.

### Interpretation

The answer to the experiment question is qualified **yes**: reviewed concept evidence
breaks the 25-way topic-only tie for books with matched and assessed concepts. However,
it does not yet distinguish the full catalog. The largest tie remains the unavailable
group: 15 Linear Algebra books and 14 Operating Systems books. Transitive prerequisite
expansion increases prerequisite coverage for seven books, but it also compresses some
mastery means: distinct C values fall from 7 to 6 in Linear Algebra and from 6 to 5 in
Operating Systems. More graph reach is not automatically more ranking resolution.

## Coverage distributions

Linear Algebra prerequisite coverage is 1.0 for all 10 applicable books. Its direct
coverage distribution is: 0.625 (1), 0.667 (2), 0.706 (2), 0.778 (1), 0.800 (1),
0.833 (1), 0.846 (1), and 0.917 (1), with 15 not applicable.

Operating Systems transitive prerequisite coverage is: 0.500 (1), 0.600 (2), 0.667
(2), and 0.714 (6), with 14 not applicable. Direct coverage is: 0.000 (1), 0.333 (1),
0.385 (1), 0.400 (1), 0.417 (1), 0.444 (1), 0.455 (1), 0.462 (1), 0.467 (1), 0.500
(2), and 0.833 (1), with 13 not applicable.

Coverage must accompany every mastery or opportunity value. A mastery mean of 1.0
over one assessed prerequisite is not equivalent to complete readiness for five
prerequisites.

## Sanity inspection

### Linear Algebra

- `Linear algebra.` has the highest prerequisite readiness (0.95, coverage 1.0) and
  highest direct learning opportunity (0.50, direct coverage 0.833).
- `Introductory linear algebra` has the lowest prerequisite readiness (0.733, coverage
  1.0).
- `Linear Algebra` has the lowest direct opportunity (0.357, direct coverage 0.778)
  while prerequisite readiness is 0.90. It is a useful direct/prerequisite divergence
  case.

### Operating Systems

- `Distributed operating systems` has the highest prerequisite readiness (0.90) but
  only 0.50 prerequisite coverage and the lowest direct opportunity (0.10).
- `Operating system concepts essentials` has the lowest prerequisite readiness (0.575,
  coverage 0.667).
- `Operating Systems` has the highest direct opportunity (0.343, coverage 0.467), but
  prerequisite readiness is 0.62 at 0.714 coverage; D1 therefore retains an
  insufficient-evidence label because direct coverage misses the existing 0.50 gate.
- `Advanced concepts in operating systems` also has direct opportunity 0.10 but high
  prerequisite readiness 0.867 at only 0.60 coverage, illustrating why the axes and
  coverage should remain visible rather than being collapsed prematurely.

## Blinded human pair-review packet

No preference or gold label is generated. The deterministic report selects at most
five informative pairs per topic:

| Topic | Pair | Reason | Left | Right |
| --- | --- | --- | --- | --- |
| Linear Algebra | 1 | Prerequisite contrast | Linear algebra. | Introductory linear algebra |
| Linear Algebra | 2 | Direct-opportunity contrast | Linear algebra. | Linear Algebra |
| Linear Algebra | 3 | D1/D2 ordering disagreement | Linear Algebra | Linear algebra |
| Linear Algebra | 4 | D1/D2 ordering disagreement | Linear algebra with applications | Linear algebra |
| Linear Algebra | 5 | D1/D2 ordering disagreement | Introductory linear algebra | Linear Algebra |
| Operating Systems | 1 | Prerequisite contrast | Distributed operating systems | Operating system concepts essentials |
| Operating Systems | 2 | Direct-opportunity contrast | Operating Systems | Distributed operating systems |
| Operating Systems | 3 | D1/D2 ordering disagreement | Advanced concepts in operating systems | Distributed operating systems & algorithms |
| Operating Systems | 4 | D1/D2 ordering disagreement | Advanced concepts in operating systems | The Design of the Unix Operating System |
| Operating Systems | 5 | D1/D2 ordering disagreement | Distributed operating systems & algorithms | Operating Systems |

For each pair a reviewer should answer: “For this fixed ReaderProfile, which book is
more appropriate to read first?” The JSON report includes covered concepts, inferred
prerequisites, assessed mastery, both coverage ratios, both diagnostic values, and
blank `human_preference`/`review_note` fields.

## Reproduction

```bash
uv run bookmatch-ml evaluate-concept-readiness-ranking \
  --concept-mapping data/reports/scale-50-concept-presence-v2-overlap.json \
  --reader data/output/reader_profile.json \
  --reader data/output/concept_matching_la_reader.json \
  --output data/reports/concept-readiness-ranking-v1.json
```

The output is deterministic and records all input hashes, graph/review hashes,
thresholds, per-book diagnostics, summaries, sanity cases, and the unlabeled pair
packet.

## Required validation before production promotion

1. Human pair review must establish whether either diagnostic ordering corresponds to
   a sensible reading order.
2. Coverage policy must be evaluated explicitly; the present large unavailable groups
   cannot be silently assigned zero readiness or renormalized into confident scores.
3. D1 and D2 need ranking-quality evaluation on more readers, not only the two
   deterministic fixtures.
4. Transitive and direct prerequisites should remain separate until the observed
   compression and graph-depth effects are understood.
5. Any production proposal must be a separate ranking-v2 change with API/regression
   validation. This experiment itself does not select a production formula.
