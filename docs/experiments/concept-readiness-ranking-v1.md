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

The experiment generator does not create a preference or gold label. It selects at
most five informative pairs per topic with blank review fields. The packet was shown
to the reviewer without the selection reason, diagnostic values, or variant ordering:

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

For each pair the reviewer answered: “For this fixed ReaderProfile, which book is more
appropriate to read first?” The blinded view included only concept readiness, covered
concepts, accepted direct and transitive prerequisites, and coverage. The generated
JSON starts with blank `human_preference`/`review_note` fields; the decisions below
were added only after both topic packets had been reviewed.

## Frozen human pair review

All ten pairs are reviewed: Linear Algebra 5/5 and Operating Systems 5/5. There are no
`tie` or `not_judgable` labels. The reviewed local report is intentionally still under
the existing ignored `data/reports/` directory and was not force-added to Git.

Reviewed report SHA-256:

```text
a5f3a77c65ce4ef0d838ff97b7a620a2927ab466d6eb01ad3e1c06024b0f3c47
```

The following frozen values make the manual review update auditable without adding a
second review artifact:

1. `linear-algebra:pair-1` — `A` — `Book A has complete prerequisite coverage with very strong readiness on matrix and vector, while still containing several concepts with substantial learning opportunity. Book B covers a much broader concept set and has weaker prerequisite readiness for this reader, so A is the more appropriate book to read first.`
2. `linear-algebra:pair-2` — `A` — `Both books have complete prerequisite assessment coverage, but Book A has slightly stronger prerequisite readiness and more room for learning among its assessed covered concepts. It is the more suitable first step for this reader.`
3. `linear-algebra:pair-3` — `A` — `Book B offers more new material, but its prerequisite readiness is lower and its direct concept assessment coverage is also lower. Book A provides a better-prepared and more conservative first step before moving to the broader Book B.`
4. `linear-algebra:pair-4` — `A` — `Book A has stronger prerequisite readiness and substantially higher direct concept assessment coverage. Book B contains more unassessed and lower-readiness material, so A is the safer and more appropriate book to read first.`
5. `linear-algebra:pair-5` — `B` — `Book B has stronger prerequisite readiness and higher direct assessment coverage for this reader. Although Book A contains more learning opportunities, B provides the better-supported first step before progressing to the broader material in A.`
6. `operating-systems:pair-1` — `A` — `Book A has strong assessed prerequisite readiness for concurrency and process. Book B has broader evidence coverage, but its prerequisite set includes computer architecture at readiness 0.0 and memory management at only 0.5. For reading order, Book A provides the safer prerequisite fit.`
7. `operating-systems:pair-2` — `A` — `Book A has broader prerequisite assessment coverage and contains a meaningful mix of already-known and lower-readiness concepts, giving this reader room to learn while retaining sufficient prerequisite support. Book B has strong assessed prerequisites but much of its assessed direct content is already well known.`
8. `operating-systems:pair-3` — `A` — `Book A has consistently strong assessed prerequisite readiness across concurrency, process, and synchronization. Book B offers somewhat more learning opportunity, but its prerequisite evidence includes computer architecture at readiness 0.0, making Book A the better-supported first step.`
9. `operating-systems:pair-4` — `A` — `This pair is relatively close because Book B has substantially higher direct concept assessment coverage. However, Book A has much stronger assessed prerequisite readiness, while Book B depends on computer architecture at readiness 0.0 and memory management at 0.5. For the question of which book should be read first, prerequisite preparedness favors Book A.`
10. `operating-systems:pair-5` — `B` — `Both books contain an assessed computer-architecture weakness, but Book B has greater prerequisite assessment coverage, slightly stronger overall assessed prerequisite readiness, and more assessed material that still provides learning opportunity. Book B is therefore the better first choice for this reader.`

## Pairwise agreement with the frozen variants

The comparison was run only after all labels were frozen. D1 predicts by its existing
status priority, then direct learning opportunity, then transitive-prerequisite
readiness. D2 predicts the higher existing unweighted combination. Direct-only uses
higher direct learning opportunity; prerequisite-only uses higher transitive-
prerequisite readiness. Book ID fallback ordering is not treated as evidence.

`not_judgable` human labels would be excluded from the comparable denominator. A human
`tie` would agree only with an evidence-value tie. An algorithmic tie against an `A` or
`B` label would be a disagreement. The present ten labels and all four compared signals
have no ties or unavailable pair values, so all ten pairs are comparable.

### Linear Algebra

| Pair | Human | D1 | D2 | Direct only | Prerequisite only |
| --- | --- | --- | --- | --- | --- |
| 1 | A | A | A | A | A |
| 2 | A | A | A | A | A |
| 3 | A | B | A | B | A |
| 4 | A | B | A | B | A |
| 5 | B | A | B | A | B |

D1 human pair agreement is 2/5 (40%); D2 is 5/5 (100%). Direct-only is
2/5 (40%), and prerequisite-only is 5/5 (100%).

### Operating Systems

| Pair | Human | D1 | D2 | Direct only | Prerequisite only |
| --- | --- | --- | --- | --- | --- |
| 1 | A | A | A | B | A |
| 2 | A | B | B | A | B |
| 3 | A | B | A | B | A |
| 4 | A | B | A | B | A |
| 5 | B | A | B | B | B |

D1 human pair agreement is 1/5 (20%); D2 is 4/5 (80%). Direct-only is
2/5 (40%), and prerequisite-only is 4/5 (80%).

### Combined

| Signal | Agreement | Rate |
| --- | ---: | ---: |
| D1 | 3/10 | 30% |
| D2 | 9/10 | 90% |
| Direct only | 4/10 | 40% |
| Prerequisite only | 9/10 | 90% |

These are diagnostic human pair-agreement results on ten deliberately informative
pairs, not an accuracy claim or a representative ranking benchmark.

## Disagreement analysis

- **Linear Algebra pair 3:** D1 follows Book B's larger direct opportunity (0.455 vs
  0.357). Human and D2 favor Book A's higher prerequisite readiness (0.90 vs 0.743),
  higher direct coverage (0.778 vs 0.625), and fewer unknown covered concepts. Both
  books have complete prerequisite coverage, and direct/transitive sets are identical.
- **Linear Algebra pair 4:** D1 again follows Book B's larger direct opportunity
  (0.455 vs 0.377). Human and D2 favor Book A's prerequisite readiness (0.88 vs 0.743)
  and direct coverage (0.846 vs 0.625). Book B has six unknown covered concepts versus
  two for Book A. Direct and transitive prerequisites are unchanged.
- **Linear Algebra pair 5:** D1 and direct-only favor Book A's larger opportunity
  (0.417 vs 0.357). Human, D2, and prerequisite-only favor Book B's readiness
  (0.90 vs 0.733), coverage (0.778 vs 0.706), and smaller unknown set. Both prerequisite
  sets have complete coverage and no transitive-only addition.
- **Operating Systems pair 2:** D1 promotes eligible Book B over insufficient-evidence
  Book A, and D2 is narrowly higher for B (0.500 vs 0.481). Human favors Book A's much
  larger direct opportunity (0.343 vs 0.100) and higher prerequisite coverage
  (0.714 vs 0.500). D2 does not include coverage. Transitive expansion adds assessed
  `process` and unknown `thread` to Book B, raising its readiness from 0.80 to 0.90
  while leaving coverage at 0.50.
- **Operating Systems pair 3:** D1 promotes Book B because its direct coverage reaches
  the 0.50 gate while Book A is at 0.444. Human, D2, and prerequisite-only favor Book
  A's readiness (0.867 vs 0.60), despite Book B's larger opportunity (0.18 vs 0.10).
  The selected books have identical direct and transitive prerequisite sets.
- **Operating Systems pair 4:** D1 prioritizes challenge-candidate Book B over
  insufficient-evidence Book A. Book B has much higher direct coverage (0.833 vs
  0.444), but human, D2, and prerequisite-only favor Book A's readiness (0.867 vs
  0.575). Book B gains unknown `thread` only through transitive expansion, reducing
  prerequisite coverage from 0.80 direct to 0.667 transitive without changing its
  assessed readiness mean.
- **Operating Systems pair 5:** D1 promotes eligible Book A because Book B's direct
  coverage is 0.462, below the 0.50 gate. Human, D2, direct-only, and
  prerequisite-only all favor Book B: its opportunity is 0.233 vs 0.18, readiness is
  0.62 vs 0.60, and prerequisite coverage is 0.714 vs 0.60. Direct and transitive
  prerequisite sets are identical for both books.

Coverage materially affects LA pairs 3-5 and OS pairs 2-5, but in different ways. It
helps the human reviewer discount means based on narrower evidence, while D1 turns the
configured minimums into categorical status boundaries. D2 and the two single-axis
comparisons retain coverage only as accompanying evidence, not as an ordering term.

All reviewed Linear Algebra books have identical direct and transitive prerequisite
sets. In the selected OS books, transitive expansion affects `Distributed operating
systems`, `Operating system concepts essentials`, and `The Design of the Unix
Operating System`; elsewhere it does not change the selected sets. The changes can
raise an assessed mean, reduce coverage by adding unknown prerequisites, or leave the
mean unchanged. This supports continuing to report both projections separately.

The 9/10 D2 and prerequisite-only agreement is not sufficient for production
promotion. The pairs were intentionally selected for contrast, only two fixed reader
profiles are represented, and the large unknown-concept groups remain. No formula,
threshold, graph, matcher, adapter, or production ranking change follows from this
review.

## Additional deterministic ReaderProfile scenarios

The next evaluation should use four interpretable scenarios per topic over the same
accepted canonical concept vocabulary. The score groups below fully specify proposed
fixtures; they are a design only and are not implemented by this experiment.

### Linear Algebra scenarios

| Scenario | User state | Proposed readiness values grouped by score |
| --- | --- | --- |
| Beginner | Comfortable with school algebra but only beginning formal linear algebra. | 0.8: high school algebra; 0.6: systems of equations; 0.5: linear system; 0.4: matrix, vector; 0.3: gaussian elimination, determinant; 0.2: vector space, linear independence; 0.1: basis, dimension, rank, inner product, orthogonality, linear transformation; 0.0: eigenvalue, eigenvector, diagonalization, least squares, singular value decomposition. |
| Intermediate | Strong fundamentals with partial abstract and spectral knowledge. | 1.0: high school algebra; 0.9: systems of equations, linear system, matrix, vector; 0.8: gaussian elimination; 0.7: determinant, vector space; 0.6: linear independence, basis, dimension, rank, linear transformation; 0.5: inner product, orthogonality, eigenvalue, eigenvector; 0.4: diagonalization, least squares; 0.3: singular value decomposition. |
| Advanced | Broad, consistently high readiness with a small remaining SVD gap. | 1.0: high school algebra, systems of equations, linear system, matrix, vector; 0.95: gaussian elimination, determinant, vector space, linear independence, basis, dimension, rank, linear transformation; 0.9: inner product, orthogonality, eigenvalue, eigenvector, diagonalization, least squares; 0.85: singular value decomposition. |
| Uneven | Computationally strong but weak on abstract vector-space and geometric foundations. | 1.0: high school algebra, systems of equations, matrix; 0.95: linear system, gaussian elimination; 0.9: determinant; 0.85: rank; 0.6: vector; 0.5: eigenvalue, eigenvector; 0.45: diagonalization; 0.4: linear transformation; 0.3: vector space; 0.25: linear independence; 0.2: basis, dimension, inner product, least squares; 0.15: orthogonality; 0.1: singular value decomposition. |

### Operating Systems scenarios

| Scenario | User state | Proposed readiness values grouped by score |
| --- | --- | --- |
| Beginner | Has programming experience but limited operating-system mechanisms. | 0.8: programming; 0.5: computer architecture; 0.4: process; 0.3: memory management, storage, input/output; 0.2: thread, scheduling, concurrency, file system; 0.1: synchronization, virtual memory, protection; 0.0: deadlock, distributed systems, security, virtualization. |
| Intermediate | Understands core execution and memory concepts but has weaker advanced systems knowledge. | 0.9: programming; 0.85: process; 0.7: computer architecture, thread, scheduling, concurrency, memory management; 0.6: synchronization, file system, storage, input/output; 0.5: virtual memory; 0.4: deadlock, protection, security, virtualization; 0.3: distributed systems. |
| Advanced | Broad systems readiness with no major prerequisite gap. | 0.95: programming, process, thread, scheduling, concurrency, synchronization, memory management, file system, storage, input/output; 0.9: computer architecture, deadlock, virtual memory, protection, security, virtualization, distributed systems. |
| Uneven | Strong in processes and concurrency but weak in architecture, memory, and storage. | 1.0: programming; 0.95: process, concurrency; 0.9: thread, scheduling, synchronization; 0.8: deadlock; 0.6: distributed systems; 0.5: protection; 0.4: file system; 0.3: storage, input/output, security; 0.25: memory management; 0.2: computer architecture, virtualization; 0.15: virtual memory. |

## Reproduction

```bash
uv run bookmatch-ml evaluate-concept-readiness-ranking \
  --concept-mapping data/reports/scale-50-concept-presence-v2-overlap.json \
  --reader data/output/reader_profile.json \
  --reader data/output/concept_matching_la_reader.json \
  --output data/reports/concept-readiness-ranking-v1.json
```

The generated output is deterministic and records all input hashes, graph/review
hashes, thresholds, per-book diagnostics, summaries, sanity cases, and the initially
unlabeled pair packet. Re-running the command intentionally recreates blank human
fields; apply only the frozen decisions above when reconstructing the reviewed local
artifact.

## Required validation before production promotion

1. Repeat the completed human pair comparison across the additional deterministic
   ReaderProfile scenarios and eventually independent reviewers.
2. Coverage policy must be evaluated explicitly; the present large unavailable groups
   cannot be silently assigned zero readiness or renormalized into confident scores.
3. D1 and D2 need ranking-quality evaluation on more readers, not only the two
   deterministic fixtures.
4. Transitive and direct prerequisites should remain separate until the observed
   compression and graph-depth effects are understood.
5. Any production proposal must be a separate ranking-v2 change with API/regression
   validation. This experiment itself does not select a production formula.
