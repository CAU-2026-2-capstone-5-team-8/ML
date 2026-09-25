# Prerequisite-first ranking-v2 candidate v1

## Purpose and boundary

`concept-prerequisite-ranking-v2-candidate-v1` is the smallest production-candidate
policy supported by the completed concept-readiness experiments. It asks first whether
the reader has the accepted prerequisite knowledge needed for a book and uses remaining
learning opportunity only as an exact-tie breaker.

This is an offline candidate. It does not replace `/ml/rank`, rank-v1, ranking
configuration, the Backend contract, ReaderProfile API, production matcher,
source-aware adapter, concept graph, graph review, or Data-Pipeline contract.

## Why this policy

- **Prerequisite readiness is primary** because it expresses whether the reader appears
  prepared for concepts inferred through the human-accepted prerequisite graph.
- **D1 is excluded** because its hard coverage gate produced a threshold cliff and only
  3/10 agreement on the frozen human pairs.
- **D2 is not promoted** because averaging readiness and opportunity did not improve
  the 9/10 prerequisite-only agreement and reversed prerequisite direction frequently
  in multi-reader comparisons.
- **Direct learning opportunity is secondary** because it usefully resolves exact
  readiness ties but should not compensate for materially lower prerequisite readiness.
- **Coverage remains diagnostic** because multiplying it into a score would introduce
  an unvalidated policy. Every readiness and opportunity value retains assessed count,
  total count, and coverage.

## Exact ordering and availability

Personalizable books are ordered lexicographically:

1. transitive accepted-prerequisite readiness, descending;
2. direct learning opportunity, descending, only when prerequisite readiness is exactly
   equal; and
3. stable `book_id` order when both diagnostics are equal.

There is no tolerance, weighted sum, coverage multiplier, evidence-occurrence weight,
or source-quality multiplier. Unknown readiness is never zero.

Books are classified into three pools:

- `personalizable`: covered concepts exist, at least one accepted transitive
  prerequisite is inferred, and at least one inferred prerequisite is assessed;
- `concept_only`: covered concepts exist but prerequisite readiness cannot be
  calculated; and
- `evidence_unavailable`: no source-aware covered concept exists.

Only `personalizable` books receive a candidate rank. The other two groups remain an
explicit fallback pool with `null` rank and readiness where unavailable.

## Scale-50 availability

| Topic | Total | Personalizable | Concept only | No concept evidence | Fallback total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear Algebra | 25 | 10 | 0 | 15 | 15 |
| Operating Systems | 25 | 11 | 1 | 13 | 14 |
| Combined | 50 | 21 | 1 | 28 | 29 |

The 28 evidence-unavailable books receive no title, ISBN, metadata-count, source-count,
or book-ID score. The single OS `concept_only` book has a covered programming concept
but no accepted prerequisite. A fallback ordering policy is intentionally not selected.

## Frozen human-pair validation

The ten pair memberships, A/B order, and human labels are unchanged. The exact candidate
was not tuned to them.

| Topic | Agreement |
| --- | ---: |
| Linear Algebra | 5/5 |
| Operating Systems | 4/5 |
| Combined | 9/10 |

This matches prerequisite-only, D2, and tolerance-E at 9/10. It does not establish
generalized accuracy from ten AI-assisted human diagnostic pairs.

## Eight-scenario regression

Statistics below consider the personalizable pool. `P values` is the count of distinct
prerequisite-readiness values. Final groups use the exact prerequisite/opportunity pair;
book ID only stabilizes order inside a tied group.

| Topic | Scenario | Personalized | P values | Final groups | Largest tie | Singletons |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LA | Beginner | 10 | 6 | 10 | 1 | 10 |
| LA | Intermediate | 10 | 6 | 9 | 2 | 8 |
| LA | Advanced | 10 | 6 | 10 | 1 | 10 |
| LA | Uneven | 10 | 6 | 10 | 1 | 10 |
| OS | Beginner | 11 | 5 | 11 | 1 | 11 |
| OS | Intermediate | 11 | 5 | 10 | 2 | 9 |
| OS | Advanced | 11 | 4 | 9 | 2 | 7 |
| OS | Uneven | 11 | 5 | 10 | 2 | 9 |

The secondary signal resolves most primary-value ties without reversing any unequal
prerequisite-readiness pair. Complete scenario vocabularies make coverage 1.0 for every
applicable axis; partial real ReaderProfiles would preserve lower coverage rather than
renormalizing it into the score.

## Behavioral examples

### Operating Systems

`Advanced concepts in operating systems` uses covered concepts including concurrency,
deadlock, distributed systems, process, scheduling, security, and synchronization. Its
accepted transitive prerequisites are concurrency, process, storage, synchronization,
and thread.

| Scenario | Rank | Group | Prerequisite readiness/coverage | Opportunity/direct coverage |
| --- | ---: | ---: | --- | --- |
| Beginner | 11 | 11 | 0.240 / 1.000 | 0.867 / 1.000 |
| Intermediate | 11 | 10 | 0.690 / 1.000 | 0.450 / 1.000 |

Its readiness rises substantially and its diagnostic group improves, while exact
prerequisite-first ordering leaves it rank 11 because the other books' prerequisites
also become more ready. The output reports that honestly rather than forcing movement.

### Linear Algebra

`Linear algebra.` (`book_09c91e7b456682eb19fb`) covers determinant,
diagonalization, eigenvalue, eigenvector, matrix, and vector space. Its accepted
transitive prerequisites are matrix and vector.

| Scenario | Rank | Group | Prerequisite readiness/coverage | Opportunity/direct coverage |
| --- | ---: | ---: | --- | --- |
| Beginner | 4 | 4 | 0.400 / 1.000 | 0.850 / 1.000 |
| Advanced | 1 | 1 | 1.000 / 1.000 | 0.067 / 1.000 |

The advanced reader is fully prepared for its prerequisites but has little remaining
opportunity over the covered concepts. The axes remain separate and explain both facts.

## Meeting demo

The demo reads the real local Scale-50 concept-mapping artifact. It does not contain
hard-coded books or ranking results.

```bash
uv run bookmatch-ml demo-concept-recommendation \
  --topic operating-systems \
  --scenario beginner \
  --limit 5

uv run bookmatch-ml demo-concept-recommendation \
  --topic operating-systems \
  --scenario intermediate \
  --limit 5

uv run bookmatch-ml demo-concept-recommendation \
  --topic linear-algebra \
  --scenario beginner \
  --limit 5

uv run bookmatch-ml demo-concept-recommendation \
  --topic linear-algebra \
  --scenario advanced \
  --limit 5
```

Each run prints the personalized/fallback counts, rank, title, prerequisite readiness
and coverage, opportunity and direct coverage, plus evidence-bounded Korean reasons.

Generate the full eight-scenario and frozen-pair report with:

```bash
uv run bookmatch-ml evaluate-prerequisite-first-candidate \
  --output data/reports/prerequisite-first-ranking-v2-candidate-v1.json
```

Generated report SHA-256:

```text
fd7b6807636f88051925c9ec2486411cbb6496e51c034949b1f639e7444cab95
```

Two consecutive runs produced byte-identical output. Generated reports stay ignored
under `data/reports/`; the evaluator, tests, commands, and this result document are
version controlled.

## Decisions still required before Backend integration

1. whether candidate validation is sufficient to replace rank-v1;
2. how partial prerequisite coverage should affect eligibility or confidence without
   becoming an implicit score;
3. which fallback-pool policy should serve the 29 non-personalizable books;
4. whether to expose both axes and coverage in a future API contract; and
5. how much broader blinded human review is required before production rollout.
