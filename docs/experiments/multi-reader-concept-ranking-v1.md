# Multi-reader concept-ranking experiment v1

## Question and boundary

This experiment asks whether prerequisite readiness and direct learning opportunity
change book suitability in a consistent, explainable way across different deterministic
reader knowledge states.

It is an offline diagnostic. It does not modify `/ml/rank`, ranking configuration,
ranking formulas, the concept graph, graph-review decisions, matcher, source-aware
adapter, Data-Pipeline, or the API contract. It does not select a ranking-v2 production
formula.

The inputs stay frozen:

- the Scale-50 book set and source-aware concept-presence report;
- book-level concept-set deduplication;
- the production overlap matcher and source-aware adapter;
- the reviewed graph projection containing exactly 18 accepted edges;
- exclusion of all rejected and `needs_revision` edges; and
- the two fixed ReaderProfiles and ten previously reviewed book pairs.

Evidence occurrence counts and source categories are not weights. Prose difficulty is
not estimated. Unknown readiness is unavailable, never zero. No matcher rule is added.

## Deterministic reader scenarios

The exact concept-level values are versioned in
`configs/multi_reader_concept_ranking_v1.yaml`. Every scenario supplies exactly the
canonical concept IDs in the accepted graph projection and uses the existing
`ReaderProfile` schema.

| Topic | Scenario | Description |
| --- | --- | --- |
| Linear Algebra | Beginner | Comfortable with school algebra but only beginning formal linear algebra. |
| Linear Algebra | Intermediate | Strong fundamentals with partial abstract and spectral knowledge. |
| Linear Algebra | Advanced | Broad, consistently high readiness with a small remaining SVD gap. |
| Linear Algebra | Uneven | Computationally strong but weak on abstract vector-space and geometric foundations. |
| Operating Systems | Beginner | Has programming experience but limited operating-system mechanisms. |
| Operating Systems | Intermediate | Understands core execution and memory concepts but has weaker advanced systems knowledge. |
| Operating Systems | Advanced | Broad systems readiness with no major prerequisite gap. |
| Operating Systems | Uneven | Strong in processes and concurrency but weak in architecture, memory, and storage. |

Profiles are deterministic fixtures, not random samples. Their dimension-level values
are the mean of their concept readiness solely to construct a valid production-schema
profile; all diagnostics in this experiment use concept readiness, not those aggregate
dimensions.

## Diagnostic definitions

- **A — topic only:** the current Scale-50 production behavior.
- **B — direct opportunity:** `mean(1 - readiness)` over assessed covered concepts.
- **C1 — direct prerequisite:** mean readiness over accepted direct prerequisites.
- **C2 — transitive prerequisite:** mean readiness over the set-deduplicated accepted
  ancestor closure.
- **D2 — unweighted:** when both axes exist,
  `(C2 readiness + B opportunity) / 2`.
- **E — prerequisite first:** order primarily by C2 readiness. Within a readiness
  tolerance band, order secondarily by B opportunity. Coverage remains separate.

E uses a primary tolerance of `0.05`, with `0.00`, `0.05`, and `0.10` sensitivity
outputs. For a full ordering, each band is anchored at its highest prerequisite
readiness; this produces transitive, deterministic groups. Pairwise comparison applies
the same two-book rule directly. Book ID only stabilizes order inside a true tie and is
never converted into a score.

An unavailable axis remains in a separate final group. It is not scored as zero.

## Scenario results

Each cell is `available / unavailable / ordering groups / largest tie / singleton
groups`. Group counts and the largest tie include the explicit unavailable group.

### Operating Systems

| Scenario | Topic only | Direct opportunity | Direct prerequisite | Transitive prerequisite | D2 | E (0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| Beginner | 25/0/1/25/0 | 12/13/13/13/12 | 11/14/8/14/4 | 11/14/6/14/3 | 11/14/11/14/9 | 11/14/12/14/11 |
| Intermediate | 25/0/1/25/0 | 12/13/11/13/8 | 11/14/6/14/1 | 11/14/6/14/3 | 11/14/11/14/9 | 11/14/10/14/7 |
| Advanced | 25/0/1/25/0 | 12/13/11/13/8 | 11/14/5/14/0 | 11/14/5/14/1 | 11/14/10/14/7 | 11/14/10/14/7 |
| Uneven | 25/0/1/25/0 | 12/13/12/13/10 | 11/14/8/14/4 | 11/14/6/14/3 | 11/14/11/14/9 | 11/14/11/14/9 |

### Linear Algebra

| Scenario | Topic only | Direct opportunity | Direct prerequisite | Transitive prerequisite | D2 | E (0.05) |
| --- | --- | --- | --- | --- | --- | --- |
| Beginner | 25/0/1/25/0 | 10/15/10/15/8 | 10/15/8/15/5 | 10/15/7/15/4 | 10/15/10/15/8 | 10/15/10/15/8 |
| Intermediate | 25/0/1/25/0 | 10/15/10/15/8 | 10/15/8/15/5 | 10/15/7/15/4 | 10/15/10/15/8 | 10/15/10/15/8 |
| Advanced | 25/0/1/25/0 | 10/15/9/15/6 | 10/15/7/15/3 | 10/15/7/15/4 | 10/15/11/15/10 | 10/15/9/15/6 |
| Uneven | 25/0/1/25/0 | 10/15/11/15/10 | 10/15/8/15/5 | 10/15/7/15/4 | 10/15/11/15/10 | 10/15/11/15/10 |

All applicable coverage is `1.0` because each deterministic scenario assesses the
complete canonical topic vocabulary. Linear Algebra has 10 direct/prerequisite
applicable books and 15 not-applicable books. Operating Systems has 12 direct-applicable
books, 11 prerequisite-applicable books, and respectively 13 and 14 not-applicable
books. One Operating Systems book has a covered programming concept but no accepted
prerequisite, so it is available for B but unavailable for C, D2, and E.

The generated JSON contains the complete ordering groups for all 25 books under every
scenario and variant, plus the direct, direct-prerequisite, and transitive-prerequisite
coverage distributions.

## E tolerance sensitivity

Each cell is `groups / largest tie / singleton groups` at tolerance
`0.00 → 0.05 → 0.10`.

| Topic/scenario | Sensitivity |
| --- | --- |
| LA Beginner | 11/15/10 → 10/15/8 → 11/15/10 |
| LA Intermediate | 10/15/8 → 10/15/8 → 10/15/8 |
| LA Advanced | 11/15/10 → 9/15/6 → 9/15/6 |
| LA Uneven | 11/15/10 → 11/15/10 → 11/15/10 |
| OS Beginner | 12/14/11 → 12/14/11 → 12/14/11 |
| OS Intermediate | 11/14/9 → 10/14/7 → 10/14/7 |
| OS Advanced | 10/14/7 → 10/14/7 → 10/14/7 |
| OS Uneven | 11/14/9 → 11/14/9 → 11/14/9 |

Tolerance changes some evidence-backed grouping but cannot change the dominant
unavailable group. The sensitivity result does not justify a production constant.

## Prerequisite-only, D2, and E comparison

Agreement below is pairwise direction agreement among books for which all three axes
are available. `C2 ties` counts prerequisite-only ties, `E breaks` counts those ties
resolved by direct opportunity, and `D2 overrides` counts pairs where D2 reverses a
non-tied prerequisite-only direction.

| Topic | Scenario | Pairs | C2=D2 | C2=E | D2=E | C2 ties | E breaks | D2 overrides |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LA | Beginner | 45 | 32 (71.1%) | 28 (62.2%) | 38 (84.4%) | 7 | 7 | 5 |
| LA | Intermediate | 45 | 34 (75.6%) | 32 (71.1%) | 43 (95.6%) | 7 | 6 | 5 |
| LA | Advanced | 45 | 35 (77.8%) | 15 (33.3%) | 25 (55.6%) | 7 | 7 | 3 |
| LA | Uneven | 45 | 34 (75.6%) | 33 (73.3%) | 40 (88.9%) | 7 | 7 | 4 |
| OS | Beginner | 55 | 24 (43.6%) | 12 (21.8%) | 40 (72.7%) | 16 | 16 | 14 |
| OS | Intermediate | 55 | 10 (18.2%) | 7 (12.7%) | 52 (94.5%) | 16 | 15 | 30 |
| OS | Advanced | 55 | 34 (61.8%) | 29 (52.7%) | 50 (90.9%) | 17 | 15 | 6 |
| OS | Uneven | 55 | 25 (45.5%) | 28 (50.9%) | 44 (80.0%) | 16 | 15 | 15 |

Direct opportunity clearly adds resolution: E breaks nearly every C2 tie, and D2
usually produces more singleton groups. That is diagnostic value, not demonstrated
recommendation quality. D2 also reverses prerequisite direction frequently, most
notably 30 of 55 comparable OS Intermediate pairs. Therefore D2 has not shown a safe,
general additional value over prerequisite-only; it shows sensitivity to learning
opportunity that needs broader validation.

## Frozen ten-pair check

The fixed pair membership, A/B ordering, human labels, and legacy predictions are
unchanged. No tolerance was tuned to the labels.

| Variant | Agreement |
| --- | ---: |
| D1 | 3/10 |
| D2 | 9/10 |
| Direct only | 4/10 |
| Prerequisite only | 9/10 |
| E prerequisite first (0.05) | 9/10 |

E matches the existing 9/10 result but does not improve it. Ten AI-assisted human
diagnostic pairs are too small to estimate generalized ranking accuracy.

## Cross-profile representative cases

The report stores each case's title, covered concepts, direct and transitive accepted
prerequisites, scenario profile, coverage, all six variant group indices, B, C1, C2,
and D2 values.

### Linear Algebra

`Linear algebra.` (`book_09c91e7b456682eb19fb`) is the clearest
Beginner-to-Intermediate movement. Its C2 readiness changes `0.400 → 0.900`, direct
opportunity changes `0.850 → 0.383`, and E moves from group 5 to group 1. At Advanced,
opportunity falls to `0.067`, showing that already-known coverage stops looking like a
large learning opportunity.

`Linear algebra` (`isbn13:9781118909584`) requires high-school algebra, inner product,
linear system, matrix, orthogonality, vector, and vector space. It has the largest
Intermediate-to-Advanced opportunity reduction (`0.400 → 0.066`) and the largest
Intermediate-to-Uneven E decline (group 5 to group 7); its C2 readiness falls from
`0.771` to `0.600` under the computationally strong but abstract/geometrically weak
profile.

No evidence-backed LA book remains in the same E group across all four profiles. The 15
books without concept evidence remain together only because they are unavailable.

### Operating Systems

`Advanced concepts in operating systems` (`isbn13:9780070575721`) has accepted
prerequisites concurrency, process, storage, synchronization, and thread. From Beginner
to Intermediate, C2 readiness changes `0.240 → 0.690`, opportunity changes
`0.867 → 0.450`, and E moves from group 11 to group 1. At Advanced, opportunity falls
to `0.072`.

`Operating system concepts` (`isbn13:9780471694663`) exposes the uneven-profile effect.
Its transitive prerequisites add computer architecture and memory management to its
execution/storage foundations. Relative to Intermediate, Uneven C2 readiness changes
`0.693 → 0.636` and E moves from group 2 to group 6 despite direct opportunity remaining
`0.417`.

No evidence-backed OS book remains in the same E group across all four profiles. The 13
books without concept evidence remain together only because they are unavailable.

## Behavioral observations

- **Beginner:** high opportunity alone is insufficient. For example, Advanced concepts
  in operating systems has opportunity `0.867` but prerequisite readiness `0.240` and
  appears in E group 11, not at the top.
- **Advanced:** introductory coverage produces little remaining opportunity. The LA
  representative falls to `0.067`; the OS Modern Operating Systems representative is
  `0.073`.
- **Intermediate:** the experiment identifies books that combine preparation and
  remaining material. LA `Linear algebra.` has readiness `0.900` and opportunity
  `0.383`; OS `Advanced concepts in operating systems` has `0.690` and `0.450`.
- **Uneven:** weak abstract/geometric LA foundations and weak architecture/memory/storage
  OS foundations change relative E groups in the expected direction for the documented
  representative cases.

These are representative observations, not automatic pass/fail claims.

## Evidence gap and service options

Twenty-eight of 50 books have no matched concept evidence: 15 Linear Algebra and 13
Operating Systems. This is the dominant tie and prevents any concept-only method from
ranking the full catalog. No title, ISBN, metadata-row, source-count, or book-ID score is
invented to fill the gap.

Possible future service policies, with none selected here, are:

1. personalize only the evidence-backed candidate pool;
2. place evidence-insufficient books in a separate fallback pool; or
3. expand Data-Pipeline concept coverage before admitting those books to the personalized
   pool.

## Production-candidate assessment

- **Reject as candidate:** D1 hard coverage gate. Its threshold cliff and 3/10 frozen
  pair agreement remain; no threshold tuning was attempted.
- **Simplest candidate for further review:** transitive prerequisite-only. It is the
  easiest result to explain and retains 9/10 on the frozen pairs, but still needs broader
  human validation and a production fallback policy.
- **Needs more validation:** D2 and E. Both add useful resolution. Neither improves the
  frozen human agreement over prerequisite-only, D2 can reverse prerequisite ordering
  frequently, and E's tolerance remains an experiment parameter.

The answer to the milestone question is a qualified **yes** for evidence-backed books:
the axes react to the intended knowledge-state changes and yield explainable examples.
It is not yet sufficiently validated to design or enable ranking-v2 in production.

## Reproduction and artifact

```bash
uv run bookmatch-ml evaluate-multi-reader-concept-ranking \
  --concept-mapping data/reports/scale-50-concept-presence-v2-overlap.json \
  --fixed-reader data/output/reader_profile.json \
  --fixed-reader data/output/concept_matching_la_reader.json \
  --output data/reports/multi-reader-concept-ranking-v1.json
```

Generated report SHA-256:

```text
a93046963219a1d259ba83a3671b770ee5786d06509cfacdf1b4013cbec3c86d
```

The command was run twice and produced byte-identical output. Generated reports under
`data/reports/` remain ignored; the config, evaluator, tests, and this result document
are version controlled.
