# Concept matching v2: TOC and measured concept readiness experiment

## Scope and reproducibility

This is a transparent **experimental baseline**, separate from the unchanged `rank-v1` CLI and
`POST /ml/rank` API. V1 fits reader vocabulary to lexical difficulty, background knowledge to
concept density and prerequisite demand, and comprehension to syntactic complexity. Its basic
fit is `1 - abs(reader level - book demand)` with configured component weights and missing-value
renormalization. We retain it as a comparison baseline and retain BookProfile, ReaderProfile,
QuestionSpec, Question Difficulty v1, and all prose features.

The current ten-book canonical snapshot has TOCs for **10/10** books but analyzable textbook
prose for **8/10**. In a larger catalog, public prose may be less consistently available. A
prose-dependent fit can therefore be incomplete for reasons unrelated to book quality. V2 uses
TOC structure and assessed concept readiness as its core inputs; lexical and syntactic values
appear only as optional diagnostics. No learned model or remote service participates.

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl

uv run bookmatch-ml reader-profile \
  --input examples/concept_matching_la_assessment.json \
  --output data/output/concept_matching_la_reader.json

uv run bookmatch-ml evaluate-concept-matching \
  --data-dir ../Data-Pipeline/data/processed \
  --reader examples/reader_profile.json \
  --books data/output/book_profiles.jsonl \
  --output data/reports/concept_matching_os.json

uv run bookmatch-ml evaluate-concept-matching \
  --data-dir ../Data-Pipeline/data/processed \
  --reader data/output/concept_matching_la_reader.json \
  --books data/output/book_profiles.jsonl \
  --output data/reports/concept_matching_la.json
```

The OS reader fixture comes from the existing assessment example. The LA assessment is
**synthetic**, created only to exercise the existing ReaderProfile scoring path. Neither fixture
establishes diagnostic validity. Reports are ignored under `data/`; they retain hashes of all
four canonical files, the reader, v1 profiles, and exact graph, feature, and matching configs.
The command rebuilds v1 profiles from canonical evidence before comparing, so stale v1 output
fails visibly.

## Three separate objects

The **TOC tree** is one book's ordered hierarchy. The canonical `parent_entry_id`, `level`, and
`order_index` restore the tree; each node retains its title, label, parent path, full path, and
depth-first traversal position. The v2 builder rejects orphans, wrong levels, duplicate sibling
positions, and unreachable nodes. The existing canonical loader also validates book/source joins.

The **domain concept graph** in `configs/concept_graph.yaml` is independent of that hierarchy.
Every edge means `prerequisite_candidate -> dependent concept`, and the graph must be a DAG.
Unknown nodes, self-loops, duplicate edges, cross-topic edges, and cycles fail validation. Each
edge carries its topic, relation type, source type, and version. The `proposed_seed` source is a
curated **candidate for human review**, not a proven prerequisite. This first graph has 17 OS
nodes, 20 LA nodes, and 24 directed edges. Graph nodes are checked against the existing feature
lexicons. The graph is not inferred from TOC parent-child links.

The **book concept profile** maps a TOC tree to graph nodes. It reuses normalized phrase aliases
from `configs/features.yaml` and records the matching alias, TOC entry ID and title, level,
order, parent path, full path, and traversal position. Distinct phrases in one title may map to
multiple concepts. If the same normalized alias would map to different concepts, that entry is
marked ambiguous. Unmatched titles remain in the artifact; neither case is guessed from context.
Only nodes in the versioned graph are mapping targets.

For a covered concept, `coverage_weight = min(1, occurrence_count / 3)`. This intentionally
simple rule has no prose input and no concept-difficulty meaning. The profile keeps occurrence
count, shallowest TOC level, first position, all TOC entry IDs and paths, matching audit rows,
unmapped rows, counts, versions, and hashes. Repeated headings can saturate this weight, so it
should be read as a structural coverage proxy rather than page depth or educational emphasis.

## Prerequisite projection and two axes

The graph's ancestor paths are projected only for concepts mapped in the book. For each target:

- If a prerequisite concept first appears **before** the target in the same book, it is a
  `taught_before_use_candidate`.
- If it is absent or its first appearance is at/after the target, it is an
  `external_or_late_candidate` and enters v2 readiness requirements.

All graph paths and target TOC IDs remain visible. Repeated paths to the same prerequisite are
deduplicated for scoring; the requirement weight is the **maximum target coverage weight**.
The same prerequisite can appear in both categories for different targets. TOC order is only a
teaching-order proxy: a heading before another does not prove that the first concept was taught
adequately or required beforehand.

V2 consumes the existing `ReaderProfile.concept_readiness` values, keyed by canonical concept
ID. The existing difficulty-weighted assessment calculation is unchanged. A score of `0` is a
measured zero; an absent concept ID means **unmeasured**.

For external/late prerequisite weights `R(c)` and measured mastery `U(c)`:

```text
readiness_score = Σ_assessed R(c) U(c) / Σ_assessed R(c)
readiness_assessment_coverage = Σ_assessed R(c) / Σ_all R(c)
```

For TOC coverage weights `B(c)`:

```text
opportunity_score = Σ_assessed B(c) (1 - U(c)) / Σ_assessed B(c)
opportunity_assessment_coverage = Σ_assessed B(c) / Σ_all B(c)
```

For example, two equal-weight assessed prerequisites at mastery `1.0` and `0.0` yield
readiness `0.5` and coverage `1.0`. If only the first is measured, readiness is `1.0` with
coverage `0.5`; the unknown second concept is never filled with zero. No external candidate
means readiness is `null` with `not_applicable` semantics. Candidates exist but none are
assessed means readiness is `null`, coverage `0`, and `unassessed` semantics. The same rules
apply to opportunity if no book concepts map or none are assessed.

Cosine similarity would rate an already-mastered book highly because its coverage vector and
user-mastery vector align. V2 instead keeps readiness and possible new learning separate.
Results expose mastered/weak/unknown prerequisite IDs, already-known/learning-candidate/unknown
book concepts, TOC evidence limitations, and both axes. The `known >= 0.8` and
`learning candidate < 0.6` display groups are experimental thresholds in
`configs/concept_matching.yaml`, not validated educational labels.

The batch display policy retains **every** book. It first marks insufficient assessment
coverage (`< 0.5` on a required axis), then a challenge candidate when measured readiness is
`< 0.6`; other candidates are eligible. Eligible books are ordered by opportunity, then
readiness and stable `book_id`. These configurable thresholds are an experiment, not a
production ranking policy. There is no v2 weighted total or easy/appropriate/hard classifier.

## Current ten-book snapshot

Counts are `matched TOC entries / total TOC entries`; `C` is distinct mapped concepts. Axes
show `score / assessment coverage`. `—` means a score cannot be interpreted. V1 is shown with
its original component-weight coverage in parentheses. Each row is one actual canonical book,
abbreviated for readability.

| Operating Systems book | TOC | C | External / taught candidates | Readiness | Opportunity | V1 | V2 status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stallings, *Operating Systems* | 60/201 | 15 | 2 / 7 | .800 / .500 | .261 / .529 | 1.000 (.35) | eligible |
| *Think OS* | 23/65 | 9 | 5 / 2 | .486 / .636 | .258 / .706 | .651 (1.00) | challenge candidate |
| *Operating Systems and Middleware* | 59/179 | 11 | 7 / 1 | .620 / .714 | .200 / .483 | .737 (1.00) | insufficient evidence |
| *xv6* | 16/99 | 8 | 6 / 2 | .360 / .417 | .190 / .714 | .697 (1.00) | insufficient evidence |
| *Operating Systems: Three Easy Pieces* | 24/62 | 11 | 5 / 3 | .250 / .182 | .130 / .455 | .705 (1.00) | insufficient evidence |

| Linear Algebra book | TOC | C | External / taught candidates | Readiness | Opportunity | V1 | V2 status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Strang, *Introduction to Linear Algebra* | 35/63 | 15 | 5 / 4 | .629 / 1.00 | .417 / .714 | 1.000 (.35) | eligible |
| *Linear Algebra* | 29/96 | 11 | 4 / 3 | .800 / 1.00 | .358 / .960 | .597 (1.00) | eligible |
| *Elementary linear algebra* | 8/9 | 8 | 2 / 1 | .667 / 1.00 | .337 / .800 | .634 (1.00) | eligible |
| *Linear Algebra with Applications* | 72/167 | 17 | 5 / 4 | .575 / 1.00 | .450 / .674 | .678 (1.00) | challenge candidate |
| *Understanding Linear Algebra* | 61/222 | 15 | 5 / 4 | .540 / 1.00 | .434 / .778 | .643 (1.00) | challenge candidate |

For every row, `unmatched = total - matched`; there were zero ambiguous rows in this snapshot.
Thus the unmatched counts are respectively 141, 42, 120, 83, 38 for OS and 28, 67, 1, 95,
161 for LA. These titles remain available in each report for lexicon review. The report also
contains every concept weight and path, external/taught candidate, learning candidate, and
unknown mastery concept, not just the table summaries.

## V1 versus v2 cases and weak signals

- Stallings receives v1 `1.000` because only topic fit is available: its v1 component-weight
  coverage is `.35`. V2 maps 15 concepts, with measured opportunity `.261` at only `.529`
  assessment coverage. The measured portion is more review-like than growth-like; eight of
  its 15 mapped concepts remain unmeasured. V2 does **not** claim low opportunity for the whole
  book.
- The LA *Linear Algebra* book receives a relatively lower v1 `.597` but has 11 mapped TOC
  concepts and `.358` measured opportunity at `.960` assessment coverage. Its measured learning
  candidates include basis, determinant, eigenvalue, eigenvector, and orthogonality. This is a
  useful content explanation that a scalar absolute gap does not provide.
- *Think OS* has readiness `.486` at `.636` coverage. Measured weak external candidates include
  computer architecture and memory management. It is a challenge candidate, while its nine
  mapped book concepts and `.258` opportunity remain visible. The status reflects the synthetic
  reader fixture and proposed graph, not verified teaching difficulty.
- *Three Easy Pieces* has prerequisite assessment coverage `.182` and opportunity coverage
  `.455`; v2 therefore withholds an eligibility judgment despite v1 `.705`. Its unmeasured
  mapped concepts include file system, security, and virtualization. This is a missing-mastery
  case, not a zero-mastery case.
- Strang's introduction is another v1 topic-only `1.000` with `.35` weight coverage. V2 sees
  15 mapped concepts and `.417` measured opportunity. The two methods are answering different
  questions. Its TOC coverage does not prove the reader will learn every mapped item.

Mapping gaps are substantial, especially for *xv6* (16/99) and *Understanding Linear Algebra*
(61/222). Some headings are organizational or outside the fixed lexicon; some may expose
missing aliases. The 9-entry TOC for *Elementary linear algebra* has high entry mapping but
coarse structural detail. Repeated TOC headings saturate weights. First occurrence can misread
an early overview as teaching. Graph paths can propagate an unreviewed edge. Self-reported or
synthetic mastery may be noisy. These limitations should be examined before any API change.

## Next decision gate

Human reviewers should first inspect graph edges, TOC mappings, and a small set of book/reader
cases. In particular, confirm whether the proposed prerequisite directions and first-occurrence
heuristic are defensible. Then collect authored concept assessments and empirical reader
feedback before calibrating eligibility thresholds or changing the production rank policy.
Only after this gate should the team test a larger Data-Pipeline snapshot. Question generation
belongs in separate work; this experiment emits no question text.
