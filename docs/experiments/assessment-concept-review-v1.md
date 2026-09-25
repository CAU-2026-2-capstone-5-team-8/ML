# Assessment concept review v1

## Purpose and boundary

This milestone adds a deterministic human-review gate immediately before assessment target
selection. It does not modify book concept extraction, prerequisite inference, the accepted
recommendation graph, matcher behavior, ReaderProfile calculation, or ranking.

These three statements are intentionally different:

- **Concept evidence**: a canonical concept appears in available book evidence.
- **Prerequisite relation**: book evidence suggests that knowing a concept may help before reading.
- **Assessment-worthy concept**: asking about the concept can meaningfully distinguish a user's
  knowledge of the selected topic.

Therefore:

> `prerequisite` does not automatically imply `assessment-worthy`.

The review key is `(topic_id, concept_id, concept_role)`. Covered and prerequisite roles are
reviewed independently.

## Versioned artifacts

- Human decisions: `configs/assessment_concept_reviews.yaml`
- Review schema: `assessment-concept-review-v1`
- Reviewed assessment config: `configs/assessment_reviewed.yaml`
- Reviewed config version: `assessment-config-v2-reviewed`
- Generated packet: `data/reviews/assessment_concept_review_operating_systems_v1.json`
- Generated packet version: `assessment-concept-review-packet-v1`

The decision file is source controlled. The evidence packet is generated and ignored. The
effective reviewed config hash is a deterministic digest over the reviewed assessment config hash,
the exact review artifact hash, and the review schema version. `QuestionSpec` remains
`question-spec-v1`, and the blueprint schema remains `assessment-blueprint-v1`.

## Status and reason codes

Statuses:

- `eligible`: may participate in reviewed assessment selection.
- `ineligible`: must not become a QuestionSpec primary target.
- `unreviewed`: has no authoritative human decision and is excluded in reviewed mode.

Every decided row requires a nonblank `review_note`. `ineligible` also requires one of:

- `too_general`: likely answerable from general computer knowledge without studying the topic.
- `too_ambiguous`: the concept boundary is too unclear for a stable diagnostic target.
- `low_diagnostic_value`: valid concept, but unlikely to separate topic knowledge levels well.
- `incidental_or_contextual`: appears as an example, environment, or side context rather than a
  meaningful assessment target.
- `misclassified_prerequisite`: inferred as a prerequisite, but not suitable as a diagnostic
  prerequisite for this topic.
- `other`: another human-documented reason.

Eligible rows have no ineligible reason code. Unreviewed rows have neither a reason nor a note.
Duplicate review keys, unknown topics, stale concept-role pairs, and invalid status/reason
combinations fail validation. Stale decisions are reported and never deleted automatically.

## Why `programming` became the current target

The current result is fully explained by the existing deterministic pipeline:

1. Canonical book `isbn13:9781985086593`, *Operating Systems: Three Easy Pieces*, contains
   introduction document `doc_b1b65cb102f21471920e`.
2. The prerequisite lexicon maps canonical `programming` to aliases including
   `computer program`.
3. The introduction says that a reader **should already** understand what a computer program does.
   `should already` is a configured explicit prerequisite cue.
4. The same alias occurrence is inside the configured first 5,000 prose characters, so it also
   receives the early-prose proxy signal.
5. One explicit occurrence at weight `1.0` plus one early occurrence at weight `0.3`, divided by
   saturation `2.0`, yields book prerequisite weight `0.65` and method
   `explicit_and_early_prose_proxy`.
6. The concept occurs as a prerequisite in 1 of 5 OS books, so coverage is `0.2`.
7. Existing assessment priority is `0.7 × 0.65 + 0.3 × 0.2 = 0.515`.
8. That is prerequisite priority rank 1. The legacy prerequisite self-assessment limit is 4, so it
   enters the selected set.
9. The background-knowledge recall quota is 3. Existing priority order makes `programming` the
   first recall target, producing `q_d2e9c3e6ca7f1ac805a4`.

This path comes directly from BookProfile prerequisite feature inference. It does **not** come from
the accepted recommendation prerequisite graph, matcher, or ranking-v2.

## Operating Systems baseline and review queue

Before review (`assessment-config-v1`):

- selected covered concepts: 8
- selected prerequisite concepts: 4
- vocabulary QuestionSpecs: 6
- background-knowledge QuestionSpecs: 3
- comprehension QuestionSpecs: 5
- total QuestionSpecs: 14
- shortages: 0

The default reserve is 3 per role. The first OS queue contains:

- covered candidates: 11
- prerequisite candidates: 5
- total concept-role rows: 16

The generated packet includes priority, coverage, mean book weight, evidence types, prerequisite
methods, supporting books, compact evidence references, legacy selection/QuestionSpec status, rank,
and the authoritative human-decision fields from the review artifact. Before review those fields
were empty; after review they contain the decisions documented below. The packet contains no
automatic eligibility suggestion.

### `programming / prerequisite`

- priority rank: 1
- assessment priority: `0.515`
- mean book weight: `0.65`
- book coverage: `1/5` (`0.2`)
- source book: `isbn13:9781985086593`
- evidence: `introduction / doc_b1b65cb102f21471920e / 1 mention`
- prerequisite method: `explicit_and_early_prose_proxy`
- legacy self-assessment selected: yes
- legacy QuestionSpec target: yes
- review status: `ineligible / too_general`

### `process / covered`

- priority rank: 1
- assessment priority: `1.0`
- mean book weight: `1.0`
- book coverage: `5/5` (`1.0`)
- evidence types: description, introduction, preface, sample chapter, TOC
- supporting books: `book_21ac29d979effd89fad2`, `book_28fad378fb6e900626f2`,
  `book_7d22aef622717f0ad24b`, `isbn13:9780130319999`, `isbn13:9781985086593`
- legacy self-assessment selected: yes
- legacy QuestionSpec target: yes
- review status: `eligible`

## User-approved Operating Systems decisions

The human decision set reviews all 16 rows in the first OS queue. These are user-approved human
review decisions, not automatic labels or heuristic suggestions.

- reviewed: `16/16`
- eligible: `11`
- ineligible: `5`
- unreviewed: `0`
- covered eligible (`9`): process, thread, scheduling, synchronization, concurrency, virtual
  memory, file system, deadlock, virtualization
- covered ineligible (`2`): security (`low_diagnostic_value`), protection (`too_ambiguous`)
- prerequisite eligible (`2`): computer architecture, assembly language
- prerequisite ineligible (`3`): programming (`too_general`), algorithms
  (`low_diagnostic_value`), data structures (`low_diagnostic_value`)

`programming` is excluded only as an assessment primary target. Its book evidence and inferred
prerequisite feature remain unchanged for recommendation-side use. The human rationale is that
general programming knowledge is too broad to distinguish operating-systems readiness.

The exact decision artifact hash is
`sha256:e816f92b81138851e048f40ef879cabd0439053f3a901e8064e1710c25005383`.
Combined with `configs/assessment_reviewed.yaml` and the schema version, it produces effective
reviewed config hash
`sha256:3d94b26001746192b43e227253746e77e9c1d3ccd17f5871b1c87e61779b6d77`.

## Reviewed selection and shortages

Reviewed mode preserves the existing assessment-priority order, filters it to explicit `eligible`
rows, and then applies the existing role limits. `ineligible`, `unreviewed`, and absent decisions
are never fallback candidates. If too few eligible concepts remain, the existing
`QuestionSpecShortage` mechanism records requested and produced counts.

With the completed OS decision set, the reviewed blueprint selects:

- covered (`8`): process, thread, scheduling, synchronization, concurrency, virtual memory, file
  system, deadlock
- prerequisite (`2`): computer architecture, assembly language

Nine covered concepts are eligible, but the existing covered self-assessment limit is eight, so
lower-priority `virtualization` remains eligible without entering this blueprint. Both eligible
prerequisites enter the selected set. The background recall quota requests three targets, so the
blueprint records one explicit shortage (`requested=3`, `produced=2`) rather than backfilling an
ineligible prerequisite.

The reviewed blueprint contains 13 QuestionSpecs:

- vocabulary: `6` — four recognition targets (process, thread, scheduling, synchronization) and
  two comparison targets (process with thread, synchronization with concurrency)
- background knowledge: `2` — computer architecture and assembly language
- comprehension: `5` — unchanged quota fulfillment over the eligible covered selection

`programming`, `security`, `protection`, `algorithms`, and `data structures` do not appear as
QuestionSpec primary targets. The legacy `assessment-config-v1` artifact remains byte-identical at
SHA-256 `3822916d24b155b50ee3efb2581b43aa5587926eb36b3f7e958bb1f740d47f9c`.

Question-Generation compatibility was checked read-only using its existing `render-prompt` CLI.
The reviewed `process` vocabulary/recognize/level-1 spec and `computer architecture` background
knowledge/recall/level-1 spec both preserve `question-spec-v1`, concept identity, difficulty,
reviewed config provenance, and `ko-KR` output policy. Live Gemini generation was not run because
the process environment did not contain `GEMINI_API_KEY`.

## Adding a topic

1. Collect canonical books and evidence.
2. Build deterministic BookProfiles and the topic concept pool.
3. Generate the bounded high-priority review packet with a small reserve.
4. Add authoritative concept-role decisions to the source-controlled review file.
5. Regenerate and validate the reviewed blueprint.
6. Send unchanged `question-spec-v1` targets to Question-Generation.
7. Human-review generated question quality separately.

If ineligible decisions exhaust a queue before quotas are met, increase the role reserve and review
the next priority window. Never backfill with unreviewed concepts.
