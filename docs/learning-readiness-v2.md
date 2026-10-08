# Learning readiness v2: implementation plan and contract

## Scope
Implement the approved concept-prerequisite reading checklist in `/ml/learning-fit`.
Do not alter `/ml/rank`, v1 ranking, frontend, production databases or source texts.
Keep the old unfinished checklist worktree untouched. No absolute book difficulty,
unvalidated mastery threshold, or claims of accuracy improvement.

### Task 1: optional ML policy
- RED: covered prerequisites cannot hide observed mistakes; unknown != incorrect;
  graph order, count evidence, duplicate evidence invariance, v1 compatibility.
- GREEN: `modelVersion=concept-learning-v2`; v1 remains the omitted default.
- v2 prerequisites are all accepted graph ancestors, including covered ones.
  Distinguish internal/external candidates; TOC presence never waives a requirement.
- Preserve prior-knowledge observation counts and the existing categorical states.
  Return a versioned checklist in prerequisite order, with next actions and caveats.
- v2 sort: foundation status, new learning vs review, established graph, practice
  availability, stable book ID. Scores and source counts never enter the sort.

### Task 2: evidence handoff and Backend
- RED: validate and preserve optional concept evidence, reject foreign concepts,
  malformed/contradictory evidence and tampered checklist; immutable replay.
- Keep existing frozen matching artifacts unchanged. An explicit enrichment tool
  joins them with the source mapping, writing a new candidate artifact.
- Backend accepts optional source references on imported covered concepts, forwards
  them only for v2, validates response and stores its exact input/output snapshot.
- Public selector is optional; existing frontend/v1 requests stay compatible.

### Task 3: verification and review
- Full ML pytest/Ruff; Backend clean build on PostgreSQL/Testcontainers.
- Actual committed 50-book snapshot comparison with synthetic readers, not a claim
  of live collection or human accuracy. Determinism and monotonic state checks.
- Fresh independent whole-change review. Commit and push feature PRs, no merge.

## Acceptance
Same book facts for every reader; absent observations are null/unmeasured, never 0.
One correct answer is an observed correct answer, not proven mastery. Source links
must identify supplied evidence and edition relation; missing evidence remains empty.
Document graph paths, TOC limitations, unsupported depth and human-review template.

## Progress / rulings
- Ruling: use an opt-in v2 instead of silently changing the running v1 policy; costs
  an explicit request/deployment choice but preserves existing clients and comparison.
- Ruling: use latest main in new worktrees; old incomplete changes remain preserved.
- Baseline Windows API execution required normal-user elevation; sandbox run stalled.
  First elevated suite had 262 passes/104 temp-directory permission errors. Re-run
  with a new explicit test-temp directory rather than changing application code.

## API and exact policy

`POST /ml/learning-fit` accepts optional `modelVersion: "concept-learning-v2"`.
Omitting it or sending `concept-learning-v1` retains the existing result contract.
Backend exposes the selector as `POST /api/learning-recommendations?modelVersion=concept-learning-v2`;
the existing JSON body (userId, topicId, profileId, ability, topK) is unchanged.
Retries with the same idempotency key cannot switch policy; GET returns the stored
snapshot, not a calculation using today's graph or book data.

For covered set C and accepted graph ancestor set A(C), v1 uses A(C) minus C;
v2 uses all of A(C). Internal candidates = A(C) intersect C, external = A(C) minus C.
These book facts are independent of the reader. No new numerical difficulty score,
beginner/intermediate/advanced label, weight, or mastery threshold is introduced.
Only prior-knowledge responses for the requested ability are used:

- No responses: `unmeasured`, null counts, `assess-concept`.
- At least one wrong response: `needs-practice`, actual counts, `review-concept`.
- All observed responses correct: `correct`, actual counts, `continue-learning`.
  One correct response is NOT sufficient evidence of mastery.

Any prerequisite candidate needing practice yields `foundation-gap`; otherwise any
unmeasured candidate (or no known prerequisites) yields `check-first`; otherwise
`ready-to-explore`. Improving the same assessed answers cannot worsen this category.
It does not promise rank monotonicity: review/new-content preference is a separate rule.
V2 ranks category first, then non-review, established foundation, presence of practice
targets, finally book ID. No claimed accuracy benefit; human evaluation is still needed.

The checklist is topological layers of known prerequisites, then stable concept ID
within a layer. Graph-disconnected concepts have no defensible relative study order;
this is a checking list, NOT a complete curriculum or personalized optimal syllabus.
`dependsOn`/`requiredFor` expose immediate edges. Counts, source URL, TOC path, matched
alias, source/evidence IDs, edition relation and provenance hash remain visible.
Duplicate evidence is removed and never multiplies rank or scores. Conflicting IDs,
foreign concept refs and non-HTTP(S) links fail validation.

TOC presence, even earlier in the TOC, does not prove adequate instruction. Internal
prerequisites therefore remain checks, not a claim that another book is required.
Every row has `teachingSufficiency=unverified`. Partial TOCs, keyword false positives,
different editions and unknown teaching depth must be reviewed at the supplied source.
We do not infer book completeness or sequence from missing source metadata.

## Evidence handoff and deployment

```
PYTHONPATH=src python scripts/prepare_learning_evidence.py \
  --candidates matching-candidates.jsonl --mapping concept-mapping.json \
  --output learning-candidates.jsonl
```

The script requires exactly matching book IDs, topics and concepts. It writes a new
artifact and hash, retaining weights and appending `-evidence-v1` to feature_version.
Never rename this output over the frozen v1 candidates. It is an optional Backend
import format, NOT a replacement for MatchingBookProfile or `/ml/rank` input.
Use an existing local-catalog-import-v1 manifest with a NEW snapshot ID, the new
candidate path and output SHA-256. Review the evidence first and use a fresh test DB.
The importer intentionally refuses replacing another active projection. Transitioning
an existing catalog requires separate explicit activation work; this PR does not
silently update live data. Old projections still work with an empty evidence list.

Deployment order: ML v2 support, then Backend support, then reviewed evidence import
and explicit v2 client choice. Frontend is unchanged, defaults stay v1. No production
deployment/database mutation is part of this change. Backend adds no schema migration.
Backend validates the counts, states, dependency consistency, concept sets and exact
source refs against its sent input before saving; ML remains the owner of graph truth.

## Reproducible real-book comparison

`PYTHONPATH=src python scripts/compare_learning_readiness.py --output new-report.json`
reproduces [the committed report](learning-readiness-comparison.json) from the historical
50-book snapshot (not the newer live catalog). It does NOT reuse fractional reader
scores as answers. It uses explicitly synthetic all-wrong/all-correct one-response
scenarios and the current reviewed graph. Ten LA and twelve OS books have concepts;
28 are unmapped and are not silently scored. No unsupported concept was excluded.

- ISBN 9780130319999: 15 covered concepts; v1 removes every covered prerequisite
  (zero remaining, check-first). V2 retains 7 internal candidates including process,
  thread, concurrency and synchronization. All-wrong gives foundation-gap; all-correct
  gives ready-to-explore. Book facts are identical in both reader scenarios.
- ISBN 9780131860612: 10 covered concepts; v1 checks high school algebra and vector.
  V2 also checks matrix and vector space despite their TOC presence. All-wrong remains
  foundation-gap, now with four explicit prerequisite candidates instead of two.

These verify rule changes, not identification accuracy, source completeness, current
availability, or benefit to real learners. The snapshot lacks source refs: comparison
results intentionally do NOT invent URLs or TOC paths. Evidence preservation is
separately tested with explicitly synthetic fixtures and source-aware mapper fixtures.

## Human review before rollout

Fill [the review template](learning-readiness-review.csv), using actual source and edition.
Judge separately: TOC match correctness; prerequisite relation validity; whether the
book teaches that prerequisite sufficiently; diagnosis coverage; usefulness of the
recommended next action. Leave unknown fields empty, not false or zero. Use at least
two independent reviewers for a small pilot and record disagreements/adjudication.
Only after genuine learner/reviewer labels exist should pairwise preference, action
agreement or learning outcomes be reported. Do not treat test passes as those metrics.

## Verification (2026-10-04)

- Baseline latest main: 366 tests passed.
- Final: 375 tests passed, Ruff check and format check passed (150 files).
- Two dependency deprecation warnings remain (Starlette/httpx and AnyIO alias).
- Tests exercise v1 compatibility, internal prerequisites, observation states/counts,
  reader-independent book facts, stable order, duplicate/conflicting evidence,
  immutable enrichment and deterministic 50-book comparison.
- New behavior is not enabled by default and real-user accuracy remains unmeasured.
