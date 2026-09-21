# Concept difficulty and reader fit (experimental v2)

## Purpose and scope

This extends the v3 TOC matcher and prerequisite graph with an explicit curriculum
rubric, an intrinsic book score, and a separate reader-dependent learning-burden calculation.
It does not infer prose complexity from chapter headings. It is a proposed baseline, not a
validated human difficulty estimator. Existing `/ml/rank` behavior remains the default;
`rankingStrategy=concept_difficulty_v2_experimental` is an opt-in comparison path.

## Criteria

`configs/concept_difficulty.yaml` assigns each configured OS and linear-algebra concept a
proposed level: 1 foundational, 2 intermediate, 3 advanced *within the topic*. These are
curriculum judgments to review, not levels learned from data. Example: process = 1,
concurrency = 2, deadlock = 3. Graph depth alone is not used as a difficulty label.

### Intrinsic book difficulty

Repeated headings never add scoring mass. Let `U` be the unique matched covered concepts and
`E` the unique external-or-late prerequisite candidates projected by the graph. The components
are:

```text
covered content C = mean((level(c) - 1) / 2 for c in U)
external prerequisite P = weighted mean(level(p) / 3 for p in E)
book difficulty D = 0.7 * C + 0.3 * P
```

If `E` is empty, active weights are renormalized and `D=C`; absence of an inferred prerequisite
does not inject a zero. `C` maps levels 1/2/3 to 0/0.5/1. `P` maps them to 1/3, 2/3, and 1 because
even a foundational prerequisite adds prior-knowledge demand. Each prerequisite's structural
weight is the number of unique covered target concepts that depend on it divided by the number of
unique covered concepts. Heading occurrence counts never enter this weight. All numbers are
initial assumptions.

| Intrinsic score | Korean meaning | API label | Operational meaning |
| --- | --- | --- | --- |
| `0.00 <= D <= 0.33` | 입문 | `introductory` | mostly foundational covered concepts and limited external prerequisite demand |
| `0.33 < D <= 0.66` | 중급 | `intermediate` | intermediate concepts or a meaningful external prerequisite structure |
| `0.66 < D <= 1.00` | 고급 | `advanced` | advanced concepts and/or high-level external prerequisite demand |

The legacy `book_level_score` (mean level, range 1–3) remains in comparison output, but ranking
uses `book_difficulty_score`. The intrinsic score is identical for every reader.

### Reader-specific burden

For reader mastery `m(c)` in [0,1]:

```text
covered learning L = mean(level(c) / 3 * (1 - m(c))) over unique covered concepts
prerequisite gap G = weighted mean(1 - m(c)) over external/late prerequisite candidates
personal burden B = 0.6 * G + 0.4 * L
```

When there is no external prerequisite candidate, active weights are renormalized and `B=L`.
This fixes the earlier v1 behavior in which such a book could never exceed burden 0.4.
Prerequisite candidates and their weights come from the existing v2 graph projection.
Concepts taught before use are not charged as external requirements; learning them still
contributes to L. If no external requirement is projected, G's contribution is zero, with
the empty prerequisite list preserved. This means no requirement was inferred, not proof
that no prerequisite exists. TOC order and proposed graph edges remain imperfect proxies.

Initial personal-burden bands:

| Burden | Label |
| --- | --- |
| ≤ 0.15 | easy |
| > 0.15 and ≤ 0.45 | manageable |
| > 0.45 | challenging |

Recommendation utility peaks at target burden 0.30. It decreases linearly to zero at burden
0 and 1. This policy targets learning something new, so a fully mastered book can be easy
but receive low learning utility; it is not a general measure of book quality. The target,
weights, bands, and minimum assessment coverage are configurable and uncalibrated.

Unknown mastery is not silently assigned zero or the user's overall score. The calculation
returns lower/upper burden bounds by assigning unknown values their best/worst possible
values. These are logical bounds, not statistical confidence intervals. A band and utility
are returned only when the bounds agree on a band and at least half of burden coefficient
mass is assessed. Utility is the worse endpoint utility. The script emits the next concepts
to assess, in descending influence order. With no mapped concepts there is no numeric fit.

The returned breakdown records concept levels, mastery, coefficients, prerequisite identities,
taught-before-use identities, every matched TOC entry/path/alias, graph paths for prerequisite
reasons, and policy/graph/mapping/input hashes. A numeric recommendation score is an experimental
utility, not a percentage probability of suitability.

## Evidence-quality rules

- A concept receives one scoring unit regardless of repeated headings. Every occurrence remains
  in `concept_evidence` for audit.
- A prerequisite appearing earlier in depth-first TOC order is `taught_before_use`; it is removed
  from external demand but remains covered learning. Appearing later remains an external-or-late
  candidate.
- Phrase matching is boundary-aware. Configured false-positive exclusions (for example,
  `compilation process` and `disk scheduling`) are applied before matching and recorded as
  `excluded_alias` with the excluded concept IDs. An alias shared by multiple concepts is
  preserved as `ambiguous_alias` and is not guessed. Exclusion events are also stored separately,
  so `Disk Scheduling` retains the suppressed `scheduling` match even while `storage` maps.
- A short or partial TOC is not penalized with an invented numeric factor. Exact total/matched
  counts, paths, hashes, and unmapped entries are returned. No mapped concepts means no score;
  it does not mean the book is easy, hard, or irrelevant.
- These rules estimate concept/prerequisite difficulty only. Prose difficulty continues to require
  actual prose evidence and remains a separate profile.

## Optional `/ml/rank` path

The default request remains `rankingStrategy=baseline_v1`. The opt-in value
`concept_difficulty_v2_experimental` requires each candidate to carry its batch-produced
`conceptProfile`. Candidates without sufficient concept evidence or reader assessment coverage
do not receive an experimental score. The response retains baseline component diagnostics and adds
`conceptDifficulty` with intrinsic score/band, personal burden interval, TOC coverage diagnostics,
unmapped/excluded entries, TOC matches, graph paths, active weights, and hashes. Top-K applies the
same topic filtering as the baseline before validating experimental profiles. This ML contract is
implemented; the current Backend PRs do not yet persist or send `conceptProfile`.
New batch artifacts use `toc-concept-profile-v3` / `concept-matching-config-v3`. Older v2
artifacts remain loadable; their newly introduced exclusion-audit fields default to empty/zero.

## Reproduce

```sh
uv sync --frozen
uv run python scripts/compare_concept_difficulty.py \
  --data-dir ../Data-Pipeline/data/processed \
  --reader examples/reader_profile.json \
  --output-dir data/reports/concept-difficulty-run1
```

The output directory must be new. Outputs are ignored local artifacts:

- `comparison.json`: v1 comparison, provided reader, and synthetic concept-mastery scenarios
  (0, 0.5, 1 for every topic concept; overall three-dimensional reader scores stay fixed).
- `blind-review.csv`: no model predictions, rubric scores, or synthetic labels.

## 25-book snapshot result (2026-09-21, rerun with v2)

Input: the earlier `toc-enriched-20260920/processed` snapshot documented in
[canonical HTTP verification](canonical-http-verification-v1.md), not a fresh crawl.
Seven books have TOC; six map to the configured OS concepts. Nineteen receive no fabricated
concept score. This does not classify those nineteen as irrelevant or difficult.

The six mapped books span legacy concept levels 1.50–2.11 and intrinsic v2 scores 0.275–0.506.
One is introductory and five are intermediate under the proposed thresholds. For synthetic
intermediate mastery their burden is 0.400–0.441 and utility 0.799–0.857, instead of the original
ranking path's all-1.0 scores.
All six are challenging with synthetic zero mastery, manageable at 0.5, and easy at 1.0.
These checks show sensitivity to the specified criteria, **not improved recommendation accuracy**.

The provided example reader leaves prerequisite concepts unassessed, so its six personal
bands still require additional assessment. Requested concepts include programming, thread,
deadlock, distributed systems, storage, file system, input/output, protection, and security.
Use the reported per-book coefficients to select useful diagnostic questions; do not mark
those concepts mastered automatically.

## Independent review and accuracy gate

1. Two reviewers independently fill the blind sheet using book evidence and the provided
   reader profile. Record topic relevance, content level 1–3, personal band, sources and reasons.
2. Assign `work_group` to group editions of the same work. Decide train/development/held-out
   splits by work group before tuning; never place editions of one work in different splits.
   `split=unassigned` deliberately requires this decision rather than silently splitting ISBNs.
3. Compare disagreements and agree labels before looking at predictions. Do not copy rubric
   levels or synthetic-reader results into human labels.
4. Tune only with development labels. On held-out books, compare ordinal-level error and
   pairwise level agreement, personal-band confusion, and top-K recommendation relevance
   against the frozen v1 and v2 baselines. Report abstention/coverage alongside metrics.
5. Also evaluate TOC concept mapping with the existing gold-review workflow. Wrong concept
   matches or wrong topic labels cannot be repaired by tuning burden weights.

No reviewer labels or accuracy numbers were invented for this change. Existing topic audit
and prose acquisition remain separate work; Data-Pipeline was not modified.

## Verification

Regression tests cover expertise/burden monotonicity, higher-level content, intrinsic prerequisite
structure, active-weight renormalization, teaching order, unknown mastery, absent concepts, topic
mismatch, repeated headings, evidence traceability, optional API ranking, learning opportunity,
and rubric completeness against the graph. The current verification result is recorded in the PR.
