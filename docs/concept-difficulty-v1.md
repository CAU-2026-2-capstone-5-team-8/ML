# Concept difficulty and reader fit (experimental v1)

## Purpose and scope

This extends the existing v2 TOC matcher and prerequisite graph with an explicit curriculum
rubric and a reader-dependent learning-burden calculation. It does not infer prose complexity
from chapter headings. It is a proposed baseline, not a validated human difficulty estimator.
Existing `/ml/rank`, v1 ranking, and v2 matching behavior remain unchanged for comparison.

## Criteria

`configs/concept_difficulty.yaml` assigns each configured OS and linear-algebra concept a
proposed level: 1 foundational, 2 intermediate, 3 advanced *within the topic*. These are
curriculum judgments to review, not levels learned from data. Example: process = 1,
concurrency = 2, deadlock = 3. Graph depth alone is not used as a difficulty label.

The intrinsic book level is the arithmetic mean of levels of unique matched taught concepts
(range 1–3). Repeated headings do not increase a concept's influence. This is a summary of
mapped content; it is not a claim about the book's explanation quality or entire contents.

For reader mastery `m(c)` in [0,1]:

```text
learning burden L = mean(level(c) / 3 * (1 - m(c))) over unique covered concepts
prerequisite gap G = weighted mean(1 - m(c)) over external/late prerequisite candidates
personal burden B = 0.6 * G + 0.4 * L
```

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
taught-before-use identities, and policy/graph/mapping/input hashes. A numeric recommendation
score is an experimental utility, not a percentage probability of suitability.

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

## 25-book snapshot result (2026-09-21)

Input: the earlier `toc-enriched-20260920/processed` snapshot documented in
[canonical HTTP verification](canonical-http-verification-v1.md), not a fresh crawl.
Seven books have TOC; six map to the configured OS concepts. Nineteen receive no fabricated
concept score. This does not classify those nineteen as irrelevant or difficult.

The six mapped books span intrinsic concept levels 1.50–2.11. For synthetic intermediate
mastery their burden is 0.400–0.441 and utility 0.799–0.857, instead of v1's all-1.0 scores.
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

Regression tests cover expertise/burden monotonicity, higher-level content, teaching order,
unknown mastery, absent concepts, topic mismatch, repeated headings, learning opportunity,
and rubric completeness against the graph. Full suite: 178 tests passed; Ruff passed.
