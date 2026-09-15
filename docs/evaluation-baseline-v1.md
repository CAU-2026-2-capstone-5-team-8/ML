# Ten-book baseline evaluation v1

Run date: 2026-09-15

This report records one reproducible vertical-slice run against the current canonical output in
the sibling `Data-Pipeline` repository. It is an engineering baseline, not a model-quality or
diagnostic-validity claim.

## Input snapshot

| Record type | Count |
| --- | ---: |
| Books | 10 |
| Documents | 28 |
| TOC entries | 1,091 |
| Sources | 50 |
| Operating Systems books | 5 |
| Linear Algebra books | 5 |

Six books produced prose-based difficulty profiles. Four books had no eligible public prose and
correctly retained `null` difficulty values.

The generated book profiles, reader profile, ranking, and evaluation report were each produced
twice and compared byte-for-byte. Every comparison was identical.

## Reproduction

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl

uv run bookmatch-ml reader-profile \
  --input examples/assessment.json \
  --output data/output/reader_profile.json

uv run bookmatch-ml rank \
  --reader data/output/reader_profile.json \
  --books data/output/book_profiles.jsonl \
  --limit 5 \
  --output data/output/ranking.json

uv run bookmatch-ml evaluate \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --reader data/output/reader_profile.json \
  --difficulty-labels examples/difficulty_judgments.json \
  --ablation-book-id isbn13:9781985086593 \
  --output data/reports/evaluation.json
```

Versioned configuration hashes from this run:

| Configuration | SHA-256 |
| --- | --- |
| Features | `sha256:6230d12facf94e46c1daa11d9d83b3929322af2f69a36614d3f553684bfb21d6` |
| Reader | `sha256:9bc1cfbc18eddb39582d7697626269cd11203bec74e2be676a5495b8f91fb73b` |
| Ranking | `sha256:4518b4af413d277e9785c4683050a467e586c35b9943513b014c6596f0d5847c` |
| Evaluation | `sha256:fa2b7093f30711e80d358c3e4d9248c51d2a35f7965452cf9f878fa3f50c4aa3` |

## Difficulty comparison

The example labels are explicitly synthetic. Of the five labeled Operating Systems books, only
three had prose-based difficulty values and were comparable.

| Metric | Result |
| --- | ---: |
| Comparable books | 3 |
| Excluded books without prose difficulty | 2 |
| Spearman correlation | 0.50 |
| Pairwise agreement | 0.667 |

These values only prove that the evaluation path works. Three comparable books and synthetic
labels are insufficient for conclusions about model quality.

## Recommendation comparison

The topic-only and readiness-aware baselines compared all five Operating Systems candidates.

| Metric | Result |
| --- | ---: |
| Comparison K | 3 |
| Top-K overlap | 2 of 3 |
| Top-K overlap rate | 0.667 |
| Candidates whose position changed | 5 of 5 |

Two books without eligible prose ranked first with a normalized score of `1.0`. Their only active
component was topic fit, and their `component_weight_coverage` was `0.35`; vocabulary, knowledge,
and comprehension fit were all unavailable. This is the configured missing-value
renormalization working as designed, but it makes the total score unsuitable for confidence-blind
sorting or display. Consumers must retain the coverage and unavailable-component fields.

Do not change the missing-evidence policy based only on this synthetic run. A separate evaluated
policy experiment could compare:

- the current pure renormalized score;
- eligibility thresholds for readiness-aware ranking;
- a two-stage presentation separating topic-only candidates from full-fit candidates.

## Evidence ablation

For *Operating Systems: Three Easy Pieces*:

- TOC to TOC+description changed the weight of `virtualization` but added no concept;
- all available evidence added `input/output`;
- all available evidence added likely prerequisite signals for `algorithms` and `programming`;
- all four prose-difficulty components became available only with prose evidence.

This result supports keeping structured prose collection separate from TOC collection. Additional
public prose can materially change both prerequisite inference and the availability of difficulty
features.

## Recorded failure cases

- `isbn13:9780070575721`: prose difficulty unavailable and ranking coverage limited to `0.35`;
- `isbn13:9780130319999`: prose difficulty unavailable and ranking coverage limited to `0.35`.

## Next evidence needed

1. Replace the synthetic order with a versioned human relative-difficulty ranking.
2. Collect multiple judgments and retain annotator protocol before interpreting correlation.
3. Decide whether low-coverage books should remain in the same ranked list through an explicit
   policy experiment.
4. Ask `Data-Pipeline` for legally usable prose evidence for the two evidence-limited OS books if
   reliable readiness-aware comparison is required.
