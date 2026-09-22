# Evidence concept human evaluation

## Purpose

This framework measures how well the existing deterministic concept matcher maps each
`book-evidence-v1` source category. It does not tune aliases, assign source weights, change the
concept graph, or alter ranking v1.

The previous `reviews/toc_concept_gold_review.json` is a 100-row blank review template: all 100
rows are `unreviewed`, no row has a human gold concept, and no row has a review note. It remains
the pinned evaluation for the older canonical TOC input. The source-aware evaluation uses the
separate, versioned `reviews/evidence_concept_gold_review_v1.json` artifact.

## Deterministic sample

The v1 sample requests 94 entries:

| Evidence type | Requested rows |
| --- | ---: |
| `toc_same_work` | 6 |
| `toc_exact` | 16 |
| `toc_public_web_exact` | 16 |
| `description` | 16 |
| `subject` | 20 |
| `metadata_minimal` | 20 |

Sampling is stratified rather than proportional to source row counts. It alternates topics and
books, caps the complete sample at eight rows per book, deduplicates identical normalized text
within a book/source group, and retains at most two generic TOC titles per TOC source group.
Generic titles are controlled rather than removed because false positives on headings such as
`Introduction`, `Summary`, and `Exercises` are useful evaluation cases.

The same input artifact and configs produce byte-identical JSON. The review pins the exact
`book-evidence-v1` hash, graph hash, feature hash, matching-config hash, generated predictions,
and evidence provenance fields.

## Prediction semantics

Each evidence row is matched independently with the existing `normalized_alias_phrase_v1`
matcher. TOC predictions use the evidence text, exactly as the existing matcher uses a TOC leaf
title. The full TOC path is retained for human context but is not silently added to model input.
Descriptions, subjects, and titles use the same aliases and exclusions without new thresholds,
embeddings, or LLM classification.

`prediction_matches` records the canonical concept ID, matching alias, and match method. An
ambiguous alias produces no prediction and is recorded as `ambiguous_alias`.

## Human decisions

An entry starts with:

```json
{
  "human_gold_concept_ids": [],
  "review_status": "unreviewed",
  "review_outcome": "pending"
}
```

Only canonical graph concept IDs are valid. A reviewer chooses one of:

- `labeled`: one or more canonical concepts are supported by the evidence.
- `no_concept`: the evidence supports none of the configured concepts.
- `not_judgable`: the evidence is too ambiguous to label; it is reviewed but excluded from
  metrics.

One decision can be applied without editing the whole file:

```bash
uv run bookmatch-ml review-evidence-concepts \
  --input PATH/TO/book-evidence.jsonl \
  --review reviews/evidence_concept_gold_review_v1.json \
  --evidence-id evidence_0123456789abcdef0123 \
  --outcome labeled \
  --gold "virtual memory,memory management" \
  --note "The heading explicitly names both concepts."
```

Use `--outcome no_concept` with no `--gold` for a reviewed empty set. Use
`--outcome not_judgable` and a note when context is insufficient. The command validates the
current evidence artifact and generated prediction before atomically updating one row.

## Metrics

Pending entries are never interpreted as empty gold. With no judgeable human labels, evaluation
returns `metric_status: unavailable`, progress counts, and empty metric lists. A partly reviewed
file returns `partial`; a fully reviewed file returns `complete`.

For judgeable rows the report contains exact-set match rate and micro, row-macro, and book-macro
precision/recall/F1 for:

- all evidence;
- each topic;
- each evidence type;
- TOC and metadata families.

Book-macro metrics give each represented book equal weight. Empty predicted or gold denominators
produce zero precision or recall; exact-set match separately credits two empty sets.

After the full sample is reviewed, policy diagnostics compare `toc_only`, `metadata_only`, and
`combined_unweighted`. They remain unavailable during partial review so missing labels cannot
distort a source comparison. Within a book, concept IDs are set-unioned before scoring, so twelve
matching TOC rows still mean only `book covers concept = true`. Raw occurrence count is diagnostic
only and is never a score or weight. Until reviews are complete, policy metrics are unavailable
rather than fabricated.
