# TOC concept gold evaluation v1

## Purpose

This workflow measures whether the current deterministic TOC concept matcher agrees with human
labels on the curated ten-book dataset. It freezes the matcher's predictions in a review file;
it does not change aliases, exclusions, concept graph edges, reader profiles, matching scores,
ranking, or `/ml/rank`.

The labels evaluate only mappings to concepts already configured for each topic. They do not
establish educational validity, book difficulty, prerequisite truth, or recommendation quality.

## Build the review source

```bash
uv run bookmatch-ml build-toc-concept-gold-review \
  --data-dir ../Data-Pipeline/data/processed \
  --output reviews/toc_concept_gold_review.json
```

The command draws 50 entries from Operating Systems and 50 from Linear Algebra. Selection is
deterministic: it allocates entries across each topic's five books, hashes the topic, book ID,
TOC entry ID, and sample-version string, and takes entries in hash order within each book. It
samples from every TOC entry, including entries with no current prediction. A small book that
cannot fill its even share leaves entries to be allocated deterministically among the remaining
books.

Every row preserves:

- topic, book ID, and book title;
- TOC entry ID, title, full path, source ID, and depth-first traversal position;
- the current predicted concept set;
- each prediction's matching alias and method;
- the human gold concept set, status, and optional note.

`reviews/toc_concept_gold_review.json` is version controlled and becomes the human source of
truth. The builder can reproduce an unchanged blank file byte for byte. Once a reviewer changes
a decision, the builder refuses to overwrite it.

## Human labeling rules

Review one TOC heading in the context of its full path. Set `human_gold_concept_ids` to the set
of configured concepts explicitly represented by that entry, then set `review_status` to
`reviewed`. Keep concept IDs sorted and unique.

An empty `human_gold_concept_ids` list with `review_status: reviewed` is a completed human label:
the entry should map to none of the configured concepts. An empty list with
`review_status: unreviewed` is still pending. `review_note` is optional and should record only
context useful for later adjudication.

Do not edit identity, provenance, traversal, or prediction fields. Validation rejects unknown
books, unknown TOC entries, changed topic/book/TOC identities, duplicate rows, unknown concepts,
changed predictions, stale hashes, and malformed statuses.

## Evaluate completed labels

```bash
uv run bookmatch-ml evaluate-toc-concept-gold \
  --data-dir ../Data-Pipeline/data/processed \
  --review reviews/toc_concept_gold_review.json \
  --output data/reports/toc_concept_gold_evaluation.json
```

Evaluation stops if even one row remains `unreviewed`. For each topic and for the combined sample,
the report contains:

- reviewed entry count and exact-set match count/accuracy;
- micro true-positive, false-positive, and false-negative assignment counts;
- micro precision, recall, and F1;
- each false-positive and false-negative concept assignment;
- entries with an empty predicted set and entries with an empty human gold set.

Predictions and gold labels are sets because one heading can explicitly name multiple concepts.
If a precision, recall, or F1 denominator is zero, this baseline reports `0.0`. Exact-set accuracy
still gives full credit when both sets are empty.

## Reproducibility and provenance

The review source records SHA-256 hashes for all four canonical Data-Pipeline files,
`features.yaml`, `concept_graph.yaml`, and `concept_matching.yaml`, plus the graph, matching,
sample, and review versions. The evaluation report adds the exact review-file SHA-256 and its own
evaluation version. There are no timestamps or runtime-random values.

The evaluator rebuilds the sample and current predictions before loading labels. A changed input,
config, matcher prediction, row identity, or sample membership fails visibly instead of silently
comparing different experiments.

## Current blank sample

The source committed with this workflow has 100 `unreviewed` decisions. Current predictions map
13 of the 50 Operating Systems entries and 25 of the 50 Linear Algebra entries.

| Topic | Book | Sampled | Predicted matched | Predicted unmatched |
| --- | --- | ---: | ---: | ---: |
| Linear Algebra | Understanding Linear Algebra | 11 | 2 | 9 |
| Linear Algebra | Linear Algebra | 10 | 2 | 8 |
| Linear Algebra | Linear Algebra with Applications | 10 | 5 | 5 |
| Linear Algebra | Elementary linear algebra | 9 | 8 | 1 |
| Linear Algebra | Introduction to Linear Algebra | 10 | 8 | 2 |
| Operating Systems | xv6: a simple, Unix-like teaching operating system | 10 | 2 | 8 |
| Operating Systems | Operating Systems and Middleware: Supporting Controlled Interaction | 10 | 3 | 7 |
| Operating Systems | Think OS | 10 | 3 | 7 |
| Operating Systems | Operating Systems | 10 | 2 | 8 |
| Operating Systems | Operating Systems: Three Easy Pieces | 10 | 3 | 7 |

The current review-file SHA-256 is
`1925ab9f01d44b9f9ead692ef8444008defa5dd236a34988cc310cef69c3eb18`.
No accuracy metric exists until people complete every label; the blank gold lists are not treated
as judgments.
