# Production matcher-v2: overlap suppression only

## Decision

The production evidence-mapping path uses `normalized_alias_span_v2`. This version contains only
the v2-A span-overlap rule: when a longer, more specific concept phrase contains a shorter concept
phrase at the same text location, the nested shorter match is suppressed. Independent occurrences
remain valid. For example, `vector spaces` maps only to `vector space`, while `vectors and vector
spaces` maps to both `vector` and `vector space`.

The v1 function and `configs/concept_matching.yaml` remain available and unchanged for frozen
experiment reproduction. Production source-aware mapping uses `configs/concept_matching_v2.yaml`.
The experimental v2-A adapter delegates to the same span implementation, and regression tests
compare production predictions with the frozen v2-A artifacts row by row.

No v2-B alias, morphology, modifier, or virtual-machine-context rule is included. No C1 or C2 TOC
path rule is included. V2-B retained modifier/rephrasing false positives on the fresh holdout, and
C2 needs additional independent path validation. C1 is rejected because it over-inherits parent
concepts.

## Fixed evaluation results

The 94-row gold and both fresh holdouts were fixed before these matcher variants were evaluated.
They were not changed for production promotion.

| Dataset | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original frozen gold (94) | 51 | 1 | 13 | 98.08% | 79.69% | 87.93% |
| Fresh General (89) | 54 | 0 | 17 | 100.00% | 76.06% | 86.40% |
| Fresh Challenge (40) | 34 | 1 | 17 | 97.14% | 66.67% | 79.07% |

These are exactly the frozen experiment v2-A results. The production rule reduces the known
nested-overlap false positives without introducing the broader experimental form or path rules.

## Scale-50 verification

Input:

`../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl`

The artifact has 50 books: 20 with TOC evidence, 30 metadata-fallback-only, and zero books without
evidence. The production mapping completed deterministically with:

| Measure | Result |
| --- | ---: |
| Imported books | 50 |
| Books with at least one configured concept match | 22 |
| Books without a configured concept match | 28 |
| TOC-bearing books with at least one match | 20 / 20 |
| Metadata-only books with at least one match | 2 / 30 |
| Unique `(book, topic, concept)` presence values | 254 |
| Raw evidence-row match occurrences (diagnostic only) | 890 |

Each presence value retains supporting evidence IDs, evidence types, provider/source identity,
edition relation, TOC path when applicable, matching alias/method, and provenance hash. Multiple
rows supporting the same concept remain traceable but produce one book-level presence value. Raw
occurrence count is not a concept-importance or ranking score.

## Current integration boundary

- **Data-Pipeline evidence importer:** connected. It validates and preserves the
  `book-evidence-v1` contract.
- **Concept matcher:** connected through `map-book-evidence-concepts`; production mapping defaults
  to overlap-only matcher-v2.
- **BookProfile:** the existing v1 `BookProfile` builder still consumes the canonical four-file
  dataset. The source-aware concept-presence report is not yet converted into a ranking profile.
- **ReaderProfile and assessment:** unchanged and operational. They do not consume book evidence.
- **Ranking:** unchanged. It consumes `MatchingBookProfile` candidates supplied by the caller and
  does not invoke the evidence importer or matcher.
- **`POST /ml/reader-profile`:** unchanged and connected to ReaderProfile calculation.
- **`POST /ml/rank`:** unchanged and connected to ranking, but expects caller-provided candidate
  book profiles; it does not build them from `book-evidence-v1`.

The smallest next milestone is a versioned, tested adapter from deduplicated source-aware concept
presence plus existing difficulty features into `MatchingBookProfile` candidate DTOs. That adapter
must define missing-difficulty handling and preserve provenance without introducing source weights
or changing ranking v1.
