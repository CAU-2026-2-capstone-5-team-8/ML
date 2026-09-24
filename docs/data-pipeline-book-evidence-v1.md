# Data-Pipeline book evidence v1 input

## Scope

ML can consume Data-Pipeline's additive `book-evidence-v1` JSONL artifact without depending on a
provider API, bulk dump, or Data-Pipeline internal database. This importer is intentionally
separate from the existing canonical four-file loader and ranking v1.

The importer does not:

- assign numeric source weights or confidence;
- infer concepts merely because evidence is present;
- convert metadata into TOC evidence;
- modify the existing concept matcher, book profile, or ranking API.

It validates and retains the source category needed for later experiments.

## Preserved evidence

Every row keeps its `evidence_type`, provider, source type and URL, retrieval time, source content
hash, edition relation, and deterministic provenance hash. When Data-Pipeline has a canonical
`EvidenceProvenance` record, the ML input also retains its source edition, source ISBNs, discovery
method, match basis, and validation status.

The supported categories are:

```text
toc_exact
toc_public_web_exact
toc_same_work
toc_unspecified
description
document
subject
metadata_minimal
```

`toc_same_work` remains explicitly separate from exact-edition TOC. Description evidence retains
document identity and cannot validate as TOC. Subject and title evidence retain their canonical
metadata field. `toc_unspecified` is the backward-compatible category for a TOC whose legacy
source lacks explicit edition provenance; provider or page type alone never promotes it to exact.

## Validate an artifact

```bash
uv run bookmatch-ml inspect-book-evidence \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl
```

The loader rejects malformed lines, unsupported schema versions, duplicate book or evidence IDs,
changed book/evidence hashes, inconsistent snapshots for the same source ID, missing TOC/document
identity fields, and contradictory evidence/edition categories.

## Concept-candidate adapter

`build_concept_candidate_inputs` exposes one unweighted candidate per evidence row. A candidate
retains the book ID, evidence ID and type, text, provider, source type and tier, edition relation,
and TOC path when present. Consequently:

- the 20 TOC-bearing Scale-50 books expose the exact, public-web, or same-Work category used by a
  later concept experiment;
- the 30 books without TOC still expose collected descriptions where present plus canonical
  subject/topic and title metadata;
- later experiments can compare TOC-only, metadata-only, combined unweighted, and source-aware
  weighting without changing or reconstructing the handoff.

The original candidate adapter still stops at candidate text. The separate
`map-book-evidence-concepts` command now applies the production overlap-only matcher-v2 and emits
deduplicated book-level concept presence with every supporting evidence row and provenance field.
It does not assign source weights or feed occurrence counts into ranking.

```bash
uv run bookmatch-ml map-book-evidence-concepts \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --output data/reports/scale-50-concept-presence-v2-overlap.json
```

This source-aware concept-presence report is intentionally not yet converted into the existing
`MatchingBookProfile` API DTO. Ranking remains unchanged until that adapter and its semantics are
defined and evaluated.

## Verified Scale-50 handoff

The Data-Pipeline artifact generated on 2026-09-22 loaded successfully with:

| Measure | Result |
| --- | ---: |
| Total books | 50 |
| Books with TOC evidence | 20 |
| Books with exact-edition TOC | 19 |
| Books with reviewed public-web exact TOC | 16 |
| Books with same-Work alternate TOC | 1 |
| Books with unspecified-edition TOC | 0 |
| Metadata-fallback-only books | 30 |
| Books with zero evidence | 0 |
| TOC evidence rows | 2,217 |
| Description rows | 27 |
| Subject rows | 100 |
| Minimal title rows | 50 |

This verifies transport and provenance preservation, not recommendation quality. The next small
experiment should compare concept coverage and human-reviewed precision for TOC-only versus
metadata-only evidence before defining any numeric source weights.
