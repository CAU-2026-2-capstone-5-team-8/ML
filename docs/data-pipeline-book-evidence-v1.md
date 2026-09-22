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
description
document
subject
metadata_minimal
```

`toc_same_work` remains explicitly separate from exact-edition TOC. Description evidence retains
document identity and cannot validate as TOC. Subject and title evidence retain their canonical
metadata field.

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

The adapter deliberately stops at candidate text. It does not call the current matcher or change
ranking v1. Choosing aliases, embeddings, source weights, thresholds, and ranking effects belongs
to a separate measured ML experiment.

## Verified Scale-50 handoff

The Data-Pipeline artifact generated on 2026-09-22 loaded successfully with:

| Measure | Result |
| --- | ---: |
| Total books | 50 |
| Books with TOC evidence | 20 |
| Metadata-fallback-only books | 30 |
| Books with zero evidence | 0 |
| TOC evidence rows | 2,217 |
| Description rows | 27 |
| Subject rows | 100 |
| Minimal title rows | 50 |

This verifies transport and provenance preservation, not recommendation quality. The next small
experiment should compare concept coverage and human-reviewed precision for TOC-only versus
metadata-only evidence before defining any numeric source weights.
