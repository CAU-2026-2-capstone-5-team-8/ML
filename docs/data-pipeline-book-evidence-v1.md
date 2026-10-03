# Data-Pipeline book evidence v1/v2/v3 input

## Scope

ML can consume Data-Pipeline's `book-evidence-v1/v2/v3` JSONL artifact without depending on a
provider API, bulk dump, or Data-Pipeline internal database. This importer is intentionally
separate from the existing canonical four-file loader and ranking v1.

The importer does not:

- assign numeric source weights or confidence;
- infer concepts merely because evidence is present;
- convert metadata into TOC evidence;
- modify the existing concept matcher, book profile, or ranking API.

It validates and retains the source category needed for later experiments.

## Preserved evidence

Version 3 adds required, nullable `en_text` to every v2 evidence row and optional English
book fields to the embedded canonical book. Canonical Book/TOC/Document loaders also retain
`en_title`, `en_subtitle`, `en_text`, omitting absent fields when serializing legacy records.
Original document/evidence text and nonblank English fields retain exact whitespace.
Provenance hashes include English text; original document hashes still describe the original.

V1/v2 cannot carry English fields. Regenerate experimental English-bearing v1 artifacts as v3
from their preserved canonical inputs. Existing original-only v1/v2 artifacts stay supported.
The candidate adapter exposes English separately; regular concept mapping, prose difficulty,
language auditing and ranking still use original text. The explicit `compare-english-evidence`
command compares paired TOC rows under identical settings and skips missing English rows on
both sides. See the [10-book offline comparison](experiments/english-evidence-comparison-2026-10-03.md).
Stored English is not a claim of translator identity or translation quality; preserve its
upstream run/cache provenance separately. No translation service is invoked here.

Version 2 (`schema_version=2`, `contract_version=book-evidence-v2`) adds required, nullable
`source_external_id`, `source_license`, `source_rights_note` and `text_extent`. The latter is
`{scope: excerpt | complete_section, basis: nonblank string}` for documents with recorded scope;
null means unknown. A complete section is the named preface/chapter, not the whole book.
Scope cannot be inferred from provider, document type or text length. Non-document evidence
cannot claim document extent.

The loader validates these fields in provenance hashes and source-snapshot consistency.
Both the concept-candidate adapter and concept-mapping supporting rows retain document identity,
extent, source URL and rights. Mapping reports declare their actual input contract version.
Missing v1 rights and extent stay unknown. Rights fields are provenance, not numeric weights or
an automatic grant of reuse permission. Mixed-version artifacts are rejected.

Old v1 files remain supported. Upgrade this reader before providing new extent-bearing canonical
data or v2 exports to strict older ML installations. Frozen gold review and holdout sampling
remain v1-only and explicitly reject v2 to avoid mislabeling new evidence IDs as old review input.
The [98-book verification](experiments/text-extent-handoff-2026-10-03.md) documents the local
boundary and how to reproduce import/profile checks.

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
