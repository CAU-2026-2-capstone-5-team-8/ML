# Ranking-v2 production integration contract

## Decision

Production ranking-v2 uses the existing `POST /ml/rank` route with an explicit
top-level selector:

```json
"rankingModel": "rank-prerequisite-first-v2"
```

If `rankingModel` is omitted, the endpoint continues to execute `rank-v1`. This
keeps current callers stable, avoids a second endpoint, and prevents a silent
production-model switch. The v2 response is a distinct, versioned shape. It does
not contain a scalar `score` because the policy is lexicographic rather than an
arithmetic formula.

The server owns the accepted prerequisite projection in `ranking_v2.yaml`.
Callers provide source-aware covered concepts but cannot submit inferred
prerequisites or graph/review hashes. The response echoes the server policy,
graph, and review versions and hashes used for the calculation.

## Eligibility and ordering

For the reader topic, each candidate belongs to exactly one internal pool:

- `personalizable`: has covered concepts, the accepted graph yields at least one
  prerequisite, and the reader has assessed readiness for at least one of those
  prerequisites.
- `concept_only`: has covered concepts but prerequisite readiness cannot be
  calculated.
- `evidence_unavailable`: has no source-aware covered concepts.

Only `personalizable` books are returned. The exact ordering is:

1. transitive prerequisite readiness descending;
2. on an exact tie, direct learning opportunity descending, with an assessed
   value ordered before an unavailable value;
3. on a remaining exact tie, `bookId` ascending.

Coverage, evidence counts, title, ISBN, and source counts do not influence this
ordering. Unknown readiness is omitted from each mean and is never converted to
zero. No tolerance, weighted sum, multiplier, or human-label tuning is applied.

## Request

```json
{
  "rankingModel": "rank-prerequisite-first-v2",
  "readerProfile": {
    "userId": 42,
    "topicId": "operating-systems",
    "vocabulary": 0.72,
    "backgroundKnowledge": 0.55,
    "comprehension": 0.68,
    "conceptReadiness": [
      {"conceptId": "process", "score": 1.0},
      {"conceptId": "concurrency", "score": 0.8}
    ],
    "profileVersion": "reader-v1",
    "configVersion": "reader-config-v1",
    "configHash": "sha256:..."
  },
  "candidateBooks": [
    {
      "bookId": "isbn13:9780132199087",
      "topicDistribution": {"operating-systems": 1.0},
      "coveredConcepts": [
        {"concept": "distributed systems", "weight": 1.0},
        {"concept": "file system", "weight": 1.0},
        {"concept": "process", "weight": 1.0},
        {"concept": "synchronization", "weight": 1.0}
      ],
      "prerequisiteConcepts": [],
      "lexicalDifficulty": null,
      "syntacticComplexity": null,
      "conceptDensity": null,
      "prerequisiteDemand": null,
      "featureVersion": "book-v1",
      "configVersion": "features-v1",
      "configHash": "sha256:..."
    }
  ],
  "limit": 5
}
```

`readerProfile` is the matching projection of the actual `/ml/reader-profile`
result. V2 reads `topicId`, `conceptReadiness`, and profile provenance; the three
aggregate readiness fields remain required by the shared request DTO and keep
rank-v1 compatibility.

For v2, `coveredConcepts` is the source-aware binary concept presence produced by
the current adapter. The server computes accepted transitive prerequisites from
it. `prerequisiteConcepts` and the four nullable prose-demand fields remain in the
shared candidate DTO for rank-v1 compatibility but are not v2 ordering inputs.

The optional top-level `bookId` requests a diagnostic for one supplied target.
The returned item retains its global rank within the full personalizable pool.

## Response

```json
{
  "userId": 42,
  "topicId": "operating-systems",
  "items": [
    {
      "bookId": "isbn13:9780132199087",
      "rank": 1,
      "availabilityStatus": "personalizable",
      "prerequisiteReadiness": 0.9,
      "prerequisiteAssessedCount": 2,
      "prerequisiteTotalCount": 4,
      "prerequisiteCoverage": 0.5,
      "directLearningOpportunity": 0.1,
      "directAssessedCount": 2,
      "directTotalCount": 4,
      "directCoverage": 0.5,
      "coveredConcepts": ["distributed systems", "file system", "process", "synchronization"],
      "inferredPrerequisites": ["concurrency", "process", "storage", "thread"],
      "reasons": ["...", "..."],
      "modelVersion": "rank-prerequisite-first-v2",
      "bookFeatureVersion": "book-v1",
      "bookConfigVersion": "features-v1",
      "bookConfigHash": "sha256:..."
    }
  ],
  "diagnostics": {
    "requestedLimit": 5,
    "returnedCount": 1,
    "topicCandidateCount": 1,
    "personalizableCount": 1,
    "conceptOnlyCount": 0,
    "evidenceUnavailableCount": 0,
    "fallbackCount": 0,
    "personalizedCandidateShortage": 4
  },
  "modelVersion": "rank-prerequisite-first-v2",
  "configVersion": "ranking-v2-config-v1",
  "configHash": "sha256:...",
  "conceptGraphVersion": "concept-graph-v1",
  "conceptGraphHash": "sha256:...",
  "graphReviewVersion": "concept-graph-reviews-v1",
  "graphReviewHash": "sha256:...",
  "readerProfileVersion": "reader-v1",
  "readerConfigVersion": "reader-config-v1",
  "readerConfigHash": "sha256:..."
}
```

`items` contains at most `limit` entries. `fallbackCount` reports excluded
fallback-pool size; it does not mean that fallback books were returned. Coverage
is diagnostic only. `directLearningOpportunity` may be `null` when no covered
concept was assessed, and this remains different from `0.0`.

## Shortage and unavailable behavior

- Fewer personalizable books than `limit`: HTTP 200, return only available
  personalizable books, and set `personalizedCandidateShortage`.
- No personalizable books: HTTP 200 with `items: []` and the full shortage in
  diagnostics.
- Requested target is `concept_only` or `evidence_unavailable`: HTTP 422; no fake
  personalized result is produced.
- Target absent from candidates, invalid/duplicate candidates, invalid profile,
  unknown covered graph concept, or target/profile topic mismatch: HTTP 422.

## Scale-50 smoke result

Using the two fixed real `ReaderProfile` artifacts and all 25 candidates per
topic produced these pools:

| Topic | Personalizable | Concept only | Evidence unavailable |
| --- | ---: | ---: | ---: |
| Operating Systems | 11 | 1 | 13 |
| Linear Algebra | 10 | 0 | 15 |

Both fixed-reader runs returned five personalizable books, preserved counts and
coverage in every item, and produced the same result after reversing candidate
input order. The production implementation also matches the reviewed candidate
implementation across all eight Beginner, Intermediate, Advanced, and Uneven
topic scenarios.

Run the same production implementation locally:

```bash
uv run bookmatch-ml demo-production-ranking-v2 \
  --reader data/output/reader_profile.json \
  --limit 5

uv run bookmatch-ml demo-production-ranking-v2 \
  --reader data/output/concept_matching_la_reader.json \
  --limit 5
```

## Backend migration from the current stub

The current Backend `MlRankRequest`/`MlRankResult` is not wire-compatible with
this v2 contract. It carries aggregate scalar features only, requires one scalar
`score` plus four fit fields, validates exactly `min(limit, candidateCount)`
items, and persists those scalar fields. Before enabling v2, Backend should:

1. Add a versioned v2 HTTP DTO rather than reusing the scalar stub item.
2. Send `rankingModel=rank-prerequisite-first-v2`, the reader `topicId`, concept
   readiness and provenance, and each candidate's topic distribution,
   source-aware covered concepts and provenance. Backend must not infer or submit
   prerequisite lists.
3. Accept `0..limit` items and validate `returnedCount`, sequential list order,
   unique requested book IDs, and shortage diagnostics. A target response may
   contain one item whose global `rank` is greater than one.
4. Persist/display canonical `rank`, prerequisite readiness/coverage, direct
   opportunity/coverage, availability, reasons, and version/hash metadata. Do
   not manufacture or sort by a scalar score.
5. Decide the public recommendation response and database migration for nullable
   legacy scalar columns before switching traffic. Keep the existing rank-v1
   gateway path active during migration.
6. Map the Backend numeric book ID to the ML string `bookId` deterministically
   and validate the echoed ID before persistence.
7. Add HTTP contract tests for an ordinary Top-5, a shortage/empty result, a
   non-personalizable target 422, and rank-v1 default compatibility. Do not add
   automatic fallback to the local stub on an ML HTTP failure.

The ML default remains rank-v1. Enabling production-v2 traffic is therefore an
explicit Backend decision after its DTO, validator, persistence, and response
contract are migrated.
