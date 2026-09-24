# Source-aware matching-profile compatibility adapter

## Pipeline boundary

The compatibility path is:

```text
Data-Pipeline book-evidence-v1
  -> normalized_alias_span_v2
  -> BookEvidenceConceptMappingReport
  -> source-aware-matching-profile-adapter-v1
  -> MatchingBookProfile
  -> existing rank_matching_books / POST /ml/rank
```

The adapter joins a source-aware mapping report and existing `BookProfile` records by exact
`book_id`. Duplicate IDs, missing records, topic disagreement, and extra records are rejected. It
does not silently drop either input.

Data-Pipeline's additive nullable `Source.evidence` provenance is accepted by the canonical ML
loader so the existing deterministic BookProfile builder can rebuild the Scale-50 profiles. The
provenance does not alter legacy concept or difficulty calculations.

## Field policy

| MatchingBookProfile field | Source |
| --- | --- |
| `book_id` | Exact one-to-one join key |
| `topic_distribution` | Existing `BookProfile.concept_profile.topic_distribution` |
| `covered_concepts` | Deduplicated source-aware `(book, topic, concept)` presence |
| `prerequisite_concepts` | Existing `BookProfile.concept_profile.prerequisite_concepts` |
| Difficulty fields | Existing `BookProfile.difficulty_profile`, including `null` |
| Feature/config metadata | Existing `BookProfile` versions and hash |

Every source-aware covered concept is represented with `weight=1.0` because the existing
`MatchingConcept` schema requires a number. This is only a binary compatibility marker meaning
`concept present = true`. It is not confidence, importance, frequency, evidence quality, or a
ranking score. Supporting evidence count and the raw matcher occurrence count never affect this
value.

The existing ranking engine does not score `covered_concepts`; it uses them only in the “주요 개념”
explanation. Ranking inputs that affect the score remain unchanged:

- topic fit uses the existing topic distribution;
- vocabulary and comprehension fit use existing lexical and syntactic difficulty;
- knowledge fit uses existing concept density, prerequisite demand, and prerequisite concepts;
- missing difficulty remains `null` and the existing renormalization policy applies.

The adapter does not infer new prerequisites from source-aware presence and does not promote
concept-graph edges into production prerequisite demand.

## Artifact and provenance

```bash
uv run bookmatch-ml build-matching-book-candidates \
  --concept-mapping data/reports/scale-50-concept-presence-v2-overlap.json \
  --book-profiles data/output/scale-50-book-profiles.jsonl \
  --output data/output/scale-50-matching-candidates.jsonl \
  --report data/reports/scale-50-matching-candidates-report.json
```

The JSONL output is the existing snake-case `MatchingBookProfile` contract. Each row validates as
the current camel-case `BookCandidateDto` without changing `/ml/rank`. The separate report records
the adapter version, concept-mapping and BookProfile hashes, matcher/config versions, candidate
artifact hash, join diagnostics, availability counts, and the explicit binary-weight semantics.

## Scale-50 verification

The current Scale-50 run produced:

| Measure | Result |
| --- | ---: |
| Mapping books | 50 |
| Existing BookProfiles rebuilt | 50 |
| Joined candidates | 50 |
| Missing or duplicate IDs | 0 |
| Candidates with covered concepts | 22 |
| Candidates without covered concepts | 28 |
| Candidates with existing prerequisite concepts | 1 |
| Operating Systems candidates | 25 |
| Linear Algebra candidates | 25 |
| Available difficulty values, per component | 0 / 50 |

All four difficulty components remain missing because the current Scale-50 canonical data has no
eligible prose documents. The ranking smoke test therefore uses the existing topic-only
renormalization for these candidates; it does not substitute zero or synthesize difficulty.

Both topic rankings accepted 25 candidates, were deterministic under reversed input ordering, and
showed source-aware concepts in the existing explanation. All 50 candidates round-tripped through
`BookCandidateDto`.

## Deferred work

Source-aware prerequisite inference, evidence-type weighting, confidence, occurrence-based
importance, and ranking changes remain separate future experiments. The smallest service step is
for Spring to load or persist the generated candidates and send them through the unchanged
`candidateBooks` field of `/ml/rank`.
