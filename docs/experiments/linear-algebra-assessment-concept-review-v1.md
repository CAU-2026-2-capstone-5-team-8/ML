# Linear Algebra assessment concept review v1

## Purpose and boundary

This experiment tests whether the existing `assessment-concept-review-v1` gate designed for
Operating Systems can be reused unchanged for Linear Algebra. It introduces no new selection
algorithm, eligibility heuristic, concept extraction, prerequisite inference, matcher, graph, or
ranking behavior.

The source-controlled artifact remains `configs/assessment_concept_reviews.yaml`. Its review key is
`topic_id + concept_id + concept_role`, so the 16 completed Operating Systems decisions coexist with
13 Linear Algebra rows. Every new Linear Algebra row is explicitly `unreviewed`; no human decision
or AI recommendation has been supplied.

The generated packet is ignored local data:

`data/reviews/assessment_concept_review_linear_algebra_v1.json`

## Real-data baseline

- canonical topic: `linear-algebra`
- source books and BookProfiles: `5`
- covered concept pool: `18`
- prerequisite concept pool: `2`
- legacy selected covered concepts (`8`): matrix, vector, linear system, orthogonality, dimension,
  determinant, gaussian elimination, basis
- legacy selected prerequisite concepts (`2`): high school algebra, systems of equations
- vocabulary QuestionSpecs: `6`
- background-knowledge QuestionSpecs: `2`
- comprehension QuestionSpecs: `5`
- total QuestionSpecs: `13`
- shortage: background knowledge / recall requested `3`, produced `2`

The five source books are *Elementary linear algebra*, *Introduction to Linear Algebra*, *Linear
Algebra*, *Understanding Linear Algebra*, and *Linear Algebra with Applications*.

The legacy `assessment-config-v1` artifact remains byte-identical at SHA-256
`b0af7c29caef5696e09da67dfa0b891a32b9cf9f795828795fd5462b6ba6a38a`.

## Candidate generation

The existing reviewed config is used without modification. Assessment priority remains the existing
`0.7 × mean book weight + 0.3 × book coverage rate` ordering signal; it is not an eligibility
decision or a difficulty estimate. The default self-assessment limits and reserve produce a covered
window of `8 + 3 = 11`. The prerequisite pool contains only two concepts, so its nominal `4 + 3`
window yields both. The resulting workload is therefore 13 rows rather than the preferred 15–25;
the pool is not padded.

- covered candidates: `11`
- prerequisite candidates: `2`
- total candidates: `13`
- packet SHA-256: `40b4e6c862b716799cc408626c1cf4742d0c9bf3d11d5f339706e88245bb7bfe`
- review artifact SHA-256:
  `a903b6b6cd4e693c294144682b0df37de78066de1957d41821c1e2a028c9dc8c`
- effective reviewed config SHA-256:
  `71bc89d82cfe19da2bb62265d96f4bd00c8cc138ad0f7d45a49298895aba9591`

Two independent CLI generations were byte-identical.

## Review candidates

`selected` and `QuestionSpec` below describe the unchanged legacy blueprint, not a review
recommendation. The generated JSON packet retains up to six exact compact evidence references for
each row.

| Role | Rank | Concept | Priority | Coverage | Mean weight | Evidence / method | Legacy selected | Legacy QuestionSpec |
|---|---:|---|---:|---:|---:|---|---|---|
| covered | 1 | matrix | 1.000000 | 5/5 | 1.000000 | preface, preview, sample_chapter, toc | yes | yes |
| covered | 2 | vector | 1.000000 | 5/5 | 1.000000 | description, preface, sample_chapter, toc | yes | yes |
| covered | 3 | linear system | 0.940000 | 4/5 | 1.000000 | preface, preview, sample_chapter, toc | yes | yes |
| covered | 4 | orthogonality | 0.896250 | 4/5 | 0.937500 | preface, sample_chapter, toc | yes | yes |
| covered | 5 | dimension | 0.852500 | 4/5 | 0.875000 | preface, preview, sample_chapter, toc | yes | no |
| covered | 6 | determinant | 0.849500 | 5/5 | 0.785000 | preface, sample_chapter, toc | yes | yes |
| covered | 7 | gaussian elimination | 0.793500 | 5/5 | 0.705000 | preface, preview, sample_chapter, toc | yes | no |
| covered | 8 | basis | 0.784750 | 5/5 | 0.692500 | preface, sample_chapter, toc | yes | no |
| covered | 9 | eigenvalue | 0.751875 | 4/5 | 0.731250 | preface, toc | no | no |
| covered | 10 | vector space | 0.745313 | 4/5 | 0.721875 | description, preface, toc | no | no |
| covered | 11 | diagonalization | 0.677500 | 4/5 | 0.625000 | preface, toc | no | no |
| prerequisite | 1 | high school algebra | 0.760000 | 1/5 | 1.000000 | preface; explicit_and_early_prose_proxy | yes | yes |
| prerequisite | 2 | systems of equations | 0.165000 | 1/5 | 0.150000 | preview; early_prose_proxy | yes | yes |

All 13 rows have `status=unreviewed`, `reason_code=null`, and an empty `review_note`.

## Relation-pair coverage

Every configured Linear Algebra relation concept exists in the full covered concept pool. Queue
membership is determined only by the existing bounded priority window:

| Concept | Pool rank | In queue | Explanation when absent |
|---|---:|---|---|
| eigenvalue | 9 | yes | — |
| eigenvector | 13 | no | outside the top-11 covered window |
| basis | 8 | yes | — |
| vector space | 10 | yes | — |
| matrix | 1 | yes | — |
| linear transformation | 14 | no | outside the top-11 covered window |
| orthogonality | 4 | yes | — |
| inner product | 15 | no | outside the top-11 covered window |
| linear system | 3 | yes | — |
| gaussian elimination | 7 | yes | — |
| determinant | 6 | yes | — |

The three absent queue concepts are not missing because of aliases, canonicalization, or absent
evidence. They are present in the pool but fall below the unchanged reserve window.

## Prerequisite audit

Only two prerequisite concepts exist in the current pool:

- `high school algebra`: inferred from *Linear Algebra with Applications*
  (`book_d061d898abc6ada0c56b`), preface `doc_63297104917172853683`, two mentions,
  `explicit_and_early_prose_proxy`, priority `0.76`.
- `systems of equations`: inferred from *Understanding Linear Algebra*
  (`book_a1364d52179f6fca0403`), preview `doc_54beabfd6f11ba8a5504`, one mention,
  `early_prose_proxy`, priority `0.165`.

Neither has been judged eligible or ineligible. The example broad concepts `algebra`, `arithmetic`,
`equation`, `mathematics`, `functions`, `geometry`, `calculus`, and `programming` do not appear as
separate prerequisite candidates in this pool.

## Fail-closed and regression results

With all LA rows unreviewed, reviewed mode selects zero concepts and produces zero QuestionSpecs.
It records five explicit shortages: vocabulary recognize, vocabulary compare, background recall,
comprehension apply, and comprehension integrate. It does not fall back to legacy concepts.

The 16 Operating Systems decisions remain 11 eligible and 5 ineligible. Its reviewed selection,
QuestionSpec target semantics, and shortage are identical before and after adding the LA rows;
their normalized semantic artifact SHA-256 is
`740a4678984b82ad46c5e63d2da6d4c702c5b08b9ad499cc19ee6423f6626434`. `programming` remains
excluded. Only review/config provenance and derived IDs change because the exact combined review
artifact hash changed.

Question-Generation main accepts the legacy LA `matrix` vocabulary/recognize/Level-1 and `high
school algebra` background-knowledge/recall/Level-1 specs. It continues to reject the LA
vocabulary/compare/Level-2 spec as unsupported. No Question-Generation code or policy changed.

## Next step

A human reviewer should label each of the 13 rows as `eligible` or `ineligible`, including a review
note and an ineligible reason code where required. Only after those decisions are recorded should a
reviewed Linear Algebra blueprint and new production QuestionSpecs be considered.
