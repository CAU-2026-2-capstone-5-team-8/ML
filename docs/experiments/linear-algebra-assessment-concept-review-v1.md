# Linear Algebra assessment concept review v1

## Purpose and boundary

This experiment tests whether the existing `assessment-concept-review-v1` gate designed for
Operating Systems can be reused unchanged for Linear Algebra. It introduces no new selection
algorithm, eligibility heuristic, concept extraction, prerequisite inference, matcher, graph, or
ranking behavior.

The source-controlled artifact remains `configs/assessment_concept_reviews.yaml`. Its review key is
`topic_id + concept_id + concept_role`, so the 16 completed Operating Systems decisions coexist with
13 completed Linear Algebra decisions. These are user-supplied human decisions, not automatic
labels or AI recommendations.

The generated packet is ignored local data:

`data/reviews/assessment_concept_review_linear_algebra_v1.json`

The distinct reviewed blueprint is generated at
`data/output/linear_algebra_assessment_blueprint_reviewed.json`; it never overwrites the legacy
artifact.

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
- packet SHA-256: `ad347b40eb10f96fbb0d12e8473d7117964c524883a3e1c9ce1715c733742e21`
- review artifact SHA-256:
  `750f6bc77e01c6e677eebf20d269148b4b22755fa6afcd82c28e3064e32e43f6`
- effective reviewed config SHA-256:
  `f64278102de445682861948fe47e01407f52306dbbf0d9e55bc8b6fb278ca063`
- reviewed blueprint SHA-256:
  `cce198d545d6f092c14ec8a1f34efcf9c43b6f81c61a89202fe32e218b45916a`

Two independent CLI generations were byte-identical.

## Review candidates

`selected` and `QuestionSpec` below describe the unchanged legacy blueprint, not a review
recommendation. The generated JSON packet retains up to six exact compact evidence references for
each row.

| Role | Rank | Concept | Priority | Coverage | Evidence / method | Status |
|---|---:|---|---:|---:|---|---|
| covered | 1 | matrix | 1.000000 | 5/5 | preface, preview, sample_chapter, toc | eligible |
| covered | 2 | vector | 1.000000 | 5/5 | description, preface, sample_chapter, toc | eligible |
| covered | 3 | linear system | 0.940000 | 4/5 | preface, preview, sample_chapter, toc | eligible |
| covered | 4 | orthogonality | 0.896250 | 4/5 | preface, sample_chapter, toc | eligible |
| covered | 5 | dimension | 0.852500 | 4/5 | preface, preview, sample_chapter, toc | eligible |
| covered | 6 | determinant | 0.849500 | 5/5 | preface, sample_chapter, toc | eligible |
| covered | 7 | gaussian elimination | 0.793500 | 5/5 | preface, preview, sample_chapter, toc | eligible |
| covered | 8 | basis | 0.784750 | 5/5 | preface, sample_chapter, toc | eligible |
| covered | 9 | eigenvalue | 0.751875 | 4/5 | preface, toc | eligible |
| covered | 10 | vector space | 0.745313 | 4/5 | description, preface, toc | eligible |
| covered | 11 | diagonalization | 0.677500 | 4/5 | preface, toc | eligible |
| prerequisite | 1 | high school algebra | 0.760000 | 1/5 | preface; explicit_and_early_prose_proxy | ineligible / too_general |
| prerequisite | 2 | systems of equations | 0.165000 | 1/5 | preview; early_prose_proxy | eligible |

The review is complete: `13/13` reviewed, `12` eligible, `1` ineligible, and `0` unreviewed.

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

`systems of equations` is eligible as specific prerequisite knowledge. `high school algebra` is
ineligible with reason `too_general`: it remains in prerequisite inference and BookProfile evidence
but cannot be an assessment primary target. The example broad concepts `algebra`, `arithmetic`,
`equation`, `mathematics`, `functions`, `geometry`, `calculus`, and `programming` do not appear as
separate prerequisite candidates in this pool.

## Reviewed selection and regression results

The covered limit selects matrix, vector, linear system, orthogonality, dimension, determinant,
gaussian elimination, and basis. Eigenvalue, vector space, and diagonalization remain eligible but
outside the existing limit. The prerequisite selection contains only systems of equations;
ineligible high school algebra is not used as a fallback.

The reviewed blueprint contains 12 QuestionSpecs: vocabulary `6`, background knowledge `1`, and
comprehension `5`. Background recall records the only shortage (`requested=3`, `produced=1`).

The 16 Operating Systems decisions remain 11 eligible and 5 ineligible. Its reviewed selection,
QuestionSpec target semantics, and shortage are identical before and after adding the LA rows;
their normalized semantic artifact SHA-256 is
`740a4678984b82ad46c5e63d2da6d4c702c5b08b9ad499cc19ee6423f6626434`. `programming` remains
excluded. Only review/config provenance and derived IDs change because the exact combined review
artifact hash changed.

Question-Generation main accepts all four reviewed vocabulary/recognize/Level-1 targets (matrix,
vector, linear system, orthogonality) and the reviewed systems of equations
background-knowledge/recall/Level-1 target. It continues to reject vocabulary/compare and
comprehension specs as unsupported. No Question-Generation code or policy changed. Live Gemini
generation was skipped because `GEMINI_API_KEY` was absent from the process environment.

## Next step

When a Gemini API key is intentionally present in the process environment, generate one Korean
question for each of the five supported reviewed targets and conduct human QA before merging the
milestone.
