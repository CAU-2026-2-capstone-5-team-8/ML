# Comprehension grounding v1

## Scope and decision

The reviewed Linear Algebra blueprint already selects canonical prose document identities, but a
QuestionSpec does not contain the corresponding text. The selected architecture is an ML-owned
artifact boundary:

```text
Data-Pipeline canonical dataset + ML QuestionSpec
    -> generation-grounding-v1
    -> Question-Generation
```

This keeps Data-Pipeline provider-independent and prevents Question-Generation from learning the
four-file canonical schema or resolving provider provenance at runtime. Compared with letting
Question-Generation read Data-Pipeline directly, the artifact adds one small versioned contract but
improves failure isolation, reproducibility, testability, and future auditability. Data-Pipeline and
the existing QuestionSpec schema require no changes.

## Reviewed comprehension coverage

All five reviewed specs resolve to exactly one canonical document. Character counts and hashes are
from the canonical dataset whose four file hashes are already bound by the blueprint.

| QuestionSpec | Operation | Primary / related concepts | Level | Document | Type | Book | Characters | Document hash | Provider | License | Edition relation | Exists |
| --- | --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| `q_375e5b6bef551015f67c` | apply | matrix / — | 2 | `doc_de934d33d551223812e8` | sample_chapter | `book_b9e14d342f56d9651d61` | 140,910 | `sha256:82e9c6f9fadf91f241cbe56fe5e3aa694cd74c94fe6eba9beade2b78be032225` | hefferon | GFDL or CC BY-SA 3.0 US | unspecified | yes |
| `q_3fae13e0df26262fb467` | apply | vector / — | 2 | `doc_de934d33d551223812e8` | sample_chapter | `book_b9e14d342f56d9651d61` | 140,910 | `sha256:82e9c6f9fadf91f241cbe56fe5e3aa694cd74c94fe6eba9beade2b78be032225` | hefferon | GFDL or CC BY-SA 3.0 US | unspecified | yes |
| `q_8b89edd4e0e5175438e3` | apply | linear system / — | 2 | `doc_de934d33d551223812e8` | sample_chapter | `book_b9e14d342f56d9651d61` | 140,910 | `sha256:82e9c6f9fadf91f241cbe56fe5e3aa694cd74c94fe6eba9beade2b78be032225` | hefferon | GFDL or CC BY-SA 3.0 US | unspecified | yes |
| `q_3d38924d402770a26911` | integrate | linear system / gaussian elimination | 3 | `doc_de934d33d551223812e8` | sample_chapter | `book_b9e14d342f56d9651d61` | 140,910 | `sha256:82e9c6f9fadf91f241cbe56fe5e3aa694cd74c94fe6eba9beade2b78be032225` | hefferon | GFDL or CC BY-SA 3.0 US | unspecified | yes |
| `q_811357a46b75142e4a15` | integrate | determinant / matrix | 3 | `doc_57e0f23c17740228d111` | sample_chapter | `isbn13:9780470458211` | 195,549 | `sha256:009b0ec217578209271fa0e5d9291e9fce686d624e5fe86d8f503abc40bf00b4` | wiley | not stated | exact-edition public excerpt; no reuse license found | yes |

Source URLs are respectively
`https://jheffero.w3.uvm.edu/linearalgebra/book.pdf` and
`https://catalogimages.wiley.com/images/db/pdf/9780470458211.excerpt.pdf`. The Hefferon source hash is
`sha256:5240f2782e645bc6351ad9eba69d8c19500142a5cca9c90450c17b3765a1a400`; the Wiley source
hash is `sha256:5630a2e2f568b539ec8800ed23f12cb2f0ae05e88178801ed4f1a99449eecfbf`.

Other explicit-license prose is present but is not selected by these five reviewed QuestionSpecs:

| Book | Representative document | Characters | Document hash | Provider | License | URL |
| --- | --- | ---: | --- | --- | --- | --- |
| Understanding Linear Algebra | `doc_73ef1334f8f1553fbe1b` preview | 22,165 | `sha256:edbee633981d4bd0f2ba64382d3b8511c92c803817d739bc30835ef9fb753a65` | understanding_linear_algebra | CC BY 4.0 | `https://understandinglinearalgebra.org/sec-expect.html` |
| Linear Algebra with Applications | `doc_e0b993c608f8fbab49c3` preview | 16,526 | `sha256:382a9224295840822d8be0a0e1aedfd26bec6507f90d505c6e5e21cf736bfb42` | libretexts | CC BY-NC-SA 4.0 | `https://math.libretexts.org/Bookshelves/Linear_Algebra/Linear_Algebra_with_Applications_(Nicholson)/01%3A_Systems_of_Linear_Equations/1.01%3A_Solutions_and_Elementary_Operations` |

## Contract and extraction

The artifact binds the QuestionSpec ID and canonical hash, topic and supported target combination,
document/book/source identities, document and source hashes, exact passage and passage hash,
license and rights note, edition relation, four canonical input hashes, and blueprint hash.

`first-concept-sentence-window-v1` finds the first sentence containing the exact primary-concept
phrase, case-insensitively, then adds immediately following source sentences until the passage is at
least 600 characters. It never exceeds 1,800 characters, never calls a model, and returns an exact
contiguous substring of the canonical document. A concept-bearing sentence over the ceiling or a
result below the floor fails closed.

## First vertical slice

- QuestionSpec: `q_375e5b6bef551015f67c`
- target: matrix, comprehension/apply, Level 2
- book: Jim Hefferon, *Linear Algebra*, Fourth edition
- document: `doc_de934d33d551223812e8`, sample chapter
- license: GFDL or CC BY-SA 3.0 US
- passage length: 617 characters
- passage hash: `sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0`

The selected passage defines a matrix as a rectangular array, explains row/column dimensions, and
illustrates entry indexing. It is sufficient for a small application question without sending the
140,910-character chapter. Two exports from identical inputs were byte-identical.

`integrate`/Level 3, multiple-document grounding, licenses outside the currently approved open
reuse families, and arbitrary fallback to another document remain unsupported.
