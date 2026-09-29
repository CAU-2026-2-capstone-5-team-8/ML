# Comprehension grounding display v2

## Decision

`generation-grounding-v1` remains the immutable raw-source contract: `passage_text` is an exact
substring of canonical `Document.text`, and `passage_hash` hashes those exact UTF-8 bytes. The new
`generation-grounding-v2` contract adds two explicit roles instead of changing that meaning:

```text
Document.content_hash
    -> source_passage_text
    -> source_passage_hash
    -> pdf-display-normalization-v1
    -> display_passage_text
    -> display_passage_hash
```

ML owns the small normalization boundary because it already selects and binds the exact passage.
Data-Pipeline's canonical four-file contract and QuestionSpec source identity remain unchanged.

## Root cause and alternatives

The Hefferon raw artifact still contains the original PDF bytes. Data-Pipeline decodes those bytes
with pypdf's default `PageObject.extract_text()` mode, joins the page text, and hashes that result as
canonical `Document.text`. PDF glyph positioning does not always encode word boundaries, so the
canonical text contains joins such as `withm`, `anentry`, and `has2` even though its provenance is
strong.

The same local PDF was also extracted with pypdf's deterministic `layout` mode. It preserved the
matrix rows more visibly and changed `anentry` to `an entry`, but still produced
`DeﬁnitionAnm×n`, `withm`, `andn`, `has2`, and `isa2,1`. It is not an exact substring of the
canonical document and would require an additional PDF-to-passage alignment contract. Replacing the
canonical extraction would also change document hashes across the dataset.

No already-collected HTML artifact contains the same chapter body. The collected author page and
license page are metadata, not an alternate representation of the selected document. Choosing a
different clean HTML textbook would change the QuestionSpec's document identity and is therefore a
future source-selection decision, not display normalization.

The considered ownership choices were:

| Choice | Result |
| --- | --- |
| ML reviewed normalization | Selected: smallest additive boundary; exact raw passage remains auditable. |
| Data-Pipeline display artifact | Possible later, but layout extraction still leaves most joins and requires alignment. |
| Add a display field to `Document` | Rejected for this slice because it expands the canonical dataset contract. |
| Replace canonical PDF extraction | Rejected because it rewrites `Document.text` and every dependent content hash. |

## Reviewed policy

`pdf-display-normalization-v1` is not a general PDF repair regex. It is an ordered allowlist keyed by
the exact source document ID and source passage hash. Every expected fragment must occur exactly
once; an unknown document, changed passage, unsupported policy, missing fragment, or hash mismatch
fails closed.

For `doc_de934d33d551223812e8` and source passage
`sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0`, it performs only these
reviewed replacements:

| Exact source fragment | Display fragment | Safety rationale |
| --- | --- | --- |
| `DeﬁnitionAnm×n` | `Definition An m×n` | Expands the `ﬁ` glyph and restores the printed heading/indefinite article boundary. |
| `withm rows\nandn columns` | `with m rows\nand n columns` | Restores spaces around the variables in the definition without touching other identifiers. |
| `anentry` | `an entry` | Restores one word boundary in a fixed prose fragment. |
| `has2 rows and3 columns and so is a2×3 matrix` | `has 2 rows and 3 columns and so is a 2×3 matrix` | Restores prose spacing while preserving `2×3`. |
| `two-by-\nthree` | `two-by-three` | Removes a PDF line wrap but preserves the visible compound-word hyphens. |
| `stated ﬁrst` | `stated first` | Expands one reviewed typographic ligature without changing the word. |
| `row and ﬁrst column` | `row and first column` | Expands the second reviewed `ﬁ` ligature in its exact prose context. |
| `isa2,1 =3` | `is a2,1 = 3` | Restores prose/operator spacing without inventing lost subscript formatting. |

The matrix rows remain separate lines. The policy intentionally leaves `a2,1` unchanged because the
canonical extraction has already lost subscript presentation and the display layer must not invent
notation. It also does not apply broad letter/digit spacing, Unicode normalization, spelling
correction, reflow, or model-based rewriting.

## Actual vertical slice

- QuestionSpec: `q_375e5b6bef551015f67c`
- target: matrix, comprehension/apply, Level 2
- source document: `doc_de934d33d551223812e8`
- source passage length: 617 characters
- source passage hash: `sha256:7220ee5501766f97edabc506e871470356fa2aeb40ec92acfd6f7e44d9ce4ce0`
- display passage length: 629 characters
- display passage hash: `sha256:c5126e3650a7c2329f2a4281efc774465bb9954e19091808d7162938926850a4`
- grounding artifact hash: `sha256:a9d7f5f1b1487040c6aff681fbfe8591800a274aa0b5e2231cf784ea8c1fe2a6`
- normalization policy: `pdf-display-normalization-v1`

Two builds from the same canonical files, blueprint, QuestionSpec, and policy were byte-identical.
No timestamp, network request, PDF rewrite, or model call participates in the result.

## Version and downstream boundary

The v2 schema uses `source_passage_text`, `source_passage_hash`, and
`source_passage_extraction_policy` for provenance, plus `display_passage_text`,
`display_passage_hash`, and `display_normalization_policy` for presentation. It retains the full v1
QuestionSpec, document, source, license, canonical file, and blueprint bindings.

Question-Generation may use the display passage for both the user presentation and provider
grounding only because the exact transformation is versioned, locally reviewed, hash-bound, and
self-validating. It must retain both hashes and the policy in its output provenance. Backend remains
unchanged until the separated passage/question contract is finalized downstream.
