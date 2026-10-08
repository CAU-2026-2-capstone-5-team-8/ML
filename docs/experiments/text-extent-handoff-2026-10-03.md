# Text extent handoff verification, 2026-10-03

The publisher's 1,362-character Korean preface excerpt for ISBN 9788961055680 now remains an
explicit excerpt through canonical loading, v2 evidence import, concept candidate projection
and difficulty-profile generation. Source rights, license (unknown), external ID and URL survive
the handoff. The structured [result](text-extent-handoff-2026-10-03.json) records exact hashes.

| Check | Observed result |
| --- | --- |
| Canonical books / books with TOC | 98 / 83 |
| v2 imported records / generated profiles | 98 / 98 |
| Target prose documents: excerpt / complete section / unknown | 1 / 0 / 0 |
| Target analyzed text scope | `excerpt_only` |
| Offline canonical and v2 replays | 2 byte-identical outputs |
| Historical v1 export | Byte-identical; all 98 records load |
| Original discovery inputs | All four hashes unchanged |
| Extent-only before/after comparison | All 98 concept profiles and four difficulty scores unchanged |

The shared synthetic fixture additionally covers complete-section and unknown documents,
provenance tampering, source-rights conflicts and excluded short excerpts. It is identical in
Data-Pipeline and ML. No collected book text is committed.

## Reproduce with the locally retained inputs

In Data-Pipeline, replay `enrich-publisher-excerpt` against the original
`data/experiments/discovery-600-20260929/topics/linear-algebra` input using the preserved raw
artifact identified in the publisher-excerpt report. Use a new experiment directory, then:

```bash
uv run data-pipeline export-ml-evidence \
  --contract-version book-evidence-v2 \
  --dataset-dir data/experiments/text-extent-handoff-20261003/replay1/processed \
  --output data/experiments/text-extent-handoff-20261003/replay1/book-evidence-v2.jsonl
```

From ML (adjust the neighboring Data-Pipeline location for worktrees):

```bash
uv run bookmatch-ml inspect-book-evidence \
  --input ../Data-Pipeline/data/experiments/text-extent-handoff-20261003/replay1/book-evidence-v2.jsonl
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/experiments/text-extent-handoff-20261003/replay1/processed \
  --output data/experiments/text-extent-handoff-20261003/book_profiles.jsonl
```

## Interpretation and next work

This verifies local transport and scope preservation, not deployment, database integration,
Korean readability validity, or whole-book difficulty. The unchanged baseline computes numeric
scores for this excerpt; its zero concept/prerequisite scores do not establish that the book
requires no concepts or prerequisites. No additional samples, translations, gold labels or
ranking changes were introduced. `complete_section` remains unused for this real sample.

Integration order: upgrade the ML reader, then enable v2 exports/explicit canonical extent.
The Data-Pipeline change follows the selection-audit and publisher-excerpt PRs. The next content
step is another identity-verified prose source or complete section with recorded boundaries;
language-aware score validation remains a separate requirement before interpreting these
numbers in the user-facing recommendation flow.
