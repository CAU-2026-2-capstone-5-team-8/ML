# Canonical data to HTTP verification — 2026-09-21

## Scope

Verified the existing local calculation API with a real, previously collected 25-book
Operating Systems search snapshot. This is transport/data-contract verification, not an
evaluation of recommendation quality or human difficulty. Reader answers are synthetic
(`examples/assessment.json`). No Spring server or service database participates.

The snapshot is `toc-enriched-20260920/processed` from the earlier Data-Pipeline collection.
It is not a claim about the latest pipeline main dataset. Raw evidence is not committed here.

## Results

| Check | Result |
| --- | --- |
| Canonical books validated and profiled | 25 |
| Books with TOC | 7 |
| TOC entries | 318 |
| Books with extracted covered concepts | 6 |
| Books with analyzed prose | 0 |
| Reader-profile and ranking HTTP requests | 200 |
| HTTP output equals local calculation | Yes |
| Repeated ranking output identical | Yes |
| Top-five component weight coverage | 0.35 for all five |
| Top-five scores | 1.0 for all five |

The five 1.0 scores are **not 100% difficulty suitability**. All three readiness-fit
components are unavailable; only topic fit is active and its weight is renormalized.
Ties do not demonstrate meaningful personalization. Consumers must display missing
components and coverage alongside the score rather than imply a validated difficulty match.

The search snapshot also includes titles such as "Power system operation" and "Computer
aided power system operation and analysis". These are topic-review candidates, not automatic
human judgments made by this verification. Topic labels need review before an end-user demo.
The script intentionally preserves inputs and does not silently filter books or change ranking.

An additional run with the repository's two-book synthetic fixture also passed; it includes
one book with analyzed prose, exercising available difficulty components.

## Reproduce

Install `uv sync --frozen`, then start the local service in a separate terminal:

```sh
uv run uvicorn bookmatch_ml.api:create_app --factory --host 127.0.0.1 --port 8012
```

Run from the repository root, replacing the input directory with your canonical snapshot:

```sh
uv run python scripts/verify_canonical_api.py \
  --data-dir ../Data-Pipeline/data/processed \
  --assessment examples/assessment.json \
  --base-url http://127.0.0.1:8012 \
  --output-dir data/reports/canonical-http-run1
```

The output directory must be new. The script validates canonical references, builds profiles,
checks profile determinism, sends reader and rank requests, compares both HTTP responses with
local calculations, and repeats the ranking request. It fails on a mismatch or HTTP error.
Use the same reader/ranking configuration on both sides; `--config-dir` configures the script,
while the server accepts `BOOKMATCH_ML_CONFIG_DIR`.

Outputs are local ignored files: `reader-request.json`, `reader-response.json`, `rank-request.json`,
and `report.json`. The report records SHA-256 hashes of all four canonical inputs, the assessment,
and configuration provenance in the outputs. It includes full rank reasons and missing components.
The script accepts local HTTP origins only and does not call public providers or paid LLMs.

## Reproducibility fix

On Windows, automatic CRLF conversion changed YAML bytes and therefore config hashes.
The documented-reader-profile regression test failed even though calculations matched.
`.gitattributes` now pins YAML to LF, preserving the original byte-hashing contract without
changing scoring or regenerating expected hashes. Existing checkouts must refresh YAML files
to LF and restart API processes after this change. Full suite: 168 passed; Ruff passed.

## Next acceptance gates

1. Review dataset topic relevance using the existing pipeline review process.
2. Evaluate existing concept-based matching separately for books with TOC but no prose.
3. Acquire valid prose evidence before claiming prose difficulty estimates.
4. Validate the Backend HTTP adapter separately before connecting the frontend.

Input fingerprints for the 25-book run:

```text
books.jsonl     2d09b55c2b72af2a77c5dc08fa90a48cf069e47cfa3e71357c2e2c730ff5ab5a
documents.jsonl 44402f7f2b90d8860b58395fddd8719c4b898aae96e727e2d0c0df46df2190a9
toc.jsonl       717d62d0042ef55e723ecb96348468f492e6596c6be6b67f590b892af85e80b4
sources.jsonl   fb5d770f844792edd7a96ec45850b28047d7f50c62d2e9c0f7bea878720b62cb
```
