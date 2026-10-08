# BookMatch ML

For concept-level response evidence grouped by question type and declared difficulty, use
`POST /ml/reader-diagnostics` with the existing reader-profile request. It returns observed
scores, question references, and the next assessment cell to review or probe, without changing
the current profile scores or ranking. See [reader depth diagnostics](docs/reader-depth-diagnostics-v1.md).

선형대수 실도서 3권의 실제 프로필·추천 HTTP 연결은
[live handoff](docs/linear-algebra-live-handoff.md)와
[REST Client 예제](examples/linear-algebra-live.http)를 참고하세요.

For a reproducible canonical-data → local HTTP verification workflow and the 25-book
integration findings, see [Canonical HTTP verification](docs/canonical-http-verification-v1.md).
The check verifies API consistency, not recommendation accuracy.

Evidence-first ML and recommendation logic for the CAU Capstone Team 8 personalized
technical-book recommendation project.

The current implementation covers the canonical-handoff, book-profile, concept-assessment
blueprint, reader-profile, matching/ranking, evaluation baseline, and thin Spring integration
milestones:

```text
books.jsonl + documents.jsonl + toc.jsonl + sources.jsonl
                            ↓
                 strict canonical loader
                            ↓
                    evidence assembler
                            ↓
               per-book coverage report
                            ↓
              concept + difficulty profiles
                            ↓
          topic concept pool + question specifications
                            ↓
              topic-specific reader profile
                            ↓
             explainable matching + ranking
                            ↓
      difficulty + recommendation + ablation evaluation
                            ↓
          stateless Spring-facing calculation API
```

It deliberately contains no scraping, source-provider adapters, database access, authentication,
recommendation persistence, remote LLM calls, learned ranking, or network-dependent calculation
logic.

## Inspect source-aware book evidence

Data-Pipeline can export `book-evidence-v1/v2/v3` JSONL artifacts for all benchmark
books, including metadata fallbacks when TOC is unavailable. Validate it without changing the
existing canonical loader, matcher, or ranking v1:

```bash
uv run bookmatch-ml inspect-book-evidence \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl
```

Map the validated evidence with the production overlap-only matcher-v2 and retain source-aware
support for each book-level concept presence:

```bash
uv run bookmatch-ml map-book-evidence-concepts \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --output data/reports/scale-50-concept-presence-v2-overlap.json
```

The output deduplicates `(book, topic, concept)` presence. Repeated evidence remains available as
diagnostic provenance and is not converted into a ranking weight.

Join that report to deterministically rebuilt legacy `BookProfile` values and emit candidates for
the existing ranking/API contract:

```bash
uv run bookmatch-ml build-matching-book-candidates \
  --concept-mapping data/reports/scale-50-concept-presence-v2-overlap.json \
  --book-profiles data/output/scale-50-book-profiles.jsonl \
  --output data/output/scale-50-matching-candidates.jsonl \
  --report data/reports/scale-50-matching-candidates-report.json
```

See [`docs/source-aware-matching-profile-adapter.md`](docs/source-aware-matching-profile-adapter.md)
for the field policy and integration boundary.

The strict importer retains exact, reviewed public-web, same-Work alternate, description,
subject/topic, and title categories together with source and edition provenance. It also exposes
unweighted concept-candidate text for later experiments. No numeric source weighting has been
selected. See
[`docs/data-pipeline-book-evidence-v1.md`](docs/data-pipeline-book-evidence-v1.md).
Production matcher-v2 scope and the current ranking integration boundary are recorded in
[`docs/production-matcher-v2-overlap.md`](docs/production-matcher-v2-overlap.md).

The source-aware handoff also has a deterministic human-review evaluation workflow. It samples
TOC and metadata evidence by source type, keeps the current alias matcher frozen, and reports
metrics only for explicit human labels:

```bash
uv run bookmatch-ml build-evidence-concept-gold-review \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --output reviews/evidence_concept_gold_review_v1.json

uv run bookmatch-ml evaluate-evidence-concept-gold \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --review reviews/evidence_concept_gold_review_v1.json \
  --output /tmp/evidence-concept-evaluation.json
```

See [`docs/evidence-concept-evaluation.md`](docs/evidence-concept-evaluation.md) for labeling
semantics, sampling controls, and the row-count-neutral policy baselines.

Fresh matcher holdouts are generated in two explicit phases. The first command fixes membership
without loading a matcher or matcher-v2 configuration. Only after those manifests are persisted
does the second command generate versioned predictions and prediction-blind review templates:

```bash
uv run bookmatch-ml build-evidence-concept-holdout-manifests \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --frozen-review reviews/evidence_concept_gold_review_v1.json \
  --general-output reviews/evidence_concept_holdout_general_v1_manifest.json \
  --challenge-output reviews/evidence_concept_holdout_challenge_v1_manifest.json

uv run bookmatch-ml build-evidence-concept-holdout-reviews \
  --input ../Data-Pipeline/data/experiments/scale-50-bulk-web-20260922/ml-evidence-v1/book-evidence.jsonl \
  --frozen-review reviews/evidence_concept_gold_review_v1.json \
  --general-manifest reviews/evidence_concept_holdout_general_v1_manifest.json \
  --challenge-manifest reviews/evidence_concept_holdout_challenge_v1_manifest.json \
  --general-predictions reviews/evidence_concept_holdout_general_v1_predictions.json \
  --challenge-predictions reviews/evidence_concept_holdout_challenge_v1_predictions.json \
  --general-review reviews/evidence_concept_holdout_general_v1.json \
  --challenge-review reviews/evidence_concept_holdout_challenge_v1.json

uv run bookmatch-ml show-evidence-concept-holdout \
  --review reviews/evidence_concept_holdout_general_v1.json \
  --limit 10

uv run bookmatch-ml evaluate-evidence-concept-holdout \
  --manifest reviews/evidence_concept_holdout_general_v1_manifest.json \
  --predictions reviews/evidence_concept_holdout_general_v1_predictions.json \
  --review reviews/evidence_concept_holdout_general_v1.json \
  --output data/reports/matcher-v2-general-holdout-v1.json
```

The packet command intentionally shows evidence and canonical concept choices but no v1/v2
predictions. Evaluation requires every row to be reviewed and validates the fixed manifest,
prediction artifact, and matcher/config hashes. See the
[`holdout design`](docs/experiments/matcher-v2-fresh-holdouts-v1.md) and
[`completed results`](docs/experiments/matcher-v2-fresh-holdout-results-v1.md).

## Evaluate deterministic reader scenarios

Run the offline multi-reader concept-ranking diagnostic without changing production ranking:

```bash
uv run bookmatch-ml evaluate-multi-reader-concept-ranking \
  --concept-mapping data/reports/scale-50-concept-presence-v2-overlap.json \
  --fixed-reader data/output/reader_profile.json \
  --fixed-reader data/output/concept_matching_la_reader.json \
  --output data/reports/multi-reader-concept-ranking-v1.json
```

It evaluates four deterministic knowledge states per topic, keeps missing concept evidence
unavailable, and compares prerequisite-only, unweighted, and prerequisite-first diagnostics.
See the [multi-reader experiment report](docs/experiments/multi-reader-concept-ranking-v1.md).

## Demo the prerequisite-first ranking candidate

Run the isolated ranking-v2 candidate against the real local Scale-50 concept mapping:

```bash
uv run bookmatch-ml demo-concept-recommendation \
  --topic operating-systems \
  --scenario beginner \
  --limit 5
```

The concise output shows personalized and fallback counts, exact prerequisite-first ranks,
both diagnostic axes, their coverage, and bounded explanations. It does not call or modify
production `/ml/rank`. Generate the complete validation report with:

```bash
uv run bookmatch-ml evaluate-prerequisite-first-candidate \
  --output data/reports/prerequisite-first-ranking-v2-candidate-v1.json
```

See the
[prerequisite-first candidate report](docs/experiments/prerequisite-first-ranking-v2-candidate-v1.md)
for the policy, Scale-50 results, human-pair agreement, and remaining production decisions.

To exercise the production-v2 implementation with an actual generated `ReaderProfile` rather
than a named scenario, run:

```bash
uv run bookmatch-ml demo-production-ranking-v2 \
  --reader data/output/reader_profile.json \
  --limit 5
```

Use `data/output/concept_matching_la_reader.json` for the fixed Linear Algebra reader. This
command consumes the same Scale-50 matching candidates and server-owned accepted prerequisite
projection used by the explicit v2 API path.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)

Install the locked development environment:

```bash
uv sync
```

## Inspect canonical data

Point the CLI at a directory containing the four canonical JSONL files produced by
Data-Pipeline:

```bash
uv run bookmatch-ml inspect-data \
  --data-dir ../Data-Pipeline/data/processed
```

The command validates the complete dataset before emitting deterministic JSON. A successful
report contains aggregate record counts and one explicit evidence-coverage object per book:

```json
{
  "book_count": 10,
  "books": [
    {
      "book_id": "isbn13:...",
      "coverage": {
        "document_count": 4,
        "has_description": true,
        "has_introduction": true,
        "has_metadata": true,
        "has_other_text": false,
        "has_preface": true,
        "has_preview": false,
        "has_sample_chapter": true,
        "has_toc": true,
        "prose_character_count": 100040,
        "prose_document_count": 3,
        "toc_entry_count": 62
      },
      "title": "Example title"
    }
  ],
  "document_count": 15,
  "source_count": 28,
  "toc_entry_count": 558
}
```

Descriptions, publisher summaries, and indexes do not count as textbook prose. Prose coverage
currently includes only `preface`, `introduction`, `preview`, `sample_chapter`, and `other`
documents. Missing optional evidence is reported with `false` and zero counts; it is not an
error and is not interpreted as poor book quality.

## Validation behavior

The loader fails visibly when it encounters:

- a missing canonical file;
- malformed JSON or a schema-invalid record, including its file and line number;
- duplicate book, source, document, TOC, or ISBN identifiers;
- a source, document, or TOC record referencing a missing book;
- a document or TOC record referencing a missing source or a source owned by another book;
- a missing or cross-book TOC parent;
- an inconsistent ISBN-based canonical identity.

TOC entries retain `parent_entry_id`, `level`, and `order_index` and are assembled in stable
depth-first section order, so the hierarchy is not irreversibly flattened. Documents retain
their identity, type, content hash, and source reference. Provider names and provider URLs are
kept at the ingestion/audit boundary and are not copied into downstream `BookEvidence`.

## Build book profiles

Generate one deterministic JSONL profile per canonical book:

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl
```

The default versioned formulas and lexicons live in `configs/features.yaml`. Use `--config` to
select another experiment configuration. Each profile records `feature_version`, concept and
difficulty subprofile versions, `config_version`, and a SHA-256 hash of the exact configuration
bytes.

### Concept baseline

The concept extractor is a local, deterministic topic-lexicon baseline. It:

- analyzes TOC entries and documents separately;
- weights matches by evidence type;
- retains each contributing TOC/document ID and mention count;
- infers likely prerequisites only from explicit cue sentences and a lower-weight early-prose
  proxy;
- labels each prerequisite method instead of presenting it as ground truth;
- uses canonical metadata topics for the topic distribution when available.

Concept weights use this configured formula:

```text
min(1, Σ(evidence-type weight × mention count) / saturation)
```

### Prose-difficulty baseline

Difficulty is computed per document and only for `preface`, `introduction`, `preview`,
`sample_chapter`, and `other`. TOCs, descriptions, publisher summaries, and indexes never
produce sentence-level difficulty.

The retained raw measurements include token and sentence counts, token length, vocabulary
diversity, long-word ratio, sentence-length statistics, clause-marker rate, concept density,
and prerequisite proxies. Configured normalization and weights produce four `[0, 1]` scores:

- lexical difficulty;
- syntactic complexity;
- concept density;
- prerequisite demand.

Documents are first analyzed independently and then combined with the recorded
`token_weighted_mean_v1` aggregation rule. Books without sufficient prose retain `null` scores;
short prose is recorded under `excluded_documents` with a reason.

`text_scope_version=text-extent-v1` adds excerpt/complete-section/unknown coverage counts and
per-document `text_extent`, including excluded documents. `analyzed_text_scope` identifies
`excerpt_only`, `complete_sections_only`, `mixed_or_unknown`, or `unavailable`. A complete named
section is not a complete book. Historical missing scope stays unknown; scores and ranking
formulas are unchanged. See the [98-book handoff verification](docs/experiments/text-extent-handoff-2026-10-03.md).

## Prose language diagnostics

For already stored English TOC fields, compare both texts with identical settings:

```bash
uv run bookmatch-ml compare-english-evidence \
  --input <book-evidence-v3.jsonl> --output data/reports/english-comparison.json
```

This opt-in comparison uses paired TOC rows only; absent English never falls back to the original.
It preserves existing analysis/ranking behavior and makes no translation calls. The
[real 10-book comparison](docs/experiments/english-evidence-comparison-2026-10-03.md) reproduces
4→10 matched books and 13→131 book/concept pairs; translation accuracy remains unreviewed.

For manual inspection, add `--review-packet` and choose a new local output file:

```bash
uv run bookmatch-ml compare-english-evidence \
  --input <book-evidence-v3.jsonl> --review-packet \
  --output data/reviews/english-evidence-review.json
```

The packet contains **exact source texts** and provenance: keep it in ignored local storage.
It includes all TOC rows (including unchanged/unmatched rows), row-level added/removed
concepts, and which rows support book-level changes. Prose samples retain their extent and
rights metadata in a separate list; missing English remains null and no prose score is
calculated. Reviewer identity, translation judgments, match judgments, and notes start null.
These fields are a manual worksheet, not automatically accepted evaluation labels.
Default comparison output remains text-free. See the
[review preparation and observed gaps](docs/experiments/english-evidence-review-2026-10-03.md).

An opt-in dot-product alias candidate is available for this comparison:

```bash
uv run bookmatch-ml compare-english-evidence \
  --input <book-evidence-v3.jsonl> \
  --matching-config configs/concept_matching_dot_product_v1.yaml \
  --output data/reports/english-dot-product-comparison.json
```

It adds only `dot product` and `dot products` to the existing linear-algebra `inner product`
concept. The production default remains the frozen overlap-only configuration. The
[candidate comparison](docs/experiments/dot-product-alias-2026-10-03.md) records four recovered
TOC matches, unchanged frozen-set predictions, and the remaining validation boundary.

Before interpreting prose scores across languages, run the independent diagnostic:

```bash
uv run bookmatch-ml audit-prose-language \
  --data-dir <canonical-directory> --baseline-language en \
  --output data/reports/prose-language-audit.json
```

It records declared-language mismatches, script counts, scope and baseline measurements without
changing scores or inventing human labels. A language match is not validation; outputs always
remain `not_evaluated`. See the [real Korean prose audit](docs/experiments/korean-prose-language-audit-2026-10-03.md).

## Evidence ablation

Compare the required evidence conditions for a rich-evidence book:

```bash
uv run bookmatch-ml ablate-book-profile \
  --data-dir ../Data-Pipeline/data/processed \
  --book-id isbn13:9781985086593 \
  --output data/reports/ostep_ablation.json
```

The report records concept names and weights, inferred prerequisites, difficulty scores, and
coverage for:

```text
TOC only
TOC + description
all available evidence
```

Generated inputs, profiles, and reports under `data/` are ignored by Git.

## Build a reader profile

Convert a structured, topic-specific assessment into readiness scores:

```bash
uv run bookmatch-ml reader-profile \
  --input examples/assessment.json
```

The command prints JSON by default. Use `--output examples/reader_profile.json` to write an
artifact atomically. The example input demonstrates boolean correctness and partial `[0, 1]`
scores, plus optional concept IDs and tags.

For each of `vocabulary`, `background_knowledge`, and `comprehension`, the baseline uses:

```text
dimension score = Σ(response score × declared-difficulty weight)
                  / Σ(declared-difficulty weight)
```

Difficulty weights, required question types, minimum responses, and versions live in
`configs/reader.yaml`. The result retains dimension-level earned/available weights, tagged
concept readiness, response count, `profile_version`, `config_version`, and the exact config
hash.

The loader rejects malformed or empty assessments, duplicate question IDs, mixed topics,
unsupported question types, missing required dimensions, unsupported difficulty labels, scores
outside `[0, 1]`, and responses that provide both or neither of `correct` and `score`.

## Build a concept-assessment blueprint

With the canonical ten-book snapshot and matching book profiles available, generate an
audit-friendly blueprint for each supported topic:

```bash
uv run bookmatch-ml build-assessment-blueprint \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --topic operating-systems \
  --output data/output/operating_systems_assessment_blueprint.json

uv run bookmatch-ml build-assessment-blueprint \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --topic linear-algebra \
  --output data/output/linear_algebra_assessment_blueprint.json
```

The command validates all four canonical files and checks that every book profile matches the
current canonical data and feature configuration. It emits separate covered and inferred
prerequisite concept roles, assessment priorities, selected self-assessment concepts, and
deterministic `QuestionSpec` targets. Specifications contain evidence references and source
document IDs, never question text or raw book prose. Eligible comprehension targets require
an analyzed prose document; missing targets appear as explicit shortages. The artifact records
configuration versions and SHA-256 hashes for configuration and inputs. Assessment rules live in
`configs/assessment.yaml`; optional `--feature-config` and `--assessment-config` arguments select
versioned alternatives.

This blueprint prepares Stage 1 concept self-report and Stage 2 quiz verification. It does not
change the existing reader-profile scoring or ranking API. See
[Question Difficulty v1](docs/assessment-difficulty-v1.md) for rules, real-data findings, and
limitations.

### Build one generation grounding artifact

QuestionSpec provenance identifies analyzed prose but intentionally does not copy book text. For
the narrow `comprehension / apply / Level 2` slice, bind one selected source document to an exact,
bounded passage before sending anything to a question provider:

```bash
uv run bookmatch-ml build-generation-grounding \
  --data-dir ../Data-Pipeline/data/processed \
  --blueprint data/output/linear_algebra_assessment_blueprint_reviewed.json \
  --question-id q_375e5b6bef551015f67c \
  --output data/output/linear_algebra_matrix_grounding.json
```

`generation-grounding-v1` verifies the blueprint's four canonical file hashes, exact
QuestionSpec, document/book/source joins, document content hash, prose type, and an approved
explicit reuse license. It then selects the first sentence containing the primary concept and
enough immediately following source sentences to reach 600 characters, with a hard 1,800-character
ceiling. The passage remains an exact contiguous substring of `Document.text`; no model or network
call is used.
The artifact records the passage hash, source and document hashes, license and rights provenance,
and contains no export timestamp, so identical inputs produce byte-identical output. `integrate`,
Level 3, multiple-source, unapproved-license, missing, altered, or irrelevant inputs fail closed.

Actual Linear Algebra availability and the architecture choice are documented in the
[comprehension grounding report](docs/experiments/comprehension-grounding-v1.md). Generated
grounding artifacts contain third-party text and therefore remain under ignored `data/output/`.

For user-facing presentation, build the additive v2 artifact without changing the v1 raw-passage
meaning:

```bash
uv run bookmatch-ml build-generation-grounding-v2 \
  --data-dir ../Data-Pipeline/data/processed \
  --blueprint data/output/linear_algebra_assessment_blueprint_reviewed.json \
  --question-id q_375e5b6bef551015f67c \
  --output data/output/linear_algebra_matrix_grounding_v2.json
```

`generation-grounding-v2` preserves the exact canonical substring as `source_passage_text` and
adds a separately hashed `display_passage_text`. `pdf-display-normalization-v1` is a reviewed,
source-hash-bound replacement policy rather than a broad spacing heuristic; unknown or changed
passages fail closed. The Data-Pipeline canonical files and their hashes are not modified. See the
[display grounding report](docs/experiments/comprehension-grounding-display-v2.md) for the PDF
extraction comparison, exact rules, hash chain, and limitations.

### Review assessment-worthy concepts

Concept evidence, prerequisite inference, and assessment eligibility are separate decisions. In
particular, `prerequisite` does not automatically mean that a concept is worth asking as a
diagnostic question. Prepare a bounded, evidence-rich human-review queue without changing the
legacy `assessment-config-v1` blueprint:

```bash
uv run bookmatch-ml prepare-assessment-concept-review \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --topic operating-systems \
  --output data/reviews/assessment_concept_review_operating_systems_v1.json

uv run bookmatch-ml prepare-assessment-concept-review \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --topic linear-algebra \
  --output data/reviews/assessment_concept_review_linear_algebra_v1.json
```

Human decisions live in the source-controlled
`configs/assessment_concept_reviews.yaml`; generated packets under `data/reviews/` stay ignored.
The review key is `topic_id + concept_id + concept_role`, so the same concept can be eligible as a
covered target and ineligible as a prerequisite target. To build the fail-closed reviewed mode:

```bash
uv run bookmatch-ml build-assessment-blueprint \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --topic operating-systems \
  --assessment-config configs/assessment_reviewed.yaml \
  --concept-reviews configs/assessment_concept_reviews.yaml \
  --output data/output/operating_systems_assessment_blueprint_reviewed.json
```

Only explicit `eligible` rows may become assessment targets. `ineligible`, explicit `unreviewed`,
and missing decisions are excluded; quota gaps remain visible shortages and are never silently
backfilled. The reviewed config hash binds the exact review artifact bytes while preserving the
existing `question-spec-v1` schema. See the
[assessment concept review report](docs/experiments/assessment-concept-review-v1.md).
The cross-domain Linear Algebra queue is documented in the
[Linear Algebra assessment concept review report](docs/experiments/linear-algebra-assessment-concept-review-v1.md).

## Experimental concept matching v2

The existing `rank` CLI and default `/ml/rank` behavior remain `absolute_gap_v1`; v2 requires an
explicit request selector. A separate batch command compares the v1 result with TOC-based book
concept coverage and the existing
`ReaderProfile.concept_readiness` values. It reports **prerequisite readiness** and **learning
opportunity** separately, each with its own mastery-assessment coverage. Missing concept mastery
is unknown, never zero. Prose difficulty remains in the report only as a v1 comparison and
optional diagnostic.

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl

uv run bookmatch-ml evaluate-concept-matching \
  --data-dir ../Data-Pipeline/data/processed \
  --reader examples/reader_profile.json \
  --books data/output/book_profiles.jsonl \
  --output data/reports/concept_matching_os.json

uv run bookmatch-ml reader-profile \
  --input examples/concept_matching_la_assessment.json \
  --output data/output/concept_matching_la_reader.json

uv run bookmatch-ml evaluate-concept-matching \
  --data-dir ../Data-Pipeline/data/processed \
  --reader data/output/concept_matching_la_reader.json \
  --books data/output/book_profiles.jsonl \
  --output data/reports/concept_matching_la.json
```

The versioned graph and display thresholds live in `configs/concept_graph.yaml` and
`configs/concept_matching.yaml`. TOC parent links and order are preserved, graph edges are
proposed prerequisite candidates, and every mapping retains its TOC path. The 10-book results,
formulas, v1 comparison, and limitations are in
[Concept matching v2](docs/concept-matching-v2.md). The LA assessment fixture is synthetic.

## Review concept graph and TOC mappings

Generate a human-review artifact for every graph edge and every matched or unmatched TOC entry:

```bash
uv run bookmatch-ml build-concept-review \
  --data-dir ../Data-Pipeline/data/processed \
  --reader examples/reader_profile.json \
  --reader data/output/concept_matching_la_reader.json \
  --review-config configs/concept_graph_reviews.yaml \
  --output data/reports/concept_validation.json \
  --review-template data/reviews/concept_graph_review.json
```

`configs/concept_graph_reviews.yaml` is the version-controlled source of human decisions. The
files under `data/reports/` and `data/reviews/` are generated snapshots. If the review config is
absent, the command still runs and reports every edge as `unreviewed`.

The command does not judge graph edges or influence ranking. It distinguishes `strict_before`,
`same_entry`, `strict_after`, and unobserved first-occurrence evidence while retaining the TOC
paths and legacy observation field. It also exposes the complete matched/unmatched queues,
per-book structural counts, mastery coverage, and descriptive opportunity fields. See
[Concept validation v1](docs/concept-validation-v1.md) for the schema, current 10-book metrics,
review procedure, and interpretation limits.

## Evaluate TOC concept mappings against human labels

Build the versioned 100-entry review source from the full canonical TOC population:

```bash
uv run bookmatch-ml build-toc-concept-gold-review \
  --data-dir ../Data-Pipeline/data/processed \
  --output reviews/toc_concept_gold_review.json
```

The deterministic sample contains 50 Operating Systems entries and 50 Linear Algebra entries.
Reviewers edit only `human_gold_concept_ids`, `review_status`, and `review_note`. An empty gold
list with `review_status: reviewed` means that the heading maps to none of the configured
concepts. The builder refuses to overwrite a file after human decisions change.

After all 100 rows are reviewed, evaluate the unchanged matcher predictions:

```bash
uv run bookmatch-ml evaluate-toc-concept-gold \
  --data-dir ../Data-Pipeline/data/processed \
  --review reviews/toc_concept_gold_review.json \
  --output data/reports/toc_concept_gold_evaluation.json
```

The evaluator rejects incomplete or stale review files and reports per-topic and combined
exact-set accuracy and micro precision, recall, and F1. See
[TOC concept gold evaluation v1](docs/toc-concept-gold-evaluation-v1.md) for the labeling rules,
sample composition, provenance fields, and metric semantics.

## Rank books or score one book

After generating `book_profiles.jsonl`, produce topic-filtered top-K recommendations:

```bash
uv run bookmatch-ml rank \
  --reader examples/reader_profile.json \
  --books data/output/book_profiles.jsonl \
  --limit 5
```

Calculate a specific-book fit, including a cross-topic mismatch, with:

```bash
uv run bookmatch-ml rank \
  --reader examples/reader_profile.json \
  --books data/output/book_profiles.jsonl \
  --book-id isbn13:9781985086593
```

The configurable baseline in `configs/ranking.yaml` uses:

```text
score = α × TopicFit
      + β × VocabularyFit
      + γ × KnowledgeFit
      + δ × ComprehensionFit
```

`KnowledgeFit` is separately decomposed into concept-density fit, prerequisite-demand fit, and
readiness for inferred prerequisite concepts that were actually assessed. Each readiness/demand
pair uses `1 - absolute gap`, as identified by `absolute_gap_v1` in configuration.

Missing values are never converted to zero. Only available components are calculated and their
configured weights are renormalized. Every item therefore includes:

- the total and all component/subcomponent scores;
- the normalized `active_weights` actually used;
- `component_weight_coverage`, the original configured weight represented by available data;
- `unavailable_components` and evidence limitations;
- deterministic Korean reasons grounded in score gaps and covered concepts;
- ranking, reader-profile, book-profile, and configuration versions and hashes.

Consequently, a topic-matched book with no public prose can receive a topic-only score of `1.0`
after renormalization. This is not hidden or treated as high-confidence readiness fit:
`component_weight_coverage` is only `0.35` under the default configuration and the unavailable
dimensions are explicit. Evaluation should compare and calibrate this baseline before changing
the missing-evidence policy.

## Spring integration API

Start the stateless calculation adapter locally after the batch profiles have been generated:

```bash
uv run uvicorn bookmatch_ml.api:create_app --factory --host 127.0.0.1 --port 8000
```

It exposes exactly two application routes:

```text
POST /ml/reader-profile
POST /ml/rank
```

The JSON boundary uses camelCase for Spring DTOs while internal Python models remain snake_case.
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs` while the process
is running.

The factory loads packaged defaults without reading repository-relative files at import time.
Set `BOOKMATCH_ML_CONFIG_DIR` to a directory containing `reader.yaml` and `ranking.yaml` to select
externally mounted, versioned configurations in deployment. Add `ranking_v2.yaml` when that
deployment should accept explicit ranking-v2 requests; v1-only operation does not require it.

`POST /ml/reader-profile` accepts the same assessment content as `examples/assessment.json`, with
camelCase keys and an optional `userId` correlation value. It returns the three readiness
dimensions, dimension details, concept readiness, and all profile/config versions. `userId` is
only echoed; it is not a predictive feature.

`POST /ml/rank` accepts this explicit boundary shape:

```json
{
  "readerProfile": {
    "userId": 1,
    "topicId": "operating-systems",
    "vocabulary": 0.72,
    "backgroundKnowledge": 0.55,
    "comprehension": 0.68,
    "conceptReadiness": [],
    "profileVersion": "reader-v1",
    "configVersion": "reader-config-v1",
    "configHash": "sha256:..."
  },
  "candidateBooks": [
    {
      "bookId": "isbn13:...",
      "topicDistribution": {"operating-systems": 1.0},
      "coveredConcepts": [],
      "prerequisiteConcepts": [],
      "lexicalDifficulty": 0.64,
      "syntacticComplexity": 0.71,
      "conceptDensity": 0.78,
      "prerequisiteDemand": 0.82,
      "featureVersion": "book-v1",
      "configVersion": "features-config-v1",
      "configHash": "sha256:..."
    }
  ],
  "limit": 5
}
```

Set the optional top-level `bookId` to score one supplied candidate even when it does not match
the selected topic. Otherwise the endpoint returns the configured topic-filtered top K. Each item
contains flat `topicFit`, `vocabularyFit`, `knowledgeFit`, and `comprehensionFit` fields plus the
subcomponents, active weights, evidence diagnostics, deterministic reasons, and version hashes.

Spring remains responsible for loading persisted assessments and candidate profiles, calling
these endpoints, and storing results. The API does not connect to PostgreSQL or upstream book
providers and does not own authentication or recommendation history. Its candidate schema is a
small matching projection rather than the complete internal `BookProfile`, so internal analysis
details are not coupled to Spring DTOs.

Ranking-v2 is available on the same route only when the request explicitly sets
`"rankingModel": "rank-prerequisite-first-v2"`. Omitting the selector preserves `rank-v1`.
V2 returns rank plus separate prerequisite-readiness and direct-opportunity axes, never a fake
scalar score, and returns at most the requested limit without filling shortages from fallback
pools. The complete request/response, error semantics, Scale-50 smoke commands, and Backend
migration checklist are in
[`docs/rank-v2-production-integration.md`](docs/rank-v2-production-integration.md).

## Evaluate the baseline

After generating the current ten-book profile artifact, run the combined evaluation report:

```bash
uv run bookmatch-ml evaluate \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --reader examples/reader_profile.json \
  --difficulty-labels examples/difficulty_judgments.json \
  --ablation-book-id isbn13:9781985086593 \
  --output data/reports/evaluation.json
```

The command is configured by `configs/evaluation.yaml` and produces three linked evaluations:

- human relative-difficulty order versus the weighted system difficulty, using Spearman rank
  correlation and strict pairwise agreement;
- deterministic topic-only ordering versus the full readiness-aware ranking, including Top-K
  overlap and per-book rank movement;
- TOC-only, TOC+description, and all-evidence ablation, including concepts, prerequisite
  concepts, weights, and newly available prose-difficulty components.

Difficulty labels use `human_rank: 1` for the easiest book and require one complete order with
no duplicate books or ranks. Labeled books without a prose-based difficulty score are excluded,
never scored as zero, and recorded as failure cases. Pairwise system-score ties are recorded and
count as non-agreements. Spearman is `null` when an ordering is constant.

The topic-only baseline sorts by topic fit and then stable `book_id`; it does not use reader or
prose features. A positive `readiness_rank_change` means the book moved upward under the
readiness-aware model. The report also identifies rankings calculated with incomplete component
weight coverage.

`examples/difficulty_judgments.json` is deliberately marked synthetic. The current public
evidence leaves only a very small comparable subset, so its metrics demonstrate the evaluation
contract and must not be presented as model quality or diagnostic validity. Replace it with a
versioned human-labeled file before drawing conclusions.

The first reproducible run against the current ten-book canonical output, including its evidence
limitations and next decision gates, is recorded in
[`docs/evaluation-baseline-v1.md`](docs/evaluation-baseline-v1.md).

## Compare ranking evidence policies

The existing `rank` CLI and default `/ml/rank` behavior retain renormalized scoring. A separate
batch experiment compares that baseline with a configurable minimum coverage rule and a two-stage
presentation:

```bash
uv run bookmatch-ml evaluate-ranking-policies \
  --data-dir ../Data-Pipeline/data/processed \
  --books data/output/book_profiles.jsonl \
  --reader data/output/reader_profile.json \
  --output data/reports/ranking_policy_evaluation.json
```

Run the profile-building commands above first. The policy parameters and versions are in
`configs/ranking_policies.yaml`; use `--policy-config` to compare a different versioned setting.
The report retains every candidate, including those below the coverage threshold, and separates
readiness scores from topic-only or ineligible entries. The current ten-book results, snapshot
hashes, tradeoffs, and next decision gate are in
[`docs/ranking-policy-evaluation-v1.md`](docs/ranking-policy-evaluation-v1.md).

## Development checks

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The tests use small versioned fixtures under `tests/fixtures/`, including a compact Scale-50
ranking snapshot; they require no network, database, external LLM, Spring service, or live
Data-Pipeline collection.

## Package layout

```text
src/bookmatch_ml/
├── schemas.py          # canonical, profile, ranking, and evaluation models
├── config.py           # strict versioned feature/reader/ranking/evaluation config
├── io.py               # deterministic atomic artifact writers
├── cli.py              # batch command entry points
├── api.py              # stateless Spring-facing FastAPI routes
├── default_configs/    # reader/ranking defaults bundled in wheels
├── book/
│   ├── concepts.py     # lexicon concept/prerequisite baseline
│   ├── difficulty.py   # per-document prose features and aggregation
│   └── profile.py      # BookProfile assembly
├── concept_v2/
│   ├── graph.py        # versioned prerequisite-candidate DAG validation
│   ├── toc.py          # ordered TOC hierarchy reconstruction
│   ├── profile.py      # TOC mapping and v2 book concept profiles
│   ├── matching.py     # separate readiness and opportunity axes
│   └── validation.py   # graph-edge and complete TOC mapping review artifacts
├── reader/
│   └── profile.py      # assessment loading and reader-readiness scoring
├── ranking/
│   ├── loader.py       # strict generated-profile loading
│   ├── matching.py     # decomposed scoring and weight renormalization
│   └── explanation.py  # deterministic Korean reason templates
├── integration/
│   ├── schemas.py      # explicit camelCase HTTP DTO boundary
│   └── service.py      # thin reader-profile/ranking orchestration
├── evaluation/
│   ├── ablation.py     # evidence-richness comparison and change summary
│   ├── difficulty.py   # Spearman and pairwise human-order evaluation
│   ├── recommendation.py # topic-only versus readiness-aware comparison
│   ├── ranking_policies.py # evidence-policy comparison over unchanged scores
│   └── report.py       # combined versioned evaluation report
└── data/
    ├── loader.py       # JSONL parsing and cross-record validation
    └── evidence.py     # deterministic BookEvidence and coverage assembly
```

Canonical and generated third-party data belongs in ignored `data/` subdirectories. Do not
commit raw book text to this repository.

## English evidence integration boundary

Canonical book/TOC/document English fields follow the current optional-field schema.
The evidence handoff accepts English only through `book-evidence-v3`; v1/v2 remain frozen.
Default concept mapping, prose difficulty, and question grounding use original text.
Stored English is compared explicitly with `compare-english-evidence` and can be inspected
with `--review-packet`; it does not silently replace the source passage or its hash.

The earlier English-first pilot configuration and helper have been retired from this PR.
CLI defaults remain `configs/features.yaml` and `configs/concept_matching_v2.yaml`.
Historical pilot outputs are local snapshots, not active Backend projections. See the
[PR #32 reconciliation](docs/account-concept-v3-integration.md) for compatibility checks.

대량 수집 471권·목차439권과 LA83 후보의 실제 연동은 [대량 데이터 후속 연결](docs/linear-algebra-live-handoff.md#대량-데이터-후속-연결)을 참고한다. 고정 snapshot 준비는 `scripts/prepare_discovery_catalog_handoff.py`, 실제 HTTP 검증은 `scripts/verify_discovery_catalog_live.py`를 사용한다.

## 개념·수행 능력 추천과 공통 지도 (2026-10-03)

`/ml/reader-profile`에 전달된 `answerMode`와 `cognitiveOperation`으로 objective 개념·능력 관찰을 만들고 자기평가를 분리한 `conceptProfile`을 반환합니다. `/ml/concepts/{topicId}`는 검토 승인된 공통 개념 그래프를 제공하고, `/ml/learning-fit`은 선택 능력의 objective 관찰·책 내용·선수관계 후보를 비교하는 `concept-learning-v1` 기준선입니다. 기존 `/ml/rank` 계약과 알고리즘은 변경하지 않습니다. 새 추천은 미평가를 0으로 채우거나 자기평가를 verified 능력으로 합치지 않으며, 선수관계가 비어 있는 경우에도 준비도 판단을 보류합니다. 상세 계약·제약·실제 앱 연결은 [Backend 설계 기록](../Backend/docs/concept-learning-v1.md)을 참고하세요.


## 개념별 진단 설계서 v2

개인별 세부 순위 실험은 [학습 순위 v3](docs/personalized-learning-order-v3.md)를 참고한다.
`/ml/learning-fit`의 명시적 v3 요청만 적용되며 앱 기본 v2 추천은 유지한다.

`build-concept-assessment`는 선형대수 6개 개념 × 뜻·성질/계산·적용/설명·추론 목표 18개를 생성합니다.
목표·오개념·설계 난도는 `configs/concept_assessment_targets.json`에 있습니다. 기존 문항 유형 할당과 별개이며,
실제 canonical 목차 연결이 없는 개념은 생성하지 않습니다. 입력 네 파일과 개념 그래프·매칭·특징 설정의 해시를 보존합니다.
목차는 평가 대상 선정 근거입니다. 정답, 학년, 책 본문 난이도의 근거로 쓰지 않습니다.

```sh
uv run bookmatch-ml build-concept-assessment --data-dir ../Data-Pipeline/data/processed --output data/output/concept-assessment-v2/blueprint.json
```

버전은 `concept-assessment-blueprint-v2` / `concept-question-spec-v2`이며 QG가 독립 계약 사본을 읽습니다.
QG의 생성·검토·Backend 등록 흐름은 [개편 계획](../Question-Generation/docs/concept-assessment-plan.md)에 있습니다.
생성은 이 저장소의 책임이 아니고, 실제 사용자 응답이나 외부 생성 API를 기본 테스트에서 호출하지 않습니다.

`concept-abilities-v2`는 `measurementContext=prior-knowledge`의 객관식 응답만 `abilities`로 집계합니다.
`provided-information`은 `providedInformationAbilities`, 조건이 없는 예전 객관식은 `legacyContextAbilities`에
분리합니다. 자기평가는 `selfReports`입니다. 뜻/적용/추론의 정답 수는 관찰이며 보정된 숙련도 확률이 아닙니다.
기존 종합 점수는 저장 호환용으로 유지되지만 새 추천에서 사용하지 않습니다.
개념 문항의 `generated-question-v5` 계약은 생성 프롬프트 v1과 Markdown·LaTeX 규칙을 추가한 v2를 모두 읽습니다.
본문 표시 형식은 개념·능력·측정 조건·목차 근거를 바꾸지 않으며, 본문 변경은 새 content ID로 검증합니다.

### v2·v3 추천 순서의 사람 평가

평가표 생성 → 독립 평가자 입력 → 파일 검증 및 동순위 보존 비교 명령은
[사람 평가 안내](docs/learning-order-human-evaluation.md)를 참고하세요.
점수 없는 평가표는 평가 대기로 처리하며, 앱의 기본 추천 모델은 변경하지 않습니다.
