# BookMatch ML

An experimental [concept difficulty and reader-fit rubric](docs/concept-difficulty-v1.md)
extends the existing TOC/graph baseline with explicit concept levels, prerequisite gaps,
learning burden, and a blind human-review export. Run it separately from the unchanged v1 API
until independent recommendation-quality evaluation is complete.

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

## Experimental concept matching v3 and difficulty v2

The existing `rank` CLI and `/ml/rank` API default remain `absolute_gap_v1`. A separate batch command
compares their v1 result with TOC-based book concept coverage and the existing
`ReaderProfile.concept_readiness` values. It reports **prerequisite readiness** and **learning
opportunity** separately, each with its own mastery-assessment coverage. Missing concept mastery
is unknown, never zero. Prose difficulty remains in the report only as a v1 comparison and
optional diagnostic.

`configs/concept_difficulty.yaml` also defines an experimental intrinsic book score and a separate
reader learning-burden interval. The exact formulas, bands, evidence rules, real-data snapshot,
and review workflow are documented in
[`docs/concept-difficulty-v1.md`](docs/concept-difficulty-v1.md).

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
Set `BOOKMATCH_ML_CONFIG_DIR` to a directory containing `reader.yaml`, `ranking.yaml`, and
`concept_difficulty.yaml` to select externally mounted, versioned configuration in deployment.
For backward compatibility, a directory containing only the original two files still starts the
baseline API; the experimental strategy returns a validation error until its third config is
mounted.

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

The optional top-level `rankingStrategy=concept_difficulty_v2_experimental` selects the new
concept-aware comparison. Every candidate must then include the batch-produced
`conceptProfile`; the response adds `conceptDifficulty` containing the reader-independent book
score/band, reader burden interval, TOC matches, prerequisite graph paths, and hashes. Omitting the
strategy preserves the existing request and response behavior. The nested `conceptProfile` is
the versioned ML batch artifact and therefore retains its canonical snake_case field names. New
artifacts use `toc-concept-profile-v3`; v2 artifacts remain readable with empty exclusion-audit
fields for compatibility.

Spring remains responsible for loading persisted assessments and candidate profiles, calling
these endpoints, and storing results. The API does not connect to PostgreSQL or upstream book
providers and does not own authentication or recommendation history. Its candidate schema is a
small matching projection rather than the complete internal `BookProfile`, so internal analysis
details are not coupled to Spring DTOs.

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

The existing `rank` CLI and `/ml/rank` API retain renormalized scoring. A separate batch experiment
compares that baseline with a configurable minimum coverage rule and a two-stage presentation:

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

The tests use only small synthetic fixtures under `tests/fixtures/`; they require no network,
database, external LLM, Spring service, or live Data-Pipeline collection.

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
