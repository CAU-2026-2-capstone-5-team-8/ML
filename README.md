# BookMatch ML

Evidence-first ML and recommendation logic for the CAU Capstone Team 8 personalized
technical-book recommendation project.

The current implementation covers the canonical-handoff, book-profile, reader-profile, and
matching/ranking baseline milestones:

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
              topic-specific reader profile
                            ↓
             explainable matching + ranking
```

It deliberately contains no scraping, source-provider adapters, database integration, remote
LLM calls, learned ranking, or network-dependent logic.

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
├── schemas.py          # canonical, evidence, and profile Pydantic models
├── config.py           # strict versioned feature configuration
├── io.py               # deterministic atomic artifact writers
├── cli.py              # batch command entry points
├── book/
│   ├── concepts.py     # lexicon concept/prerequisite baseline
│   ├── difficulty.py   # per-document prose features and aggregation
│   └── profile.py      # BookProfile assembly
├── reader/
│   └── profile.py      # assessment loading and reader-readiness scoring
├── ranking/
│   ├── loader.py       # strict generated-profile loading
│   ├── matching.py     # decomposed scoring and weight renormalization
│   └── explanation.py  # deterministic Korean reason templates
├── evaluation/
│   └── ablation.py     # evidence-richness comparison
└── data/
    ├── loader.py       # JSONL parsing and cross-record validation
    └── evidence.py     # deterministic BookEvidence and coverage assembly
```

Canonical and generated third-party data belongs in ignored `data/` subdirectories. Do not
commit raw book text to this repository.
