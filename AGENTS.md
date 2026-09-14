# AGENTS.md

## 1. Project Overview

This repository is the ML/NLP and recommendation-logic component for the CAU Capstone Team 8 personalized technical-book recommendation project.

The overall system compares:

- what a reader currently understands in a selected topic,
- what concepts and prerequisite knowledge a book appears to require,
- how difficult the book's prose is,
- what concepts the book covers,
- and how well the reader's current readiness matches the book.

The project is not trying to assign one universal difficulty score to every book.

The main goal is:

> Convert canonical book evidence into explainable book profiles, convert assessment results into topic-specific reader profiles, and match the two to produce explainable recommendations or book-fit scores.

The initial supported topics are:

```text
Computer Science
└── Operating Systems

Mathematics
└── Linear Algebra
```

The first ML milestone uses the current ten-book dataset produced by the `Data-Pipeline` repository.

Do not optimize for hundreds or thousands of books before the ten-book vertical slice works end-to-end.

---

## 2. System Boundary

The intended architecture is:

```text
External Public Sources
        ↓
Data-Pipeline
        ↓
books.jsonl
documents.jsonl
toc.jsonl
sources.jsonl
        ↓
ML
├── Evidence Assembly
├── Book Concept Profile
├── Text Difficulty Profile
├── Reader Profile
├── Matching / Ranking
└── Evaluation
        ↓
Spring Boot
├── Service orchestration
├── PostgreSQL
├── Assessment lifecycle
├── Recommendation persistence
└── API delivery
```

This repository owns calculation and evaluation logic.

This repository does **not** own:

- web scraping,
- external book-source adapters,
- canonical book metadata collection,
- service databases,
- user authentication,
- Spring Boot entities or repositories,
- frontend state,
- OCR,
- persistent recommendation history.

The ML repository must not depend on Open Library, Wiley, eCampus, OSTEP, Google Books, or any other upstream provider-specific representation.

It consumes only the provider-independent canonical output of `Data-Pipeline`.

---

## 3. Core Design Principles

### 3.1 Evidence first, inference second

Never hide the distinction between collected evidence and inferred features.

For every inferred result, preserve enough metadata to answer:

> Which evidence was available, which evidence was actually used, and which algorithm/version produced this value?

Do not infer unavailable information merely to make every book look complete.

---

### 3.2 Missing evidence is valid data

Different books expose different public evidence.

A book may have:

```text
metadata       yes
description    yes
TOC            yes
preface        no
introduction   no
sample text    no
```

Another book may have all of them.

This is expected.

If a feature cannot be calculated from the available evidence, store `null` or mark it unavailable.

Never:

- replace missing values with zero unless zero is semantically correct,
- invent prose-based features from a TOC,
- treat missing evidence as poor book quality,
- silently substitute one feature for another.

---

### 3.3 Explainable baseline before complex ML

The first working version should prefer:

- deterministic transformations,
- transparent formulas,
- explicit feature values,
- simple configuration,
- reproducible ranking,
- inspectable recommendation reasons.

Do not begin with:

- deep neural ranking models,
- opaque embeddings as the only representation,
- end-to-end LLM recommendation,
- reinforcement learning,
- collaborative filtering requiring large user histories,
- complicated psychometric models.

More complex models are comparison candidates after the baseline works and evaluation data exists.

---

### 3.4 Separate concept evidence from prose difficulty

Do not compress every book immediately into four scalar difficulty values.

A book profile has at least two distinct components:

```text
Book Profile
├── Concept Profile
└── Text Difficulty Profile
```

The Concept Profile can often be built from:

- TOC,
- description,
- preface,
- introduction,
- sample text,
- other public text.

The Text Difficulty Profile requires actual prose.

This distinction is mandatory.

---

### 3.5 Internal scores use [0, 1]

Unless a concrete experiment requires otherwise, internal normalized scores should use:

```text
0.0 <= score <= 1.0
```

UI code may convert these values to percentages.

Do not mix `[0, 1]` and `[0, 100]` internally.

---

### 3.6 Version everything that changes interpretation

At minimum retain:

```text
feature_version
concept_profile_version
difficulty_profile_version
profile_version
model_version
config_version
```

If an algorithm, weight, threshold, prompt, feature formula, or normalization rule changes, the corresponding version must change.

---

## 4. Upstream Data Contract

The upstream `Data-Pipeline` repository emits:

```text
books.jsonl
documents.jsonl
toc.jsonl
sources.jsonl
```

The ML repository must consume those files without source-specific logic.

### 4.1 `books.jsonl`

Provides normalized bibliographic identity such as:

- `book_id`
- ISBN
- title
- authors
- publisher
- publication year
- language
- topics

`book_id` is the primary join key.

---

### 4.2 `documents.jsonl`

Provides normalized textual evidence.

Typical document types may include:

```text
description
publisher_summary
preface
introduction
preview
sample_chapter
index
other
```

A document contains at least:

- `document_id`
- `book_id`
- `document_type`
- `text`
- `source_id`
- `content_hash`

Do not assume every type exists for every book.

---

### 4.3 `toc.jsonl`

Provides hierarchical table-of-contents entries.

Important fields include:

- `toc_entry_id`
- `book_id`
- `parent_entry_id`
- `level`
- `order_index`
- `label`
- `title`
- `source_id`

Preserve hierarchy.

Do not flatten the TOC irreversibly during ingestion.

---

### 4.4 `sources.jsonl`

Provides provenance.

The ML repository normally does not use provider identity as a predictive feature.

It may use source metadata only for:

- audit,
- debugging,
- evidence coverage,
- reproducibility,
- filtering invalid/missing references.

Never train or rank books based on provider name.

---

## 5. Canonical ML Input Model

The first transformation inside this repository should be:

```text
Canonical JSONL
      ↓
EvidenceAssembler
      ↓
BookEvidence
```

Suggested logical shape:

```json
{
  "book_id": "isbn13:...",
  "metadata": {},
  "toc": [],
  "documents": [],
  "coverage": {
    "has_toc": true,
    "has_description": true,
    "has_preface": false,
    "has_introduction": false,
    "has_preview": false,
    "has_sample_chapter": false,
    "has_other_text": false,
    "toc_entry_count": 57,
    "prose_character_count": 0
  }
}
```

The exact Pydantic model may evolve, but the following are required:

- stable `book_id`,
- deterministic assembly,
- validated cross-file references,
- explicit evidence coverage,
- no provider-specific fields in downstream feature code.

The loader must fail visibly on broken references or conflicting canonical identities.

---

## 6. Evidence Coverage

Evidence coverage must be a first-class object, not an incidental log line.

Suggested fields:

```text
has_metadata
has_toc
has_description
has_preface
has_introduction
has_preview
has_sample_chapter
has_other_text
toc_entry_count
document_count
prose_document_count
prose_character_count
```

This object is important because downstream interpretation depends on how much evidence existed.

For example:

```text
Book A
TOC: 201 entries
Description: yes
Prose: none

Book B
TOC: 62 entries
Description: yes
Preface: yes
Introduction: yes
Sample: yes
Prose: 100k+ chars
```

Both may receive a Concept Profile.

Only Book B should receive a full prose-based Text Difficulty Profile.

---

## 7. Book Profile

A book profile should logically resemble:

```json
{
  "book_id": "isbn13:...",
  "concept_profile": {},
  "difficulty_profile": {},
  "evidence_coverage": {},
  "feature_version": "book-v1"
}
```

Do not force both subprofiles to be equally complete.

---

## 8. Concept Profile

### 8.1 Goal

Represent:

- what the book covers,
- what concepts appear central,
- what concepts may be prerequisites,
- and how strongly the evidence supports each conclusion.

Suggested output:

```json
{
  "book_id": "isbn13:...",
  "covered_concepts": [
    {
      "concept": "virtual memory",
      "weight": 0.92,
      "evidence_types": ["toc", "description"]
    }
  ],
  "prerequisite_concepts": [
    {
      "concept": "process",
      "weight": 0.73,
      "method": "explicit_or_proxy"
    }
  ],
  "topic_distribution": {
    "operating-systems": 0.96
  },
  "profile_version": "concept-v1"
}
```

---

### 8.2 Evidence use

Concept extraction may use:

```text
TOC
description
preface
introduction
sample chapter
other public prose
```

Do not concatenate all evidence into one irreversible string and lose structure.

Preserve:

- evidence type,
- document identity,
- TOC hierarchy,
- section order when useful.

The extraction logic should be able to answer whether a concept came primarily from:

- structural evidence,
- descriptive evidence,
- or prose evidence.

---

### 8.3 Covered concepts

For the first baseline, prefer a reproducible and inspectable approach.

Possible components include:

- normalized TOC terms,
- noun/keyphrase extraction,
- topic-specific concept lexicons,
- frequency across TOC branches,
- repetition across independent evidence types.

Do not require a remote LLM for the core baseline.

An LLM-assisted extractor may be added later as an experiment, but it must not replace the reproducible baseline.

---

### 8.4 Prerequisite concepts

`Prerequisite Demand` and `prerequisite_concepts` are inferred signals, not ground truth.

Do not claim:

> The reader must know X.

Prefer:

> The available evidence suggests X is a likely prerequisite.

Baseline signals may include:

- explicit prerequisite wording in preface/introduction/description,
- concepts assumed without local explanation,
- advanced concepts appearing early,
- terminology dependencies,
- topic-specific prerequisite lexicons.

Every prerequisite inference should retain its method or evidence category.

---

## 9. Text Difficulty Profile

### 9.1 Goal

Estimate the difficulty of actual book prose.

Suggested output:

```json
{
  "book_id": "isbn13:...",
  "lexical_difficulty": 0.64,
  "syntactic_complexity": 0.71,
  "concept_density": 0.78,
  "prerequisite_demand": 0.82,
  "analyzed_document_count": 3,
  "analyzed_character_count": 100040,
  "feature_version": "difficulty-v1"
}
```

All scalar values should be normalized to `[0, 1]` only after the raw subfeatures are retained or reproducible.

---

### 9.2 Valid input

Use actual prose such as:

```text
preface
introduction
preview
sample_chapter
other
```

A description may be used for some semantic features, but should not be treated as equivalent to textbook prose for syntactic difficulty.

A TOC must never be used to calculate sentence-level syntactic complexity.

---

### 9.3 Lexical Difficulty

Possible raw subfeatures:

- rare-word ratio,
- domain-term ratio,
- average lexical frequency,
- vocabulary diversity,
- token-length statistics.

Do not hide all lexical information behind one opaque score.

Keep the primitive measurements available for evaluation and debugging.

---

### 9.4 Syntactic Complexity

Possible raw subfeatures:

- mean sentence length,
- median sentence length,
- long-sentence ratio,
- punctuation/clause proxies,
- parse-tree or dependency-based measures if a justified parser is introduced.

Start simple.

Do not add a heavyweight parser until the simple baseline has been evaluated.

---

### 9.5 Concept Density

Possible baseline interpretation:

> How densely domain-relevant concepts occur in the analyzed prose.

Possible subfeatures:

- concept mentions per 1,000 tokens,
- unique concept count per 1,000 tokens,
- repeated concept concentration,
- concept coverage relative to the book's Concept Profile.

The exact formula must live in versioned code/config and be testable.

---

### 9.6 Prerequisite Demand

Treat this as a proxy.

Possible signals include:

- advanced concepts introduced without explanation,
- domain-term density near the beginning of the text,
- ratio of prerequisite-concept mentions,
- concept references whose definitions are absent from the analyzed window.

Do not present the score as direct measurement of human prior knowledge.

---

### 9.7 Multiple-document aggregation

When multiple prose documents exist:

```text
preface
introduction
sample chapter
appendix
```

calculate per-document features first.

Then aggregate.

Do not simply concatenate arbitrarily large documents and treat them as one sample.

Aggregation must preserve:

- document type,
- document size,
- per-document feature vector,
- final aggregation rule.

The aggregation rule must be versioned and deterministic.

---

## 10. Reader Profile

### 10.1 Goal

Convert topic-specific assessment results into a reader-readiness profile.

Initial dimensions:

```text
vocabulary
background_knowledge
comprehension
```

Suggested output:

```json
{
  "topic_id": "operating-systems",
  "vocabulary": 0.72,
  "background_knowledge": 0.55,
  "comprehension": 0.68,
  "profile_version": "reader-v1"
}
```

Reader Profile is topic-specific.

The same user may have very different profiles for Operating Systems and Linear Algebra.

---

### 10.2 Input

Expected question metadata may include:

```text
question_id
topic_id
concept_id or concept_tag
question_type
difficulty
correct / score
```

Initial question types:

```text
vocabulary
background_knowledge
comprehension
```

---

### 10.3 Baseline scoring

Start with a transparent scoring method based on:

- correctness,
- question type,
- declared difficulty.

If difficulty weighting is used, weights must live in configuration.

Do not bury arbitrary weights inside implementation code.

Do not introduce IRT, CAT, or another psychometric model until real response data exists.

---

## 11. Reader-Book Matching

### 11.1 Baseline structure

The initial explainable structure is:

```text
Score =
    α * TopicFit
  + β * VocabularyFit
  + γ * KnowledgeFit
  + δ * ComprehensionFit
```

Conceptual mapping:

```text
Reader vocabulary
    ↔ lexical difficulty

Reader background knowledge
    ↔ concept density + prerequisite demand + prerequisite concept overlap

Reader comprehension
    ↔ syntactic complexity

Selected topic / learning goal
    ↔ topic fit + covered concepts
```

The exact formula must be configurable and versioned.

---

### 11.2 Missing feature handling

Missing values must not automatically become zero.

Example:

```text
VocabularyFit      available
KnowledgeFit       available
ComprehensionFit   unavailable
```

The ranking function should:

1. calculate available components,
2. renormalize active weights,
3. record which components were unavailable,
4. expose this limitation in the result.

A book must not be unfairly penalized merely because public prose was unavailable.

---

### 11.3 Score decomposition

Every ranking result must preserve component scores.

Suggested output:

```json
{
  "book_id": "isbn13:...",
  "score": 0.81,
  "components": {
    "topic_fit": 1.0,
    "vocabulary_fit": 0.82,
    "knowledge_fit": 0.67,
    "comprehension_fit": 0.79
  },
  "unavailable_components": [],
  "reasons": [
    "Vocabulary demand is close to the reader profile.",
    "Prerequisite demand is somewhat above the current background-knowledge score."
  ],
  "model_version": "rank-v1"
}
```

Do not return only one total score.

---

## 12. Recommendation Reasons

The first version should use deterministic templates based on score gaps and evidence.

Examples:

```text
어휘 수준은 현재 준비도와 대체로 맞습니다.
선행지식 요구도는 현재 배경지식보다 다소 높습니다.
실제 본문 샘플이 없어 문장 난이도 평가는 제한적입니다.
운영체제 핵심 개념 중 가상 메모리와 동기화를 폭넓게 다룹니다.
```

Do not use a generative model solely to make reasons sound natural.

The reason must remain faithful to actual component scores and available evidence.

A generative explanation layer may be added later, but the deterministic structured reasons remain the source of truth.

---

## 13. Evidence Ablation

The current dataset contains books with different levels of evidence richness.

Use this intentionally.

For books with rich evidence, compare:

```text
TOC only
TOC + Description
TOC + Description + Preface/Introduction/Sample
```

Questions to measure:

- Do extracted concepts change?
- Do prerequisite predictions change?
- Does the difficulty profile stabilize?
- Are recommendation results sensitive to richer evidence?

This experiment is important because it determines whether additional data collection is worth the engineering cost.

---

## 14. Evaluation

Evaluation code belongs in this repository from the beginning.

Do not wait until the end of the semester to make metrics reproducible.

### 14.1 Book difficulty

For a small human-labeled set, compare:

```text
human relative difficulty order
vs
system difficulty order
```

Initial metrics:

- Spearman rank correlation,
- pairwise agreement.

Do not over-interpret results from very small samples.

---

### 14.2 Recommendation

Compare at least:

```text
Topic-only baseline
vs
Topic + Reader Readiness / Book Difficulty
```

When enough interaction data exists, possible ranking metrics include:

- Precision@K,
- Recall@K,
- Hit Rate@K,
- NDCG@K.

These are not required for the first ten-book vertical slice.

---

### 14.3 Reader assessment

When real responses exist, inspect:

- accuracy by declared question difficulty,
- score distribution,
- whether question types distinguish users,
- repeated-test consistency.

Do not claim diagnostic validity from synthetic data.

---

### 14.4 Explanation quality

For user studies, evaluate whether the explanation matches perceived difficulty and the reader's experience.

This may remain qualitative in the first MVP.

---

## 15. LLM Usage Policy

Remote LLMs are optional experimental components, not core infrastructure.

If an LLM is used for:

- concept extraction,
- prerequisite inference,
- concept normalization,
- explanation rewriting,

then:

1. use structured output,
2. record model name,
3. record prompt version,
4. use deterministic settings when possible,
5. cache or preserve the raw response for reproducibility,
6. never require live LLM access in the default test suite,
7. keep a non-LLM baseline for comparison.

Do not send copyrighted raw book text to a third-party model unless the project has explicitly confirmed that the usage is allowed.

Prefer using only the minimum necessary excerpt when an external model is used.

---

## 16. Repository Structure

Recommended initial structure:

```text
ML/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
│
├── configs/
│   ├── features.yaml
│   ├── reader.yaml
│   └── ranking.yaml
│
├── src/
│   └── bookmatch_ml/
│       ├── __init__.py
│       ├── schemas.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   └── evidence.py
│       │
│       ├── book/
│       │   ├── concepts.py
│       │   ├── difficulty.py
│       │   └── profile.py
│       │
│       ├── reader/
│       │   └── profile.py
│       │
│       ├── ranking/
│       │   ├── matching.py
│       │   └── explanation.py
│       │
│       ├── evaluation/
│       │   ├── difficulty.py
│       │   ├── recommendation.py
│       │   └── ablation.py
│       │
│       └── cli.py
│
├── tests/
│   ├── fixtures/
│   ├── test_loader.py
│   ├── test_evidence.py
│   ├── test_book_profile.py
│   ├── test_reader_profile.py
│   ├── test_matching.py
│   └── test_evaluation.py
│
├── examples/
│   ├── assessment.json
│   └── reader_profile.json
│
└── data/
    ├── input/
    ├── output/
    └── reports/
```

Generated or third-party data under `data/` should normally be excluded from Git.

Commit only small, legally appropriate fixtures.

---

## 17. Technology Stack

Use:

```text
Python 3.12+
uv
pydantic
typer
numpy
scipy
scikit-learn
pytest
ruff
```

Add NLP dependencies only when a concrete baseline requires them.

Examples that may be considered later:

```text
wordfreq
spaCy
sentence-transformers
```

Do not add them preemptively.

Do not add:

- FastAPI before the CLI/batch path works,
- PostgreSQL access to this repository,
- Redis,
- Celery,
- Kafka,
- distributed training infrastructure,
- a vector database without a demonstrated need.

---

## 18. CLI

The repository must work without manually editing notebooks.

Suggested commands:

```bash
uv run bookmatch-ml inspect-data \
  --data-dir ../Data-Pipeline/data/processed
```

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl
```

```bash
uv run bookmatch-ml reader-profile \
  --input examples/assessment.json
```

```bash
uv run bookmatch-ml rank \
  --reader examples/reader_profile.json \
  --books data/output/book_profiles.jsonl \
  --limit 5
```

```bash
uv run bookmatch-ml evaluate \
  --config configs/evaluation.yaml
```

Exact command names may evolve.

Reproducible commands are mandatory.

---

## 19. Batch First, API Later

Book analysis should initially run as batch processing.

Do not create a real-time `/ml/book-features` endpoint for the first milestone.

The first Spring integration only needs thin endpoints or callable functions for:

```text
reader-profile
rank
```

Even those should be introduced only after the underlying pure functions and CLI work.

The integration boundary should remain thin.

---

## 20. Spring Integration Contract

Spring Boot is the service and persistence owner.

The ML repository returns calculations.

It does not write directly to the service database.

Expected logical endpoints later:

```text
POST /ml/reader-profile
POST /ml/rank
```

Possible ranking request:

```json
{
  "readerProfile": {
    "userId": 1,
    "topicId": "operating-systems",
    "vocabulary": 0.72,
    "backgroundKnowledge": 0.55,
    "comprehension": 0.68,
    "profileVersion": "reader-v1"
  },
  "candidateBooks": [],
  "limit": 5
}
```

Possible response:

```json
{
  "items": [
    {
      "bookId": "isbn13:...",
      "score": 0.81,
      "topicFit": 1.0,
      "vocabularyFit": 0.82,
      "knowledgeFit": 0.67,
      "comprehensionFit": 0.79,
      "reasons": [],
      "modelVersion": "rank-v1"
    }
  ]
}
```

Do not couple internal Python classes directly to Spring DTOs.

Use explicit schemas at the boundary.

---

## 21. Configuration

Weights and thresholds belong in config, not scattered through code.

Examples:

```text
configs/features.yaml
configs/reader.yaml
configs/ranking.yaml
```

Configuration may include:

- difficulty weights,
- feature normalization,
- matching weights,
- missing-component rules,
- explanation thresholds,
- concept extraction thresholds.

A result must be reproducible from:

```text
input data
+
code version
+
config version
+
model/feature version
```

---

## 22. Tests

Tests must focus on deterministic logic.

Prioritize:

### Data loading

- four JSONL files load correctly,
- invalid cross-file references fail,
- duplicate IDs fail visibly,
- TOC hierarchy is preserved,
- missing optional evidence is accepted.

### Evidence assembly

- documents attach to the correct book,
- document types remain distinguishable,
- coverage metrics are correct,
- provider-specific fields do not leak into book feature logic.

### Book profile

- concept extraction is deterministic,
- prose features are calculated only from valid prose,
- TOC-only books do not receive fake syntactic scores,
- multiple documents aggregate deterministically,
- missing features remain missing.

### Reader profile

- question types map correctly,
- weighting is deterministic,
- scores stay within `[0, 1]`,
- empty or invalid assessments fail clearly.

### Ranking

- score decomposition matches the configured formula,
- missing components trigger weight renormalization,
- missing values are not treated as zero,
- ranking order is deterministic,
- explanations match the calculated gaps.

### Evaluation

- Spearman calculation is correct,
- pairwise agreement is correct,
- ablation configurations are reproducible.

The default test suite must not require:

- network access,
- Data-Pipeline live collection,
- a database,
- an external LLM,
- Spring Boot.

---

## 23. Logging and Error Handling

Distinguish:

- invalid canonical input,
- missing optional evidence,
- unsupported document type,
- feature-unavailable conditions,
- parsing/tokenization failure,
- invalid configuration,
- model/prompt failure,
- evaluation-data insufficiency.

A missing preface is not an exception.

A broken `book_id` reference is an exception.

A book with no prose may still have a valid Concept Profile.

A book with no usable evidence beyond metadata should be represented explicitly rather than assigned fabricated features.

---

## 24. Data and Copyright Boundaries

Do not commit large third-party book text.

Do not copy raw public documents into the repository merely for convenience.

Use the Data-Pipeline output or local ignored data directories.

Small test fixtures should be:

- synthetic,
- heavily minimized,
- or clearly safe to redistribute.

The ML repository is not a second crawler.

If more source data is needed, change `Data-Pipeline`, not this repository.

---

## 25. First Milestone: Canonical Handoff

### Goal

Prove that the ML repository can consume the current ten-book canonical dataset without source-specific logic.

### Required work

```text
Canonical JSONL
    ↓
loader
    ↓
BookEvidence
    ↓
coverage report
```

### Definition of Done

- 10 books load successfully.
- `book_id` joins are validated.
- TOC hierarchy is preserved.
- documents are attached correctly.
- missing evidence is represented explicitly.
- no source-specific branches exist in downstream feature code.
- CLI command reproduces the result.
- tests pass.

Do not implement recommendation before this layer is trustworthy.

---

## 26. Second Milestone: Book Profile Baseline

### Goal

Generate useful book profiles for the ten-book dataset.

### Required outputs

For every book where evidence permits:

```text
Concept Profile
```

For books with sufficient prose:

```text
Text Difficulty Profile
```

### Required experiment

For at least one rich-evidence book, compare:

```text
TOC only
TOC + Description
All available evidence
```

Record how the generated profile changes.

### Definition of Done

- concept profile generation works,
- difficulty profile does not fabricate missing prose features,
- evidence coverage accompanies every profile,
- version information is stored,
- output JSONL is deterministic,
- tests pass.

---

## 27. Third Milestone: Reader Profile Baseline

### Goal

Convert assessment results into a topic-specific profile.

### Initial dimensions

```text
vocabulary
background_knowledge
comprehension
```

### Definition of Done

- structured assessment JSON can be loaded,
- per-type scores are produced in `[0, 1]`,
- difficulty handling is config-driven,
- result contains `profile_version`,
- invalid assessments fail clearly,
- tests pass.

Synthetic fixtures may be used before real user data exists.

---

## 28. Fourth Milestone: Matching and Ranking

### Goal

Produce explainable recommendations from Reader Profile + Book Profile.

### Initial cases

Support both:

```text
topic-based top-K recommendation
specific-book fit score
```

### Definition of Done

- deterministic baseline score,
- component scores,
- missing-component handling,
- recommendation reasons,
- model/config version,
- reproducible CLI example,
- tests pass.

Use a small number of books first.

---

## 29. Evaluation Milestone

After the baseline works:

1. collect a small human relative-difficulty ranking,
2. calculate Spearman and pairwise agreement,
3. compare Topic-only vs Readiness-aware ranking,
4. run evidence ablation,
5. inspect failure cases.

Only after these results should the team decide whether to add:

- Linear Regression,
- Random Forest,
- XGBoost / LightGBM,
- sentence embeddings,
- LLM-based concept extraction,
- adaptive assessment,
- learned ranking weights.

---

## 30. Non-Goals for the Initial ML Repository

Do not implement yet:

- book crawling,
- OCR,
- camera processing,
- book-spine detection,
- Spring persistence,
- authentication,
- frontend code,
- collaborative filtering,
- real-time feature extraction,
- online learning,
- neural ranking,
- RAG infrastructure,
- vector databases,
- full psychometric assessment,
- end-to-end LLM recommendation.

These can be reconsidered only after the baseline vertical slice is evaluated.

---

## 31. Development Workflow for the Agent

Before implementation:

1. Read this `AGENTS.md` completely.
2. Inspect the repository.
3. Inspect the current `Data-Pipeline` canonical schema or fixtures.
4. Produce a concise implementation plan.
5. Identify assumptions.
6. Implement the smallest end-to-end slice.
7. Run tests and linting.
8. Inspect actual generated output.
9. Update README when commands or contracts change.

After each meaningful step:

```text
run tests
run ruff
inspect output
verify determinism
update docs
```

Do not claim a feature works without running it.

---

## 32. Preferred Implementation Order

Use this order:

```text
project setup
    ↓
schemas
    ↓
canonical loader
    ↓
evidence assembler
    ↓
coverage inspection
    ↓
concept profile baseline
    ↓
difficulty subfeatures
    ↓
book profile aggregation
    ↓
reader profile baseline
    ↓
matching / ranking
    ↓
explanations
    ↓
evaluation
    ↓
Spring integration
```

Do not start with FastAPI or model training.

---

## 33. Engineering Philosophy

Prefer:

- explicit schemas,
- pure functions,
- deterministic outputs,
- provider-independent inputs,
- transparent feature formulas,
- meaningful missing values,
- versioned configuration,
- small reproducible experiments,
- evidence-aware explanations.

Avoid:

- hidden magic,
- arbitrary hard-coded weights,
- opaque single-number scores,
- pretending missing data exists,
- premature model training,
- source-specific logic,
- notebook-only workflows,
- unversioned prompts,
- untestable remote dependencies.

For this capstone, a transparent baseline that explains why a book fits a reader is more valuable than a sophisticated model whose behavior cannot be justified.

---

## 34. Current Success Criterion

The immediate goal is not:

> Build the final recommendation model.

It is:

> Demonstrate, with the current canonical ten-book dataset, that public book evidence can be transformed into reproducible book profiles and matched against topic-specific reader profiles to produce explainable recommendation results.

Once that works, evaluate it before expanding the model or the data scale.
