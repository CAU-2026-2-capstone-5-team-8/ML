# Concept assessment and Question Difficulty v1

## Scope and flow

The current deterministic book profiles supply a topic concept pool. The pool keeps **covered**
concepts separate from **inferred prerequisites**. Stage 1 asks a reader whether they know selected
concepts. Stage 2 can later verify important or uncertain concepts with authored or generated
questions. This milestone emits **question specifications** for Stage 2, not question text, answer
keys, reader mastery, or a new ranking signal.

```text
Canonical evidence → Book Profiles → Topic Concept Pool → Stage 1 self-report targets
                                                 ↘ Stage 2 QuestionSpecs
Self-report + future quiz evidence → future Concept Mastery + Confidence → future matching
```

`ConceptSelfAssessment` records one strict boolean `knows_concept` per canonical concept and
checks duplicate and mixed-topic responses. It can be validated against the topic pool. The
`ConceptMastery` schema reserves separate self-report, quiz, mastery, confidence, and evidence
fields, but deliberately has no fusion formula. The existing reader-profile and `/ml/rank` paths
are unchanged.

## Three meanings of difficulty

| Name | Meaning in this repository | Status |
| --- | --- | --- |
| Book Difficulty | Reading demands measured from eligible book prose | Existing Book Profile; unchanged |
| Concept Difficulty | An intrinsic ease or hardness of a concept | **Not assigned** |
| Question Difficulty | Intended cognitive work in a future question | Level 1, 2, or 3 target in QuestionSpec |

A concept's book occurrence is evidence about coverage and assessment priority. It is not a
measure of how difficult that concept is. A one-book concept may be a valid assessment target
and can receive a Level 1 question target.

## Pool and selection rules

The pool merges the canonical concept names from `configs/features.yaml` by `(topic, name, role)`.
Each pool entry retains contributing book IDs, per-book concept weights, book coverage count and
rate, evidence types and IDs with mention counts, and prerequisite inference methods. The input
Book Profiles must match all canonical book IDs, evidence references, feature config hash, and a
fresh deterministic build from the canonical files. No upstream provider fields enter selection.

The configured `assessment_priority` baseline is:

```text
0.7 × mean per-book concept weight + 0.3 × topic book coverage rate
```

Weights apply within each role. Stable tie-breaking uses the canonical concept name. The
configured Stage 1 cap is eight covered and four prerequisite concepts per topic. This is a
selection heuristic, not concept difficulty, learner mastery, or pedagogical validation. Book
frequency has a limited role: it contributes 30% of priority and does not determine question
difficulty. Evidence-type diversity remains visible for audit but is not yet a selection term.

## Question Difficulty v1

`QuestionType` remains the existing assessment dimension: `vocabulary`,
`background_knowledge`, or `comprehension`. `cognitive_operation` is separate and supplies the
primary target-difficulty rule. The mapping, quotas, structural conditions, relation candidates,
and versions are in `configs/assessment.yaml`.

| Level | Configured operations | Intended work |
| --- | --- | --- |
| 1 | recognize, recall | Direct term recognition or prerequisite recall; one primary concept, no relation or multiple steps |
| 2 | compare, relate, apply | Direct comparison, one relation step, or simple application |
| 3 | integrate, infer | Multiple reasoning steps or inference from authored context |

The current blueprint emits `recognize` and `compare` vocabulary targets, `recall` background
targets, and `apply` and `integrate` comprehension targets. `relate` and `infer` are defined for
future specifications but are not emitted by the v1 quotas. A direct comparison of **two**
concepts stays Level 2. An `integrate` target is Level 3 because the intended question requires
multiple reasoning steps, not because it contains two concepts. A future author must confirm
that the actual question really performs the specified operation.

Each specification retains primary and related concept counts, relation and multiple-step flags,
question type, operation, explicit rationale, evidence references, supporting books, source
document IDs, config version/hash, and a deterministic question ID. `prerequisite_depth` and
`abstraction_level` remain `null`: the current baseline has no validated prerequisite graph or
abstraction model. Per-document syntactic complexity is retained as an optional **diagnostic**
for prose-grounded specs and never changes the Level 1/2/3 assignment.

Vocabulary uses covered concepts; background knowledge uses inferred prerequisite concepts and
states the inference method. Relation pairs in configuration are **candidates for author
verification**, not proven pedagogical relations. `compare` requires both concepts to appear in
one book. `integrate` requires both concepts in the **same analyzed eligible prose document**.
`apply` also requires an analyzed eligible prose document mentioning its concept. Eligible types
are preface, introduction, preview, sample chapter, and other actual prose. TOCs, descriptions,
publisher summaries, and indexes cannot ground comprehension. Unfilled quotas are reported as
shortages rather than replaced with invented evidence. Specifications store references, never
raw book prose.

`target_difficulty` is the intended operation level. Future `empirical_difficulty` may be
estimated from real responses, timing, error patterns, and stability. It is not present or
inferred in this artifact, and target difficulty must not be interpreted as observed difficulty.

## Reproduce the fixed ten-book snapshot

```bash
uv run bookmatch-ml build-book-profiles \
  --data-dir ../Data-Pipeline/data/processed \
  --output data/output/book_profiles.jsonl

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

Both JSON artifacts are audit-friendly: `concept_pool` shows every candidate and its role,
`selected_assessment_concepts` shows the Stage 1 selection, and every `question_specs` entry
shows the target operation, level, rationale, related concepts, evidence IDs, books, and any
eligible source document. Generated artifacts under `data/` stay ignored by Git. Input file
SHA-256 hashes and exact configuration hashes are included for reproducibility. Both topic
artifacts were regenerated from the same inputs and compared byte-for-byte; each comparison
matched.

## Results from the current canonical ten books

The snapshot has five books per topic. Four books per topic have analyzed eligible prose (nine
analyzed documents for Operating Systems; ten for Linear Algebra).

| Measure | Operating Systems | Linear Algebra |
| --- | ---: | ---: |
| Pool: covered / prerequisite / total | 15 / 5 / 20 | 18 / 2 / 20 |
| Stage 1 selected: covered / prerequisite | 8 / 4 | 8 / 2 |
| QuestionSpecs | 14 | 13 |
| Level 1 / 2 / 3 | 7 / 5 / 2 | 6 / 5 / 2 |
| Vocabulary / background / comprehension | 6 / 3 / 5 | 6 / 2 / 5 |
| Prose-grounded comprehension specs | 5 | 5 |
| Shortages | 0 | 1 background recall target |

Operating Systems selected covered concepts: process, thread, scheduling, synchronization,
concurrency, virtual memory, file system, security. Its selected inferred prerequisites are
programming, computer architecture, algorithms, and assembly language.

Linear Algebra selected covered concepts: matrix, vector, linear system, orthogonality,
dimension, determinant, gaussian elimination, basis. Its inferred prerequisites are high school
algebra and systems of equations. Only two inferred prerequisites exist in this pool, so the
configured three-target background quota produces two specs and one explicit shortage.

The emitted comparison specs are Level 2 despite using two concepts. The two integration specs
per topic are Level 3 because their target operation requires multiple steps; source documents
show co-occurrence only. Each topic's five comprehension specs refer to two analyzed documents,
so an author should review whether those excerpts support sufficiently varied situations.

## Suspicious cases and limitations

- Operating Systems `programming`, `computer architecture`, and `algorithms` each appear as an
  inferred prerequisite in only one book. `programming` has an explicit cue plus an early-prose
  proxy; `computer architecture` and `algorithms` have only the weaker early-prose proxy. They
  receive Level 1 recall targets, **not** a hard-concept label. An author should confirm their
  prerequisite role before quiz authoring.
- Linear Algebra `high school algebra` appears in one book but has high source weight and ranks
  first among prerequisites. `systems of equations` is supported only by an early-prose proxy.
  The 70% mean-weight term can elevate a strong single-book signal; the pool exposes that fact.
- Several common covered concepts are omitted by the eight-concept cap, including Operating
  Systems `memory management` and Linear Algebra `eigenvalue`/`eigenvector`. Because pair
  candidates require selected concepts, those relation targets are omitted too. This suggests
  reviewing coverage and diversity before choosing a later question-generation strategy.
- Configured relation pairs and same-document co-occurrence do not prove a useful comparison or
  a genuine multi-step task. Level 3 is an authoring target, not a measured cognitive demand.
- Current concept names come from a fixed lexicon; the pool cannot discover out-of-lexicon
  concepts. Priority weights, caps, quotas, and relation pairs are transparent engineering
  defaults, not educationally validated values. No learner responses, correctness, response
  times, or empirical question difficulty exist yet.

The next decision gate is a human review of these QuestionSpec targets and their evidence:
**Does Question Difficulty v1 and the distribution look reasonable enough to proceed to
structured natural-language question generation?** That generation is outside this milestone.
