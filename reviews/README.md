# Human concept graph reviews

The version-controlled source of truth is
[`configs/concept_graph_reviews.yaml`](../configs/concept_graph_reviews.yaml). Reviewers update
`review_status` and `review_note` there in a dedicated change. Allowed statuses are `unreviewed`,
`accepted`, `rejected`, and `needs_revision`; a decided edge requires a non-blank note.

Generated evidence and review snapshots belong under ignored `data/reports/` and
`data/reviews/`. Do not record decisions by editing those files. The CLI reads the config and
copies each decision beside its automatic evidence.

The human review config is separate from `configs/concept_graph.yaml` and is never loaded by
matching or ranking. Promoting an accepted edge from `proposed_seed` to `human_reviewed_seed`
requires a later explicit graph configuration and version change.
