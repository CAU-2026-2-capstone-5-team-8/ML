# Human concept graph reviews

Generated evidence and blank review templates belong under ignored `data/reports/` and
`data/reviews/`. After a person reviews every intended edge, copy the completed file here as
`concept_graph_review.json` in a dedicated review change. Record a reviewer ID, timezone-aware
review time, and rationale for every `accepted`, `rejected`, or `uncertain` decision.

The review file is separate from `configs/concept_graph.yaml`. Current matching code does not
load it, so review status cannot silently change v1 or v2 scores. Promoting an accepted edge from
`proposed_seed` to `human_reviewed_seed` requires a later explicit graph configuration change,
version change, and review.

The committed [example template](../examples/concept_graph_review.json) contains only
`unreviewed` decisions and shows the strict schema.
