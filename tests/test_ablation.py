from pathlib import Path

from bookmatch_ml.config import load_feature_config
from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import load_canonical_dataset
from bookmatch_ml.evaluation.ablation import build_evidence_ablation_report

ROOT = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "canonical"


def test_ablation_uses_three_reproducible_evidence_modes() -> None:
    evidence = assemble_book_evidence(load_canonical_dataset(FIXTURE_DIR))[0]
    config = load_feature_config(ROOT / "configs" / "features.yaml")

    report = build_evidence_ablation_report(evidence, config)

    assert [variant.evidence_mode for variant in report.variants] == [
        "toc_only",
        "toc_description",
        "all_available",
    ]
    assert report.variants[0].syntactic_complexity is None
    assert report.variants[1].syntactic_complexity is None
    assert report.variants[2].syntactic_complexity is not None
    assert report.variants[0].evidence_coverage.prose_document_count == 0
    assert report.variants[2].evidence_coverage.prose_document_count == 1
    toc_weights = {item.concept: item.weight for item in report.variants[0].covered_concepts}
    all_weights = {item.concept: item.weight for item in report.variants[2].covered_concepts}
    assert all_weights["process"] > toc_weights["process"]
