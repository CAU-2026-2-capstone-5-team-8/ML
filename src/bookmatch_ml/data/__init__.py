"""Canonical input loading and evidence assembly."""

from bookmatch_ml.data.evidence import assemble_book_evidence
from bookmatch_ml.data.loader import CanonicalDataError, load_canonical_dataset

__all__ = ["CanonicalDataError", "assemble_book_evidence", "load_canonical_dataset"]
