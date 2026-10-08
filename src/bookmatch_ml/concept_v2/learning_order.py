"""Optional v3 ordering of observed learning readiness, not absolute book difficulty."""

from collections import Counter
from fractions import Fraction

POLICY = "observed-readiness-lexicographic-v1"
CRITERIA = [
    "established-prerequisite-graph-first",
    "foundation-status",
    "observed-prerequisite-error-rate-ascending",
    "prerequisite-assessment-coverage-descending",
    "observed-practice-concept-coverage-descending",
    "observed-practice-density-descending",
]


def _ratio(numerator, denominator):
    return None if denominator == 0 else Fraction(numerator, denominator)


def personalize(plans, observations, ability):
    """Return a sorted copy; stable IDs order only genuinely tied recommendations."""
    observed = {c: o for (c, a), o in observations.items() if a == ability}
    practice = {c for c, o in observed.items() if o.correct_count < o.response_count}
    keyed = []
    priority = {"ready-to-explore": 0, "check-first": 1, "foundation-gap": 2}
    for plan in plans:
        covered = set(plan["coveredConcepts"])
        prerequisite = set(plan["inferredPrerequisites"])
        assessed = prerequisite & observed.keys()
        errors = [
            Fraction(
                observed[c].response_count - observed[c].correct_count, observed[c].response_count
            )
            for c in sorted(assessed)
        ]
        error_rate = sum(errors, Fraction()) / len(errors) if errors else None
        coverage = _ratio(len(assessed), len(prerequisite))
        relevant_practice = covered & practice
        practice_coverage = _ratio(len(relevant_practice), len(practice))
        practice_density = _ratio(len(relevant_practice), len(covered))
        correct_targets = {
            c
            for c in covered & observed.keys()
            if observed[c].correct_count == observed[c].response_count
        }
        metrics = {
            "prerequisiteCount": len(prerequisite),
            "prerequisiteObservedCount": len(assessed),
            "prerequisiteUnknownCount": len(prerequisite - assessed),
            "prerequisitePracticeCount": len(prerequisite & practice),
            "prerequisiteResponseCount": sum(observed[c].response_count for c in assessed),
            "prerequisiteErrorRate": None if error_rate is None else float(error_rate),
            "prerequisiteCoverage": None if coverage is None else float(coverage),
            "targetCount": len(covered),
            "targetObservedCount": len(covered & observed.keys()),
            "targetUnknownCount": len(covered - observed.keys()),
            "targetPracticeCount": len(relevant_practice),
            "observedCorrectTargetCount": len(correct_targets),
            "observedCorrectTargetShare": float(_ratio(len(correct_targets), len(covered))),
            "readerPracticeConceptCount": len(practice),
            "practiceCoverage": None if practice_coverage is None else float(practice_coverage),
            "practiceDensity": float(practice_density),
            "goalSource": "none-supplied-observed-practice-only",
        }
        key = (
            not prerequisite,
            priority[plan["status"]],
            Fraction(1) if error_rate is None else error_rate,
            -(coverage or Fraction()),
            -(practice_coverage or Fraction()),
            -(practice_density or Fraction()),
        )
        item = {**plan, "personalization": metrics}
        item["reasons"] = [
            *plan["reasons"],
            f"선수개념 {len(prerequisite)}개 중 {len(assessed)}개를 진단했고, "
            f"오답을 확인한 개념 {len(prerequisite & practice)}개가 있어요.",
            f"선택한 능력에서 오답을 확인한 개념 {len(practice)}개 중 "
            f"{len(relevant_practice)}개가 이 책에서 다루는 개념과 연결돼요.",
        ]
        keyed.append((key, item))
    keyed.sort(key=lambda pair: (pair[0], pair[1]["bookId"]))
    counts = Counter(key for key, _ in keyed)
    group = 0
    previous = None
    for key, item in keyed:
        if key != previous:
            group += 1
        item.update(rankGroup=group, tieCount=counts[key])
        previous = key
    return [item for _, item in keyed]
