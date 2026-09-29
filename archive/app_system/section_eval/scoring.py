"""
Weighted scoring system for section and overall paper assessment.
"""

from typing import Any, Dict, List, Optional


# Section importance multipliers by paper type.
# Scores for each section are multiplied by these weights when computing the overall score.
SECTION_IMPORTANCE: Dict[str, Dict[str, float]] = {
    "empirical": {
        "data":              1.2,
        "methodology":       1.3,
        "results":           1.2,
        "robustness_checks": 1.1,
        "identification_strategy": 1.2,
        "introduction":      1.0,
        "literature_review": 0.9,
        "discussion":        0.9,
        "conclusion":        0.8,
        "abstract":          0.7,
    },
    "theoretical": {
        "model_setup":       1.3,
        "proofs":            1.4,
        "extensions":        1.1,
        "introduction":      1.0,
        "discussion":        0.9,
        "conclusion":        0.9,
        "abstract":          0.7,
    },
    "policy": {
        "policy_context":    1.2,
        "recommendations":   1.3,
        "introduction":      1.1,
        "background":        1.0,
        "discussion":        1.0,
        "conclusion":        1.1,
        "abstract":          0.7,
    },
    "finance": {
        "identification_strategy": 1.3,
        "robustness_checks":       1.2,
        "results":                 1.2,
        "data":                    1.1,
        "methodology":             1.1,
        "introduction":            1.0,
        "literature_review":       0.9,
        "conclusion":              0.8,
        "abstract":                0.7,
    },
    "macro": {
        "calibration":       1.3,
        "simulations":       1.2,
        "model_setup":       1.2,
        "stylized_facts":    1.1,
        "introduction":      1.0,
        "conclusion":        0.9,
        "abstract":          0.7,
    },
    "systematic_review": {
        "search_methodology":  1.2,
        "inclusion_criteria":  1.2,
        "synthesis":           1.3,
        "introduction":        1.0,
        "discussion":          1.0,
        "conclusion":          0.9,
        "abstract":            0.7,
    },
}

_READINESS_THRESHOLDS = [
    (4.5, 3.5, "Ready for submission"),
    (4.0, 3.0, "Minor revisions needed"),
    (3.0, 2.5, "Major revisions needed"),
    (0.0, 0.0, "Substantial work required"),
]


def _infer_section_type(section_name: str) -> str:
    """Map a section header to a canonical type key for importance lookup."""
    from .criteria.base import _canonical_section_type
    return _canonical_section_type(section_name)


def compute_section_score(
    criteria_evaluations: List[Dict[str, Any]],
    section_name: str,
    paper_type: str,
) -> Dict[str, Any]:
    """
    Compute weighted section score from criteria evaluation results.

    Fatal-flaw rule: if any criterion marked critical=True scores ≤ FATAL_FLAW_SCORE_THRESHOLD,
    the raw_score is capped at FATAL_FLAW_SCORE_CAP regardless of the weighted average.
    This prevents high scores on other criteria from masking a fundamental flaw.

    Args:
        criteria_evaluations: list of dicts with keys: criterion, score, weight, is_fatal_criterion
        section_name: human-readable section name
        paper_type: e.g. "empirical"

    Returns dict with:
        raw_score: weighted average of criteria scores (1-5 scale), possibly capped
        adjusted_score: raw_score * section importance multiplier
        importance_multiplier: float
        criteria_breakdown: {criterion: score}
        weight_breakdown: {criterion: weight}
        fatal_flaw_triggered: bool
        fatal_flaw_criteria: list of criterion names that triggered the cap
    """
    from .criteria.base import FATAL_FLAW_SCORE_THRESHOLD, FATAL_FLAW_SCORE_CAP

    if not criteria_evaluations:
        return {
            "raw_score": 3.0,
            "adjusted_score": 3.0,
            "importance_multiplier": 1.0,
            "criteria_breakdown": {},
            "weight_breakdown": {},
            "fatal_flaw_triggered": False,
            "fatal_flaw_criteria": [],
        }

    total_weight = sum(c.get("weight", 1.0) for c in criteria_evaluations)
    weighted_sum = sum(c.get("score", 3) * c.get("weight", 1.0) for c in criteria_evaluations)
    raw_score = weighted_sum / total_weight if total_weight > 0 else 3.0

    # Fatal-flaw check: any critical criterion at or below the threshold caps the score
    fatal_flaw_criteria = [
        c["criterion"]
        for c in criteria_evaluations
        if c.get("is_fatal_criterion", False) and c.get("score", 3) <= FATAL_FLAW_SCORE_THRESHOLD
    ]
    fatal_flaw_triggered = len(fatal_flaw_criteria) > 0
    if fatal_flaw_triggered:
        raw_score = min(raw_score, FATAL_FLAW_SCORE_CAP)

    section_type = _infer_section_type(section_name)
    importance_map = SECTION_IMPORTANCE.get(paper_type, {})
    importance = importance_map.get(section_type, 1.0)
    adjusted_score = raw_score * importance

    return {
        "raw_score": round(raw_score, 2),
        "adjusted_score": round(adjusted_score, 2),
        "importance_multiplier": importance,
        "criteria_breakdown": {c["criterion"]: c.get("score", 3) for c in criteria_evaluations},
        "weight_breakdown": {c["criterion"]: c.get("weight", 0.25) for c in criteria_evaluations},
        "fatal_flaw_triggered": fatal_flaw_triggered,
        "fatal_flaw_criteria": fatal_flaw_criteria,
    }


def compute_overall_score(
    section_scores: Dict[str, Dict[str, Any]],
    paper_type: str,
) -> Dict[str, Any]:
    """
    Aggregate section scores into an overall paper score.

    Args:
        section_scores: {section_name: result_of_compute_section_score}
        paper_type: e.g. "empirical"

    Returns dict with:
        overall_score: float (1–5 scale)
        publication_readiness: str
        section_scores: passed through
    """
    if not section_scores:
        return {
            "overall_score": 3.0,
            "publication_readiness": "Insufficient data",
            "section_scores": section_scores,
        }

    importance_map = SECTION_IMPORTANCE.get(paper_type, {})
    weighted_sum = 0.0
    total_weight = 0.0

    for section_name, scores in section_scores.items():
        section_type = _infer_section_type(section_name)
        weight = importance_map.get(section_type, 1.0)
        weighted_sum += scores.get("raw_score", 3.0) * weight
        total_weight += weight

    overall = weighted_sum / total_weight if total_weight > 0 else 3.0
    min_score = min(s.get("raw_score", 3.0) for s in section_scores.values())

    readiness = "Substantial work required"
    for overall_thresh, min_thresh, label in _READINESS_THRESHOLDS:
        if overall >= overall_thresh and min_score >= min_thresh:
            readiness = label
            break

    return {
        "overall_score": round(overall, 2),
        "publication_readiness": readiness,
        "section_scores": section_scores,
    }


def score_bar_html(score: float, max_score: float = 5.0) -> str:
    """Return a simple percentage string for Streamlit progress bars."""
    return min(score / max_score, 1.0)
