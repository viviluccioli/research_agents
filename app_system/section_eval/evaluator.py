"""
Core evaluation orchestration: paper-type-aware, multi-pass evaluation with quote extraction.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .utils import parse_json_from_text, hash_text, safe_query, extract_short_phrases
from .criteria.base import get_criteria, PAPER_TYPE_LABELS as _pt_labels
from .prompts.templates import build_evaluation_prompt
from .scoring import compute_section_score


# ---------------------------------------------------------------------------
# Quote validation
# ---------------------------------------------------------------------------

def _validate_quote(quote_text: str, section_text: str) -> bool:
    """
    Check if a quote appears in the section text with some tolerance
    for minor OCR/formatting differences.
    Strips whitespace and lowercases both before comparison.
    """
    if not quote_text or not section_text:
        return False
    # Normalize: collapse whitespace
    norm_quote = re.sub(r'\s+', ' ', quote_text.strip().lower())
    norm_section = re.sub(r'\s+', ' ', section_text.lower())
    # Exact match
    if norm_quote in norm_section:
        return True
    # Partial match: first 40 chars of quote
    partial = norm_quote[:40]
    return len(partial) > 10 and partial in norm_section


def _mark_quotes(criteria_evaluations: List[Dict], section_text: str) -> List[Dict]:
    """
    Validate quote_1 and quote_2 for each criterion evaluation in-place.
    Adds 'quote_1_valid' and 'quote_2_valid' boolean flags.
    """
    for ev in criteria_evaluations:
        q1 = ev.get("quote_1", {})
        q2 = ev.get("quote_2", {})
        ev["quote_1"]["valid"] = _validate_quote(q1.get("text", ""), section_text)
        ev["quote_2"]["valid"] = _validate_quote(q2.get("text", ""), section_text)
    return criteria_evaluations


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class SectionEvaluator:
    """
    Paper-type-aware section evaluator.

    Evaluation flow per section:
      1. Build paper-type + section-specific criteria
      2. Single LLM pass: qualitative + criterion scores + quotes
      3. Validate quotes against source text
      4. Compute weighted section score
      5. Cache result by content hash
    """

    CACHE_PREFIX = "se_cache_v3"

    def __init__(self, llm, cache_store: Optional[Dict] = None):
        """
        llm: ConversationManager instance (from eval/utils.py)
        cache_store: dict-like to persist results (e.g. st.session_state sub-dict)
        """
        self.llm = llm
        self._cache: Dict[str, Any] = cache_store if cache_store is not None else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_section(
        self,
        section_name: str,
        section_text: str,
        paper_type: str,
        paper_context: str = "",
        figures_external: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate a single section.

        Returns dict with keys:
            qualitative_assessment: str
            criteria_evaluations: list of dicts (criterion, score, weight, justification, quote_1, quote_2)
            improvements: list of dicts (priority, suggestion, rationale)
            section_score: dict from compute_section_score()
            paper_type: str
            section_name: str
        """
        cache_key = hash_text(f"{paper_type}|{section_name}|{section_text[:50000]}|{figures_external}")
        if cache_key in self._cache:
            return self._cache[cache_key]

        criteria = get_criteria(paper_type, section_name)

        # When figures are external, zero out the 'presentation' criterion weight
        # and redistribute its weight proportionally to remaining criteria.
        if figures_external:
            criteria = self._adjust_criteria_for_external_figures(criteria)

        # Determine canonical section type for the prompt
        from .criteria.base import _canonical_section_type
        section_type = _canonical_section_type(section_name)

        prompt = build_evaluation_prompt(
            paper_type=paper_type,
            paper_context=paper_context,
            section_name=section_name,
            section_type=section_type,
            section_text=section_text,
            criteria=criteria,
            figures_external=figures_external,
        )

        # Record model version and wall-clock timestamp before calling LLM
        try:
            from utils import model_selection as _model_ver
        except ImportError:
            _model_ver = "unknown"
        eval_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        raw_response = safe_query(self.llm, prompt, max_chars=16000)
        parsed = parse_json_from_text(raw_response)

        if not isinstance(parsed, dict):
            # Fallback: build a minimal result
            result = self._fallback_result(
                section_name, section_text, paper_type, criteria, raw_response,
                eval_timestamp=eval_timestamp, model_version=_model_ver,
            )
            self._cache[cache_key] = result
            return result

        # Extract and validate fields
        qualitative = str(parsed.get("qualitative_assessment", "")).strip()
        criteria_evals = parsed.get("criteria_evaluations", [])
        improvements = parsed.get("improvements", [])

        # Ensure criteria_evals is a list of dicts with required fields
        criteria_evals = self._sanitize_criteria_evals(criteria_evals, criteria)

        # Validate quotes against source text
        criteria_evals = _mark_quotes(criteria_evals, section_text)

        # Improvements: ensure list of dicts
        improvements = self._sanitize_improvements(improvements)

        # Compute weighted score
        section_score = compute_section_score(criteria_evals, section_name, paper_type)

        result = {
            "qualitative_assessment": qualitative,
            "criteria_evaluations": criteria_evals,
            "improvements": improvements,
            "section_score": section_score,
            "paper_type": paper_type,
            "section_name": section_name,
            "eval_timestamp": eval_timestamp,
            "model_version": _model_ver,
        }

        self._cache[cache_key] = result
        return result

    def generate_context_summary(self, paper_text: str) -> str:
        """
        Summarize the full paper to extract its key message and purpose.
        Used as paper_context when evaluating individual sections.
        """
        prompt = f"""Read the following economics paper and provide a 3-5 sentence summary covering:
- The research question
- The main methodology (e.g. empirical/theoretical/policy)
- The key finding or contribution

Paper text (first 12000 characters):
{paper_text[:12000]}

Provide the summary in plain text, 3-5 sentences."""

        return safe_query(self.llm, prompt, max_chars=14000).strip()

    def generate_overall_assessment(
        self,
        section_results: Dict[str, Dict[str, Any]],
        paper_type: str,
    ) -> str:
        """
        Generate an overall assessment from per-section scores.
        """
        from .scoring import compute_overall_score
        overall = compute_overall_score(
            {name: res["section_score"] for name, res in section_results.items()},
            paper_type,
        )

        lines = []
        for name, res in section_results.items():
            sc = res.get("section_score", {})
            lines.append(
                f"{name}: raw={sc.get('raw_score', '?')}, "
                f"adjusted={sc.get('adjusted_score', '?')}"
            )

        prompt = f"""You are an experienced editor for top economics journals.
Paper type: {_pt_labels.get(paper_type, paper_type)}
Overall score (1–5): {overall['overall_score']}
Publication readiness: {overall['publication_readiness']}

Per-section scores:
{chr(10).join(lines)}

Produce EXACTLY this output format:

## Key Strengths
1.
2.
3.

## Key Weaknesses
1.
2.
3.

## Priority Improvements
1.
2.
3.

## Publication Readiness
{overall['publication_readiness']} — [one sentence justification]."""

        return safe_query(self.llm, prompt, max_chars=4000)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjust_criteria_for_external_figures(self, criteria: List[dict]) -> List[dict]:
        """
        When figures/tables are in an appendix, zero out the 'presentation' criterion
        and redistribute its weight proportionally to the remaining criteria.
        """
        presentation_weight = sum(c["weight"] for c in criteria if c["name"] == "presentation")
        if presentation_weight == 0:
            return criteria  # no presentation criterion, nothing to adjust

        remaining = [c for c in criteria if c["name"] != "presentation"]
        if not remaining:
            return criteria

        total_remaining = sum(c["weight"] for c in remaining)
        adjusted = []
        for c in criteria:
            if c["name"] == "presentation":
                continue  # drop it
            new_weight = c["weight"] + (c["weight"] / total_remaining) * presentation_weight
            adjusted.append({**c, "weight": round(new_weight, 4)})
        return adjusted

    def _sanitize_criteria_evals(
        self, evals: Any, criteria: List[dict]
    ) -> List[Dict[str, Any]]:
        """
        Ensure criteria_evals is a clean list of dicts with all required fields.
        Falls back to default values if LLM output is malformed.
        """
        result = []
        criteria_by_name = {c["name"]: c for c in criteria}

        if isinstance(evals, list):
            for item in evals:
                if not isinstance(item, dict):
                    continue
                criterion = str(item.get("criterion", ""))
                meta = criteria_by_name.get(criterion, {})
                score = item.get("score", 3)
                try:
                    score = max(1, min(5, int(score)))
                except (ValueError, TypeError):
                    score = 3

                q1 = item.get("quote_1", {})
                q2 = item.get("quote_2", {})
                if isinstance(q1, str):
                    q1 = {"text": q1, "supports_assessment": True}
                if isinstance(q2, str):
                    q2 = {"text": q2, "supports_assessment": True}
                if not isinstance(q1, dict):
                    q1 = {"text": "", "supports_assessment": True}
                if not isinstance(q2, dict):
                    q2 = {"text": "", "supports_assessment": True}

                result.append({
                    "criterion": criterion,
                    "score": score,
                    "weight": float(meta.get("weight", item.get("weight", 0.25))),
                    "description": meta.get("description", ""),
                    "justification": str(item.get("justification", "")),
                    "quote_1": {"text": str(q1.get("text", "")), "supports_assessment": q1.get("supports_assessment", True)},
                    "quote_2": {"text": str(q2.get("text", "")), "supports_assessment": q2.get("supports_assessment", True)},
                    "is_fatal_criterion": bool(meta.get("critical", False)),
                })

        # If we got fewer criteria than expected, fill gaps with defaults
        evaluated_names = {r["criterion"] for r in result}
        for c in criteria:
            if c["name"] not in evaluated_names:
                result.append({
                    "criterion": c["name"],
                    "score": 3,
                    "weight": float(c["weight"]),
                    "description": c["description"],
                    "justification": "(Not evaluated)",
                    "quote_1": {"text": "", "supports_assessment": True},
                    "quote_2": {"text": "", "supports_assessment": True},
                    "is_fatal_criterion": bool(c.get("critical", False)),
                })

        return result

    def _sanitize_improvements(self, improvements: Any) -> List[Dict[str, Any]]:
        result = []
        if isinstance(improvements, list):
            for i, item in enumerate(improvements[:4]):
                if isinstance(item, str):
                    result.append({"priority": i + 1, "suggestion": item, "rationale": ""})
                elif isinstance(item, dict):
                    result.append({
                        "priority": item.get("priority", i + 1),
                        "suggestion": str(item.get("suggestion", "")),
                        "rationale": str(item.get("rationale", "")),
                    })
        return result

    def _fallback_result(
        self,
        section_name: str,
        section_text: str,
        paper_type: str,
        criteria: List[dict],
        raw_response: str,
        eval_timestamp: str = "",
        model_version: str = "",
    ) -> Dict[str, Any]:
        """Build a minimal result when LLM output cannot be parsed."""
        qualitative = raw_response[:500] if raw_response else "Evaluation could not be parsed."

        # Use heuristic phrase extraction for improvements
        improvements_text = extract_short_phrases(qualitative, ("suggest", "fix", "improv", "recommend"))
        improvements = [{"priority": i+1, "suggestion": s, "rationale": ""} for i, s in enumerate(improvements_text)]

        # Default scores = 3 for all criteria
        criteria_evals = [
            {
                "criterion": c["name"],
                "score": 3,
                "weight": float(c["weight"]),
                "description": c["description"],
                "justification": "(Fallback — LLM output not parseable)",
                "quote_1": {"text": "", "supports_assessment": True, "valid": False},
                "quote_2": {"text": "", "supports_assessment": True, "valid": False},
                "is_fatal_criterion": bool(c.get("critical", False)),
            }
            for c in criteria
        ]

        section_score = compute_section_score(criteria_evals, section_name, paper_type)

        return {
            "qualitative_assessment": qualitative,
            "criteria_evaluations": criteria_evals,
            "improvements": improvements,
            "section_score": section_score,
            "paper_type": paper_type,
            "section_name": section_name,
            "eval_timestamp": eval_timestamp,
            "model_version": model_version,
            "_fallback": True,
        }
