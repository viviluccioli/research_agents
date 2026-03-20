"""
Criteria registry. Maps (paper_type, section_type) -> list of {name, weight, description}.
Weights within a section sum to 1.0.
"""

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paper type registry
# ---------------------------------------------------------------------------

PAPER_TYPES = ["empirical", "theoretical", "policy"]

PAPER_TYPE_LABELS = {
    "empirical": "Empirical",
    "theoretical": "Theoretical",
    "policy": "Policy",
}

# Default section lists per paper type
SECTION_DEFAULTS: Dict[str, List[str]] = {
    "empirical": ["Abstract", "Introduction", "Literature Review", "Data", "Methodology", "Results", "Discussion", "Conclusion"],
    "theoretical": ["Abstract", "Introduction", "Literature Review", "Model Setup", "Proofs / Derivations", "Extensions", "Conclusion"],
    "policy": ["Abstract", "Introduction", "Background", "Policy Context", "Analysis", "Recommendations", "Conclusion"],
}

# ---------------------------------------------------------------------------
# Universal criteria (all paper types × all sections)
# ---------------------------------------------------------------------------

_UNIVERSAL: Dict[str, List[dict]] = {
    "abstract": [
        {"name": "completeness",  "weight": 0.25, "description": "Covers purpose, method, findings, and implications"},
        {"name": "clarity",       "weight": 0.25, "description": "Accessible to a broad economics audience"},
        {"name": "accuracy",      "weight": 0.25, "description": "Claims match the paper content"},
        {"name": "conciseness",   "weight": 0.25, "description": "No unnecessary information"},
    ],
    "introduction": [
        {"name": "territory",     "weight": 0.15, "description": "Topic importance and relevance established (Move 1)"},
        {"name": "niche",         "weight": 0.20, "description": "Gap or problem in existing research identified (Move 2)"},
        {"name": "contribution",  "weight": 0.20, "description": "Paper's contribution clearly stated (Move 3)"},
        {"name": "thesis",        "weight": 0.20, "description": "Research question and thesis clearly stated"},
        {"name": "roadmap",       "weight": 0.10, "description": "Paper organization previewed at the end"},
        {"name": "scope",         "weight": 0.15, "description": "Limitations or scope briefly acknowledged"},
    ],
    "literature_review": [
        {"name": "coverage",      "weight": 0.20, "description": "Key relevant studies included"},
        {"name": "organization",  "weight": 0.20, "description": "Logical structure (thematic/methodological, not just chronological)"},
        {"name": "synthesis",     "weight": 0.25, "description": "Critical engagement — evaluates and synthesizes, not just summarizes"},
        {"name": "gap",           "weight": 0.20, "description": "Shows what is missing that this paper addresses"},
        {"name": "recency",       "weight": 0.15, "description": "Includes recent and relevant developments"},
    ],
    "conclusion": [
        {"name": "consistency",       "weight": 0.25, "description": "Aligns with introduction and body findings"},
        {"name": "contribution",      "weight": 0.20, "description": "Main findings clearly restated"},
        {"name": "limitations",       "weight": 0.20, "description": "Caveats and limitations honestly acknowledged"},
        {"name": "future_research",   "weight": 0.15, "description": "Suggests directions for further research"},
        {"name": "implications",      "weight": 0.20, "description": "Broader policy or theoretical implications noted"},
    ],
    "discussion": [
        {"name": "interpretation",    "weight": 0.30, "description": "Results interpreted in context of existing literature"},
        {"name": "mechanisms",        "weight": 0.25, "description": "Underlying mechanisms or explanations proposed"},
        {"name": "limitations",       "weight": 0.25, "description": "Limitations acknowledged honestly"},
        {"name": "implications",      "weight": 0.20, "description": "Practical or theoretical implications discussed"},
    ],
}

# ---------------------------------------------------------------------------
# Paper-type-specific section criteria
# ---------------------------------------------------------------------------

_EMPIRICAL: Dict[str, List[dict]] = {
    "data": [
        {"name": "source",        "weight": 0.15, "description": "Data source clearly identified and described"},
        {"name": "appropriateness","weight": 0.25, "description": "Data suitable for the research question with clear justification (5=ideal data with detailed justification, 3=adequate but with acknowledged limitations, 1=questionable fit)"},
        {"name": "limitations",   "weight": 0.30, "description": "Selection bias, measurement error, coverage gaps honestly acknowledged with depth (5=thorough discussion of implications, 3=mentioned but superficial, 1=ignored or dismissed)"},
        {"name": "sample",        "weight": 0.15, "description": "Sample size, time period, geographic scope clear"},
        {"name": "variables",     "weight": 0.15, "description": "Key variables operationally defined with measurement details"},
    ],
    "methodology": [
        {"name": "specification",      "weight": 0.20, "description": "Econometric model clearly specified"},
        {"name": "identification",     "weight": 0.30, "description": "Causal identification approach rigorously justified (5=compelling argument with falsification tests, 3=standard approach adequately justified, 1=weak or circular reasoning)"},
        {"name": "assumptions",        "weight": 0.20, "description": "Key assumptions stated and defended, not merely asserted"},
        {"name": "robustness_depth",   "weight": 0.20, "description": "Robustness approach is comprehensive and pre-specified (5=extensive sensitivity analysis across specifications/samples/measures, 3=basic checks, 1=minimal or ad hoc)"},
        {"name": "replicability",      "weight": 0.10, "description": "Sufficient detail for replication"},
    ],
    "results": [
        {"name": "alignment",          "weight": 0.20, "description": "Results directly address the stated research question"},
        {"name": "interpretation_depth","weight": 0.30, "description": "Results interpreted with depth: economic magnitude quantified, mechanisms explained, compared to prior work (5=multi-faceted interpretation with decomposition, 3=basic magnitude discussion, 1=only reports significance stars)"},
        {"name": "honesty",            "weight": 0.20, "description": "Unexpected, null, or contradictory results acknowledged and explained (5=transparent discussion with potential explanations, 3=acknowledged briefly, 1=ignored or cherry-picked)"},
        {"name": "calibration",        "weight": 0.15, "description": "Results calibrated to real-world context: effect sizes compared to benchmarks, policy implications quantified"},
        {"name": "presentation",       "weight": 0.15, "description": "Tables and figures integrate well with narrative"},
    ],
}

_THEORETICAL: Dict[str, List[dict]] = {
    "model_setup": [
        {"name": "assumptions",        "weight": 0.25, "description": "All assumptions explicitly and clearly stated"},
        {"name": "notation",           "weight": 0.10, "description": "Mathematical notation consistent and defined"},
        {"name": "motivation",         "weight": 0.25, "description": "Modeling choices economically motivated with depth (5=profound justification connecting theory to real-world phenomena, 3=standard textbook motivation, 1=ad hoc)"},
        {"name": "sophistication",     "weight": 0.25, "description": "Model represents theoretical advancement beyond routine application (5=novel framework or creative extension, 3=competent standard approach, 1=trivial or derivative)"},
        {"name": "relation_to_lit",    "weight": 0.15, "description": "Builds meaningfully on existing models with clear differentiation"},
    ],
    "proofs": [
        {"name": "correctness",        "weight": 0.20, "description": "No mathematical errors in derivations"},
        {"name": "rigor",              "weight": 0.25, "description": "Technical conditions handled properly: convergence explicitly derived, transversality stated, edge cases addressed (5=complete rigorous treatment, 3=basic correct derivation, 1=hand-waving)"},
        {"name": "economic_depth",     "weight": 0.25, "description": "Economic mechanisms explained with depth beyond surface intuition: provides decomposition, welfare analysis, multi-layered interpretation (5=profound insights, 3=adequate one-sentence explanation, 1=tautological)"},
        {"name": "sophistication",     "weight": 0.15, "description": "Analysis advances beyond routine techniques (5=novel theoretical insight, 3=competent standard analysis, 1=trivial)"},
        {"name": "clarity",            "weight": 0.15, "description": "Derivation clear and accessible to expert readers"},
    ],
    "extensions": [
        {"name": "meaningful_variation","weight": 0.30, "description": "Extensions address interesting and non-trivial cases"},
        {"name": "comparative_statics", "weight": 0.25, "description": "Parameter changes analyzed systematically"},
        {"name": "robustness",          "weight": 0.25, "description": "Core results hold under extensions"},
        {"name": "relevance",           "weight": 0.20, "description": "Extensions have real-world or theoretical applicability"},
    ],
}

_POLICY: Dict[str, List[dict]] = {
    "policy_context": [
        {"name": "institutional",      "weight": 0.30, "description": "Policy environment correctly and accurately described"},
        {"name": "historical",         "weight": 0.20, "description": "Relevant policy history covered"},
        {"name": "stakeholders",       "weight": 0.25, "description": "Affected parties and interests identified"},
        {"name": "current_debate",     "weight": 0.25, "description": "Paper situated within ongoing policy discussions"},
    ],
    "recommendations": [
        {"name": "evidence_basis",     "weight": 0.30, "description": "Recommendations follow from the analysis"},
        {"name": "feasibility",        "weight": 0.25, "description": "Implementation practicality considered"},
        {"name": "tradeoffs",          "weight": 0.25, "description": "Costs and distributional effects acknowledged"},
        {"name": "specificity",        "weight": 0.20, "description": "Concrete and actionable recommendations"},
    ],
    "background": [
        {"name": "context",            "weight": 0.35, "description": "Relevant institutional and economic context provided"},
        {"name": "data_landscape",     "weight": 0.30, "description": "Empirical landscape of the policy area described"},
        {"name": "framing",            "weight": 0.35, "description": "Problem clearly framed for policy audience"},
    ],
}

_FINANCE: Dict[str, List[dict]] = {
    "identification_strategy": [
        {"name": "endogeneity",        "weight": 0.25, "description": "Sources of endogeneity comprehensively identified with theoretical grounding"},
        {"name": "instrument_validity","weight": 0.30, "description": "IV, RDD, or DiD design rigorously justified (5=compelling with tests, 3=adequate defense, 1=assertions without evidence)"},
        {"name": "exclusion",          "weight": 0.25, "description": "Exclusion restriction argued convincingly with institutional detail and economic reasoning (5=thorough multi-pronged argument, 3=standard defense, 1=assumed without discussion)"},
        {"name": "falsification",      "weight": 0.20, "description": "Falsification tests strengthen identification (5=multiple pre-specified tests, 3=basic placebo/pre-trend, 1=none or weak)"},
    ],
    "robustness_checks": [
        {"name": "comprehensiveness",  "weight": 0.35, "description": "Robustness checks are extensive and address key threats (5=systematic exploration of specifications/samples/measures, 3=basic standard checks, 1=minimal or cherry-picked)"},
        {"name": "specification",      "weight": 0.25, "description": "Alternative econometric specifications tested meaningfully"},
        {"name": "sample_variation",   "weight": 0.20, "description": "Results tested across subsamples, time periods, outlier treatments"},
        {"name": "transparency",       "weight": 0.20, "description": "All results reported transparently including cases where core findings weaken (5=complete transparency with discussion, 3=selective but honest, 1=only confirmatory tests)"},
    ],
}

_MACRO: Dict[str, List[dict]] = {
    "calibration": [
        {"name": "justification",      "weight": 0.35, "description": "Parameter values empirically grounded or theoretically justified (5=micro-founded estimates with citations, 3=reasonable values with basic defense, 1=ad hoc without justification)"},
        {"name": "moments",            "weight": 0.25, "description": "Calibration targets clearly stated and economically motivated"},
        {"name": "fit",                "weight": 0.25, "description": "Model fit rigorously evaluated: moments matched quantitatively, discrepancies acknowledged (5=formal statistical evaluation, 3=qualitative comparison, 1=fit claimed without evidence)"},
        {"name": "sensitivity",        "weight": 0.15, "description": "Sensitivity to key parameter choices systematically explored"},
    ],
    "simulations": [
        {"name": "relevance",          "weight": 0.25, "description": "Simulated scenarios are economically meaningful and policy-relevant"},
        {"name": "counterfactual",     "weight": 0.25, "description": "Baseline vs. counterfactual clearly distinguished with economic interpretation"},
        {"name": "welfare_depth",      "weight": 0.25, "description": "Welfare implications rigorously computed and interpreted (5=decomposed welfare effects with distributional analysis, 3=basic welfare calculation, 1=mentioned without quantification)"},
        {"name": "policy_experiments", "weight": 0.25, "description": "Policy experiments are realistic and address relevant questions"},
    ],
    "stylized_facts": [
        {"name": "documentation",      "weight": 0.40, "description": "Key empirical regularities documented with evidence"},
        {"name": "relevance",          "weight": 0.35, "description": "Facts motivate the model being developed"},
        {"name": "sources",            "weight": 0.25, "description": "Data sources for stylized facts cited"},
    ],
}

_SYSREV: Dict[str, List[dict]] = {
    "search_methodology": [
        {"name": "databases",          "weight": 0.25, "description": "Multiple relevant databases searched"},
        {"name": "search_terms",       "weight": 0.30, "description": "Exact search strings reported transparently"},
        {"name": "date_range",         "weight": 0.20, "description": "Time period specified and justified"},
        {"name": "reproducibility",    "weight": 0.25, "description": "Another researcher could replicate the search"},
    ],
    "inclusion_criteria": [
        {"name": "clarity",            "weight": 0.30, "description": "Inclusion and exclusion criteria explicitly stated"},
        {"name": "justification",      "weight": 0.30, "description": "Criteria choices are defended"},
        {"name": "consistency",        "weight": 0.25, "description": "Criteria applied uniformly across studies"},
        {"name": "flow",               "weight": 0.15, "description": "PRISMA-style flow or equivalent reported"},
    ],
    "synthesis": [
        {"name": "integration",        "weight": 0.30, "description": "Findings synthesized and compared, not just listed"},
        {"name": "heterogeneity",      "weight": 0.25, "description": "Differences across studies explained"},
        {"name": "quality_weighting",  "weight": 0.25, "description": "Study quality considered in drawing conclusions"},
        {"name": "gaps",               "weight": 0.20, "description": "Research gaps clearly identified"},
    ],
}

# ---------------------------------------------------------------------------
# Master criteria lookup
# ---------------------------------------------------------------------------

# Maps section_type (canonical lowercase key) -> criteria list
# First checks paper-type-specific, then falls back to universal.

_ALL_CRITERIA: Dict[str, Dict[str, List[dict]]] = {
    "empirical":         {**_UNIVERSAL, **_EMPIRICAL},
    "theoretical":       {**_UNIVERSAL, **_THEORETICAL},
    "policy":            {**_UNIVERSAL, **_POLICY},
}

# Canonical section-type aliases: map common header strings -> canonical key
# Keys are lowercase; values are canonical type keys used in _ALL_CRITERIA dicts.
_SECTION_ALIASES: Dict[str, str] = {
    # Abstract
    "abstract": "abstract",
    "summary": "abstract",
    "executive summary": "abstract",
    # Introduction
    "introduction": "introduction",
    "intro": "introduction",
    "motivation": "introduction",
    "overview": "introduction",
    # Literature review
    "literature review": "literature_review",
    "literature": "literature_review",
    "related work": "literature_review",
    "related works": "literature_review",
    "related literature": "literature_review",
    "prior work": "literature_review",
    "prior literature": "literature_review",
    "previous work": "literature_review",
    # Background
    "background": "background",
    "institutional background": "background",
    "institutional context": "background",
    "context": "background",
    # Data
    "data": "data",
    "data description": "data",
    "data and variables": "data",
    "dataset": "data",
    "datasets": "data",
    "data sources": "data",
    "data and sample": "data",
    "sample": "data",
    "empirical setting": "data",
    "variable construction": "data",
    # Methodology
    "methodology": "methodology",
    "empirical methodology": "methodology",
    "empirical strategy": "methodology",
    "empirical approach": "methodology",
    "methods": "methodology",
    "method": "methodology",
    "estimation": "methodology",
    "estimation strategy": "methodology",
    "identification": "identification_strategy",
    "identification strategy": "identification_strategy",
    "causal identification": "identification_strategy",
    "instrumental variables": "identification_strategy",
    "natural experiment": "identification_strategy",
    # Model / Theory
    "model": "model_setup",
    "model setup": "model_setup",
    "the model": "model_setup",
    "theory": "model_setup",
    "theory/model": "model_setup",
    "theoretical framework": "model_setup",
    "theoretical model": "model_setup",
    "model and theoretical predictions": "model_setup",
    "model and theory": "model_setup",
    "theoretical predictions": "model_setup",
    "framework": "model_setup",
    "setup": "model_setup",
    "economic model": "model_setup",
    "equilibrium": "model_setup",
    # Proofs
    "proofs": "proofs",
    "proofs / derivations": "proofs",
    "appendix proofs": "proofs",
    "derivations": "proofs",
    "formal analysis": "proofs",
    # Extensions
    "extensions": "extensions",
    "extension": "extensions",
    "robustness and extensions": "extensions",
    # Results
    "results": "results",
    "empirical results": "results",
    "main results": "results",
    "findings": "results",
    "estimates": "results",
    "main findings": "results",
    "experimental results": "results",
    # Discussion
    "discussion": "discussion",
    "interpretation": "discussion",
    "implications": "discussion",
    # Robustness
    "robustness": "robustness_checks",
    "robustness checks": "robustness_checks",
    "robustness analysis": "robustness_checks",
    "sensitivity analysis": "robustness_checks",
    "additional results": "robustness_checks",
    "alternative specifications": "robustness_checks",
    # Calibration / Simulation (macro)
    "calibration": "calibration",
    "parameterization": "calibration",
    "simulations": "simulations",
    "simulation": "simulations",
    "quantitative analysis": "simulations",
    "welfare analysis": "simulations",
    "stylized facts": "stylized_facts",
    "facts": "stylized_facts",
    # Policy
    "policy context": "policy_context",
    "policy background": "policy_context",
    "policy environment": "policy_context",
    "recommendations": "recommendations",
    "policy recommendations": "recommendations",
    "policy implications": "recommendations",
    # Systematic review
    "search methodology": "search_methodology",
    "search strategy": "search_methodology",
    "search process": "search_methodology",
    "inclusion / exclusion criteria": "inclusion_criteria",
    "inclusion criteria": "inclusion_criteria",
    "exclusion criteria": "inclusion_criteria",
    "study selection": "inclusion_criteria",
    "synthesis": "synthesis",
    "meta-analysis": "synthesis",
    # Conclusion
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "concluding remarks": "conclusion",
    "conclusion and future work": "conclusion",
    "conclusion and policy implications": "conclusion",
    "summary and conclusion": "conclusion",
    "summary and conclusions": "conclusion",
}

# Keyword-based fallback: if a keyword appears anywhere in the section name,
# map to that canonical type. Checked in priority order.
_KEYWORD_MAP: List[Tuple[str, str]] = [
    ("abstract", "abstract"),
    ("introduction", "introduction"),
    ("motivation", "introduction"),
    ("literature", "literature_review"),
    ("related work", "literature_review"),
    ("prior work", "literature_review"),
    ("background", "background"),
    ("institutional", "background"),
    ("data", "data"),
    ("dataset", "data"),
    ("sample", "data"),
    ("variable", "data"),
    ("identification", "identification_strategy"),
    ("instrumental", "identification_strategy"),
    ("natural experiment", "identification_strategy"),
    ("methodology", "methodology"),
    ("empirical strategy", "methodology"),
    ("empirical approach", "methodology"),
    ("estimation", "methodology"),
    ("method", "methodology"),
    ("theoretical prediction", "model_setup"),
    ("theoretical framework", "model_setup"),
    ("theoretical model", "model_setup"),
    ("model", "model_setup"),
    ("theory", "model_setup"),
    ("framework", "model_setup"),
    ("setup", "model_setup"),
    ("equilibrium", "model_setup"),
    ("proof", "proofs"),
    ("derivation", "proofs"),
    ("extension", "extensions"),
    ("result", "results"),
    ("finding", "results"),
    ("estimate", "results"),
    ("discussion", "discussion"),
    ("interpretation", "discussion"),
    ("implication", "discussion"),
    ("robustness", "robustness_checks"),
    ("sensitivity", "robustness_checks"),
    ("alternative specification", "robustness_checks"),
    ("calibration", "calibration"),
    ("parameterization", "calibration"),
    ("simulation", "simulations"),
    ("welfare", "simulations"),
    ("stylized fact", "stylized_facts"),
    ("policy context", "policy_context"),
    ("policy background", "policy_context"),
    ("recommendation", "recommendations"),
    ("policy implication", "recommendations"),
    ("search strategy", "search_methodology"),
    ("search methodology", "search_methodology"),
    ("inclusion", "inclusion_criteria"),
    ("exclusion", "inclusion_criteria"),
    ("study selection", "inclusion_criteria"),
    ("synthesis", "synthesis"),
    ("meta-analysis", "synthesis"),
    ("conclusion", "conclusion"),
    ("concluding", "conclusion"),
    ("summary", "conclusion"),
]

# Default fallback criteria when no specific match exists
_FALLBACK_CRITERIA: List[dict] = [
    {"name": "clarity",          "weight": 0.25, "description": "Writing is clear and accessible"},
    {"name": "depth",            "weight": 0.25, "description": "Topic treated with appropriate depth"},
    {"name": "relevance",        "weight": 0.25, "description": "Content is relevant to the paper's goals"},
    {"name": "technical_quality","weight": 0.25, "description": "Technical content is accurate and well-executed"},
]

# Numbering prefix pattern: strips "1.", "2.1", "I.", "A.", "II." etc. from the front
_PREFIX_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*[\.\):\s]+|[IVXivx]{1,4}[\.\):\s]+|[A-Za-z][\.\):\s]+)"
)


def _canonical_section_type(section_name: str) -> str:
    """Map a section header string to a canonical section type key."""
    normalized = section_name.strip().lower()

    # Strip numbering prefix (e.g. "3. ", "II. ", "A. ", "4.1 ")
    stripped = _PREFIX_RE.sub("", normalized).strip()

    # 1. Exact match on stripped name
    if stripped in _SECTION_ALIASES:
        return _SECTION_ALIASES[stripped]

    # 2. Exact match on full normalized name
    if normalized in _SECTION_ALIASES:
        return _SECTION_ALIASES[normalized]

    # 3. Alias key contained within the section name (e.g. "model and theoretical predictions" contains "model")
    #    Use the longest matching alias to avoid spurious short matches (e.g. "data" matching "foundation")
    best_alias = None
    best_len = 0
    for alias, canonical in _SECTION_ALIASES.items():
        if alias in stripped and len(alias) > best_len:
            best_alias = canonical
            best_len = len(alias)
    if best_alias:
        return best_alias

    # 4. Keyword scan in priority order
    for keyword, canonical in _KEYWORD_MAP:
        if keyword in stripped:
            return canonical

    return "other"


def get_criteria(paper_type: str, section_name: str) -> List[dict]:
    """
    Return the list of criteria dicts for a given paper type and section name.
    Each dict has keys: name, weight, description.
    Weights sum to 1.0.
    """
    canonical = _canonical_section_type(section_name)
    type_map = _ALL_CRITERIA.get(paper_type, _UNIVERSAL)
    return type_map.get(canonical, _FALLBACK_CRITERIA)
