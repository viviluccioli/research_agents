"""
Prompt templates for paper-type-aware evaluation.
"""

from typing import List
from ..criteria.base import PAPER_TYPE_LABELS

# ---------------------------------------------------------------------------
# Paper-type context prompts
# ---------------------------------------------------------------------------

PAPER_TYPE_CONTEXTS = {
    "empirical": """This is an EMPIRICAL economics paper. Key characteristics:
- Uses data to test hypotheses or estimate causal relationships
- Must have a clear identification strategy for any causal claims
- Results should discuss both statistical and economic significance
- Robustness checks are expected
- Data limitations should be acknowledged

Evaluation emphasis: Data quality, identification strategy validity, statistical rigor, honest limitation acknowledgment.""",

    "theoretical": """This is a THEORETICAL economics paper. Key characteristics:
- Develops mathematical model(s) from explicit assumptions
- Derives propositions or theorems with formal proofs
- Proofs may appear in appendix, but intuition must be in main text
- Model extensions explore robustness of core results
- Economic interpretation must accompany mathematical results

Evaluation emphasis: Assumption clarity, mathematical correctness, logical consistency, economic intuition.""",

    "policy": """This is a POLICY-FOCUSED economics paper. Key characteristics:
- Addresses a real-world policy question or debate
- Recommendations must be grounded in evidence
- Should acknowledge trade-offs and distributional effects
- Implementation feasibility matters
- Should engage with current policy discourse

Evaluation emphasis: Evidence-recommendation linkage, feasibility, trade-off acknowledgment, practical applicability.""",

    "finance": """This is a FINANCE / MICROECONOMICS paper. Key characteristics:
- Studies firm, market, or individual-level phenomena
- Identification strategy is crucial (endogeneity must be addressed)
- Robustness checks across specifications are expected
- May use event studies, DiD, RDD, or IV approaches
- Should discuss economic magnitude, not just statistical significance

Evaluation emphasis: Identification strategy rigor, robustness of findings, economic magnitude interpretation.""",

    "macro": """This is a MACROECONOMICS paper. Key characteristics:
- Studies aggregate economic phenomena
- Often uses DSGE or other structural models
- Calibration or estimation of model parameters is critical
- Should match stylized facts or moments in the data
- Policy experiments and welfare analysis are common

Evaluation emphasis: Model calibration justification, moment matching, policy experiment validity, welfare analysis.""",

    "systematic_review": """This is a SYSTEMATIC REVIEW paper. Key characteristics:
- Synthesizes existing literature on a topic
- Must have transparent and reproducible search methodology
- Inclusion and exclusion criteria must be explicit
- Should identify gaps for future research
- Quality of included studies should affect conclusions

Evaluation emphasis: Search comprehensiveness, methodological transparency, synthesis quality, gap identification.""",
}

# ---------------------------------------------------------------------------
# Section-type guidance prompts
# ---------------------------------------------------------------------------

SECTION_TYPE_PROMPTS = {
    "abstract": """The ABSTRACT should:
- State the research question or purpose
- Briefly describe the methodology
- Present key findings
- Note implications or contributions

Common weaknesses: Vague claims, missing findings, overclaiming, inconsistency with paper content.""",

    "introduction": """The INTRODUCTION should follow the "3 moves" structure:
1. ESTABLISH TERRITORY: Show topic importance and relevance
2. ESTABLISH NICHE: Identify gap or problem in existing work
3. OCCUPY NICHE: State this paper's contribution clearly

Must include: Clear research question, contribution statement, paper roadmap.
Common weaknesses: Unclear contribution, missing roadmap, inconsistency with conclusion.""",

    "literature_review": """The LITERATURE REVIEW should:
- Cover relevant prior work comprehensively
- Be organized thematically or methodologically (not merely chronologically)
- Critically engage — evaluate and synthesize, do not just summarize
- Identify the gap this paper fills

Common weaknesses: Mere listing, missing key citations, no critical synthesis, unclear connection to present study.""",

    "data": """The DATA section should:
- Identify and describe the data source clearly
- Explain why this data is appropriate for the question
- Acknowledge limitations (selection bias, measurement error, coverage gaps)
- Provide or reference summary statistics

Common weaknesses: Inadequate source description, hidden limitations, missing summary statistics.""",

    "methodology": """The METHODOLOGY section should:
- Clearly specify the model or empirical approach
- Justify the methodological choice
- State key assumptions explicitly
- Address identification (for causal claims)
- Discuss potential threats to validity

Common weaknesses: Unclear specification, unjustified choices, hidden assumptions, weak identification argument.""",

    "model_setup": """The MODEL SETUP section should:
- State all assumptions explicitly and clearly
- Use consistent and defined mathematical notation
- Justify modeling choices economically
- Relate to existing models in literature

Common weaknesses: Hidden assumptions, inconsistent notation, insufficient motivation for modeling choices.""",

    "proofs": """The PROOFS section should:
- Be mathematically correct with no errors
- Follow logically from stated assumptions
- Provide economic intuition alongside mathematics
- Be complete with no missing steps

Common weaknesses: Mathematical errors, logical gaps, missing economic intuition.""",

    "extensions": """The EXTENSIONS section should:
- Address non-trivial and interesting cases
- Analyze how changes in parameters affect results (comparative statics)
- Show that core results are robust
- Have real-world or theoretical applicability

Common weaknesses: Trivial extensions, no comparative statics, results not robust.""",

    "results": """The RESULTS section should:
- Present findings that directly answer the research question
- Interpret both statistical and economic significance
- Acknowledge unexpected or null results
- Integrate tables and figures with the narrative text

Common weaknesses: Over-reliance on significance stars, ignoring economic magnitude, selective reporting.""",

    "discussion": """The DISCUSSION section should:
- Interpret results in the context of existing literature
- Explain the underlying mechanisms
- Acknowledge limitations honestly
- Suggest practical or theoretical implications

Common weaknesses: Mere repetition of results, overclaiming, ignoring limitations.""",

    "robustness_checks": """The ROBUSTNESS CHECKS section should:
- Test alternative model specifications
- Check different samples and subgroups
- Use alternative variable definitions
- Report results honestly, including when core results weaken

Common weaknesses: Only confirmatory tests, insufficient variation, no explanation for cases where results change.""",

    "identification_strategy": """The IDENTIFICATION STRATEGY section should:
- Identify sources of endogeneity
- Justify the IV, DiD, RDD, or other approach chosen
- Argue for the exclusion restriction convincingly
- Include falsification or pre-trend tests

Common weaknesses: Unaddressed endogeneity, weak instruments, missing pre-trends test.""",

    "calibration": """The CALIBRATION section should:
- Source or estimate parameter values with justification
- Clearly state calibration targets (moments)
- Evaluate model fit to data
- Explore sensitivity to key parameter values

Common weaknesses: Unjustified parameters, unclear targets, no fit assessment.""",

    "simulations": """The SIMULATIONS section should:
- Present economically meaningful scenarios
- Clearly distinguish baseline from counterfactual
- Compute welfare implications
- Include policy-relevant experiments

Common weaknesses: Irrelevant scenarios, unclear counterfactual, missing welfare analysis.""",

    "stylized_facts": """The STYLIZED FACTS section should:
- Document key empirical regularities with evidence
- Motivate the model being developed
- Cite data sources for all facts

Common weaknesses: Facts not connected to the model, missing sources, cherry-picked facts.""",

    "policy_context": """The POLICY CONTEXT section should:
- Accurately describe the policy environment and institutions
- Provide relevant policy history
- Identify affected stakeholders
- Situate the paper in current policy debates

Common weaknesses: Inaccurate institutional description, missing stakeholder analysis.""",

    "recommendations": """The RECOMMENDATIONS section should:
- Link recommendations directly to the analysis
- Consider implementation feasibility
- Acknowledge costs and distributional effects
- Be specific and actionable

Common weaknesses: Recommendations disconnected from evidence, ignoring distributional effects.""",

    "search_methodology": """The SEARCH METHODOLOGY section should:
- Report searches across multiple relevant databases
- Provide exact search strings
- Specify and justify the time period
- Be reproducible by another researcher

Common weaknesses: Single database, vague search terms, missing date range.""",

    "inclusion_criteria": """The INCLUSION / EXCLUSION CRITERIA section should:
- State criteria explicitly
- Defend criteria choices
- Apply criteria consistently across all studies
- Report a PRISMA-style or equivalent flow

Common weaknesses: Vague criteria, inconsistent application, no flow diagram.""",

    "synthesis": """The SYNTHESIS section should:
- Integrate and compare findings across studies
- Explain heterogeneity across studies
- Weight conclusions by study quality
- Identify research gaps

Common weaknesses: Mere listing of results, no quality weighting, gaps not identified.""",

    "conclusion": """The CONCLUSION should:
- Restate the main contribution clearly
- Summarize key findings
- Acknowledge limitations honestly
- Suggest future research directions
- Note broader policy or theoretical implications

Common weaknesses: Inconsistency with introduction, overclaiming, no future directions.""",

    "background": """The BACKGROUND section should:
- Provide the institutional or economic context needed
- Describe the relevant empirical landscape
- Frame the problem for the intended audience

Common weaknesses: Too much irrelevant history, insufficient context for the specific question.""",
}

# ---------------------------------------------------------------------------
# Master evaluation prompt
# ---------------------------------------------------------------------------

def build_evaluation_prompt(
    paper_type: str,
    paper_context: str,
    section_name: str,
    section_type: str,
    section_text: str,
    criteria: List[dict],
    figures_external: bool = False,
) -> str:
    """
    Build the master evaluation prompt for a given paper type, section, and criteria.
    """
    paper_type_context = PAPER_TYPE_CONTEXTS.get(paper_type, "")
    section_guidance = SECTION_TYPE_PROMPTS.get(section_type, "")

    criteria_lines = "\n".join(
        f"  - **{c['name']}** (weight {c['weight']:.0%}): {c['description']}"
        for c in criteria
    )

    figures_note = (
        "\n> **Note**: The author has indicated that tables and figures for this paper "
        "are located in an appendix or are not embedded in the submitted text. "
        "Do NOT penalize this section for the absence of tables or figures in the text. "
        "Evaluate the narrative discussion and interpretation of results on its own merits.\n"
        if figures_external else ""
    )

    prompt = f"""You are a senior reviewer for top economics journals.

## Paper Context
**Paper type**: {PAPER_TYPE_LABELS.get(paper_type, paper_type)}

{paper_type_context}

**Paper summary / context**: {paper_context or "(not provided)"}

## Section Being Evaluated
**Section name**: {section_name}
{figures_note}
{section_guidance}

## Evaluation Criteria
{criteria_lines}

## Section Text
{section_text[:20000]}

---

## Your Task

### Part 1: Qualitative Assessment (3–5 sentences)
Provide a concise overall assessment of this section's quality in the context of a {PAPER_TYPE_LABELS.get(paper_type, paper_type)} paper.
Cover: the section's purpose, 1–2 main strengths, 1–2 main shortcomings, and the single most impactful improvement.

### Part 2: Criterion-by-Criterion Evaluation
For EACH criterion listed above, provide:
1. **score** (integer 1–5): 1=Poor, 2=Below average, 3=Adequate, 4=Good, 5=Excellent
2. **justification** (1–2 sentences)
3. **quote_1**: An EXACT quote from the section text that supports or illustrates your assessment (10–60 words)
4. **quote_2**: A SECOND EXACT quote from the section text — either further supporting or complicating your assessment

### Part 3: Actionable Improvements
List 2–4 specific, actionable improvements ranked by importance.

---

## Output Format
Return a single JSON object with EXACTLY this structure:
{{
  "qualitative_assessment": "...",
  "criteria_evaluations": [
    {{
      "criterion": "criterion_name",
      "score": 1-5,
      "weight": 0.XX,
      "justification": "...",
      "quote_1": {{"text": "exact quote from section", "supports_assessment": true}},
      "quote_2": {{"text": "exact quote from section", "supports_assessment": true_or_false}}
    }}
  ],
  "improvements": [
    {{"priority": 1, "suggestion": "...", "rationale": "..."}},
    {{"priority": 2, "suggestion": "...", "rationale": "..."}}
  ]
}}

Return valid JSON only. No additional text outside the JSON."""

    return prompt


# ---------------------------------------------------------------------------
# Quote validation prompt
# ---------------------------------------------------------------------------

QUOTE_VALIDATION_PROMPT = """You provided this quote as evidence:

"{quote}"

Check whether this quote appears (possibly with minor OCR/formatting differences) in the section text below.
Return JSON: {{"found": true/false, "closest_match": "the closest actual text if not found exactly"}}

Section text:
{section_text}

Return valid JSON only."""
