"""
Prompt templates for paper-type-aware evaluation.

All prompts are now loaded from external versioned files.
To update prompts, edit files in prompts/section_evaluator/ and update config.yaml.
"""

from typing import List
from ..criteria.base import PAPER_TYPE_LABELS
from prompts.section_evaluator.prompt_loader import get_section_evaluator_prompt_loader

# ==========================================
# INITIALIZE PROMPT LOADER
# ==========================================
# Load all prompts from external versioned files
_prompt_loader = get_section_evaluator_prompt_loader()

# Load prompts (keeping original variable names for backward compatibility)
PAPER_TYPE_CONTEXTS = _prompt_loader.get_all_paper_type_contexts()
SECTION_TYPE_PROMPTS = _prompt_loader.get_all_section_type_guidance()

# All prompts are now loaded from external files via the PromptLoader
# To update prompts:
#   1. Edit the .txt files in prompts/section_evaluator/
#   2. To create a new version, copy an existing file (e.g., v1.0.txt) to a new version (e.g., v1.1.txt)
#   3. Update config.yaml to point to the new version
#   4. Call _prompt_loader.reload_prompts() to load the changes (or restart the app)

# Legacy hardcoded prompts (DEPRECATED - kept for reference only)
_DEPRECATED_PAPER_TYPE_CONTEXTS = {
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
- Connect prior findings to build toward this paper's contribution

**Quality levels:**
- **Excellent (5)**: Comprehensive coverage organized thematically; critical synthesis evaluating strengths/limitations of prior work; clear identification of gap with explicit connection to this paper; recent developments included; builds narrative toward contribution
- **Adequate (3)**: Key papers covered; organized beyond mere chronology; gap identified; some synthesis though may lean toward summary
- **Poor (1)**: Mere listing of papers without synthesis; missing key citations; no clear gap identification; purely chronological; no critical engagement

Common weaknesses: Mere listing, missing key citations, no critical synthesis, unclear connection to present study.""",

    "data": """The DATA section should:
- Identify and describe the data source clearly
- Explain why this data is appropriate for the question
- Acknowledge limitations (selection bias, measurement error, coverage gaps) honestly
- Provide or reference summary statistics
- Define key variables operationally

**Quality levels:**
- **Excellent (5)**: Ideal or near-ideal data with detailed justification of fit; thorough discussion of limitations and their implications for inference; selection processes and measurement details transparent; limitations acknowledged not dismissed
- **Adequate (3)**: Data adequately described and reasonably appropriate; limitations mentioned but discussion somewhat superficial; summary statistics provided
- **Poor (1)**: Questionable data fit with limitations ignored or dismissed; inadequate source description; selection processes opaque; no discussion of measurement issues

Common weaknesses: Inadequate source description, hidden limitations, missing summary statistics, limitations dismissed.""",

    "methodology": """The METHODOLOGY section should:
- Clearly specify the model or empirical approach
- Justify the methodological choice
- State key assumptions explicitly
- Address identification (for causal claims)
- Discuss potential threats to validity
- Plan for robustness checks

**Quality levels:**
- **Excellent (5)**: Compelling identification argument with institutional detail; falsification tests planned; assumptions defended not just stated; comprehensive robustness plan pre-specified; replication-ready detail
- **Adequate (3)**: Standard identification approach adequately explained; basic assumptions stated; typical robustness checks mentioned; sufficient detail for replication
- **Poor (1)**: Weak or circular identification reasoning; endogeneity unaddressed; assumptions hidden; robustness checks absent or ad hoc

Common weaknesses: Unclear specification, unjustified choices, hidden assumptions, weak identification argument.""",

    "model_setup": """The MODEL SETUP section should:
- State all assumptions explicitly and clearly
- Use consistent and defined mathematical notation
- Justify modeling choices economically (not just mathematically)
- Relate to existing models in literature

**Quality levels:**
- **Excellent (5)**: Assumptions deeply motivated by economic phenomena; parameter restrictions economically justified; creative modeling choices that advance theory; clear differentiation from prior models with specific citations
- **Adequate (3)**: Assumptions stated clearly; standard modeling approach competently applied; basic economic motivation
- **Poor (1)**: Hidden assumptions; ad hoc choices without justification; notation inconsistent; no connection to literature

Common weaknesses: Hidden assumptions, inconsistent notation, insufficient motivation for modeling choices.""",

    "proofs": """The PROOFS section should:
- Be mathematically correct with no errors
- Follow logically from stated assumptions
- Provide economic intuition alongside mathematics
- Be complete with no missing steps
- Address technical conditions (convergence, transversality, parameter restrictions)

**Quality levels:**
- **Excellent (5)**: Rigorous derivation with convergence conditions explicitly derived; transversality conditions stated; edge cases handled; multi-layered economic interpretation with decomposition or welfare analysis
- **Adequate (3)**: Mathematically correct with standard steps; basic one-sentence intuition per result; parameters chosen without empirical grounding
- **Poor (1)**: Mathematical errors; logical gaps; hand-waving instead of rigor; tautological interpretation ("X increases because investors want more X")

Common weaknesses: Mathematical errors, logical gaps, missing economic intuition, omitted convergence conditions.""",

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
- Calibrate effect sizes to real-world context

**Quality levels:**
- **Excellent (5)**: Economic magnitudes quantified and calibrated to real-world benchmarks; mechanisms explained with potential decomposition; unexpected results addressed transparently with discussion; effect sizes compared to prior literature
- **Adequate (3)**: Basic discussion of economic magnitude beyond significance stars; main findings clearly presented; tables adequately described
- **Poor (1)**: Only reports significance stars without interpretation; ignores null or contradictory results; no discussion of economic magnitude; selective reporting

Common weaknesses: Over-reliance on significance stars, ignoring economic magnitude, selective reporting.""",

    "discussion": """The DISCUSSION section should:
- Interpret results in the context of existing literature
- Explain the underlying mechanisms
- Acknowledge limitations honestly
- Suggest practical or theoretical implications

Common weaknesses: Mere repetition of results, overclaiming, ignoring limitations.""",

    "robustness_checks": """The ROBUSTNESS CHECKS section should:
- Test alternative model specifications systematically
- Check different samples and subgroups
- Use alternative variable definitions
- Report results honestly, including when core results weaken
- Address key threats to inference pre-identified in methodology

**Quality levels:**
- **Excellent (5)**: Extensive pre-specified robustness checks systematically exploring specifications, samples, and measures; transparent reporting including cases where results weaken with discussion; addresses all major threats to inference
- **Adequate (3)**: Standard robustness checks included (alternative specs, subsamples, variable definitions); results reported honestly
- **Poor (1)**: Only confirmatory tests shown; insufficient variation tested; no explanation when results change; cherry-picked robustness checks

Common weaknesses: Only confirmatory tests, insufficient variation, no explanation for cases where results change, selective reporting.""",

    "identification_strategy": """The IDENTIFICATION STRATEGY section should:
- Identify sources of endogeneity comprehensively
- Justify the IV, DiD, RDD, or other approach chosen
- Argue for the exclusion restriction convincingly with institutional detail
- Include falsification or pre-trend tests

**Quality levels:**
- **Excellent (5)**: Thorough multi-pronged argument for exclusion restriction with institutional detail and economic reasoning; multiple pre-specified falsification tests; potential confounds explicitly addressed; instrument relevance and validity demonstrated
- **Adequate (3)**: Standard identification approach adequately justified; basic pre-trend or placebo test included; main endogeneity concerns addressed
- **Poor (1)**: Exclusion restriction merely asserted without evidence; endogeneity sources ignored; no falsification tests; weak instruments not acknowledged

Common weaknesses: Unaddressed endogeneity, weak instruments, missing pre-trends test, assumed exclusion restriction.""",

    "calibration": """The CALIBRATION section should:
- Source or estimate parameter values with justification
- Clearly state calibration targets (moments)
- Evaluate model fit to data
- Explore sensitivity to key parameter values

**Quality levels:**
- **Excellent (5)**: Parameters empirically grounded with micro-founded estimates and citations; moments matched quantitatively with formal statistical evaluation; discrepancies acknowledged and discussed; systematic sensitivity analysis
- **Adequate (3)**: Reasonable parameter values with basic defense; calibration targets clearly stated; qualitative comparison of model to data
- **Poor (1)**: Ad hoc parameters without justification; fit claimed without evidence; no sensitivity analysis; targets unclear

Common weaknesses: Unjustified parameters, unclear targets, no fit assessment.""",

    "simulations": """The SIMULATIONS section should:
- Present economically meaningful scenarios
- Clearly distinguish baseline from counterfactual
- Compute welfare implications rigorously
- Include policy-relevant experiments

**Quality levels:**
- **Excellent (5)**: Policy-relevant scenarios with clear economic motivation; welfare effects decomposed and interpreted (e.g., distributional analysis); baseline vs counterfactual clearly distinguished with quantitative comparisons
- **Adequate (3)**: Meaningful scenarios simulated; basic welfare calculation provided; baseline and counterfactual distinguished
- **Poor (1)**: Irrelevant scenarios; welfare mentioned without quantification; unclear counterfactual; no economic interpretation

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
    # Load master prompt templates
    scoring_philosophy = _prompt_loader.get_master_prompt("scoring_philosophy")
    sophistication_assessment = _prompt_loader.get_master_prompt("sophistication_assessment")
    task_instructions = _prompt_loader.get_master_prompt("task_instructions")

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

    # Build sophistication assessment for theoretical/empirical papers
    sophistication_check = ""
    if paper_type in ["theoretical", "empirical", "finance", "macro"]:
        sophistication_check = f"\n---\n\n{sophistication_assessment}\n\n---\n"

    # Format task instructions with paper type label substitution
    formatted_task_instructions = task_instructions.replace("{paper_type_label}", PAPER_TYPE_LABELS.get(paper_type, paper_type))

    prompt = f"""{scoring_philosophy}

---

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

**Note**: Criteria descriptions with "(5=..., 3=..., 1=...)" provide concrete scoring anchors. Use them.

## Section Text
{section_text[:20000]}

{sophistication_check}

{formatted_task_instructions}"""

    return prompt


# ---------------------------------------------------------------------------
# Quote validation prompt
# ---------------------------------------------------------------------------

# Load quote validation prompt from external file
QUOTE_VALIDATION_PROMPT = _prompt_loader.get_master_prompt("quote_validation")
