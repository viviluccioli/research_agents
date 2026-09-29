# engine.py - Multi-Agent Debate Engine (Experiment 5)
"""
Enhanced debate engine with Severity Delta (Δ) and Layman Translation.

NEW FEATURES IN EXP-5:
1. **Severity Delta (Δ) System**: Structured flaw categorization using
   counterfactual reasoning ("If this flaw were fixed, would the conclusion change?")
   - Δ-High: Conclusion breaks (fatal flaws)
   - Δ-Medium: Evidence halved (significant issues)
   - Δ-Low: Presentational only (minor issues)

2. **Layman Translation**: Each persona must translate their technical critique
   into plain English for cross-domain understanding.

3. **Δ-Based Filtering**: Round 2A cross-examination only allows cross-domain
   verdict changes for Δ-High flaws with Confidence > 8.

This module orchestrates the 5-round debate process between AI personas
to evaluate research papers. It handles persona selection, debate rounds,
consensus calculation, and metadata tracking.

Features:
- 10 available personas (Theorist, Econometrician, ML_Expert, Data_Scientist,
  CS_Expert, Historian, Visionary, Policymaker, Ethicist, Perspective)
- Round 0: Automatic persona selection (selects 3)
- Rounds 1-2C: Multi-agent debate with parallel execution
- Round 3: Weighted consensus and editor decision
- Caching support for cost-efficient re-runs
"""
import asyncio
import datetime
import json
import re
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
import yaml
import streamlit as st
from utils import single_query, referee_query, count_tokens
from config import (
    MODEL_PRIMARY,
    model_selection,
    API_BASE
)

# ==========================================
# PROMPT LOADING UTILITIES
# ==========================================
def load_paper_type_context(paper_type: str) -> str:
    """Load paper type context guidance for persona selection."""
    if not paper_type or paper_type not in ["empirical", "theoretical", "policy"]:
        return ""

    try:
        prompt_path = Path(__file__).parent.parent / "prompts" / "multi_agent_debate" / "additional_context" / "paper_type_contexts" / paper_type / "v1.0.txt"
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load paper type context for {paper_type}: {e}")
        return ""

def load_custom_context_guide() -> str:
    """Load custom context integration instructions."""
    try:
        prompt_path = Path(__file__).parent.parent / "prompts" / "multi_agent_debate" / "custom_context_integration.txt"
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load custom context guide: {e}")
        return ""

def load_error_severity_guide() -> str:
    """Load error severity classification guide."""
    try:
        prompt_path = Path(__file__).parent.parent / "prompts" / "multi_agent_debate" / "additional_context" / "error_severity" / "v1.0.txt"
        with open(prompt_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load error severity guide: {e}")
        return ""

def get_severity_delta_guide() -> str:
    """
    Return the Severity Delta (Δ) scoring guide for EXP-5.

    This guide uses counterfactual reasoning to categorize flaws.
    """
    return """
### SEVERITY SCORING (COUNTERFACTUAL SEVERITY DELTA - Δ)

You must evaluate your primary critique using the **Counterfactual Severity Delta (Δ)**.
Ask yourself: **"If this flaw were fixed, would the primary conclusion change?"**

- **Δ-High (Fatal Flaw)**: The conclusion mathematically/empirically BREAKS.
  *Example: An IV regression where the excluded instrument is logically related to the error term.
  Fixing this collapses the causal claim entirely.*

- **Δ-Medium (Significant Issue)**: The conclusion remains plausible, but evidence is halved or weakened substantially.
  *Example: A transformer AI model uses a standard learning rate without sensitivity analysis.
  Results might hold, but robustness is questionable.*

- **Δ-Low (Minor/Presentational)**: The core data/math is unaffected; the flaw is presentational or framing-related.
  *Example: A paper claims "Causal Impact" in the abstract but performs a robust "Descriptive Correlation."
  Changing the abstract fixes the error without touching the data.*

**CRITICAL RULE FOR ROUND 2A**: You may ONLY change your verdict based on a peer's critique if they:
1. Identified a **Δ-High** flaw, AND
2. Reported **Confidence Score > 8/10**

Ignore Δ-Low cascading errors from peers' domains.
"""

# ==========================================
# PERSONA SELECTION PROMPT (ROUND 0) - Enhanced for EXP-5
# ==========================================
SELECTION_PROMPT = """
You are the Chief Editor of an economics journal. Select exactly {N} expert personas to review the provided paper.

### SELECTION CRITERIA (PRIORITIZE VALUE PILLARS - EXP-5)

**IMPORTANT**: You MUST select from this EXACT list. Do not create new names or variations:

1. **"Theorist"**: Guards STRUCTURAL NOVELTY. Activated ONLY for new mathematical setups, new estimators, or formal derivations. Ignore if setup is standard (OLS/standard utility).

2. **"Econometrician"**: Guards EMPIRICAL MAPPING. Audits the validity of the empirical tool for the specific hypothesis. Does not assume causality unless claimed.

3. **"ML_Expert"**: Guards ALGORITHMIC INTEGRITY. Activated for LLMs, Neural Nets, or complex ML architectures. Focuses on model logic and interpretability.

4. **"Data_Scientist"**: Guards EMPIRICAL INTEGRITY. Focuses on data quality, leakage, and sampling.

5. **"CS_Expert"**: Guards COMPUTATIONAL FEASIBILITY. Focuses on scale and algorithmic complexity.

6. **"Visionary"**: Guards INTELLECTUAL NOVELTY. Mandatory if paper claims a paradigm shift.

7. **"Policymaker"**: Guards PRAGMATIC FEASIBILITY. Audits the "Implementation Gap" and global relevance (not US-centric).

8. **"Ethicist"**: Guards PROCEDURAL FAIRNESS. Focuses on moral hazard and privacy.

9. **"Perspective"**: Guards SOCIAL IMPACT. Focuses on marginalized groups and equity.

10. **"Historian"**: Guards INTELLECTUAL LINEAGE. Audits subject-matter-corpus context. Ensures the paper correctly frames its contribution within the existing lineage of thought.

### MANDATORY SELECTION RULES:
- If a paper makes **policy suggestions**, you MUST include **Policymaker**.
- If it introduces a **new math framework**, you MUST include **Theorist**.

### OBJECTIVE

Select EXACTLY {N} personas from the list above that are most crucial for reviewing THIS SPECIFIC PAPER.
Assign them weights based on relevance. Weights must sum exactly to 1.0.

**CRITICAL**: Use the EXACT persona names as shown above (e.g., "ML_Expert" not "Machine Learning Expert", "Theorist" not "Mathematician").

OUTPUT FORMAT (STRICT JSON):
{{
  "selected_personas": ["Persona1", "Persona2", ...],
  "weights": {{"Persona1": 0.5, "Persona2": 0.5, ...}},
  "justification": "1 sentence explaining why this specific mix audits the paper's core intent."
}}
"""

# ==========================================
# SYSTEM PROMPTS FOR EACH AGENT
# ==========================================
def load_persona_prompt(persona_name: str) -> str:
    """Load persona system prompt from file."""
    try:
        # Try loading from the new persona directories
        persona_dir_map = {
            "Theorist": "theorist",
            "Econometrician": "econometrician",
            "ML_Expert": "ml_expert",
            "Data_Scientist": "data_scientist",
            "CS_Expert": "cs_expert",
            "Historian": "historian",
            "Visionary": "visionary",
            "Policymaker": "policymaker",
            "Ethicist": "ethicist",
            "Perspective": "perspective"
        }

        persona_dir = persona_dir_map.get(persona_name)
        if not persona_dir:
            raise ValueError(f"Unknown persona: {persona_name}")

        prompt_path = Path(__file__).parent.parent / "prompts" / "multi_agent_debate" / "personas" / persona_dir / "v1.0.txt"
        with open(prompt_path, 'r') as f:
            prompt_content = f.read()

        # Inject error severity guide and EXP-5 enhancements
        error_severity = load_error_severity_guide()
        severity_delta = get_severity_delta_guide()
        layman_guide = """
### LAYMAN TRANSLATION REQUIREMENT (EXP-5)

**MANDATORY**: You MUST include a "Layman Translation" section in your output.
Translate your technical critique into 1-2 sentences of plain English so non-experts
on the panel can understand exactly what is right/wrong with the paper.

This enables cross-domain synthesis during Round 2A cross-examination.
"""

        # Replace placeholders
        prompt_content = prompt_content.replace("{error_severity}", error_severity)

        # Append EXP-5 enhancements
        prompt_content += "\n\n" + severity_delta + "\n\n" + layman_guide

        return prompt_content

    except Exception as e:
        print(f"Warning: Could not load persona prompt for {persona_name}: {e}")
        # Fallback to hardcoded prompts with EXP-5 enhancements
        return FALLBACK_SYSTEM_PROMPTS.get(persona_name, f"ROLE: {persona_name}. Evaluate the paper.")

# Fallback system prompts (used if file loading fails) - Enhanced for EXP-5
_DELTA_AND_LAYMAN_SUFFIX = """

""" + get_severity_delta_guide() + """

### MANDATORY OUTPUT FORMAT (EXP-5)
- **Structural Strength**: [Identify one core methodology, dataset, or mathematical derivation that is robust]
- **Domain Audit**: [Your technical critique strictly within your assigned domain]
- **Severity Delta (Δ)**: [Select Δ-High, Δ-Medium, or Δ-Low based on counterfactual reasoning]
- **Layman Translation**: [Translate your technical critique into 1-2 sentences of plain English for cross-domain understanding]
- **Confidence Score (1-10)**: [10 = Absolute certainty; 1 = Speculative hunch]
- **Source Evidence**: [MANDATORY: verbatim quotes/equations/tables from the paper]
- **Verdict**: [PASS/REVISE/FAIL. *Rule: You may ONLY issue a FAIL if you have a Δ-High flaw with Confidence Score > 8*]
"""

FALLBACK_SYSTEM_PROMPTS = {
    "Theorist": "ROLE: Pure Economic Theorist. Focus ONLY on mathematical logic, proofs, and the soundness of derivations." + _DELTA_AND_LAYMAN_SUFFIX,

    "Econometrician": "ROLE: Econometrician. Focus ONLY on causal inference, endogeneity, identification strategies, and the robustness of results and interpretation." + _DELTA_AND_LAYMAN_SUFFIX,

    "ML_Expert": "ROLE: Machine Learning/AI Expert. Focus ONLY on the model architecture decisions, structure, and execution (e.g., transformers, dimensionality reduction algorithms), hyperparameter tuning, train/test validity, model explanation, interpretability, and relevance; keep Occam's Razor in mind." + _DELTA_AND_LAYMAN_SUFFIX,

    "Data_Scientist": "ROLE: Data Science Expert. Focus ONLY on the data pipeline: data cleaning decisions, feature engineering, exploratory data analysis (EDA), data leakage, preprocessing biases, and potential mistakes." + _DELTA_AND_LAYMAN_SUFFIX,

    "CS_Expert": "ROLE: Computer Science Expert. Focus ONLY on algorithm creation if it is not SOLELY ML, computational complexity and efficiency, memory efficiency, hardware constraints." + _DELTA_AND_LAYMAN_SUFFIX,

    "Historian": "ROLE: Historian of Thought. Focus ONLY on literature lineage and how well the author represents that lineage, characterizes their work within the lineage, and contributes to the lineage." + _DELTA_AND_LAYMAN_SUFFIX,

    "Visionary": "ROLE: Visionary. Focus ONLY on paradigm-shifting potential. Does this challenge existing frameworks, or is it merely incremental? View economics from both an insider and outsider perspective when answering these questions." + _DELTA_AND_LAYMAN_SUFFIX,

    "Policymaker": "ROLE: Policymaker. Focus ONLY on real-world utility. Can a central bank or regulator use this? Are there welfare implications, and are they actionable?" + _DELTA_AND_LAYMAN_SUFFIX,

    "Ethicist": "ROLE: Ethicist. Focus ONLY on the adherence of this premise and construction on moral and social values, privacy, consent, fairness, accountability, philosophical implications of the research." + _DELTA_AND_LAYMAN_SUFFIX,

    "Perspective": "ROLE: Perspective/DEI Expert. Focus ONLY on distributional consequences. Does this dataset contain inherent biases? Does the algorithm lack fairness? How does this impact marginalized groups? Are marginalized groups represented?" + _DELTA_AND_LAYMAN_SUFFIX
}

# Build SYSTEM_PROMPTS dict by loading all personas
SYSTEM_PROMPTS = {
    persona: load_persona_prompt(persona)
    for persona in [
        "Theorist", "Econometrician", "ML_Expert", "Data_Scientist", "CS_Expert",
        "Historian", "Visionary", "Policymaker", "Ethicist", "Perspective"
    ]
}

DEBATE_PROMPTS = {
    "Round_2A_Cross_Examination": """
    ### CONTEXT: You are the {role}. Read your peers' Round 1 evaluations:
    {peer_reports}

    ### OBJECTIVE
    Engage in cross-domain examination. You must base your understanding of your peers' critiques
    heavily on their **"Layman Translation"** sections for cross-domain clarity.

    Employ the **Principle of Charity**: assume your peers' points are valid until proven otherwise,
    but push back vigorously (yet respectfully) if they violate the truth of your domain.

    ### CRITICAL RULE (EXP-5 Δ-BASED FILTERING):
    **DO NOT** penalize the paper or change your verdict based on an error outside your domain
    UNLESS a peer explicitly assigned it:
    1. **Δ-High** (fatal flaw), AND
    2. **Confidence Score > 8/10**

    **Ignore Δ-Low cascading errors** - they are presentational only and do not warrant cross-domain verdict changes.

    ### OUTPUT FORMAT
    - **Bayesian Update**: [How your prior belief shifted based on your peers' evidence and their Layman Translations]
    - **Constructive Pushback**: [Where you disagree with peers' assessments from your domain perspective]
    - **Clarification Requests**: [Ask 1 specific question to each peer]
    """,

    "Round_2B_Direct_Examination": """
    ### CONTEXT: You are the {role}. Here are the questions directed at you:
    {r2a_transcript}

    ### OBJECTIVE
    Answer the questions directed at you directly using textual evidence. Do not dodge.

    ### OUTPUT FORMAT
    - **Direct Responses**: [Answer each peer's question]
    - **Concession or Defense**: [Explicitly state if you concede a flaw or defend your ground]
    """,

    "Round_2C_Final_Amendment": """
    ### CONTEXT: Full Debate Transcript:
    {debate_transcript}

    ### OBJECTIVE
    Submit your Final Amended Report.

    ⚠️ **CRITICAL: You MUST include BOTH your Final Verdict AND Final Score in the output below.**

    ### MANDATORY OUTPUT FORMAT (include these exact lines first):
    ```
    - **Final Verdict**: [PASS / REVISE / FAIL]
    - **Final Score**: [X/10]
    ```

    Then provide:
    - **Insights Absorbed**: [How the debate changed your evaluation]
    - **Final Rationale**: [3-sentence justification incorporating debate context]

    Score Scale Reminder:
    - 1-3 = FAIL zone
    - 4-7 = REVISE zone
    - 8-10 = PASS zone
    """,

    "Round_3_Editor": """
    ### CONTEXT: You are the Senior Editor. Calculate the endogenous weighted consensus of the panel and write the final decision letter.

    ### PANEL CONTEXT & WEIGHTS
    {weights_json}

    ### AMENDED REPORTS
    {final_reports_text}

    ### THE ENDOGENOUS WEIGHTING SYSTEM (STRICT)
    1. Assign values to verdicts: PASS = 1.0, REVISE = 0.5, FAIL = 0.0.
    2. Multiply each persona's value by their assigned weight.
    3. Sum the weighted values to get the Final Consensus Score (out of 1.0).
    4. Thresholds: > 0.75 : ACCEPT | 0.40 - 0.75 : REJECT AND RESUBMIT | < 0.40 : REJECT

    ### OUTPUT FORMAT
    - **Weight Calculation**: [Show math]
    - **Debate Synthesis**: [2-3 sentences summarizing alignment]
    - **Final Decision**: [ACCEPT / REJECT AND RESUBMIT / REJECT]
    - **Official Referee Report**: [Synthesized letter drawing ONLY from panel findings]
    """
}

# ==========================================
# QUOTE VALIDATION
# ==========================================
def should_enable_quote_validation() -> bool:
    """
    Check if quote validation should be enabled.
    Defaults to True. Can be disabled by setting DISABLE_QUOTE_VALIDATION env var.
    """
    import os
    return os.environ.get('DISABLE_QUOTE_VALIDATION', '').lower() != 'true'

# ==========================================
# TEMPERATURE CONFIGURATION
# ==========================================
ROUND_TEMPERATURES = {
    'round_0': 0.4,   # Persona selection - needs consistency (same personas for similar papers)
    'round_1': 0.7,   # Independent analysis - needs thoughtful, creative evaluation
    'round_2a': 0.7,  # Cross-examination - needs insightful questions and synthesis
    'round_2b': 0.6,  # Direct answers - focused responses to specific questions
    'round_2c': 0.6,  # Final amendments - refined evaluation after debate
    'round_3': 0.4,   # Editor synthesis - faithful consensus calculation, no new ideas
}

def get_round_temperature(round_id: str) -> float:
    """
    Get the appropriate temperature for a given round.

    Args:
        round_id: Round identifier (e.g., 'round_0', 'round_1', 'round_2a')

    Returns:
        Temperature value (0.0-1.0)
    """
    return ROUND_TEMPERATURES.get(round_id, 0.7)  # Default to 0.7 if not specified

# ==========================================
# ORCHESTRATION FUNCTIONS
# ==========================================
async def call_llm_async(
    system_prompt: str,
    user_prompt: str,
    role: str,
    paper_text: str,
    custom_context: Optional[str] = None,
    round_id: str = 'round_1'
) -> str:
    """
    Async wrapper for LLM calls with round-specific temperature.

    Args:
        system_prompt: The role-specific system prompt
        user_prompt: The round-specific user prompt
        role: The persona name
        paper_text: The paper text
        custom_context: Optional user-provided evaluation priorities
        round_id: Round identifier for temperature selection (e.g., 'round_1', 'round_2a')

    Returns:
        LLM response string
    """
    # Build the full prompt
    full_prompt = user_prompt

    # Add custom context if provided
    if custom_context and custom_context.strip():
        custom_guide = load_custom_context_guide()
        full_prompt += f"\n\n{custom_guide}\n\nUSER EVALUATION PRIORITIES:\n{custom_context}\n"

    full_prompt += f"\n\nPAPER TEXT:\n{paper_text}"

    # Get round-specific temperature
    temperature = get_round_temperature(round_id)

    # Call the LLM (running in thread to avoid blocking)
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    return await asyncio.to_thread(referee_query, combined_prompt, temperature=temperature)

async def run_round_0_selection(
    paper_text: str,
    N: int = 3,
    paper_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None
) -> dict:
    """
    Round 0: Dynamically selects the N most relevant personas and their weights.

    Args:
        paper_text: The paper to evaluate
        N: Number of personas to select (default 3)
        paper_type: Optional paper type (empirical/theoretical/policy) for context
        custom_context: Optional user-provided evaluation priorities
        manual_personas: If provided, skip LLM selection and use these personas
        manual_weights: If provided with manual_personas, use these exact weights

    Returns:
        Dictionary with selected_personas, weights, and justification
    """
    print(f"[Round 0] Starting persona selection (N={N})...")

    # If manual selection is provided, use it directly
    if manual_personas:
        if manual_weights:
            # User provided both personas and weights
            print(f"[Round 0] Using manual selection: {manual_personas} with manual weights: {manual_weights}")
            return {
                "selected_personas": manual_personas,
                "weights": manual_weights,
                "justification": "Manually selected by user with specified weights."
            }
        else:
            # User provided personas, LLM assigns weights
            print(f"[Round 0] Using manual personas {manual_personas}, LLM will assign weights...")
            weight_prompt = f"{SELECTION_PROMPT.format(N=len(manual_personas))}\n\n"
            weight_prompt += f"The user has pre-selected these personas: {', '.join(manual_personas)}\n"
            weight_prompt += "Your task is ONLY to assign appropriate weights to these personas (must sum to 1.0).\n\n"

            # Add paper type context if available
            if paper_type:
                paper_context = load_paper_type_context(paper_type)
                if paper_context:
                    weight_prompt += f"\n{paper_context}\n\n"

            # Add custom context if available
            if custom_context and custom_context.strip():
                weight_prompt += f"\nUSER EVALUATION PRIORITIES:\n{custom_context}\n\n"

            weight_prompt += f"\nPAPER TEXT:\n{paper_text}\n\n"
            weight_prompt += f"OUTPUT FORMAT: Return ONLY valid JSON with these personas {manual_personas} and their weights."

            # Use round_0 temperature for consistency
            temperature = get_round_temperature('round_0')
            response = await asyncio.to_thread(referee_query, weight_prompt, temperature=temperature)

            try:
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    data = json.loads(response)

                weights = data.get("weights", {})

                # Validate that weights are for the correct personas
                if set(weights.keys()) == set(manual_personas):
                    print(f"[Round 0] LLM assigned weights: {weights}")
                    return {
                        "selected_personas": manual_personas,
                        "weights": weights,
                        "justification": data.get("justification", "Weights assigned by LLM for user-selected personas.")
                    }
                else:
                    # Fallback to equal weights
                    equal_weight = 1.0 / len(manual_personas)
                    weights = {p: equal_weight for p in manual_personas}
                    print(f"[Round 0] LLM weights invalid, using equal weights: {weights}")
                    return {
                        "selected_personas": manual_personas,
                        "weights": weights,
                        "justification": "Equal weights assigned due to LLM parsing error."
                    }
            except Exception as e:
                print(f"[Round 0] Failed to parse LLM weights: {e}")
                equal_weight = 1.0 / len(manual_personas)
                weights = {p: equal_weight for p in manual_personas}
                return {
                    "selected_personas": manual_personas,
                    "weights": weights,
                    "justification": "Equal weights assigned due to parsing error."
                }

    # Full automatic selection by LLM
    selection_prompt = f"{SELECTION_PROMPT.format(N=N)}\n\n"

    # Add paper type context if available
    if paper_type:
        paper_context = load_paper_type_context(paper_type)
        if paper_context:
            selection_prompt += f"{paper_context}\n\n"

    # Add custom context if available
    if custom_context and custom_context.strip():
        selection_prompt += f"USER EVALUATION PRIORITIES:\n{custom_context}\n\n"

    selection_prompt += f"PAPER TEXT:\n{paper_text}"

    # Use round_0 temperature for consistency in persona selection
    temperature = get_round_temperature('round_0')
    response = await asyncio.to_thread(referee_query, selection_prompt, temperature=temperature)

    try:
        # Extract JSON from response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            selection_data = json.loads(json_match.group())
        else:
            selection_data = json.loads(response)

        personas = selection_data.get("selected_personas", [])
        weights = selection_data.get("weights", {})

        # CRITICAL: Validate that all selected personas exist in SYSTEM_PROMPTS
        valid_personas = list(SYSTEM_PROMPTS.keys())
        invalid_personas = [p for p in personas if p not in valid_personas]

        if invalid_personas:
            print(f"[WARNING] LLM selected invalid personas: {invalid_personas}")
            print(f"[WARNING] Valid personas are: {valid_personas}")
            # Filter out invalid personas and normalize weights
            personas = [p for p in personas if p in valid_personas]

            # If we don't have enough valid personas, fall back to defaults
            if len(personas) < N:
                print(f"[WARNING] Only {len(personas)} valid personas, need {N}. Using defaults.")
                default_personas = ["Econometrician", "ML_Expert", "Policymaker"][:N]
                equal_weight = round(1.0 / N, 2)
                return {
                    "selected_personas": default_personas,
                    "weights": {p: equal_weight for p in default_personas},
                    "justification": "Default selection due to invalid LLM selections."
                }

            # Recalculate weights for valid personas only
            total_weight = sum(weights.get(p, 0) for p in personas)
            if total_weight > 0:
                weights = {p: weights.get(p, 0) / total_weight for p in personas}
            else:
                equal_weight = 1.0 / len(personas)
                weights = {p: equal_weight for p in personas}

        if len(personas) != N:
            print(f"[WARNING] Requested {N} personas, but LLM returned {len(personas)}.")
            # Proceed anyway if we got valid data

        print(f"[Round 0] Selected personas: {personas}")
        print(f"[Round 0] Weights: {weights}")
        return {
            "selected_personas": personas,
            "weights": weights,
            "justification": selection_data.get("justification", "Selected by LLM based on paper content.")
        }

    except Exception as e:
        print(f"[Round 0] Failed to parse selection. Defaulting to first {N} personas. Error: {e}")
        default_personas = ["Econometrician", "ML_Expert", "Policymaker"][:N]
        equal_weight = round(1.0 / N, 2)
        return {
            "selected_personas": default_personas,
            "weights": {p: equal_weight for p in default_personas},
            "justification": "Default selection due to parsing error."
        }

async def run_round_1(active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 1: Independent evaluation by selected agents."""
    print("[Round 1] Independent evaluations starting...")

    # Updated prompt to request numeric score
    user_prompt = """Evaluate this paper based on your role.

⚠️ **REQUIRED: You MUST provide both a categorical verdict AND a numeric quality score.**

**SCORING REQUIREMENT**: Rate the paper's quality within your domain on a 1-10 scale:
- **1-3 (FAIL zone)**: Fatal flaws, fundamental problems, reject
- **4-7 (REVISE zone)**: Significant to moderate issues, needs revision
- **8-10 (PASS zone)**: Minor issues to exceptional, accept

Your score reflects the paper's quality in your domain, NOT your confidence in the verdict.

**MANDATORY OUTPUT FORMAT** (include these exact lines):
```
- **Verdict**: [PASS/REVISE/FAIL]
- **Score**: [X/10]
```

[Then continue with your domain audit]

Example:
```
- **Verdict**: REVISE
- **Score**: 6/10
```
"""

    tasks = {}
    for role in active_personas:
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], user_prompt, role, paper_text, custom_context, round_id='round_1')

    # Use return_exceptions=True to get partial results even if some personas fail
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # Filter out exceptions and log them
    successful_results = {}
    for role, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"[Round 1] ERROR: {role} failed with exception: {result}")
        else:
            successful_results[role] = result

    print(f"[Round 1] Completed {len(successful_results)}/{len(active_personas)} evaluations")
    return successful_results

async def run_round_2a(r1_reports: Dict[str, str], active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 2A: Cross-examination between agents."""
    print("[Round 2A] Cross-examination starting...")
    tasks = {}
    for role in active_personas:
        # Only include peers that successfully completed Round 1
        peers = [p for p in active_personas if p != role and p in r1_reports]
        peer_reports_text = "\n".join([f"--- {p} Report ---\n{r1_reports[p]}\n" for p in peers])

        prompt_2a = DEBATE_PROMPTS["Round_2A_Cross_Examination"].format(
            role=role,
            peer_reports=peer_reports_text
        )
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2a, role, paper_text, custom_context, round_id='round_2a')

    # Use return_exceptions=True to get partial results even if some personas fail
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # Filter out exceptions and log them
    successful_results = {}
    for role, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"[Round 2A] ERROR: {role} failed with exception: {result}")
        else:
            successful_results[role] = result

    print(f"[Round 2A] Completed {len(successful_results)}/{len(active_personas)} cross-examinations")
    return successful_results

async def run_round_2b(r2a_reports: Dict[str, str], active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 2B: Direct examination - answering clarification questions."""
    print("[Round 2B] Answering clarifications starting...")
    # Build R2A transcript (only from personas that completed R2A)
    r2a_transcript = "\n".join([f"[{r} CROSS-EXAMINATION]:\n{text}\n" for r, text in r2a_reports.items()])

    tasks = {}
    for role in active_personas:
        # Only include personas that completed R2A
        if role in r2a_reports:
            prompt_2b = DEBATE_PROMPTS["Round_2B_Direct_Examination"].format(
                role=role,
                r2a_transcript=r2a_transcript
            )
            tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2b, role, paper_text, custom_context, round_id='round_2b')

    # Use return_exceptions=True to get partial results even if some personas fail
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # Filter out exceptions and log them
    successful_results = {}
    for role, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"[Round 2B] ERROR: {role} failed with exception: {result}")
        else:
            successful_results[role] = result

    print(f"[Round 2B] Completed {len(successful_results)}/{len(tasks)} answers")
    return successful_results

async def run_round_2c(r1_reports: Dict[str, str], r2a_reports: Dict[str, str],
                       r2b_reports: Dict[str, str], active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 2C: Final amendments after full debate."""
    print("[Round 2C] Final amendments starting...")
    # Build complete debate transcript (only from personas that completed each round)
    transcript = "ROUND 1 REPORTS:\n" + "\n".join([f"[{r}]:\n{t}" for r, t in r1_reports.items()])
    transcript += "\nCROSS-EXAMINATION:\n" + "\n".join([f"[{r}]:\n{t}" for r, t in r2a_reports.items()])
    transcript += "\nANSWERS:\n" + "\n".join([f"[{r}]:\n{t}" for r, t in r2b_reports.items()])

    tasks = {}
    for role in active_personas:
        # Only include personas that completed all previous rounds
        if role in r1_reports and role in r2a_reports and role in r2b_reports:
            prompt_2c = DEBATE_PROMPTS["Round_2C_Final_Amendment"].format(
                role=role,
                debate_transcript=transcript
            )
            tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2c, role, paper_text, custom_context, round_id='round_2c')

    # Use return_exceptions=True to get partial results even if some personas fail
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)

    # Filter out exceptions and log them
    successful_results = {}
    for role, result in zip(tasks.keys(), results):
        if isinstance(result, Exception):
            print(f"[Round 2C] ERROR: {role} failed with exception: {result}")
        else:
            successful_results[role] = result

    print(f"[Round 2C] Completed {len(successful_results)}/{len(tasks)} final reports")
    return successful_results

def extract_verdict_from_report(report: str) -> str:
    """Extract the final verdict from a Round 2C report."""
    # Try to find "Final Verdict:" first
    final_verdict_patterns = [
        r'\*\*Final Verdict\*\*\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'Final Verdict\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
    ]

    for pattern in final_verdict_patterns:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback to any verdict
    generic_patterns = [
        r'\*\*Verdict\*\*\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'Verdict\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
    ]

    for pattern in generic_patterns:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find all and take the last one
    all_matches = re.findall(r'\b(PASS|REVISE|REJECT|FAIL)\b', report, re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()

    return "UNKNOWN"


def extract_score_from_report(report: str) -> Optional[float]:
    """
    Extract the numeric score from a report.

    Looks for patterns like:
    - **Final Score**: 7/10
    - **Score**: 8/10
    - Score: 6/10
    - **Final Score**: 8 (without /10)

    Returns:
        Score as float (1-10), or None if not found
    """
    # Try to find "Final Score:" or "Score:" first (with /10)
    score_patterns_with_denominator = [
        r'\*\*Final Score\*\*\s*:+\s*([0-9\.]+)\s*/\s*10',
        r'Final Score\s*:+\s*([0-9\.]+)\s*/\s*10',
        r'\*\*Score\*\*\s*:+\s*([0-9\.]+)\s*/\s*10',
        r'Score\s*:+\s*([0-9\.]+)\s*/\s*10',
        # Keep these for backwards compatibility with old format
        r'\*\*Confidence Score\*\*\s*:+\s*([0-9\.]+)\s*/\s*10',
        r'Confidence Score\s*:+\s*([0-9\.]+)\s*/\s*10',
    ]

    for pattern in score_patterns_with_denominator:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Validate range
                if 1 <= score <= 10:
                    return score
            except ValueError:
                continue

    # Fallback: try patterns without /10 (must be followed by whitespace or end of line)
    score_patterns_without_denominator = [
        r'\*\*Final Score\*\*\s*:+\s*([0-9\.]+)(?:\s|$)',
        r'Final Score\s*:+\s*([0-9\.]+)(?:\s|$)',
        r'\*\*Score\*\*\s*:+\s*([0-9\.]+)(?:\s|$)',
        r'Score\s*:+\s*([0-9\.]+)(?:\s|$)',
        # Keep these for backwards compatibility
        r'\*\*Confidence Score\*\*\s*:+\s*([0-9\.]+)(?:\s|$)',
        r'Confidence Score\s*:+\s*([0-9\.]+)(?:\s|$)',
    ]

    for pattern in score_patterns_without_denominator:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Validate range
                if 1 <= score <= 10:
                    return score
            except ValueError:
                continue

    return None

def calculate_consensus(r2c_reports: Dict[str, str], selection_data: dict) -> dict:
    """Calculate the weighted consensus from Round 2C reports."""
    weights = selection_data["weights"]
    verdicts = {}
    scores = {}

    # Extract verdict and score from each report
    for role, report in r2c_reports.items():
        verdict = extract_verdict_from_report(report)
        score = extract_score_from_report(report)
        verdicts[role] = verdict
        scores[role] = score

    # Calculate weighted score (categorical method)
    verdict_values = {"PASS": 1.0, "REVISE": 0.5, "FAIL": 0.0, "REJECT": 0.0, "UNKNOWN": 0.0}
    weighted_score_categorical = 0.0

    for role, verdict in verdicts.items():
        weight = weights.get(role, 0)
        value = verdict_values.get(verdict, 0.0)
        weighted_score_categorical += weight * value

    # Calculate weighted score (numeric method if scores available)
    weighted_score_numeric = None
    if all(score is not None for score in scores.values()):
        # Normalize scores to 0-1 scale (divide by 10)
        weighted_score_numeric = 0.0
        for role, score in scores.items():
            weight = weights.get(role, 0)
            weighted_score_numeric += weight * (score / 10.0)

    # Determine decision based on thresholds (using categorical)
    if weighted_score_categorical > 0.75:
        decision = "ACCEPT"
    elif weighted_score_categorical >= 0.40:
        decision = "REJECT AND RESUBMIT"
    else:
        decision = "REJECT"

    return {
        "verdicts": verdicts,
        "scores": scores,
        "weighted_score": weighted_score_categorical,  # For backwards compatibility
        "weighted_score_categorical": weighted_score_categorical,
        "weighted_score_numeric": weighted_score_numeric,
        "decision": decision
    }

async def run_round_3(r2c_reports: Dict[str, str], selection_data: dict) -> str:
    """Round 3: Editor synthesizes with weighted consensus."""
    print("[Round 3] Editor decision starting...")
    weights_json = json.dumps(selection_data["weights"], indent=2)

    final_reports_text = ""
    for role, report in r2c_reports.items():
        final_reports_text += f"\n--- {role.upper()} FINAL REPORT ---\n{report}\n"

    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(
        weights_json=weights_json,
        final_reports_text=final_reports_text
    )

    system_prompt = "You are the Senior Editor. Follow the mathematical weighting instructions strictly."
    # Use round_3 temperature for faithful synthesis
    temperature = get_round_temperature('round_3')
    result = await asyncio.to_thread(referee_query, f"{system_prompt}\n\n{prompt_3}", temperature=temperature)
    print("[Round 3] Editor decision completed")
    return result

def calculate_token_usage_and_cost(paper_text: str, results: Dict, num_personas: int = 3) -> Dict:
    """
    Calculate estimated token usage and cost for the debate pipeline.

    Args:
        paper_text: The full paper text
        results: The debate results dictionary
        num_personas: Number of active personas (default 3)

    Returns:
        Dictionary with token counts and cost estimates
    """
    # Count paper tokens
    paper_tokens = count_tokens(paper_text)

    # Estimate input tokens for each round
    # Round 0: paper only
    round_0_input = paper_tokens + 500  # selection prompt overhead

    # Round 1: paper + system prompt per persona
    round_1_input = num_personas * (paper_tokens + 1000)

    # Round 2A: paper + system prompt + Round 1 reports per persona
    r1_output_tokens = sum(count_tokens(report) for report in results.get('round_1', {}).values())
    round_2a_input = num_personas * (paper_tokens + 1000 + r1_output_tokens)

    # Round 2B: paper + system prompt + Round 2A transcript per persona
    r2a_output_tokens = sum(count_tokens(report) for report in results.get('round_2a', {}).values())
    round_2b_input = num_personas * (paper_tokens + 1000 + r2a_output_tokens)

    # Round 2C: paper + system prompt + full debate transcript per persona
    r2b_output_tokens = sum(count_tokens(report) for report in results.get('round_2b', {}).values())
    full_transcript_tokens = r1_output_tokens + r2a_output_tokens + r2b_output_tokens
    round_2c_input = num_personas * (paper_tokens + 1000 + full_transcript_tokens)

    # Round 3: all final reports
    r2c_output_tokens = sum(count_tokens(report) for report in results.get('round_2c', {}).values())
    round_3_input = r2c_output_tokens + 1500

    # Summarization: estimate input tokens for all summary calls
    summary_input_r1 = num_personas * r1_output_tokens
    summary_input_r2a = num_personas * r2a_output_tokens
    summary_input_r2b = num_personas * r2b_output_tokens
    summary_input_r2c = num_personas * r2c_output_tokens
    editor_report_tokens = count_tokens(results.get('final_decision', ''))
    summary_input_editor = editor_report_tokens + 500

    # Total input tokens
    total_debate_input = (round_0_input + round_1_input + round_2a_input +
                         round_2b_input + round_2c_input + round_3_input)
    total_summary_input = (summary_input_r1 + summary_input_r2a + summary_input_r2b +
                          summary_input_r2c + summary_input_editor)
    total_input_tokens = total_debate_input + total_summary_input

    # Output tokens (actual from results)
    total_debate_output = (r1_output_tokens + r2a_output_tokens + r2b_output_tokens +
                          r2c_output_tokens + editor_report_tokens)

    # Estimate summary output tokens (3-4 calls × 13 rounds, ~300-500 tokens each)
    total_summary_output = num_personas * 4 * 400 + 768  # ~5200 tokens

    total_output_tokens = total_debate_output + total_summary_output

    # Cost calculation (Claude 4.5 Sonnet pricing)
    # Input: $3.00 per million tokens
    # Output: $15.00 per million tokens
    input_cost_per_million = 3.00
    output_cost_per_million = 15.00

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    # Count LLM calls
    debate_calls = 1 + (num_personas * 4) + 1  # Round 0 + (R1, R2A, R2B, R2C) + Round 3
    summary_calls = (num_personas * 4) + 1  # R1, R2A, R2B, R2C summaries + editor
    total_calls = debate_calls + summary_calls

    return {
        'paper_tokens': paper_tokens,
        'input_tokens': {
            'debate': total_debate_input,
            'summarization': total_summary_input,
            'total': total_input_tokens
        },
        'output_tokens': {
            'debate': total_debate_output,
            'summarization': total_summary_output,
            'total': total_output_tokens
        },
        'total_tokens': total_input_tokens + total_output_tokens,
        'llm_calls': {
            'debate': debate_calls,
            'summarization': summary_calls,
            'total': total_calls
        },
        'cost_usd': {
            'input': round(input_cost, 4),
            'output': round(output_cost, 4),
            'total': round(total_cost, 4)
        },
        'model': MODEL_PRIMARY
    }

# ==========================================
# MAIN PIPELINE
# ==========================================
async def execute_debate_pipeline(
    paper_text: str,
    progress_callback=None,
    paper_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None,
    enable_quote_validation: bool = True,
    use_cache: bool = True,
    force_refresh: bool = False
) -> Dict:
    """
    Execute the complete multi-agent debate pipeline.

    Args:
        paper_text: The paper text to evaluate
        progress_callback: Optional callback function to report progress
        paper_type: Optional paper type for context
        custom_context: Optional user evaluation priorities
        manual_personas: Optional manual persona selection
        manual_weights: Optional manual weight assignment
        enable_quote_validation: Whether to run quote validation
        use_cache: Whether to use cached results (not implemented in exp_4)
        force_refresh: Force recomputation even if cache exists (not implemented in exp_4)

    Returns:
        Dictionary containing all round outputs and metadata
    """
    start_time = datetime.datetime.now()

    # Round 0: Selection
    if progress_callback:
        progress_callback("Round 0: Selecting Personas", 0.05)

    selection_data = await run_round_0_selection(
        paper_text,
        N=3,
        paper_type=paper_type,
        custom_context=custom_context,
        manual_personas=manual_personas,
        manual_weights=manual_weights
    )
    active_personas = selection_data["selected_personas"]

    # Round 1: Independent evaluations
    if progress_callback:
        progress_callback("Round 1: Independent Evaluation", 0.15)

    r1_reports = await run_round_1(active_personas, paper_text, custom_context)

    # Round 2A: Cross-examination
    if progress_callback:
        progress_callback("Round 2A: Cross-Examination", 0.35)

    r2a_reports = await run_round_2a(r1_reports, active_personas, paper_text, custom_context)

    # Round 2B: Direct examination
    if progress_callback:
        progress_callback("Round 2B: Answering Questions", 0.55)

    r2b_reports = await run_round_2b(r2a_reports, active_personas, paper_text, custom_context)

    # Round 2C: Final amendments
    if progress_callback:
        progress_callback("Round 2C: Final Amendments", 0.75)

    r2c_reports = await run_round_2c(r1_reports, r2a_reports, r2b_reports, active_personas, paper_text, custom_context)

    # Round 3: Editor decision
    if progress_callback:
        progress_callback("Round 3: Editor Decision", 0.90)

    final_decision = await run_round_3(r2c_reports, selection_data)

    # Calculate consensus
    consensus_data = calculate_consensus(r2c_reports, selection_data)

    # Quote validation (if enabled)
    quote_validation_results = {}
    if enable_quote_validation and should_enable_quote_validation():
        if progress_callback:
            progress_callback("Validating quotes...", 0.95)

        try:
            from referee._utils.quote_validator import validate_quotes_in_reports
            print("[Quote Validation] Running validation...")

            # Validate Round 1 reports
            r1_validation = validate_quotes_in_reports(
                reports=r1_reports,
                paper_text=paper_text
            )

            # Validate Round 2C reports
            r2c_validation = validate_quotes_in_reports(
                reports=r2c_reports,
                paper_text=paper_text
            )

            # Combine results
            quote_validation_results = {
                'round_1': r1_validation,
                'round_2c': r2c_validation
            }

            total_personas = len(r1_validation) + len(r2c_validation)
            print(f"[Quote Validation] Completed: {total_personas} reports validated (R1: {len(r1_validation)}, R2C: {len(r2c_validation)})")
        except ImportError:
            print("[Quote Validation] Skipped: thefuzz library not installed")
        except Exception as e:
            print(f"[Quote Validation] Error: {e}")

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Format duration as MM:SS
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    duration_formatted = f"{minutes:02d}:{seconds:02d}"

    # Load prompt versions from config
    prompt_versions = {}
    try:
        config_path = Path(__file__).parent.parent / "prompts" / "multi_agent_debate" / "config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Extract persona prompt versions
            for persona_name, persona_config in config.get('personas', {}).items():
                prompt_versions[f'persona_{persona_name}'] = persona_config.get('version', 'unknown')
            # Extract debate round prompt versions
            for round_name, round_config in config.get('debate_rounds', {}).items():
                prompt_versions[f'round_{round_name}'] = round_config.get('version', 'unknown')
    except Exception as e:
        prompt_versions['error'] = f'Could not load prompt versions: {e}'

    # Package results
    results = {
        'selection': selection_data,
        'round_0': selection_data,  # Alias for compatibility
        'round_1': r1_reports,
        'round_2a': r2a_reports,
        'round_2b': r2b_reports,
        'round_2c': r2c_reports,
        'final_decision': final_decision,
        'consensus': consensus_data,
        'metadata': {
            'timestamp': start_time.isoformat(),
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'total_runtime_formatted': duration_formatted,
            'num_personas': len(active_personas),
            'model': MODEL_PRIMARY,
            'model_version': MODEL_PRIMARY,  # Alias for Excel export
            'api_base': API_BASE,
            'temperature_system': 'per_round',  # Per-round temperature control enabled
            'round_temperatures': ROUND_TEMPERATURES.copy(),  # Temperature by round
            'thinking_enabled': False,  # Not currently implemented (requires temp=1.0)
            'thinking_budget_tokens': 0,  # Not enabled
            'max_retries': 3,
            'retry_delay_seconds': 5,
            'prompt_versions': prompt_versions
        }
    }

    # Add quote validation results at top level if available
    if quote_validation_results:
        results['round_1_quote_validation'] = quote_validation_results.get('round_1', {})
        results['round_2c_quote_validation'] = quote_validation_results.get('round_2c', {})
        results['metadata']['quote_validation'] = {
            'enabled': True,
            'rounds_validated': ['round_1', 'round_2c']
        }

    # Calculate token usage
    token_usage = calculate_token_usage_and_cost(paper_text, results, len(active_personas))
    results['metadata']['token_usage'] = token_usage

    if progress_callback:
        progress_callback("Complete!", 1.0)

    print(f"\n[Pipeline Complete] Duration: {duration:.1f}s | Cost: ${token_usage['cost_usd']['total']:.4f}")

    return results
