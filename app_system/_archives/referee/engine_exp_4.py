# engine_exp_4.py - Multi-Agent Debate Engine (Experiment 4 Version)
"""
Experimental version with 10 personas (selects 3) based on mad_experiments/exp_4.

This module orchestrates the 5-round debate process between AI personas
to evaluate research papers. It handles persona selection, debate rounds,
consensus calculation, and metadata tracking.

Key differences from base engine.py:
- 10 available personas instead of 5 (still selects 3)
- Updated persona descriptions and system prompts
- Slightly modified debate prompt formatting
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
from utils import single_query, count_tokens
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

# ==========================================
# PERSONA SELECTION PROMPT (ROUND 0)
# ==========================================
SELECTION_PROMPT = """
You are the Chief Editor of an economics journal. You must select exactly {N} expert personas to review the provided paper.

**IMPORTANT**: You MUST select from this EXACT list of available personas. Do not create new persona names or use variations:

1. "Theorist" - Rigorous mathematical logic, logically airtight explanations and proof

2. "Econometrician" - Compelling causal inference, well-defined and constructed identification and estimation strategies, robust interpretation of results without overclaiming

3. "ML_Expert" - Fundamental machine learning models, traditional ML, neural architecture, modern ML, well-justified hyperparameter decisions

4. "Data_Scientist" - Data cleaning, processing, engineering, manipulation visualization, analysis, interpretation, cleanliness and interpretability of data manipulations

5. "CS_Expert" - Sophistication in algorithm creation, computability, complexity, specification/implementation duality, recursion, fixpoint, scale, function/data duality, static/dynamic duality, modeling, interaction

6. "Historian" - Literary history, subject-matter-corpus-specific context, accurate framing of the research narrative

7. "Visionary" - Potential for paradigm shifts; broad intellectual novelty

8. "Policymaker" - Real-world applicability, regulatory use, welfare implications, political and policy-making relevance

9. "Ethicist" - Moral hazard, adverse selection, selection bias, overrepresented/underrepresented literature, privacy, consent, following a standard of conduct, fairness, accountability, adherence to moral and social values

10. "Perspective" - Distributional consequences, algorithmic fairness, impact of research on marginalized groups and coverage of marginalized groups within research, racism, sexism, homophobia, transphobia, etc.

### OBJECTIVE

Select EXACTLY {N} personas from the list above that are most crucial for reviewing THIS SPECIFIC PAPER. Assign them weights based on relevance. Weights must sum exactly to 1.0.

**CRITICAL**: Use the EXACT persona names as shown above (e.g., "ML_Expert" not "Machine Learning Expert", "Theorist" not "Mathematician").

OUTPUT FORMAT (STRICT JSON):
{{
  "selected_personas": ["Persona1", "Persona2", ...],
  "weights": {{"Persona1": 0.5, "Persona2": 0.5, ...}},
  "justification": "1 sentence explaining the choice."
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

        # Inject error severity guide
        error_severity = load_error_severity_guide()
        return prompt_content.replace("{error_severity}", error_severity)

    except Exception as e:
        print(f"Warning: Could not load persona prompt for {persona_name}: {e}")
        # Fallback to hardcoded prompts
        return FALLBACK_SYSTEM_PROMPTS.get(persona_name, f"ROLE: {persona_name}. Evaluate the paper.")

# Fallback system prompts (used if file loading fails)
FALLBACK_SYSTEM_PROMPTS = {
    "Theorist": "ROLE: Pure Economic Theorist. Focus ONLY on mathematical logic, proofs, and the soundness of derivations.",

    "Econometrician": "ROLE: Econometrician. Focus ONLY on causal inference, endogeneity, identification strategies, and the robustness of results and interpretation.",

    "ML_Expert": "ROLE: Machine Learning/AI Expert. Focus ONLY on the model architecture decisions, structure, and execution (e.g., transformers, dimensionality reduction algorithms), hyperparameter tuning, train/test validity, model explanation, interpretability, and relevance; keep Occam's Razor in mind.",

    "Data_Scientist": "ROLE: Data Science Expert. Focus ONLY on the data pipeline: data cleaning decisions, feature engineering, exploratory data analysis (EDA), data leakage, preprocessing biases, and potential mistakes.",

    "CS_Expert": "ROLE: Computer Science Expert. Focus ONLY on algorithm creation if it is not SOLELY ML, computational complexity and efficiency, memory efficiency, hardware constraints.",

    "Historian": "ROLE: Historian of Thought. Focus ONLY on literature lineage and how well the author represents that lineage, characterizes their work within the lineage, and contributes to the lineage.",

    "Visionary": "ROLE: Visionary. Focus ONLY on paradigm-shifting potential. Does this challenge existing frameworks, or is it merely incremental? View economics from both an insider and outsider perspective when answering these questions.",

    "Policymaker": "ROLE: Policymaker. Focus ONLY on real-world utility. Can a central bank or regulator use this? Are there welfare implications, and are they actionable?",

    "Ethicist": "ROLE: Ethicist. Focus ONLY on the adherence of this premise and construction on moral and social values, privacy, consent, fairness, accountability, philosophical implications of the research.",

    "Perspective": "ROLE: Perspective/DEI Expert. Focus ONLY on distributional consequences. Does this dataset contain inherent biases? Does the algorithm lack fairness? How does this impact marginalized groups? Are marginalized groups represented?"
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
    Engage in cross-domain examination. Employ the **Principle of Charity**: assume your peers' points are valid until proven otherwise, but push back vigorously (yet respectfully) if they violate the truth of your domain.

    ### OUTPUT FORMAT
    - **Insights Absorbed**: [How their views change your perspective]
    - **Constructive Pushback**: [Clashes between your domain and theirs]
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

    ### OUTPUT FORMAT
    - **Insights Absorbed**: [How the debate changed your evaluation]
    - **Final Verdict**: [PASS / REVISE / FAIL]
    - **Final Rationale**: [3-sentence justification incorporating debate context]
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
# ORCHESTRATION FUNCTIONS
# ==========================================
async def call_llm_async(
    system_prompt: str,
    user_prompt: str,
    role: str,
    paper_text: str,
    custom_context: Optional[str] = None
) -> str:
    """
    Async wrapper for LLM calls.

    Args:
        system_prompt: The role-specific system prompt
        user_prompt: The round-specific user prompt
        role: The persona name
        paper_text: The paper text
        custom_context: Optional user-provided evaluation priorities

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

    # Call the LLM (running in thread to avoid blocking)
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    return await asyncio.to_thread(single_query, combined_prompt)

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

            response = await asyncio.to_thread(single_query, weight_prompt)

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
    response = await asyncio.to_thread(single_query, selection_prompt)

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
    user_prompt = "Evaluate this paper based on your role."

    tasks = {}
    for role in active_personas:
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], user_prompt, role, paper_text, custom_context)

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
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2a, role, paper_text, custom_context)

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
            tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2b, role, paper_text, custom_context)

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
            tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2c, role, paper_text, custom_context)

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

def calculate_consensus(r2c_reports: Dict[str, str], selection_data: dict) -> dict:
    """Calculate the weighted consensus from Round 2C reports."""
    weights = selection_data["weights"]
    verdicts = {}

    # Extract verdict from each report
    for role, report in r2c_reports.items():
        verdict = extract_verdict_from_report(report)
        verdicts[role] = verdict

    # Calculate weighted score
    verdict_values = {"PASS": 1.0, "REVISE": 0.5, "FAIL": 0.0, "REJECT": 0.0, "UNKNOWN": 0.0}
    weighted_score = 0.0

    for role, verdict in verdicts.items():
        weight = weights.get(role, 0)
        value = verdict_values.get(verdict, 0.0)
        weighted_score += weight * value

    # Determine decision based on thresholds
    if weighted_score > 0.75:
        decision = "ACCEPT"
    elif weighted_score >= 0.40:
        decision = "REJECT AND RESUBMIT"
    else:
        decision = "REJECT"

    return {
        "verdicts": verdicts,
        "weighted_score": weighted_score,
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
    result = await asyncio.to_thread(single_query, f"{system_prompt}\n\n{prompt_3}")
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
            'temperature': 1.0,  # From single_query in utils.py
            'thinking_enabled': True,
            'thinking_budget_tokens': 2048,
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
