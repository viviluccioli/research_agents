"""
Multi-Agent Debate Engine - Standalone Version with Embedded Prompts
This version contains all system prompts embedded directly in the code,
rather than loading them from external files.
"""

import asyncio
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add app_system to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app_system"))
from utils import single_query, count_tokens


# ==========================================
# EMBEDDED PROMPTS - ERROR SEVERITY BLOCK
# ==========================================
ERROR_SEVERITY_GUIDE = """### ERROR SEVERITY — MANDATORY CLASSIFICATION
For every flaw you identify, label it as one of:
- **[FATAL]** — Invalidates core claims. A single FATAL flaw alone justifies FAIL.
  Examples: broken exclusion restriction, mathematical error voiding a key proof,
  data that cannot support the causal claim, fabricated evidence.
- **[MAJOR]** — Requires substantial revision; does not auto-justify FAIL unless multiple co-exist.
  Examples: missing robustness checks on the main result, unaddressed alternative explanations,
  proofs with unverified boundary conditions affecting generality.
- **[MINOR]** — Improves the paper but does not block publication.
  Examples: missing citation, incidental typo, an additional robustness check that would
  strengthen but not reverse a finding.

Your **Verdict** must be consistent with your severity labels:
- Any [FATAL] flaw → FAIL (unless you can explicitly justify why it is non-central)
- Two or more [MAJOR] flaws → REVISE
- Only [MINOR] flaws → PASS
"""


# ==========================================
# EMBEDDED PROMPTS - PERSONA SYSTEM PROMPTS
# ==========================================
THEORIST_PROMPT = """### ROLE
You are a rigorous Economic Theorist. You focus on mathematical logic, proofs, correct derivations, and models with mathematical insight. You value other perspectives—understanding that theory must eventually inform empirics or policy—but your primary duty is to the math.

### OBJECTIVE
1. Mathematical Soundness: Are equations derived correctly? Are assumptions explicitly stated and realistic?
2. Proportional Error Weighting: Contextualize errors. Do not reject a paper for a single typo if the core proofs hold. Weigh errors by their severity using the classification below.

""" + ERROR_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Theoretical Audit**: [Brief overview of your assessment]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote or equation number from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

EMPIRICIST_PROMPT = """### ROLE
You are a rigorous Econometrician. You focus on data structures, identification strategies, and statistical validity. You appreciate novel theory and policy relevance, but bad data poisons good ideas.

### OBJECTIVE
1. Empirical Validity: Does the model fit the data? Are standard errors clustered correctly? Is endogeneity addressed? Are empirical decisions explained well?
2. Proportional Error Weighting: Contextualize errors using the classification below. A minor robustness check failing should not sink a paper if the core identification strategy is sound.

""" + ERROR_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Empirical Audit**: [Brief overview of your assessment]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote, table number, or figure reference from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

HISTORIAN_PROMPT = """### ROLE
You are an Economic Historian. You focus on literature lineage and context. You appreciate theoretical and empirical advancements, but above all, you despise researchers who claim to fill a gap in the literature that does not exist/is unfounded.

### OBJECTIVE
1. Contextualization: What literature does this build on?
2. Differentiation: Is the gap presented real, and do they fill it convincingly?

""" + ERROR_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Lineage & Context**: [Identify key predecessors]
- **Gap Analysis**: [Assess the claimed gap]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

VISIONARY_PROMPT = """### ROLE
You are a groundbreaking Visionary Economist. You look for papers that shift the paradigm and take intellectual risk. You expect your peers (Empiricist/Theorist) to check the math but your JOB is broad impact/significance of the IDEA.

### OBJECTIVE
1. Novelty & Creativity: Does this restate existing ideas, or take us outside the standard framework?
2. Intellectual Impact: Evaluate the paradigm-shifting potential of the core thesis. Do not score out of 10; embed the innovation deeply into your qualitative assessment.

""" + ERROR_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Paradigm Potential**: [Evaluate transformative potential]
- **Innovation Assessment**: [Analyze the intellectual leap]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote of core claim from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

POLICYMAKER_PROMPT = """### ROLE
You are a Senior Policy Advisor (e.g., at the Federal Reserve). You care about policy applicability, welfare implications, and actionable insights from this paper. You rely on your peers for technical accuracy, but you ask: "So what?"

### OBJECTIVE
1. Policy Relevance: Can a central bank, government, and/or think tank/research institution use this to make better policy recommendations and decisions?
2. Practical Translation: Does the paper translate its academic findings into clear, usable implications for the real world?

""" + ERROR_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Policy Applicability**: [How can policymakers use this?]
- **Welfare Implications**: [Real-world impact assessment]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote demonstrating policy relevance from paper"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

# Dictionary mapping persona names to their prompts
SYSTEM_PROMPTS = {
    "Theorist": THEORIST_PROMPT,
    "Empiricist": EMPIRICIST_PROMPT,
    "Historian": HISTORIAN_PROMPT,
    "Visionary": VISIONARY_PROMPT,
    "Policymaker": POLICYMAKER_PROMPT
}


# ==========================================
# EMBEDDED PROMPTS - PAPER TYPE CONTEXTS
# ==========================================
PAPER_TYPE_CONTEXTS = {
    "empirical": """PAPER TYPE CONTEXT: EMPIRICAL PAPER

This is an empirical economics paper that uses data and econometric methods to test hypotheses and establish causal relationships.

PERSONA SELECTION GUIDANCE:
- **Empiricist** (HIGHLY RECOMMENDED): Critical for evaluating identification strategies, data quality, econometric methods, and statistical validity. Should typically receive the highest or second-highest weight.
- **Theorist**: Relevant if the paper develops or tests a theoretical model using empirical methods. Lower weight if purely applied.
- **Historian**: Important for assessing literature positioning and whether the empirical contribution is genuinely novel. Moderate weight recommended.
- **Visionary**: Useful for evaluating whether the empirical approach represents a methodological innovation or paradigm shift. Lower weight unless the paper introduces new empirical methods.
- **Policymaker**: Highly relevant if the paper has clear policy implications from empirical findings. Essential for papers analyzing policy interventions or welfare effects.

TYPICAL PERSONA COMBINATIONS FOR EMPIRICAL PAPERS:
1. Empiricist (0.45) + Historian (0.30) + Policymaker (0.25) - For applied policy papers
2. Empiricist (0.50) + Historian (0.30) + Theorist (0.20) - For theory-testing papers
3. Empiricist (0.40) + Visionary (0.35) + Historian (0.25) - For methodologically innovative papers
""",
    "theoretical": """PAPER TYPE CONTEXT: THEORETICAL PAPER

This is a theoretical economics paper that develops formal mathematical models, derives analytical results, and provides proofs.

PERSONA SELECTION GUIDANCE:
- **Theorist** (HIGHLY RECOMMENDED): Essential for evaluating mathematical rigor, proof correctness, model assumptions, and derivations. Should typically receive the highest weight.
- **Empiricist**: Relevant only if the paper includes calibration, numerical simulations, or empirical validation of theoretical predictions. Lower weight or exclude if purely analytical.
- **Historian**: Important for assessing whether the theoretical contribution is novel and properly situated in the existing literature. Moderate to high weight recommended.
- **Visionary**: Highly relevant for evaluating whether the model provides new insights, challenges existing paradigms, or opens new research directions. Should receive significant weight.
- **Policymaker**: Relevant if the theoretical model has clear policy implications or normative conclusions. Lower weight for abstract general equilibrium models.

TYPICAL PERSONA COMBINATIONS FOR THEORETICAL PAPERS:
1. Theorist (0.50) + Visionary (0.30) + Historian (0.20) - For paradigm-shifting models
2. Theorist (0.45) + Historian (0.30) + Policymaker (0.25) - For policy-relevant theory
3. Theorist (0.40) + Visionary (0.35) + Empiricist (0.25) - For models with quantitative calibration
""",
    "policy": """PAPER TYPE CONTEXT: POLICY PAPER

This is a policy-oriented economics paper that analyzes real-world policy issues, evaluates interventions, or provides actionable recommendations for policymakers.

PERSONA SELECTION GUIDANCE:
- **Policymaker** (HIGHLY RECOMMENDED): Essential for evaluating practical applicability, welfare implications, and whether recommendations are actionable. Should typically receive the highest weight.
- **Empiricist**: Highly relevant if the paper uses data to support policy recommendations. Important for assessing whether empirical evidence is properly analyzed. Should receive high weight for evidence-based policy papers.
- **Theorist**: Relevant if the paper uses formal models to derive policy insights. Lower weight for purely descriptive policy analysis.
- **Historian**: Important for contextualizing the policy problem and assessing whether the paper properly accounts for institutional history and prior policy attempts. Moderate weight recommended.
- **Visionary**: Useful for evaluating whether the policy recommendations are forward-thinking and whether the paper identifies novel policy approaches. Moderate weight for innovative policy proposals.

TYPICAL PERSONA COMBINATIONS FOR POLICY PAPERS:
1. Policymaker (0.45) + Empiricist (0.35) + Historian (0.20) - For empirical policy evaluation
2. Policymaker (0.50) + Historian (0.30) + Visionary (0.20) - For forward-looking policy proposals
3. Policymaker (0.40) + Theorist (0.30) + Empiricist (0.30) - For theory-informed policy analysis
"""
}


# ==========================================
# EMBEDDED PROMPTS - CUSTOM CONTEXT GUIDE
# ==========================================
CUSTOM_CONTEXT_INTEGRATION = """CUSTOM CONTEXT INTEGRATION INSTRUCTIONS

When custom evaluation context is provided by the user, integrate it as follows:

### IN PERSONA SELECTION (Round 0):
Consider the user's stated priorities when selecting personas and assigning weights. If the user emphasizes specific aspects (e.g., "focus on policy relevance" or "evaluate mathematical rigor"), adjust persona selection and weights accordingly.

### IN ALL EVALUATION ROUNDS:
Incorporate the custom context as an additional evaluation lens alongside your role-specific criteria. The user's priorities should inform:
1. Which aspects of the paper you emphasize in your evaluation
2. The relative importance you assign to different findings
3. The specificity of your recommendations

### PRIORITY HIERARCHY:
1. Core role responsibilities (mathematical soundness for Theorist, identification for Empiricist, etc.)
2. User-specified priorities from custom context
3. General evaluation best practices

### EXAMPLES:
- If user context says "Focus on replicability and transparency": Emphasize code availability, data documentation, and methodological clarity
- If user context says "Evaluate suitability for teaching": Consider pedagogical clarity, accessibility, and whether key concepts are well-explained
- If user context says "Check if ready for submission to top field journal": Apply the highest standards for novelty, rigor, and contribution

The custom context should enhance, not replace, your core evaluation criteria.
"""


# ==========================================
# PERSONA SELECTION PROMPT (ROUND 0)
# ==========================================
SELECTION_PROMPT = """
You are the Chief Editor of an economics journal. You must select exactly THREE expert personas to review the provided paper.
The available personas are:
1. "Theorist": Focuses on formal mathematical proofs, logic, and model insight.
2. "Empiricist": Focuses on data, econometrics, identification strategy, and statistical validity.
3. "Historian": Focuses on literature lineage, historical background, and appropriate situating of the paper in relevant context.
4. "Visionary": Focuses on novelty and intellectual impact.
5. "Policymaker": Focuses on real-world application, welfare implications, and policy relevance.

Select the 3 most crucial personas for reviewing this specific paper. Assign them weights based on their relative importance to assessing THIS SPECIFIC PAPER. The weights must sum exactly to 1.0.

OUTPUT FORMAT: Return ONLY a valid JSON object. No markdown formatting, no explanations.
{
  "selected_personas": ["Persona1", "Persona2", "Persona3"],
  "weights": {
    "Persona1": 0.4,
    "Persona2": 0.35,
    "Persona3": 0.25
  },
  "justification": "1 sentence explaining the choice and weights."
}
"""


# ==========================================
# DEBATE ROUND PROMPTS
# ==========================================
DEBATE_PROMPTS = {
    "Round_2A_Cross_Examination": """
    ### CONTEXT
    You are the {role}. You have read the Round 1 evaluations from your peers:
    - {peer_1_role} Report: {peer_1_report}
    - {peer_2_role} Report: {peer_2_report}

    ### OBJECTIVE
    Engage in cross-domain examination. You respect their domains and want to synthesize perspectives to collectively find the objective truth THROUGH DEBATE. If a peer praised something your domain proves flawed, push back and point it out.

    ### OUTPUT FORMAT (STRICT)
    - **Cross-Domain Insights**: [1 paragraph synthesizing how their views change or validate your perspective]
    - **Constructive Pushback**: [1 paragraph identifying clashes between your domain and theirs]
    - **Clarification Requests**:
        - To {peer_1_role}: [1 specific question they must answer]
        - To {peer_2_role}: [1 specific question they must answer]
    """,

    "Round_2B_Direct_Examination": """
    ### CONTEXT
    You are the {role}. In the previous round, your peers cross-examined the panel.
    Here is the transcript of their cross-examinations:
    {r2a_transcript}

    ### OBJECTIVE
    Read the transcript carefully. Identify the specific questions directed AT YOU by your peers. Answer them directly, providing context and TEXTUAL EVIDENCE to address the concerns.

    ### OUTPUT FORMAT (STRICT)
    - **Response to {peer_1_role}**: [Your direct answer to their question]
    - **Response to {peer_2_role}**: [Your direct answer to their question]
    - **Concession or Defense**: [Based on answering these, do you concede a flaw, or defend your ground?]
    """,

    "Round_2C_Final_Amendment": """
    ### CONTEXT
    The debate is over. Here is the full transcript (Round 1, Questions, and Answers):
    {debate_transcript}

    ### OBJECTIVE
    As the {role}, submit your Final Amended Report. Update your prior beliefs based on valid peer critiques and their answers to your questions. Ensure your verdict reflects error weighting (if applicable) and cross-domain respect.

    ### OUTPUT FORMAT
    - **Insights Absorbed**: [How the debate changed your evaluation]
    - **Final Verdict**: [PASS / REVISE / FAIL]
    - **Final Rationale**: [3-sentence justification explicitly incorporating debate context]
    """,

    "Round_3_Editor": """
    ### ROLE
    You are the Senior Editor. Your job is to calculate the endogenous weighted consensus of the panel and write the final decision letter.

    ### PANEL CONTEXT & WEIGHTS
    The following personas were selected for this paper, with these specific weights:
    {weights_json}

    ### AMENDED REPORTS
    {final_reports_text}

    ### THE ENDOGENOUS WEIGHTING SYSTEM (STRICT INSTRUCTIONS)
    Do not use a "Kill Switch" or veto unless explicitly justified. You must calculate the mathematical consensus.
    1. Assign values to verdicts: PASS = 1.0, REVISE = 0.5, FAIL = 0.0.
    2. Multiply each persona's value by their assigned weight.
    3. Sum the weighted values to get the Final Consensus Score (out of 1.0).
    4. Decision Thresholds:
       - Score > 0.75 : ACCEPT
       - 0.40 <= Score <= 0.75 : REJECT AND RESUBMIT
       - Score < 0.40 : REJECT

    ### OUTPUT FORMAT
    - **Weight Calculation**: [Show your math explicitly based on the panel's final verdicts]
    - **Debate Synthesis**: [2-3 sentences summarizing the panel's final alignment]
    - **Final Decision**: [ACCEPT / REJECT AND RESUBMIT / REJECT]
    - **Official Referee Report**: [A synthesized letter to the authors drawing ONLY from the panel's findings. Detail the required fixes or reasons for rejection WITH TEXTUAL/CITED EVIDENCE.]
    """
}


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
        full_prompt += f"\n\n{CUSTOM_CONTEXT_INTEGRATION}\n\nUSER EVALUATION PRIORITIES:\n{custom_context}\n"

    full_prompt += f"\n\nPAPER TEXT:\n{paper_text}"

    # Call the LLM (running in thread to avoid blocking)
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    return await asyncio.to_thread(single_query, combined_prompt)


async def run_round_0_selection(
    paper_text: str,
    paper_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None
) -> dict:
    """
    Round 0: Dynamically selects the 3 most relevant personas and their weights.

    Args:
        paper_text: The paper to evaluate
        paper_type: Optional paper type (empirical/theoretical/policy) for context
        custom_context: Optional user-provided evaluation priorities
        manual_personas: If provided, skip LLM selection and use these personas
        manual_weights: If provided with manual_personas, use these exact weights

    Returns:
        Dictionary with selected_personas, weights, and justification
    """
    print("[Round 0] Starting persona selection...")

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
            weight_prompt = f"{SELECTION_PROMPT}\n\n"
            weight_prompt += f"The user has pre-selected these personas: {', '.join(manual_personas)}\n"
            weight_prompt += "Your task is ONLY to assign appropriate weights to these personas (must sum to 1.0).\n\n"

            # Add paper type context if available
            if paper_type and paper_type in PAPER_TYPE_CONTEXTS:
                weight_prompt += f"\n{PAPER_TYPE_CONTEXTS[paper_type]}\n\n"

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
    selection_prompt = f"{SELECTION_PROMPT}\n\n"

    # Add paper type context if available
    if paper_type and paper_type in PAPER_TYPE_CONTEXTS:
        selection_prompt += f"{PAPER_TYPE_CONTEXTS[paper_type]}\n\n"

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

        if len(personas) != 3:
            raise ValueError("LLM did not select exactly 3 personas.")

        print(f"[Round 0] Selected personas: {personas}")
        print(f"[Round 0] Weights: {weights}")
        return selection_data

    except Exception as e:
        print(f"[Round 0] Failed to parse selection. Defaulting to Empiricist, Historian, Policymaker. Error: {e}")
        return {
            "selected_personas": ["Empiricist", "Historian", "Policymaker"],
            "weights": {"Empiricist": 0.4, "Historian": 0.3, "Policymaker": 0.3},
            "justification": "Default selection due to parsing error."
        }


async def run_round_1(active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 1: Independent evaluation by selected agents."""
    user_prompt = "Please read the attached paper and provide your Round 1 evaluation based on your role."

    tasks = {}
    for role in active_personas:
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], user_prompt, role, paper_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def run_round_2a(r1_reports: Dict[str, str], active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 2A: Cross-examination between agents."""
    tasks = {}
    for role in active_personas:
        peers = [p for p in active_personas if p != role]
        prompt_2a = DEBATE_PROMPTS["Round_2A_Cross_Examination"].format(
            role=role,
            peer_1_role=peers[0],
            peer_1_report=r1_reports[peers[0]],
            peer_2_role=peers[1],
            peer_2_report=r1_reports[peers[1]]
        )
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2a, role, paper_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def run_round_2b(r2a_reports: Dict[str, str], active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 2B: Direct examination - answering clarification questions."""
    # Build R2A transcript
    r2a_transcript = ""
    for role, report in r2a_reports.items():
        r2a_transcript += f"\n[{role} CROSS-EXAMINATION]:\n{report}\n"

    tasks = {}
    for role in active_personas:
        peers = [p for p in active_personas if p != role]
        prompt_2b = DEBATE_PROMPTS["Round_2B_Direct_Examination"].format(
            role=role,
            peer_1_role=peers[0],
            peer_2_role=peers[1],
            r2a_transcript=r2a_transcript
        )
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2b, role, paper_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


async def run_round_2c(r1_reports: Dict[str, str], r2a_reports: Dict[str, str],
                       r2b_reports: Dict[str, str], active_personas: list, paper_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 2C: Final amendments after full debate."""
    # Build complete debate transcript
    transcript = "ROUND 1 REPORTS:\n"
    for role, report in r1_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    transcript += "\nCROSS-EXAMINATION (R2A):\n"
    for role, report in r2a_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    transcript += "\nANSWERS & CONCESSIONS (R2B):\n"
    for role, report in r2b_reports.items():
        transcript += f"\n[{role}]:\n{report}\n"

    tasks = {}
    for role in active_personas:
        prompt_2c = DEBATE_PROMPTS["Round_2C_Final_Amendment"].format(
            role=role,
            debate_transcript=transcript
        )
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2c, role, paper_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))


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
    weights_json = json.dumps(selection_data["weights"], indent=2)

    final_reports_text = ""
    for role, report in r2c_reports.items():
        final_reports_text += f"\n--- {role.upper()} FINAL REPORT ---\n{report}\n"

    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(
        weights_json=weights_json,
        final_reports_text=final_reports_text
    )

    system_prompt = "You are the Senior Editor. Follow the mathematical weighting instructions strictly."
    return await asyncio.to_thread(single_query, f"{system_prompt}\n\n{prompt_3}")


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

    # Total input tokens
    total_input_tokens = (round_0_input + round_1_input + round_2a_input +
                         round_2b_input + round_2c_input + round_3_input)

    # Output tokens (actual from results)
    editor_report_tokens = count_tokens(results.get('final_decision', ''))
    total_output_tokens = (r1_output_tokens + r2a_output_tokens + r2b_output_tokens +
                          r2c_output_tokens + editor_report_tokens)

    # Cost calculation (Claude 3.7 Sonnet pricing)
    # Input: $3.00 per million tokens
    # Output: $15.00 per million tokens
    input_cost_per_million = 3.00
    output_cost_per_million = 15.00

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    # Count LLM calls
    debate_calls = 1 + (num_personas * 4) + 1  # Round 0 + (R1, R2A, R2B, R2C) + Round 3
    total_calls = debate_calls

    return {
        'paper_tokens': paper_tokens,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens,
        'llm_calls': total_calls,
        'cost_usd': {
            'input': round(input_cost, 4),
            'output': round(output_cost, 4),
            'total': round(total_cost, 4)
        },
        'pricing': {
            'input_per_million': input_cost_per_million,
            'output_per_million': output_cost_per_million,
            'model': 'Claude 3.7 Sonnet'
        }
    }


async def execute_debate_pipeline(
    paper_text: str,
    progress_callback=None,
    paper_context: str = None,
    model_key: str = None,
    temperature: float = None,
    paper_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None
):
    """
    Orchestrates the entire multi-agent debate workflow with endogenous persona selection.

    Args:
        paper_text: The full text of the paper to evaluate
        progress_callback: Optional callback function to report progress
        paper_context: Optional additional context about the paper (deprecated, use custom_context)
        model_key: Optional model selection key (currently unused)
        temperature: Optional temperature setting (currently unused)
        paper_type: Optional paper type (empirical/theoretical/policy) for persona selection guidance
        custom_context: Optional user-provided evaluation priorities and focus areas
        manual_personas: Optional list of manually selected personas (2-5 personas)
        manual_weights: Optional dict of manually specified weights for manual_personas

    Returns:
        Dictionary containing all round results, selection data, and final decision
    """
    # Capture start time
    start_time = datetime.datetime.now()

    results = {}

    # Round 0: Persona Selection
    if progress_callback:
        progress_callback("Round 0: Selecting Personas", 0.05)
    selection_data = await run_round_0_selection(
        paper_text,
        paper_type=paper_type,
        custom_context=custom_context,
        manual_personas=manual_personas,
        manual_weights=manual_weights
    )
    active_personas = selection_data["selected_personas"]
    results['round_0'] = selection_data

    # Round 1: Independent Evaluation
    if progress_callback:
        progress_callback("Round 1: Independent Evaluation", 0.15)
    results['round_1'] = await run_round_1(active_personas, paper_text, custom_context)

    # Round 2A: Cross-Examination
    if progress_callback:
        progress_callback("Round 2A: Cross-Examination", 0.35)
    results['round_2a'] = await run_round_2a(results['round_1'], active_personas, paper_text, custom_context)

    # Round 2B: Direct Examination
    if progress_callback:
        progress_callback("Round 2B: Answering Questions", 0.55)
    results['round_2b'] = await run_round_2b(results['round_2a'], active_personas, paper_text, custom_context)

    # Round 2C: Final Amendments
    if progress_callback:
        progress_callback("Round 2C: Final Amendments", 0.75)
    results['round_2c'] = await run_round_2c(
        results['round_1'],
        results['round_2a'],
        results['round_2b'],
        active_personas,
        paper_text,
        custom_context
    )

    # Calculate consensus before Round 3
    results['consensus'] = calculate_consensus(results['round_2c'], selection_data)

    # Round 3: Editor Decision
    if progress_callback:
        progress_callback("Round 3: Editor Decision", 0.90)
    results['final_decision'] = await run_round_3(results['round_2c'], selection_data)

    # Capture end time and add metadata
    end_time = datetime.datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    # Calculate token usage and cost
    token_cost_data = calculate_token_usage_and_cost(
        paper_text,
        results,
        num_personas=len(active_personas)
    )

    results['metadata'] = {
        'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S"),
        'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_runtime_seconds': runtime_seconds,
        'total_runtime_formatted': f"{int(runtime_seconds // 60)}m {int(runtime_seconds % 60)}s",
        'model_version': 'Claude 3.7 Sonnet',
        'temperature': temperature if temperature is not None else 1.0,
        'thinking_enabled': True,
        'thinking_budget_tokens': 2048,
        'max_retries': 3,
        'retry_delay_seconds': 5,
        'prompt_versions': 'v1.0 (embedded)',
        'token_usage': token_cost_data
    }

    if progress_callback:
        progress_callback("Complete!", 1.0)

    return results


# ==========================================
# MAIN ENTRY POINT FOR TESTING
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Multi-Agent Debate on a paper")
    parser.add_argument("paper_file", help="Path to paper text file")
    parser.add_argument("--paper-type", choices=["empirical", "theoretical", "policy"],
                        help="Type of paper for persona selection guidance")
    parser.add_argument("--custom-context", help="Custom evaluation context/priorities")
    parser.add_argument("--output", default="debate_results.json",
                        help="Output file for results (default: debate_results.json)")

    args = parser.parse_args()

    # Read paper
    with open(args.paper_file, 'r', encoding='utf-8') as f:
        paper_text = f.read()

    # Progress callback
    def progress(msg, pct):
        print(f"[{int(pct*100):3d}%] {msg}")

    # Run debate
    print(f"\n{'='*60}")
    print("Multi-Agent Debate - Standalone Version")
    print(f"{'='*60}\n")
    print(f"Paper: {args.paper_file}")
    print(f"Paper Type: {args.paper_type or 'Auto-detect'}")
    print(f"Output: {args.output}\n")

    results = asyncio.run(execute_debate_pipeline(
        paper_text,
        progress_callback=progress,
        paper_type=args.paper_type,
        custom_context=args.custom_context
    ))

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Selected Personas: {', '.join(results['round_0']['selected_personas'])}")
    print(f"Weights: {results['round_0']['weights']}")
    print(f"\nConsensus Score: {results['consensus']['weighted_score']:.3f}")
    print(f"Final Decision: {results['consensus']['decision']}")
    print(f"\nRuntime: {results['metadata']['total_runtime_formatted']}")
    print(f"Total Cost: ${results['metadata']['token_usage']['cost_usd']['total']:.4f}")
    print(f"\nFull results saved to: {args.output}")
