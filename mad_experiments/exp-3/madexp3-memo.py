"""
Multi-Agent Debate Engine - Policy Memo Evaluation Version
This version evaluates economic policy memos through multi-agent debate.
Adapted from the research paper evaluation system.
Includes Financial Stability Analyst persona.
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
# EMBEDDED PROMPTS - ISSUE SEVERITY BLOCK
# ==========================================
ISSUE_SEVERITY_GUIDE = """### ISSUE SEVERITY — MANDATORY CLASSIFICATION
For every flaw you identify, label it as one of:
- **[FATAL]** — Invalidates the core policy recommendation or renders the memo unusable for decision-making.
  Examples: fabricated evidence, fundamental misunderstanding of the policy problem, recommendations 
  that are demonstrably infeasible or illegal, missing critical stakeholder impact analysis that would 
  reverse the recommendation, mathematical errors in cost-benefit calculations that void conclusions.
  
- **[MAJOR]** — Requires substantial revision; does not auto-justify REJECT unless multiple co-exist.
  Examples: insufficient evidence for key recommendations, missing analysis of unintended consequences,
  unclear action items, inadequate cost-benefit analysis, poor problem definition, recommendations 
  that lack political feasibility assessment, missing critical stakeholders.
  
- **[MINOR]** — Improves the memo but does not block its use for decision-making.
  Examples: formatting inconsistencies, missing secondary citations, could benefit from additional 
  data visualization, minor organizational improvements, supplementary context that would strengthen 
  but not change recommendations.

Your **Verdict** must be consistent with your severity labels:
- Any [FATAL] flaw → REJECT (unless you can explicitly justify why it is non-central)
- Two or more [MAJOR] flaws → MAJOR REVISION REQUIRED
- Only [MINOR] flaws → ACCEPT WITH MINOR REVISIONS
"""

# ==========================================
# EMBEDDED PROMPTS - PERSONA SYSTEM PROMPTS
# ==========================================
COMMUNICATION_SPECIALIST_PROMPT = """### ROLE
You are a Policy Communication Specialist. You focus on clarity, accessibility, structure, and whether the memo effectively communicates to decision-makers with limited time and expertise. You value evidence and feasibility, but your primary duty is ensuring the memo can be quickly understood and acted upon.

### OBJECTIVE
1. **Clarity & Accessibility**: Is the memo written in clear, jargon-free language? Can a decision-maker grasp the main points in 5 minutes? Is the structure logical (inverted pyramid with most important information first)?
2. **Professional Presentation**: Is it professionally written? Are headings clear and descriptive? Is the executive summary truly a standalone summary? Can the reader efficiently navigate the document?
3. **Problem Definition**: Is the policy problem clearly and concisely defined? Does it answer "So what?" and "Why now?"

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Communication Audit**: [Brief overview of clarity and accessibility]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote from memo or specific section reference"
- **Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

EVIDENCE_ANALYST_PROMPT = """### ROLE
You are an Evidence & Data Analyst. You focus on whether the memo's recommendations are evidence-based, whether the cost-benefit analysis is rigorous, and whether data/facts are accurately presented and properly contextualized. You appreciate clear communication and feasibility, but bad evidence undermines good recommendations.

### OBJECTIVE
1. **Evidence Quality**: Are recommendations supported by credible evidence? Are facts accurate? Are data sources cited appropriately?
2. **Cost-Benefit Analysis**: Is there a clear, quantifiable analysis of costs and benefits? Are success metrics defined? Are alternative options evaluated against clear criteria?
3. **Transparency**: Are limitations acknowledged? Are potential weaknesses of the analysis addressed? Does the memo avoid cherry-picking data?

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Evidence Audit**: [Brief overview of evidence quality]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote, data citation, or specific analysis reference from memo"
- **Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

IMPLEMENTATION_STRATEGIST_PROMPT = """### ROLE
You are an Implementation Strategy Expert. You focus on whether recommendations are practical, feasible, and politically viable. You evaluate whether action items are specific enough, whether implementation timelines are realistic, and whether the memo addresses potential obstacles to implementation.

### OBJECTIVE
1. **Feasibility**: Are recommendations realistic given current political, economic, bureaucratic, and social constraints? Are they based on what is actually happening, not hypothetical scenarios?
2. **Action Items**: Are strategic recommendations translated into specific, practical action items? Is it clear WHO should do WHAT and WHEN?
3. **Obstacle Assessment**: Does the memo anticipate implementation challenges? Are strategies for overcoming resistance or obstacles provided?

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Feasibility Audit**: [Brief overview of implementation practicality]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote from recommendations section or analysis"
- **Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

STAKEHOLDER_ANALYST_PROMPT = """### ROLE
You are a Stakeholder Impact Analyst. You focus on identifying who benefits and who may be harmed by policy recommendations. You evaluate whether the memo acknowledges unintended consequences, assesses distributional impacts, and considers which groups will support or oppose the recommendations.

### OBJECTIVE
1. **Stakeholder Identification**: Are all relevant stakeholders identified? Does the memo clearly delineate who benefits and who may be harmed?
2. **Unintended Consequences**: Does the memo address the Law of Unintended Consequences? Are secondary effects and potential negative externalities analyzed?
3. **Political Economy**: Does the memo anticipate which groups will support or oppose recommendations? Are strategies for building coalitions or compensating losers addressed?

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Stakeholder Audit**: [Brief overview of impact assessment]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote about stakeholders or impacts from memo"
- **Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

STRATEGIC_ADVISOR_PROMPT = """### ROLE
You are a Senior Strategic Policy Advisor. You evaluate the big picture: Is this memo addressing the right problem? Are the recommendations strategic and transformative, or merely incremental? Does the analysis demonstrate sophisticated understanding of the policy landscape? You rely on your peers for technical details, but you ask: "Is this the right approach to the right problem?"

### OBJECTIVE
1. **Problem Framing**: Is the policy problem correctly identified and framed? Does the memo demonstrate understanding of root causes vs. symptoms?
2. **Strategic Value**: Are recommendations transformative or merely incremental? Do they address the core policy challenge or peripheral issues?
3. **Analytical Sophistication**: Does the analysis go beyond pro/con dichotomies? Does it identify gray areas and complexity? Is the issue analysis thorough?

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Strategic Assessment**: [Evaluate problem framing and strategic value]
- **Analytical Depth**: [Assess sophistication of analysis]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote demonstrating problem framing or recommendations"
- **Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

FINANCIAL_STABILITY_ANALYST_PROMPT = """### ROLE
You are a Financial Stability Analyst with expertise in systemic risk, macroprudential policy, and financial sector resilience. You focus on whether the memo adequately considers financial stability implications, systemic risk, and potential impacts on the banking sector, capital markets, and overall financial system health. You value clear recommendations, but your primary duty is ensuring policy proposals don't inadvertently create financial vulnerabilities.

### OBJECTIVE
1. **Systemic Risk Assessment**: Does the memo identify and analyze potential systemic risks? Are financial contagion channels considered? Does it assess impacts on financial sector interconnectedness?
2. **Financial Sector Impacts**: Are effects on banks, shadow banking, capital markets, and financial intermediation properly analyzed? Are credit conditions, liquidity, and market functioning addressed?
3. **Macroprudential Considerations**: Does the memo consider procyclicality, leverage dynamics, asset price effects, and regulatory implications? Are financial stability trade-offs with other policy objectives explicitly discussed?

### CRITICAL FINANCIAL STABILITY CONSIDERATIONS TO EVALUATE:
- **Banking Sector Resilience**: Capital adequacy, liquidity buffers, stress scenarios
- **Credit Conditions**: Impact on credit availability, lending standards, credit growth
- **Market Stability**: Effects on market liquidity, volatility, price discovery, investor confidence
- **Systemic Interconnections**: Counterparty risks, cross-border spillovers, financial contagion pathways
- **Shadow Banking & Nonbanks**: Impacts on money market funds, insurance companies, pension funds
- **Asset Prices**: Bubble formation risks, fire sale dynamics, collateral valuation issues
- **Regulatory Framework**: Compliance costs, regulatory arbitrage, unintended gaps
- **Crisis Amplification**: Potential to exacerbate financial stress during downturns
- **Moral Hazard**: Risk-taking incentives, implicit guarantees, too-big-to-fail dynamics
- **Financial Cycle Dynamics**: Procyclical effects, credit booms/busts, leverage cycles

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **Financial Stability Audit**: [Brief overview of systemic risk and financial sector analysis]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote about financial impacts or risk analysis from memo"
- **Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]

CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section.
"""

# Dictionary mapping persona names to their prompts
SYSTEM_PROMPTS = {
    "Communication_Specialist": COMMUNICATION_SPECIALIST_PROMPT,
    "Evidence_Analyst": EVIDENCE_ANALYST_PROMPT,
    "Implementation_Strategist": IMPLEMENTATION_STRATEGIST_PROMPT,
    "Stakeholder_Analyst": STAKEHOLDER_ANALYST_PROMPT,
    "Strategic_Advisor": STRATEGIC_ADVISOR_PROMPT,
    "Financial_Stability_Analyst": FINANCIAL_STABILITY_ANALYST_PROMPT
}

# ==========================================
# EMBEDDED PROMPTS - MEMO TYPE CONTEXTS
# ==========================================
MEMO_TYPE_CONTEXTS = {
    "descriptive": """MEMO TYPE CONTEXT: DESCRIPTIVE POLICY MEMO

This memo answers "What is happening?" It provides analysis of the current situation without prescribing specific policy actions.

PERSONA SELECTION GUIDANCE:
- **Evidence_Analyst** (HIGHLY RECOMMENDED): Critical for evaluating data quality, accuracy of facts, and whether evidence appropriately describes the condition.
- **Communication_Specialist**: Essential for ensuring the description is clear, accessible, and efficiently structured. High weight recommended.
- **Strategic_Advisor**: Important for assessing whether the right aspects of the situation are being described and whether context is adequate. Moderate weight.
- **Financial_Stability_Analyst**: Relevant if describing financial sector conditions, market dynamics, or systemic risk indicators. Include if memo addresses financial sector topics.
- **Stakeholder_Analyst**: Useful if describing who is affected by the current situation. Lower weight for purely descriptive memos.
- **Implementation_Strategist**: Generally not relevant for descriptive memos unless they include preliminary feasibility discussion.

TYPICAL PERSONA COMBINATIONS FOR DESCRIPTIVE MEMOS:
1. Evidence_Analyst (0.45) + Communication_Specialist (0.35) + Strategic_Advisor (0.20)
2. Communication_Specialist (0.40) + Evidence_Analyst (0.40) + Stakeholder_Analyst (0.20)
3. Evidence_Analyst (0.40) + Financial_Stability_Analyst (0.35) + Communication_Specialist (0.25) - For financial sector analysis
""",
    
    "evaluative": """MEMO TYPE CONTEXT: EVALUATIVE POLICY MEMO

This memo answers "What is working?" It evaluates the effectiveness of existing policies, programs, or interventions.

PERSONA SELECTION GUIDANCE:
- **Evidence_Analyst** (HIGHLY RECOMMENDED): Essential for evaluating the quality of evaluation data, causal claims, and whether effectiveness is properly measured. Should receive highest weight.
- **Communication_Specialist**: Important for ensuring evaluation findings are clearly presented. Moderate to high weight.
- **Strategic_Advisor**: Relevant for assessing whether the evaluation addresses the right questions and provides strategic insights. Moderate weight.
- **Stakeholder_Analyst**: Important for evaluating who benefits from existing policies and whether distributional impacts are assessed. Moderate weight recommended.
- **Financial_Stability_Analyst**: Critical if evaluating financial regulations, macroprudential policies, or financial sector programs. Essential for assessing whether policy changes affected financial stability.
- **Implementation_Strategist**: Useful if the memo discusses why programs are or aren't working from an implementation perspective. Lower weight.

TYPICAL PERSONA COMBINATIONS FOR EVALUATIVE MEMOS:
1. Evidence_Analyst (0.50) + Communication_Specialist (0.30) + Stakeholder_Analyst (0.20)
2. Evidence_Analyst (0.45) + Strategic_Advisor (0.30) + Communication_Specialist (0.25)
3. Evidence_Analyst (0.40) + Financial_Stability_Analyst (0.35) + Stakeholder_Analyst (0.25) - For financial policy evaluation
4. Evidence_Analyst (0.40) + Stakeholder_Analyst (0.30) + Implementation_Strategist (0.30)
""",
    
    "prescriptive": """MEMO TYPE CONTEXT: PRESCRIPTIVE POLICY MEMO

This memo answers "What should be done next?" It recommends specific policy actions with supporting evidence and implementation guidance.

PERSONA SELECTION GUIDANCE:
- **Implementation_Strategist** (HIGHLY RECOMMENDED): Critical for evaluating feasibility, action items, and whether recommendations can actually be executed. Should typically receive highest or second-highest weight.
- **Evidence_Analyst**: Essential for ensuring recommendations are evidence-based and that cost-benefit analysis is rigorous. Should receive high weight.
- **Stakeholder_Analyst**: Highly important for assessing unintended consequences and political viability. Essential weight recommended.
- **Financial_Stability_Analyst**: CRITICAL for any memo involving financial sector reforms, banking regulation, monetary policy, fiscal policy with financial implications, or macroprudential measures. Must be included if recommendations could affect financial stability.
- **Communication_Specialist**: Important for ensuring recommendations are clearly articulated and actionable. Moderate weight.
- **Strategic_Advisor**: Relevant for assessing whether recommendations address root causes and are strategically sound. Moderate weight.

TYPICAL PERSONA COMBINATIONS FOR PRESCRIPTIVE MEMOS:
1. Implementation_Strategist (0.35) + Evidence_Analyst (0.35) + Stakeholder_Analyst (0.30) - Balanced approach
2. Evidence_Analyst (0.40) + Implementation_Strategist (0.35) + Communication_Specialist (0.25) - Evidence-focused
3. Stakeholder_Analyst (0.35) + Implementation_Strategist (0.35) + Strategic_Advisor (0.30) - Politics-aware
4. Financial_Stability_Analyst (0.40) + Evidence_Analyst (0.35) + Implementation_Strategist (0.25) - For financial sector reforms
5. Financial_Stability_Analyst (0.35) + Stakeholder_Analyst (0.35) + Implementation_Strategist (0.30) - For macroprudential policy
"""
}

# ==========================================
# EMBEDDED PROMPTS - CUSTOM CONTEXT GUIDE
# ==========================================
CUSTOM_CONTEXT_INTEGRATION = """CUSTOM CONTEXT INTEGRATION INSTRUCTIONS

When custom evaluation context is provided by the user, integrate it as follows:

### IN PERSONA SELECTION (Round 0):
Consider the user's stated priorities when selecting personas and assigning weights. If the user emphasizes specific aspects (e.g., "focus on feasibility" or "evaluate evidence quality"), adjust persona selection and weights accordingly.

### IN ALL EVALUATION ROUNDS:
Incorporate the custom context as an additional evaluation lens alongside your role-specific criteria. The user's priorities should inform:
1. Which aspects of the memo you emphasize in your evaluation
2. The relative importance you assign to different findings
3. The specificity of your recommendations

### PRIORITY HIERARCHY:
1. Core role responsibilities (clarity for Communication Specialist, evidence for Evidence Analyst, etc.)
2. User-specified priorities from custom context
3. General policy memo best practices

### EXAMPLES:
- If user context says "Evaluate for submission to Treasury Secretary": Apply highest standards for clarity, executive-level appropriateness, and political sensitivity
- If user context says "Check readiness for Congressional briefing": Emphasize accessibility for non-experts, anticipation of tough questions, political feasibility
- If user context says "Focus on implementation barriers": Weight Implementation Strategist and Stakeholder Analyst more heavily
- If user context says "Assess financial stability risks": Ensure Financial_Stability_Analyst is included and heavily weighted

The custom context should enhance, not replace, your core evaluation criteria.
"""

# ==========================================
# PERSONA SELECTION PROMPT (ROUND 0)
# ==========================================
SELECTION_PROMPT = """
You are the Chief Editor of a policy advisory firm. You must select exactly THREE expert personas to review the provided policy memo.

The available personas are:
1. "Communication_Specialist": Focuses on clarity, accessibility, structure, and professional presentation for decision-makers.
2. "Evidence_Analyst": Focuses on evidence quality, data accuracy, cost-benefit analysis, and analytical rigor.
3. "Implementation_Strategist": Focuses on feasibility, action items, political viability, and practical implementation.
4. "Stakeholder_Analyst": Focuses on identifying beneficiaries and those harmed, unintended consequences, and political economy.
5. "Strategic_Advisor": Focuses on problem framing, strategic value of recommendations, and analytical sophistication.
6. "Financial_Stability_Analyst": Focuses on systemic risk, financial sector impacts, macroprudential considerations, and financial stability implications.

Select the 3 most crucial personas for reviewing this specific memo. Assign them weights based on their relative importance to assessing THIS SPECIFIC MEMO. The weights must sum exactly to 1.0.

**IMPORTANT**: If the memo addresses financial sector policy, banking regulation, monetary policy, fiscal policy with financial implications, macroprudential measures, or could materially affect financial stability, you MUST include Financial_Stability_Analyst as one of the three personas.

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
    Engage in cross-domain examination. You respect their domains and want to synthesize perspectives to collectively determine whether this memo is ready for decision-makers. If a peer praised something your domain proves flawed, push back and point it out.

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
    Read the transcript carefully. Identify the specific questions directed AT YOU by your peers. Answer them directly, providing context and TEXTUAL EVIDENCE from the memo to address the concerns.

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
    As the {role}, submit your Final Amended Report. Update your prior assessment based on valid peer critiques and their answers to your questions. Ensure your verdict reflects issue severity weighting and cross-domain insights.

    ### OUTPUT FORMAT
    - **Insights Absorbed**: [How the debate changed your evaluation]
    - **Final Verdict**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]
    - **Final Rationale**: [3-sentence justification explicitly incorporating debate context]
    """,

    "Round_3_Editor": """
    ### ROLE
    You are the Senior Editorial Director. Your job is to calculate the endogenous weighted consensus of the panel and write the final decision letter to the memo author and their client.

    ### PANEL CONTEXT & WEIGHTS
    The following personas were selected for this memo, with these specific weights:
    {weights_json}

    ### AMENDED REPORTS
    {final_reports_text}

    ### THE ENDOGENOUS WEIGHTING SYSTEM (STRICT INSTRUCTIONS)
    Do not use a "Kill Switch" or veto unless explicitly justified. You must calculate the mathematical consensus.
    1. Assign values to verdicts: ACCEPT WITH MINOR REVISIONS = 1.0, MAJOR REVISION REQUIRED = 0.5, REJECT = 0.0.
    2. Multiply each persona's value by their assigned weight.
    3. Sum the weighted values to get the Final Consensus Score (out of 1.0).
    4. Decision Thresholds:
       - Score > 0.75 : ACCEPT WITH MINOR REVISIONS
       - 0.40 <= Score <= 0.75 : MAJOR REVISION REQUIRED
       - Score < 0.40 : REJECT

    ### OUTPUT FORMAT
    - **Weight Calculation**: [Show your math explicitly based on the panel's final verdicts]
    - **Debate Synthesis**: [2-3 sentences summarizing the panel's final alignment]
    - **Final Decision**: [ACCEPT WITH MINOR REVISIONS / MAJOR REVISION REQUIRED / REJECT]
    - **Official Evaluation Report**: [A synthesized letter to the authors drawing ONLY from the panel's findings. Detail required revisions or reasons for rejection WITH TEXTUAL/CITED EVIDENCE from the memo.]
    """
}

# ==========================================
# ORCHESTRATION FUNCTIONS
# ==========================================
async def call_llm_async(
    system_prompt: str,
    user_prompt: str,
    role: str,
    memo_text: str,
    custom_context: Optional[str] = None
) -> str:
    """
    Async wrapper for LLM calls.

    Args:
        system_prompt: The role-specific system prompt
        user_prompt: The round-specific user prompt
        role: The persona name
        memo_text: The policy memo text
        custom_context: Optional user-provided evaluation priorities

    Returns:
        LLM response string
    """
    # Build the full prompt
    full_prompt = user_prompt

    # Add custom context if provided
    if custom_context and custom_context.strip():
        full_prompt += f"\n\n{CUSTOM_CONTEXT_INTEGRATION}\n\nUSER EVALUATION PRIORITIES:\n{custom_context}\n"

    full_prompt += f"\n\nPOLICY MEMO TEXT:\n{memo_text}"

    # Call the LLM (running in thread to avoid blocking)
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    return await asyncio.to_thread(single_query, combined_prompt)

async def run_round_0_selection(
    memo_text: str,
    memo_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None
) -> dict:
    """
    Round 0: Dynamically selects the 3 most relevant personas and their weights.

    Args:
        memo_text: The policy memo to evaluate
        memo_type: Optional memo type (descriptive/evaluative/prescriptive) for context
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

            # Add memo type context if available
            if memo_type and memo_type in MEMO_TYPE_CONTEXTS:
                weight_prompt += f"\n{MEMO_TYPE_CONTEXTS[memo_type]}\n\n"

            # Add custom context if available
            if custom_context and custom_context.strip():
                weight_prompt += f"\nUSER EVALUATION PRIORITIES:\n{custom_context}\n\n"

            weight_prompt += f"\nPOLICY MEMO TEXT:\n{memo_text}\n\n"
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

    # Add memo type context if available
    if memo_type and memo_type in MEMO_TYPE_CONTEXTS:
        selection_prompt += f"{MEMO_TYPE_CONTEXTS[memo_type]}\n\n"

    # Add custom context if available
    if custom_context and custom_context.strip():
        selection_prompt += f"USER EVALUATION PRIORITIES:\n{custom_context}\n\n"

    selection_prompt += f"POLICY MEMO TEXT:\n{memo_text}"
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
        print(f"[Round 0] Failed to parse selection. Defaulting to Evidence_Analyst, Communication_Specialist, Implementation_Strategist. Error: {e}")
        return {
            "selected_personas": ["Evidence_Analyst", "Communication_Specialist", "Implementation_Strategist"],
            "weights": {"Evidence_Analyst": 0.4, "Communication_Specialist": 0.3, "Implementation_Strategist": 0.3},
            "justification": "Default selection due to parsing error."
        }

async def run_round_1(active_personas: list, memo_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 1: Independent evaluation by selected agents."""
    user_prompt = "Please read the attached policy memo and provide your Round 1 evaluation based on your role."

    tasks = {}
    for role in active_personas:
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], user_prompt, role, memo_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

async def run_round_2a(r1_reports: Dict[str, str], active_personas: list, memo_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
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
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2a, role, memo_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

async def run_round_2b(r2a_reports: Dict[str, str], active_personas: list, memo_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
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
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2b, role, memo_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

async def run_round_2c(r1_reports: Dict[str, str], r2a_reports: Dict[str, str],
                       r2b_reports: Dict[str, str], active_personas: list, memo_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
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
        tasks[role] = call_llm_async(SYSTEM_PROMPTS[role], prompt_2c, role, memo_text, custom_context)

    results = await asyncio.gather(*tasks.values())
    return dict(zip(tasks.keys(), results))

def extract_verdict_from_report(report: str) -> str:
    """Extract the final verdict from a Round 2C report."""
    # Try to find "Final Verdict:" first
    final_verdict_patterns = [
        r'\*\*Final Verdict\*\*\s*:+\s*(ACCEPT WITH MINOR REVISIONS|MAJOR REVISION REQUIRED|REJECT)',
        r'Final Verdict\s*:+\s*(ACCEPT WITH MINOR REVISIONS|MAJOR REVISION REQUIRED|REJECT)',
    ]

    for pattern in final_verdict_patterns:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback to any verdict
    generic_patterns = [
        r'\*\*Verdict\*\*\s*:+\s*(ACCEPT WITH MINOR REVISIONS|MAJOR REVISION REQUIRED|REJECT)',
        r'Verdict\s*:+\s*(ACCEPT WITH MINOR REVISIONS|MAJOR REVISION REQUIRED|REJECT)',
    ]

    for pattern in generic_patterns:
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find all and take the last one
    all_matches = re.findall(r'\b(ACCEPT WITH MINOR REVISIONS|MAJOR REVISION REQUIRED|REJECT)\b', report, re.IGNORECASE)
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
    verdict_values = {
        "ACCEPT WITH MINOR REVISIONS": 1.0,
        "MAJOR REVISION REQUIRED": 0.5,
        "REJECT": 0.0,
        "UNKNOWN": 0.0
    }
    weighted_score = 0.0

    for role, verdict in verdicts.items():
        weight = weights.get(role, 0)
        value = verdict_values.get(verdict, 0.0)
        weighted_score += weight * value

    # Determine decision based on thresholds
    if weighted_score > 0.75:
        decision = "ACCEPT WITH MINOR REVISIONS"
    elif weighted_score >= 0.40:
        decision = "MAJOR REVISION REQUIRED"
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

    system_prompt = "You are the Senior Editorial Director. Follow the mathematical weighting instructions strictly."
    return await asyncio.to_thread(single_query, f"{system_prompt}\n\n{prompt_3}")

def calculate_token_usage_and_cost(memo_text: str, results: Dict, num_personas: int = 3) -> Dict:
    """
    Calculate estimated token usage and cost for the debate pipeline.

    Args:
        memo_text: The full policy memo text
        results: The debate results dictionary
        num_personas: Number of active personas (default 3)

    Returns:
        Dictionary with token counts and cost estimates
    """
    # Count memo tokens
    memo_tokens = count_tokens(memo_text)

    # Estimate input tokens for each round
    # Round 0: memo only
    round_0_input = memo_tokens + 500  # selection prompt overhead

    # Round 1: memo + system prompt per persona
    round_1_input = num_personas * (memo_tokens + 1000)

    # Round 2A: memo + system prompt + Round 1 reports per persona
    r1_output_tokens = sum(count_tokens(report) for report in results.get('round_1', {}).values())
    round_2a_input = num_personas * (memo_tokens + 1000 + r1_output_tokens)

    # Round 2B: memo + system prompt + Round 2A transcript per persona
    r2a_output_tokens = sum(count_tokens(report) for report in results.get('round_2a', {}).values())
    round_2b_input = num_personas * (memo_tokens + 1000 + r2a_output_tokens)

    # Round 2C: memo + system prompt + full debate transcript per persona
    r2b_output_tokens = sum(count_tokens(report) for report in results.get('round_2b', {}).values())
    full_transcript_tokens = r1_output_tokens + r2a_output_tokens + r2b_output_tokens
    round_2c_input = num_personas * (memo_tokens + 1000 + full_transcript_tokens)

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
        'memo_tokens': memo_tokens,
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
    memo_text: str,
    progress_callback=None,
    memo_context: str = None,
    model_key: str = None,
    temperature: float = None,
    memo_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None
):
    """
    Orchestrates the entire multi-agent debate workflow for policy memo evaluation.

    Args:
        memo_text: The full text of the policy memo to evaluate
        progress_callback: Optional callback function to report progress
        memo_context: Optional additional context about the memo (deprecated, use custom_context)
        model_key: Optional model selection key (currently unused)
        temperature: Optional temperature setting (currently unused)
        memo_type: Optional memo type (descriptive/evaluative/prescriptive) for persona selection guidance
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
        progress_callback("Round 0: Selecting Evaluators", 0.05)
    selection_data = await run_round_0_selection(
        memo_text,
        memo_type=memo_type,
        custom_context=custom_context,
        manual_personas=manual_personas,
        manual_weights=manual_weights
    )
    active_personas = selection_data["selected_personas"]
    results['round_0'] = selection_data

    # Round 1: Independent Evaluation
    if progress_callback:
        progress_callback("Round 1: Independent Evaluation", 0.15)
    results['round_1'] = await run_round_1(active_personas, memo_text, custom_context)

    # Round 2A: Cross-Examination
    if progress_callback:
        progress_callback("Round 2A: Cross-Examination", 0.35)
    results['round_2a'] = await run_round_2a(results['round_1'], active_personas, memo_text, custom_context)

    # Round 2B: Direct Examination
    if progress_callback:
        progress_callback("Round 2B: Answering Questions", 0.55)
    results['round_2b'] = await run_round_2b(results['round_2a'], active_personas, memo_text, custom_context)

    # Round 2C: Final Amendments
    if progress_callback:
        progress_callback("Round 2C: Final Amendments", 0.75)
    results['round_2c'] = await run_round_2c(
        results['round_1'],
        results['round_2a'],
        results['round_2b'],
        active_personas,
        memo_text,
        custom_context
    )

    # Calculate consensus before Round 3
    results['consensus'] = calculate_consensus(results['round_2c'], selection_data)

    # Round 3: Editor Decision
    if progress_callback:
        progress_callback("Round 3: Editorial Decision", 0.90)
    results['final_decision'] = await run_round_3(results['round_2c'], selection_data)

    # Capture end time and add metadata
    end_time = datetime.datetime.now()
    runtime_seconds = (end_time - start_time).total_seconds()

    # Calculate token usage and cost
    token_cost_data = calculate_token_usage_and_cost(
        memo_text,
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
        'prompt_versions': 'v1.0 (policy memo evaluation with financial stability)',
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

    parser = argparse.ArgumentParser(description="Run Multi-Agent Debate on a policy memo")
    parser.add_argument("memo_file", help="Path to policy memo text file")
    parser.add_argument("--memo-type", choices=["descriptive", "evaluative", "prescriptive"],
                        help="Type of memo for persona selection guidance")
    parser.add_argument("--custom-context", help="Custom evaluation context/priorities")
    parser.add_argument("--output", default="memo_evaluation_results.json",
                        help="Output file for results (default: memo_evaluation_results.json)")

    args = parser.parse_args()

    # Read memo
    with open(args.memo_file, 'r', encoding='utf-8') as f:
        memo_text = f.read()

    # Progress callback
    def progress(msg, pct):
        print(f"[{int(pct*100):3d}%] {msg}")

    # Run debate
    print(f"\n{'='*60}")
    print("Multi-Agent Debate - Policy Memo Evaluation")
    print(f"{'='*60}\n")
    print(f"Memo: {args.memo_file}")
    print(f"Memo Type: {args.memo_type or 'Auto-detect'}")
    print(f"Output: {args.output}\n")

    results = asyncio.run(execute_debate_pipeline(
        memo_text,
        progress_callback=progress,
        memo_type=args.memo_type,
        custom_context=args.custom_context
    ))

    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Selected Evaluators: {', '.join(results['round_0']['selected_personas'])}")
    print(f"Weights: {results['round_0']['weights']}")
    print(f"\nConsensus Score: {results['consensus']['weighted_score']:.3f}")
    print(f"Final Decision: {results['consensus']['decision']}")
    print(f"\nRuntime: {results['metadata']['total_runtime_formatted']}")
    print(f"Total Cost: ${results['metadata']['token_usage']['cost_usd']['total']:.4f}")
    print(f"\nFull results saved to: {args.output}")