"""
Memo-specific Multi-Agent Debate Engine

This module wraps the standard debate engine with memo-specific prompts.
It uses the same orchestration logic but with memo evaluation personas.
"""

import asyncio
import datetime
import json
import re
from typing import Dict, List, Optional
from pathlib import Path

from utils import single_query, count_tokens
from referee.memo_prompts import (
    MEMO_SYSTEM_PROMPTS,
    MEMO_SELECTION_PROMPT,
    MEMO_TYPE_CONTEXTS,
    MEMO_CUSTOM_CONTEXT_INTEGRATION,
    ISSUE_SEVERITY_GUIDE
)

# Import debate round prompts from standard engine (these are domain-agnostic)
from referee.engine import DEBATE_PROMPTS

# Use memo-specific prompts
SYSTEM_PROMPTS = MEMO_SYSTEM_PROMPTS
SELECTION_PROMPT = MEMO_SELECTION_PROMPT


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
        memo_text: The memo text
        custom_context: Optional user-provided evaluation priorities

    Returns:
        LLM response string
    """
    # Build the full prompt
    full_prompt = user_prompt

    # Add custom context if provided
    if custom_context and custom_context.strip():
        full_prompt += f"\n\n{MEMO_CUSTOM_CONTEXT_INTEGRATION}\n\nUSER EVALUATION PRIORITIES:\n{custom_context}\n"

    full_prompt += f"\n\nMEMO TEXT:\n{memo_text}"

    # Call the LLM (running in thread to avoid blocking)
    combined_prompt = f"{system_prompt}\n\n{full_prompt}"
    return await asyncio.to_thread(single_query, combined_prompt)


def load_memo_type_context(memo_type: str) -> str:
    """Load memo type context guidance for analyst selection."""
    if not memo_type or memo_type not in MEMO_TYPE_CONTEXTS:
        return ""
    return MEMO_TYPE_CONTEXTS[memo_type]


async def run_round_0_selection(
    memo_text: str,
    memo_type: Optional[str] = None,
    custom_context: Optional[str] = None,
    manual_personas: Optional[List[str]] = None,
    manual_weights: Optional[Dict[str, float]] = None
) -> dict:
    """
    Round 0: Dynamically selects the 3 most relevant analysts and their weights.

    Args:
        memo_text: The memo to evaluate
        memo_type: Optional memo type (policy_recommendation/analytical_briefing/decision_memo)
        custom_context: Optional user-provided evaluation priorities
        manual_personas: If provided, skip LLM selection and use these personas
        manual_weights: If provided with manual_personas, use these exact weights

    Returns:
        Dictionary with selected_personas, weights, and justification
    """
    print("[Round 0] Starting analyst selection...")

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
            weight_prompt += f"The user has pre-selected these analysts: {', '.join(manual_personas)}\n"
            weight_prompt += "Your task is ONLY to assign appropriate weights to these analysts (must sum to 1.0).\n\n"

            # Add memo type context if available
            if memo_type:
                memo_context = load_memo_type_context(memo_type)
                if memo_context:
                    weight_prompt += f"\n{memo_context}\n\n"

            # Add custom context if available
            if custom_context and custom_context.strip():
                weight_prompt += f"\nUSER EVALUATION PRIORITIES:\n{custom_context}\n\n"

            weight_prompt += f"\nMEMO TEXT:\n{memo_text}\n\n"
            weight_prompt += f"OUTPUT FORMAT: Return ONLY valid JSON with these analysts {manual_personas} and their weights."

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
                        "justification": data.get("justification", "Weights assigned by LLM for user-selected analysts.")
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
    if memo_type:
        memo_context = load_memo_type_context(memo_type)
        if memo_context:
            selection_prompt += f"{memo_context}\n\n"

    # Add custom context if available
    if custom_context and custom_context.strip():
        selection_prompt += f"USER EVALUATION PRIORITIES:\n{custom_context}\n\n"

    selection_prompt += f"MEMO TEXT:\n{memo_text}"
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
            raise ValueError("LLM did not select exactly 3 analysts.")

        print(f"[Round 0] Selected analysts: {personas}")
        print(f"[Round 0] Weights: {weights}")
        return selection_data

    except Exception as e:
        print(f"[Round 0] Failed to parse selection. Defaulting to Policy Analyst, Data Analyst, Implementation Analyst. Error: {e}")
        return {
            "selected_personas": ["Policy Analyst", "Data Analyst", "Implementation Analyst"],
            "weights": {"Policy Analyst": 0.4, "Data Analyst": 0.3, "Implementation Analyst": 0.3},
            "justification": "Default selection due to parsing error."
        }


async def run_round_1(active_personas: list, memo_text: str, custom_context: Optional[str] = None) -> Dict[str, str]:
    """Round 1: Independent evaluation by selected agents."""
    user_prompt = "Please read the attached memo and provide your Round 1 evaluation based on your role."

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
    """Round 3: Senior reviewer synthesizes with weighted consensus."""
    weights_json = json.dumps(selection_data["weights"], indent=2)

    final_reports_text = ""
    for role, report in r2c_reports.items():
        final_reports_text += f"\n--- {role.upper()} FINAL REPORT ---\n{report}\n"

    prompt_3 = DEBATE_PROMPTS["Round_3_Editor"].format(
        weights_json=weights_json,
        final_reports_text=final_reports_text
    )

    system_prompt = "You are a Senior Executive reviewing the memo evaluation. Follow the mathematical weighting instructions strictly."
    return await asyncio.to_thread(single_query, f"{system_prompt}\n\n{prompt_3}")


def calculate_token_usage_and_cost(memo_text: str, results: Dict, num_personas: int = 3) -> Dict:
    """
    Calculate estimated token usage and cost for the debate pipeline.

    Args:
        memo_text: The full memo text
        results: The debate results dictionary
        num_personas: Number of active personas (default 3)

    Returns:
        Dictionary with token counts and cost estimates
    """
    # Count memo tokens
    memo_tokens = count_tokens(memo_text)

    # Estimate input tokens for each round
    round_0_input = memo_tokens + 500
    round_1_input = num_personas * (memo_tokens + 1000)

    r1_output_tokens = sum(count_tokens(report) for report in results.get('round_1', {}).values())
    round_2a_input = num_personas * (memo_tokens + 1000 + r1_output_tokens)

    r2a_output_tokens = sum(count_tokens(report) for report in results.get('round_2a', {}).values())
    round_2b_input = num_personas * (memo_tokens + 1000 + r2a_output_tokens)

    r2b_output_tokens = sum(count_tokens(report) for report in results.get('round_2b', {}).values())
    full_transcript_tokens = r1_output_tokens + r2a_output_tokens + r2b_output_tokens
    round_2c_input = num_personas * (memo_tokens + 1000 + full_transcript_tokens)

    r2c_output_tokens = sum(count_tokens(report) for report in results.get('round_2c', {}).values())
    round_3_input = r2c_output_tokens + 1500

    total_input_tokens = (round_0_input + round_1_input + round_2a_input +
                         round_2b_input + round_2c_input + round_3_input)

    editor_report_tokens = count_tokens(results.get('final_decision', ''))
    total_output_tokens = (r1_output_tokens + r2a_output_tokens + r2b_output_tokens +
                          r2c_output_tokens + editor_report_tokens)

    # Cost calculation (Claude 4.5 Sonnet pricing)
    input_cost_per_million = 3.00
    output_cost_per_million = 15.00

    input_cost = (total_input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (total_output_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost

    debate_calls = 1 + (num_personas * 4) + 1
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
            'model': 'Claude 4.5 Sonnet'
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
    Orchestrates the entire multi-agent debate workflow for memo evaluation.

    Args:
        memo_text: The full text of the memo to evaluate
        progress_callback: Optional callback function to report progress
        memo_context: Optional additional context about the memo (deprecated, use custom_context)
        model_key: Optional model selection key (currently unused)
        temperature: Optional temperature setting (currently unused)
        memo_type: Optional memo type for analyst selection guidance
        custom_context: Optional user-provided evaluation priorities and focus areas
        manual_personas: Optional list of manually selected personas (2-5 personas)
        manual_weights: Optional dict of manually specified weights for manual_personas

    Returns:
        Dictionary containing all round results, selection data, and final decision
    """
    # Capture start time
    start_time = datetime.datetime.now()

    results = {}

    # Round 0: Analyst Selection
    if progress_callback:
        progress_callback("Round 0: Selecting Analysts", 0.05)
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

    # Round 3: Senior Reviewer Decision
    if progress_callback:
        progress_callback("Round 3: Senior Reviewer Decision", 0.90)
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
        'model_version': 'Claude 4.5 Sonnet',
        'temperature': temperature if temperature is not None else 1.0,
        'thinking_enabled': True,
        'thinking_budget_tokens': 2048,
        'max_retries': 3,
        'retry_delay_seconds': 5,
        'prompt_versions': 'v1.0 (memo-specific)',
        'token_usage': token_cost_data
    }

    if progress_callback:
        progress_callback("Complete!", 1.0)

    return results
