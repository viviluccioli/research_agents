"""
debate_summarizer.py - Compress debate outputs for cleaner UI display

This module provides LLM-powered summarization of multi-agent debate results.
Each round's output is condensed to key points while preserving full details.
"""

import asyncio
import re
from typing import Dict, List
from utils import single_query


def summarize_round_1_report(persona: str, full_report: str) -> Dict[str, str]:
    """
    Summarize a Round 1 independent evaluation to 3-4 key bullet points.

    Returns:
        Dict with 'summary' (3-4 bullets) and 'verdict'
    """
    prompt = f"""You are summarizing a peer review report for UI display. Extract ONLY the most critical points.

FULL REPORT FROM {persona.upper()}:
{full_report}

Your task: Create a 3-4 bullet point summary capturing:
1. Overall assessment (1 sentence)
2. Most severe issues (1-2 bullets with severity labels)
3. Verdict

OUTPUT FORMAT:
**Summary:**
- [Overall assessment in 1 sentence]
- [Most critical issue with [SEVERITY] label]
- [Second most critical issue with [SEVERITY] label if applicable]

**Verdict:** [PASS/REVISE/FAIL]

Be CONCISE. Each bullet should be 1 sentence max. Focus on actionable findings only.
"""

    summary = single_query(prompt, max_tokens=1024)

    # Extract verdict from "Verdict:" line specifically
    verdict = "UNKNOWN"

    # Try to extract from "Verdict:" line specifically
    verdict_patterns = [
        r'\*\*Verdict\*\*\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'Verdict\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'\*\*Verdict:\*\*\s+(PASS|REVISE|REJECT|FAIL)',
    ]

    for pattern in verdict_patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            break

    # Fallback: if still unknown, look for last occurrence of verdict words
    if verdict == "UNKNOWN":
        # Find all occurrences and take the last one
        all_matches = re.findall(r'\b(PASS|REVISE|FAIL|REJECT)\b', summary.upper())
        if all_matches:
            verdict = all_matches[-1]

    return {
        'summary': summary,
        'verdict': verdict
    }


def summarize_round_2a_report(persona: str, full_report: str) -> str:
    """
    Summarize a Round 2A cross-examination to 2-3 key points.

    Returns:
        Compressed markdown string
    """
    prompt = f"""You are summarizing a peer cross-examination for UI display. Extract ONLY the most critical challenges.

FULL CROSS-EXAMINATION FROM {persona.upper()}:
{full_report}

Your task: Create a 2-3 bullet point summary capturing:
1. Main insight from other reviews (1 bullet)
2. Key challenge or question raised (1-2 bullets)

OUTPUT FORMAT:
**Key Points:**
- [Main cross-domain insight in 1 sentence]
- [Most important challenge or question in 1 sentence]

Be EXTREMELY CONCISE. Focus only on what matters for understanding the debate flow.
"""

    return single_query(prompt, max_tokens=512)


def summarize_round_2b_report(persona: str, full_report: str) -> str:
    """
    Summarize a Round 2B response to 2-3 key points.

    Returns:
        Compressed markdown string
    """
    prompt = f"""You are summarizing peer responses for UI display. Extract ONLY the key position.

FULL RESPONSES FROM {persona.upper()}:
{full_report}

Your task: Create a 2-3 bullet point summary capturing:
1. Main position (concede or defend)
2. Key reasoning (1-2 bullets)

OUTPUT FORMAT:
**Position:** [CONCEDE or DEFEND key points]
- [Main reasoning in 1 sentence]
- [Supporting point in 1 sentence if applicable]

Be EXTREMELY CONCISE.
"""

    return single_query(prompt, max_tokens=512)


def summarize_round_2c_report(persona: str, full_report: str) -> Dict[str, str]:
    """
    Summarize a Round 2C final amendment to key insights and verdict change.

    Returns:
        Dict with 'summary' and 'verdict'
    """
    prompt = f"""You are summarizing a final amended verdict for UI display. Extract ONLY the essential changes.

FULL FINAL REPORT FROM {persona.upper()}:
{full_report}

Your task: Create a 2-3 bullet point summary capturing:
1. Did verdict change? From what to what?
2. Key reason for change (or maintaining position)

OUTPUT FORMAT:
**Verdict Change:** [OLD → NEW, or "Maintained: VERDICT"]
- [1-sentence explanation of why it changed or held]
- [1-sentence key finding from debate that influenced decision]

**Final Verdict:** [PASS/REVISE/FAIL]

Be EXTREMELY CONCISE.
"""

    summary = single_query(prompt, max_tokens=768)

    # Extract verdict from "Final Verdict:" line specifically (not from "Verdict Change:" line)
    verdict = "UNKNOWN"

    # Try to extract from "Final Verdict:" line specifically
    final_verdict_patterns = [
        r'\*\*Final Verdict\*\*\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'Final Verdict\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'\*\*Final Verdict:\*\*\s+(PASS|REVISE|REJECT|FAIL)',
    ]

    for pattern in final_verdict_patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            verdict = match.group(1).upper()
            break

    # Fallback: if still unknown, look for last occurrence of verdict words
    if verdict == "UNKNOWN":
        # Find all occurrences and take the last one (most likely the final verdict)
        all_matches = re.findall(r'\b(PASS|REVISE|FAIL|REJECT)\b', summary.upper())
        if all_matches:
            verdict = all_matches[-1]

    return {
        'summary': summary,
        'verdict': verdict
    }


def summarize_editor_report(full_report: str, consensus_decision: str, consensus_score: float) -> str:
    """
    Summarize editor's report to essential decision rationale.

    Returns:
        Compressed markdown string
    """
    prompt = f"""You are summarizing an editor's final decision for UI display.

DECISION: {consensus_decision} (Score: {consensus_score:.3f})

FULL EDITOR REPORT:
{full_report}

Your task: Create a concise summary with:
1. Top 2-3 strengths (1 bullet each, max 1 sentence)
2. Top 2-3 critical issues (1 bullet each, max 1 sentence)
3. Recommendation (1 sentence)

OUTPUT FORMAT:
**Decision:** {consensus_decision}

**Strengths:**
- [Strength 1]
- [Strength 2]

**Critical Issues:**
- [Issue 1]
- [Issue 2]
- [Issue 3]

**Recommendation:** [1 sentence next steps]

Focus on ACTIONABLE items. Be EXTREMELY CONCISE.
"""

    return single_query(prompt, max_tokens=1536)


async def summarize_all_rounds(results: Dict) -> Dict[str, any]:
    """
    Asynchronously summarize all debate rounds for cleaner UI display.

    Args:
        results: Full debate results from execute_debate_pipeline()

    Returns:
        Dict with same structure but 'summary' fields added to each round
    """
    active_personas = results['round_0']['selected_personas']

    # Build summary tasks
    tasks = {}

    # Round 1 summaries
    for persona in active_personas:
        tasks[f'r1_{persona}'] = asyncio.to_thread(
            summarize_round_1_report,
            persona,
            results['round_1'][persona]
        )

    # Round 2A summaries
    for persona in active_personas:
        tasks[f'r2a_{persona}'] = asyncio.to_thread(
            summarize_round_2a_report,
            persona,
            results['round_2a'][persona]
        )

    # Round 2B summaries
    for persona in active_personas:
        tasks[f'r2b_{persona}'] = asyncio.to_thread(
            summarize_round_2b_report,
            persona,
            results['round_2b'][persona]
        )

    # Round 2C summaries
    for persona in active_personas:
        tasks[f'r2c_{persona}'] = asyncio.to_thread(
            summarize_round_2c_report,
            persona,
            results['round_2c'][persona]
        )

    # Editor summary
    tasks['editor'] = asyncio.to_thread(
        summarize_editor_report,
        results['final_decision'],
        results['consensus']['decision'],
        results['consensus']['weighted_score']
    )

    # Execute all summarizations in parallel
    print("[Summarizer] Compressing debate outputs for UI...")
    summaries = await asyncio.gather(*[t for t in tasks.values()])
    summary_map = dict(zip(tasks.keys(), summaries))

    # Build summary structure
    summarized = {
        'round_1_summaries': {p: summary_map[f'r1_{p}'] for p in active_personas},
        'round_2a_summaries': {p: summary_map[f'r2a_{p}'] for p in active_personas},
        'round_2b_summaries': {p: summary_map[f'r2b_{p}'] for p in active_personas},
        'round_2c_summaries': {p: summary_map[f'r2c_{p}'] for p in active_personas},
        'editor_summary': summary_map['editor']
    }

    print("[Summarizer] Compression complete!")
    return summarized
