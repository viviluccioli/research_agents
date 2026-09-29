#!/usr/bin/env python3
"""
Quick test script for referee report with minimal paper.
Run from app_system/: python test_referee_quick.py
"""
import asyncio
from multi_agent_debate import execute_debate_pipeline
import json

# Minimal test paper (just 2 paragraphs)
MINIMAL_PAPER = """
Title: The Effect of Monetary Policy on Inflation

This paper examines the relationship between monetary policy and inflation using a simple regression model.
We find that interest rate increases are associated with lower inflation rates in the following quarter.

Our data comes from the Federal Reserve Economic Data (FRED) covering 2000-2020. We use OLS regression
with robust standard errors. The results are statistically significant at the 5% level.
"""

async def test_minimal_paper():
    print("=" * 60)
    print("Testing Referee Report with Minimal Paper")
    print("=" * 60)
    print(f"\nPaper length: {len(MINIMAL_PAPER)} characters")
    print("\nRunning debate pipeline...")

    results = await execute_debate_pipeline(MINIMAL_PAPER)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    # Show metadata
    if 'metadata' in results:
        print("\n📊 METADATA:")
        print(json.dumps(results['metadata'], indent=2))

    # Show Round 0 selection
    print("\n🎯 ROUND 0 - Selected Personas:")
    print(f"  Personas: {results['round_0']['selected_personas']}")
    print(f"  Weights: {results['round_0']['weights']}")

    # Show Round 1 verdicts
    print("\n⚡ ROUND 1 - Initial Verdicts:")
    for persona, report in results['round_1'].items():
        verdict = "UNKNOWN"
        for v in ["PASS", "REVISE", "FAIL", "REJECT"]:
            if v in report.upper():
                verdict = v
                break
        print(f"  {persona}: {verdict} ({len(report)} chars)")

    # Check Round 2C outputs (the key test)
    print("\n⚖️ ROUND 2C - Final Verdicts (KEY TEST):")
    for persona, report in results['round_2c'].items():
        is_blank = len(report.strip()) < 20
        verdict = "UNKNOWN"
        for v in ["PASS", "REVISE", "FAIL", "REJECT"]:
            if v in report.upper():
                verdict = v
                break

        status = "❌ BLANK" if is_blank else "✅ OK"
        print(f"  {persona}: {status} - {verdict} ({len(report)} chars)")

        if is_blank:
            print(f"    WARNING: Output too short!")
        else:
            # Check for required sections
            has_insights = "Insights Absorbed" in report or "insights absorbed" in report.lower()
            has_changes = "Changes to Original" in report or "changes" in report.lower()
            has_verdict = "Final Verdict" in report or "verdict" in report
            has_rationale = "Final Rationale" in report or "rationale" in report.lower()

            print(f"    Sections: Insights={has_insights}, Changes={has_changes}, Verdict={has_verdict}, Rationale={has_rationale}")

    # Show final decision
    print("\n📜 ROUND 3 - Final Decision:")
    final = results['final_decision']
    for decision in ["ACCEPT", "REJECT AND RESUBMIT", "REJECT"]:
        if decision in final.upper():
            print(f"  Decision: {decision}")
            break

    print(f"\n✅ Test completed! Total runtime: {results['metadata'].get('total_runtime_formatted', 'N/A')}")

    return results

if __name__ == "__main__":
    results = asyncio.run(test_minimal_paper())
