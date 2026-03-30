#!/usr/bin/env python3
"""
Test the deterministic consensus calculation with the bug scenario.
Run from app_system/: python test_consensus_calculation.py
"""

from multi_agent_debate import compute_weighted_consensus

# Simulate the bug scenario from the user's report
mock_r2c_reports = {
    'Empiricist': """
**Insights Absorbed:**
The Theorist and Visionary confirmed my concerns about statistical inference.

**Changes to Original Assessment:**
My verdict has changed from PASS to FAIL based on fatal methodological flaws.

**Final Verdict:** FAIL

**Final Rationale:**
The paper contains three fatal flaws that invalidate its core empirical claims.
    """,

    'Theorist': """
**Insights Absorbed:**
The Empiricist exposed critical flaws I initially underweighted.

**Changes to Original Assessment:**
I maintain REVISE but recognize the severity is higher than initially assessed.

**Final Verdict:** REVISE

**Final Rationale:**
Complete absence of statistical inference is a major flaw requiring substantial revision.
    """,

    'Visionary': """
**Insights Absorbed:**
The Empiricist and Theorist provided devastating critiques.

**Changes to Original Assessment:**
My verdict has changed from PASS to FAIL.

**Final Verdict:** FAIL

**Final Rationale:**
Methodologically fatal - the central claims cannot be substantiated.
    """
}

weights = {
    'Empiricist': 0.45,
    'Theorist': 0.35,
    'Visionary': 0.20
}

# Compute consensus
print("=" * 70)
print("TESTING CONSENSUS CALCULATION - BUG SCENARIO")
print("=" * 70)

consensus = compute_weighted_consensus(mock_r2c_reports, weights)

print("\n📊 EXTRACTED VERDICTS:")
for persona, verdict in consensus['verdicts'].items():
    print(f"  {persona}: {verdict}")

print("\n🔢 CALCULATION:")
for detail in consensus['calculation_details']:
    print(f"  {detail}")

print(f"\n📈 WEIGHTED SCORE: {consensus['weighted_score']:.3f}")

print("\n📋 DECISION THRESHOLDS:")
print("  > 0.75: ACCEPT")
print("  0.40 - 0.75: REJECT AND RESUBMIT")
print("  < 0.40: REJECT")

print(f"\n✅ COMPUTED DECISION: {consensus['decision']}")

# Verify the expected result
expected_score = (0.45 * 0.0) + (0.35 * 0.5) + (0.20 * 0.0)
expected_decision = "REJECT"

print("\n" + "=" * 70)
print("VERIFICATION:")
print("=" * 70)
print(f"Expected Score: {expected_score:.3f}")
print(f"Actual Score: {consensus['weighted_score']:.3f}")
print(f"Match: {'✅ YES' if abs(expected_score - consensus['weighted_score']) < 0.001 else '❌ NO'}")

print(f"\nExpected Decision: {expected_decision}")
print(f"Actual Decision: {consensus['decision']}")
print(f"Match: {'✅ YES' if consensus['decision'] == expected_decision else '❌ NO'}")

if consensus['decision'] == expected_decision and abs(expected_score - consensus['weighted_score']) < 0.001:
    print("\n🎉 TEST PASSED! Consensus calculation is deterministic and correct.")
else:
    print("\n❌ TEST FAILED! Consensus calculation has issues.")
