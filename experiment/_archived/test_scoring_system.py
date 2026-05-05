#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Scoring System Implementation

Validates that:
1. Score extraction works on sample reports
2. CSV columns are properly configured
3. Consensus calculation handles scores correctly
4. Backwards compatibility is maintained
"""

import sys
from pathlib import Path

# Add app_system to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app_system"))

from referee.engine import extract_verdict_from_report, extract_score_from_report, calculate_consensus

print("=" * 80)
print("SCORING SYSTEM TEST")
print("=" * 80)

# ==============================================================================
# TEST 1: Score Extraction from Reports
# ==============================================================================

print("\n[TEST 1] Score Extraction from Sample Reports")
print("-" * 80)

# Sample report formats that might be generated
test_reports = {
    "Format 1 (Bold + Colon)": """
## Domain Audit
[Analysis here...]

- **Verdict**: REVISE
- **Confidence Score**: 6/10
- **Rationale**: Moderate revisions needed across methodology section.
""",

    "Format 2 (Final Report)": """
## Insights Absorbed
[Discussion...]

- **Final Verdict**: FAIL
- **Final Score**: 3/10
- **Final Rationale**: Fundamental methodological flaws that cannot be fixed.
""",

    "Format 3 (No Bold)": """
Verdict: PASS
Confidence Score: 9/10

The paper demonstrates strong identification strategy...
""",

    "Format 4 (Round 1 with bullet)": """
## Assessment

**Verdict**: REVISE
**Confidence Score**: 7/10

The identification strategy is sound but requires additional robustness checks.
""",

    "Format 5 (No score provided)": """
**Verdict**: FAIL

This paper has fundamental issues with the identification strategy.
""",

    "Format 6 (Score without /10)": """
**Final Verdict**: PASS
**Final Score**: 8

Strong empirical work with minor presentation issues.
"""
}

extraction_results = []

for name, report in test_reports.items():
    verdict = extract_verdict_from_report(report)
    score = extract_score_from_report(report)

    extraction_results.append({
        'format': name,
        'verdict': verdict,
        'score': score,
        'success': verdict != "UNKNOWN" and (score is not None or "no score" in name.lower())
    })

    status = "[PASS]" if extraction_results[-1]['success'] else "[FAIL]"
    print(f"\n{status} {name}")
    print(f"   Extracted verdict: {verdict}")
    print(f"   Extracted score: {score}")

# Summary
successful = sum(1 for r in extraction_results if r['success'])
total = len(extraction_results)
print(f"\n{'='*80}")
print(f"Extraction Test: {successful}/{total} formats handled correctly")

if successful == total:
    print("[PASS] All extraction patterns working!")
else:
    print("[FAIL] Some extraction patterns failed - review regex patterns")

# ==============================================================================
# TEST 2: Consensus Calculation with Scores
# ==============================================================================

print("\n" + "="*80)
print("[TEST 2] Consensus Calculation with Hybrid Scores")
print("-" * 80)

# Mock Round 2C reports with scores
mock_r2c_reports = {
    "Econometrician": """
**Final Verdict**: REVISE
**Final Score**: 6/10
**Final Rationale**: Identification strategy is sound but needs robustness checks.
""",
    "Policymaker": """
**Final Verdict**: FAIL
**Final Score**: 3/10
**Final Rationale**: Limited policy applicability, major revisions needed.
""",
    "ML_Expert": """
**Final Verdict**: REVISE
**Final Score**: 5/10
**Final Rationale**: Model architecture choices need better justification.
"""
}

mock_selection_data = {
    "weights": {
        "Econometrician": 0.4,
        "Policymaker": 0.3,
        "ML_Expert": 0.3
    }
}

print("\nMock scenario:")
print("  Econometrician (0.4 weight): REVISE (6/10)")
print("  Policymaker (0.3 weight): FAIL (3/10)")
print("  ML_Expert (0.3 weight): REVISE (5/10)")

consensus = calculate_consensus(mock_r2c_reports, mock_selection_data)

print("\nConsensus results:")
print(f"  Verdicts: {consensus['verdicts']}")
print(f"  Scores: {consensus['scores']}")
print(f"  Weighted score (categorical): {consensus['weighted_score_categorical']:.4f}")
print(f"  Weighted score (numeric): {consensus.get('weighted_score_numeric', 'N/A')}")
print(f"  Decision: {consensus['decision']}")

# Validate calculations
expected_categorical = 0.4 * 0.5 + 0.3 * 0.0 + 0.3 * 0.5  # REVISE=0.5, FAIL=0.0
print(f"\nValidation:")
print(f"  Expected categorical: {expected_categorical:.4f}")
print(f"  Match: {'[PASS]' if abs(consensus['weighted_score_categorical'] - expected_categorical) < 0.001 else '[FAIL]'}")

if consensus.get('weighted_score_numeric') is not None:
    expected_numeric = 0.4 * (6/10) + 0.3 * (3/10) + 0.3 * (5/10)  # Weighted avg of scores/10
    print(f"  Expected numeric: {expected_numeric:.4f}")
    print(f"  Match: {'[PASS]' if abs(consensus['weighted_score_numeric'] - expected_numeric) < 0.001 else '[FAIL]'}")

# ==============================================================================
# TEST 3: Backwards Compatibility (Reports without Scores)
# ==============================================================================

print("\n" + "="*80)
print("[TEST 3] Backwards Compatibility (Old Format)")
print("-" * 80)

mock_r2c_reports_old = {
    "Econometrician": """
**Final Verdict**: REVISE
**Final Rationale**: Identification strategy needs work.
""",
    "Policymaker": """
**Final Verdict**: FAIL
**Final Rationale**: Limited policy applicability.
"""
}

mock_selection_data_old = {
    "weights": {
        "Econometrician": 0.5,
        "Policymaker": 0.5
    }
}

print("\nOld format (no scores):")
print("  Econometrician (0.5 weight): REVISE (no score)")
print("  Policymaker (0.5 weight): FAIL (no score)")

consensus_old = calculate_consensus(mock_r2c_reports_old, mock_selection_data_old)

print("\nConsensus results:")
print(f"  Verdicts: {consensus_old['verdicts']}")
print(f"  Scores: {consensus_old['scores']}")
print(f"  Weighted score (categorical): {consensus_old['weighted_score_categorical']:.4f}")
print(f"  Weighted score (numeric): {consensus_old.get('weighted_score_numeric', 'N/A')}")
print(f"  Decision: {consensus_old['decision']}")

# Check that None scores don't break consensus
all_none = all(s is None for s in consensus_old['scores'].values())
print(f"\nValidation:")
print(f"  All scores are None: {'[PASS]' if all_none else '[FAIL]'}")
print(f"  Categorical score still works: {'[PASS]' if consensus_old['weighted_score_categorical'] > 0 else '[FAIL]'}")
print(f"  Numeric score is None: {'[PASS]' if consensus_old.get('weighted_score_numeric') is None else '[FAIL]'}")

# ==============================================================================
# TEST 4: CSV Column Configuration
# ==============================================================================

print("\n" + "="*80)
print("[TEST 4] CSV Column Configuration")
print("-" * 80)

# Import batch script to check columns
try:
    sys.path.insert(0, str(Path(__file__).parent))

    # Read the batch_referee_reports.py file to check column definitions
    batch_script_path = Path(__file__).parent / "batch_referee_reports.py"
    with open(batch_script_path, 'r') as f:
        content = f.read()

    # Check for score-related columns
    required_columns = [
        'persona_1_round1_score',
        'persona_1_final_score',
        'persona_2_round1_score',
        'persona_2_final_score',
        'persona_3_round1_score',
        'persona_3_final_score',
        'round1_consensus_score_numeric',
        'round2c_consensus_score_numeric',
        'consensus_delta_numeric'
    ]

    print("\nChecking CSV column configuration:")
    all_present = True
    for col in required_columns:
        present = f"'{col}'" in content
        status = "[PASS]" if present else "[FAIL]"
        print(f"  {status} {col}")
        if not present:
            all_present = False

    if all_present:
        print("\n[PASS] All required columns are configured in batch script!")
    else:
        print("\n[FAIL] Some columns missing - check batch_referee_reports.py")

except Exception as e:
    print(f"\n[FAIL] Error checking batch script: {e}")

# ==============================================================================
# TEST 5: Edge Cases
# ==============================================================================

print("\n" + "="*80)
print("[TEST 5] Edge Cases")
print("-" * 80)

edge_cases = [
    ("Score out of range (11/10)", "**Final Score**: 11/10", "Should return None"),
    ("Score out of range (0/10)", "**Final Score**: 0/10", "Should return None"),
    ("Decimal score (7.5/10)", "**Final Score**: 7.5/10", "Should extract 7.5"),
    ("Multiple scores", "Score: 3/10\nLater: Final Score: 8/10", "Should extract last (8)"),
    ("No verdict", "This is a good paper", "Should return UNKNOWN"),
]

print("\nTesting edge cases:")
for name, report, expected_behavior in edge_cases:
    verdict = extract_verdict_from_report(report)
    score = extract_score_from_report(report)
    print(f"\n  {name}")
    print(f"    Verdict: {verdict}, Score: {score}")
    print(f"    Expected: {expected_behavior}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

print("""
[PASS] Tests completed. Review results above.

If all tests passed:
  → System is ready for pilot run
  → Use: python batch_referee_reports.py --pdf-dir ... --limit 3

If any tests failed:
  → Fix extraction patterns in engine.py
  → Fix column configuration in batch_referee_reports.py
  → Re-run this test script

Next step: Run pilot with 1-3 papers to validate end-to-end
""")

print("="*80)
