#!/usr/bin/env python3
"""
Analyze Tool Calibration - Focused Insights

Shows exactly what you need to know:
1. Is the tool calibrated?
2. Where is it miscalibrated?
3. Does debate help?
4. Which papers are problematic?

Usage:
    python analyze_calibration.py
"""

import pandas as pd

# Load calibrated CSV
df = pd.read_csv('results/referee_batch_results_20260423_160156_calibrated.csv')

print("=" * 80)
print("TOOL CALIBRATION ANALYSIS")
print("=" * 80)
print(f"\nTotal papers: {len(df)}\n")

# =============================================================================
# 1. IS THE TOOL CALIBRATED?
# =============================================================================

print("=" * 80)
print("1. IS THE TOOL CALIBRATED?")
print("   (Are scores aligned with tier quality?)")
print("=" * 80)

print("\nExpected behavior:")
print("  Tier 1 (top journal)     → score near 0.90")
print("  Tier 2 (good journal)    → score near 0.70")
print("  Tier 3 (mediocre journal)→ score near 0.30")
print("  Tier 4 (not accepted)    → score near 0.10")

print("\nActual scores:")
score_summary = df.groupby('tier').agg({
    'consensus_score_r2c': 'mean',
    'expected_score_for_tier': 'first',
    'calibration_error': 'mean'
}).round(3)
score_summary.columns = ['Actual Score', 'Expected Score', 'Avg Error']
print(score_summary)

print("\n📊 CALIBRATION VERDICT:")
avg_error = df['calibration_error'].mean()
if avg_error < 0.2:
    print(f"   ✓ WELL CALIBRATED (avg error: {avg_error:.3f})")
elif avg_error < 0.4:
    print(f"   ⚠️  MODERATELY CALIBRATED (avg error: {avg_error:.3f})")
else:
    print(f"   ✗ POORLY CALIBRATED (avg error: {avg_error:.3f})")

# =============================================================================
# 2. WHERE IS IT MISCALIBRATED?
# =============================================================================

print("\n" + "=" * 80)
print("2. WHERE IS THE TOOL MISCALIBRATED?")
print("   (Papers with large calibration errors)")
print("=" * 80)

high_error = df[df['calibration_error'] > 0.4].sort_values('calibration_error', ascending=False)

if len(high_error) > 0:
    print("\n⚠️  Papers with calibration error > 0.4:")
    print("-" * 80)
    for _, row in high_error.iterrows():
        tier = row['tier']
        doc_id = row['doc_id']
        actual = row['consensus_score_r2c']
        expected = row['expected_score_for_tier']
        error = row['calibration_error']
        agreement = row['agreement_level_r2c']
        final = row['final_verdict']

        direction = "TOO HARSH" if actual < expected else "TOO LENIENT"

        print(f"\n{doc_id} ({tier}):")
        print(f"  Expected score: {expected:.2f} | Actual score: {actual:.2f} | Error: {error:.2f}")
        print(f"  Final verdict: {final} | Agreement: {agreement:.1f}")
        print(f"  → Tool is {direction} for this paper")

        # Persona breakdown
        personas = []
        for i in range(1, 4):
            name = row.get(f'persona_{i}_name', '')
            if name:
                verdict = row.get(f'persona_{i}_final_verdict', '')
                weight = row.get(f'persona_{i}_weight', 0)
                personas.append(f"{name} ({weight:.2f}): {verdict}")
        print(f"  Personas: {', '.join(personas)}")
else:
    print("\n✓ No major calibration errors detected")

# =============================================================================
# 3. TIER-BY-TIER BREAKDOWN
# =============================================================================

print("\n" + "=" * 80)
print("3. TIER-BY-TIER BREAKDOWN")
print("=" * 80)

for tier in ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']:
    tier_df = df[df['tier'] == tier]
    if len(tier_df) == 0:
        continue

    print(f"\n{tier} ({len(tier_df)} papers):")
    print("-" * 60)

    avg_score = tier_df['consensus_score_r2c'].mean()
    expected = tier_df['expected_score_for_tier'].iloc[0]
    avg_error = tier_df['calibration_error'].mean()
    avg_agreement = tier_df['agreement_level_r2c'].mean()

    print(f"  Avg score: {avg_score:.3f} (expected: {expected:.2f}, error: {avg_error:.3f})")
    print(f"  Avg agreement: {avg_agreement:.2f} (1.0 = unanimous)")

    # Direction
    if avg_score < expected - 0.2:
        print(f"  → Tool is TOO STRICT for {tier} papers")
    elif avg_score > expected + 0.2:
        print(f"  → Tool is TOO LENIENT for {tier} papers")
    else:
        print(f"  → Tool is WELL CALIBRATED for {tier} papers")

    # Verdict distribution
    verdicts = tier_df['final_verdict'].value_counts()
    print(f"  Final verdicts: {dict(verdicts)}")

# =============================================================================
# 4. DOES DEBATE HELP?
# =============================================================================

print("\n" + "=" * 80)
print("4. DOES DEBATE IMPROVE CALIBRATION?")
print("=" * 80)

debate_stats = df['debate_improved_calibration'].value_counts()
total = len(df)

print(f"\nDebate improved calibration: {debate_stats.get('Yes', 0)}/{total} papers ({debate_stats.get('Yes', 0)/total*100:.1f}%)")
print(f"Debate worsened calibration: {debate_stats.get('No', 0)}/{total} papers ({debate_stats.get('No', 0)/total*100:.1f}%)")
print(f"No change: {debate_stats.get('No Change', 0)}/{total} papers")

print("\nBy tier:")
debate_by_tier = df.groupby(['tier', 'debate_improved_calibration']).size().unstack(fill_value=0)
print(debate_by_tier)

print("\nAgreement change (R2C - R1):")
agreement_change_summary = df.groupby('tier')['agreement_change'].agg(['mean', 'min', 'max']).round(2)
print(agreement_change_summary)
print("  Positive = debate increased confidence")
print("  Negative = debate increased disagreement")

# =============================================================================
# 5. CONFIDENCE ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("5. TOOL CONFIDENCE ANALYSIS")
print("   (Does tool have appropriate confidence?)")
print("=" * 80)

print("\nAgreement levels by tier (1.0 = unanimous):")
agreement_by_tier = df.groupby('tier').agg({
    'agreement_level_r1': 'mean',
    'agreement_level_r2c': 'mean'
}).round(2)
agreement_by_tier.columns = ['R1 Agreement', 'R2C Agreement']
print(agreement_by_tier)

print("\nIdeal pattern:")
print("  Tier 1 & 4: High agreement (clear quality signals)")
print("  Tier 2 & 3: Lower agreement (genuinely ambiguous)")

# =============================================================================
# 6. VERDICT STABILITY
# =============================================================================

print("\n" + "=" * 80)
print("6. VERDICT STABILITY (How much do verdicts change?)")
print("=" * 80)

stability_counts = df['verdict_stability'].value_counts()
print("\nOverall:")
for status, count in stability_counts.items():
    pct = count / len(df) * 100
    print(f"  {status}: {count} papers ({pct:.1f}%)")

print("\nBy tier:")
stability_by_tier = df.groupby(['tier', 'verdict_stability']).size().unstack(fill_value=0)
print(stability_by_tier)

# =============================================================================
# 7. PROBLEM PAPERS (Need Investigation)
# =============================================================================

print("\n" + "=" * 80)
print("7. PAPERS NEEDING INVESTIGATION")
print("=" * 80)

print("\nCriteria for investigation:")
print("  - High calibration error (>0.5)")
print("  - OR: Tier 1 with FAIL verdict")
print("  - OR: Tier 4 with PASS verdict")

problem_papers = df[
    (df['calibration_error'] > 0.5) |
    ((df['tier'] == 'Tier 1') & (df['final_verdict'] == 'FAIL')) |
    ((df['tier'] == 'Tier 4') & (df['final_verdict'] == 'PASS'))
]

if len(problem_papers) > 0:
    print(f"\n⚠️  {len(problem_papers)} papers need investigation:")
    for _, row in problem_papers.iterrows():
        print(f"\n  {row['doc_id']} ({row['tier']}):")
        print(f"    Final verdict: {row['final_verdict']}")
        print(f"    Consensus score: {row['consensus_score_r2c']:.3f} (expected: {row['expected_score_for_tier']:.2f})")
        print(f"    Calibration error: {row['calibration_error']:.3f}")
        print(f"    Agreement: {row['agreement_level_r2c']:.2f}")
else:
    print("\n✓ No major problem papers detected")

# =============================================================================
# 8. KEY INSIGHTS SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Overall calibration
tier1_error = df[df['tier'] == 'Tier 1']['calibration_error'].mean()
tier4_error = df[df['tier'] == 'Tier 4']['calibration_error'].mean()

print(f"\n1. OVERALL CALIBRATION:")
print(f"   - Average calibration error: {avg_error:.3f}")
if tier1_error > tier4_error:
    print(f"   - Tool is BETTER at identifying bad papers (Tier 4) than good papers (Tier 1)")
else:
    print(f"   - Tool is BETTER at identifying good papers (Tier 1) than bad papers (Tier 4)")

# Debate value
debate_helped_pct = debate_stats.get('Yes', 0) / total * 100
print(f"\n2. DEBATE VALUE:")
print(f"   - Debate improved calibration in {debate_helped_pct:.0f}% of papers")
if debate_helped_pct >= 50:
    print(f"   - Debate adds SIGNIFICANT value")
else:
    print(f"   - Debate adds LIMITED value")

# Confidence appropriateness
tier1_agreement = df[df['tier'] == 'Tier 1']['agreement_level_r2c'].mean()
tier4_agreement = df[df['tier'] == 'Tier 4']['agreement_level_r2c'].mean()
tier2_agreement = df[df['tier'] == 'Tier 2']['agreement_level_r2c'].mean()
tier3_agreement = df[df['tier'] == 'Tier 3']['agreement_level_r2c'].mean()

print(f"\n3. CONFIDENCE CALIBRATION:")
print(f"   - Clear cases (Tier 1 & 4): {(tier1_agreement + tier4_agreement)/2:.2f} avg agreement")
print(f"   - Ambiguous cases (Tier 2 & 3): {(tier2_agreement + tier3_agreement)/2:.2f} avg agreement")
if (tier1_agreement + tier4_agreement) > (tier2_agreement + tier3_agreement):
    print(f"   - Tool has APPROPRIATE confidence (higher for clear cases)")
else:
    print(f"   - Tool confidence is NOT well calibrated")

# Most common issue
if tier1_error > 0.5:
    print(f"\n4. PRIMARY ISSUE:")
    print(f"   - Tool is TOO HARSH on high-quality papers (Tier 1)")
    print(f"   - Consider: Are tier labels accurate? Or does tool need recalibration?")
elif tier4_error > 0.5:
    print(f"\n4. PRIMARY ISSUE:")
    print(f"   - Tool is TOO LENIENT on low-quality papers (Tier 4)")
    print(f"   - Consider: Stricter persona selection or weight adjustment")
else:
    print(f"\n4. PERFORMANCE:")
    print(f"   - Tool shows reasonable calibration across quality tiers")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Review miscalibrated papers (Section 2)")
print("  2. Check tier-specific patterns (Section 3)")
print("  3. Investigate problem papers (Section 7)")
print("  4. Open calibrated CSV for detailed analysis:")
print("     results/referee_batch_results_20260423_160156_calibrated.csv")
print("=" * 80)
