#!/usr/bin/env python3
"""
Quick variance analysis script for enhanced CSV results.

Usage:
    python analyze_variance.py
"""

import pandas as pd

# Load enhanced CSV
df = pd.read_csv('results/referee_batch_results_20260423_160156_enhanced.csv')

print("=" * 80)
print("VARIANCE ANALYSIS: Referee Report Debate Dynamics")
print("=" * 80)
print(f"\nTotal papers analyzed: {len(df)}")

# =============================================================================
# 1. VARIANCE BY TIER
# =============================================================================

print("\n" + "=" * 80)
print("1. VARIANCE METRICS BY TIER")
print("=" * 80)

variance_by_tier = df.groupby('tier').agg({
    'round1_verdict_variance': 'mean',
    'round1_verdict_entropy': 'mean',
    'round2c_verdict_variance': 'mean',
    'round2c_verdict_entropy': 'mean',
    'variance_delta': 'mean',
    'convergence_magnitude': 'mean'
}).round(4)

print(variance_by_tier)

print("\nInterpretation:")
print("- Positive variance_delta = personas diverged (increased disagreement)")
print("- Negative variance_delta = personas converged (decreased disagreement)")
print("- Higher entropy = more diverse opinions")

# =============================================================================
# 2. CONVERGENCE PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("2. CONVERGENCE PATTERNS BY TIER")
print("=" * 80)

convergence_patterns = df.groupby(['tier', 'convergence_pattern']).size().unstack(fill_value=0)
print(convergence_patterns)

# =============================================================================
# 3. TRAJECTORY PATTERNS
# =============================================================================

print("\n" + "=" * 80)
print("3. TRAJECTORY PATTERNS BY TIER")
print("=" * 80)

trajectory_patterns = df.groupby(['tier', 'trajectory_pattern']).size().unstack(fill_value=0)
print(trajectory_patterns)

# =============================================================================
# 4. PAPERS WITH HIGHEST/LOWEST VARIANCE
# =============================================================================

print("\n" + "=" * 80)
print("4. PAPERS WITH HIGHEST INITIAL VARIANCE (Most Disagreement)")
print("=" * 80)

high_variance = df.nlargest(5, 'round1_verdict_variance')[
    ['doc_id', 'tier', 'round1_verdict_variance', 'round1_verdict_entropy',
     'convergence_pattern', 'num_personas_changed']
]
print(high_variance.to_string(index=False))

print("\n" + "=" * 80)
print("5. PAPERS WITH MOST CONVERGENCE (Biggest Disagreement Drop)")
print("=" * 80)

most_convergent = df.nsmallest(5, 'variance_delta')[
    ['doc_id', 'tier', 'round1_verdict_variance', 'round2c_verdict_variance',
     'variance_delta', 'trajectory_pattern']
]
print(most_convergent.to_string(index=False))

# =============================================================================
# 6. CONSENSUS SHIFT ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("6. CONSENSUS SHIFTS BY TIER")
print("=" * 80)

consensus_by_tier = df.groupby('tier').agg({
    'round1_consensus_score': 'mean',
    'round2c_consensus_score': 'mean',
    'consensus_delta': 'mean',
    'consensus_shift_magnitude': 'mean'
}).round(4)

print(consensus_by_tier)

# =============================================================================
# 7. VERDICT CHANGE FREQUENCY
# =============================================================================

print("\n" + "=" * 80)
print("7. HOW OFTEN DO PERSONAS CHANGE VERDICTS?")
print("=" * 80)

print(f"\nAverage personas changed per paper: {df['num_personas_changed'].mean():.2f}")
print(f"\nPapers where 0 personas changed: {len(df[df['num_personas_changed'] == 0])}")
print(f"Papers where 1 persona changed: {len(df[df['num_personas_changed'] == 1])}")
print(f"Papers where 2 personas changed: {len(df[df['num_personas_changed'] == 2])}")
print(f"Papers where 3 personas changed: {len(df[df['num_personas_changed'] == 3])}")

# =============================================================================
# 8. PERSONA TRAJECTORIES
# =============================================================================

print("\n" + "=" * 80)
print("8. MOST COMMON PERSONA TRAJECTORIES")
print("=" * 80)

# Collect all persona trajectories
trajectories = []
for i in range(1, 4):
    traj_col = f'persona_{i}_trajectory'
    if traj_col in df.columns:
        trajectories.extend(df[df[traj_col] != ''][traj_col].tolist())

from collections import Counter
traj_counts = Counter(trajectories)

print("\nTop 10 most common trajectory patterns:")
for traj, count in traj_counts.most_common(10):
    pct = count / len(trajectories) * 100
    print(f"  {traj:15s}: {count:2d} ({pct:5.1f}%)")

# =============================================================================
# 9. CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("9. CORRELATIONS WITH TIER")
print("=" * 80)

# Convert tier to numeric for correlation
tier_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3, 'Tier 4': 4}
df['tier_numeric'] = df['tier'].map(tier_map)

correlations = df[[
    'tier_numeric',
    'round1_verdict_variance',
    'round2c_verdict_variance',
    'variance_delta',
    'round1_consensus_score',
    'round2c_consensus_score',
    'consensus_delta'
]].corr()['tier_numeric'].drop('tier_numeric').sort_values(ascending=False)

print("\nCorrelation with tier (higher tier = lower quality):")
print(correlations.round(4))

print("\nInterpretation:")
print("- Negative correlation = higher tiers (better papers) have lower values")
print("- Positive correlation = higher tiers (better papers) have higher values")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nEnhanced CSV location:")
print("  results/referee_batch_results_20260423_160156_enhanced.csv")
print("\nYou can now open this CSV in Excel or pandas for deeper analysis!")
print("=" * 80)
