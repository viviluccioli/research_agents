#!/usr/bin/env python3
"""
Persona-Level Analysis for Enhanced CSV Results

Analyzes:
1. Persona selection frequency (who gets chosen most often)
2. Persona verdict distributions (pass/fail rates)
3. Persona influence on final verdict
4. Trajectory change metrics per persona

Usage:
    python analyze_personas.py
"""

import pandas as pd
from collections import Counter

# Load enhanced CSV
df = pd.read_csv('results/referee_batch_results_20260423_160156_enhanced.csv')

print("=" * 80)
print("PERSONA-LEVEL ANALYSIS")
print("=" * 80)
print(f"\nTotal papers analyzed: {len(df)}\n")

# =============================================================================
# 1. PERSONA SELECTION FREQUENCY
# =============================================================================

print("=" * 80)
print("1. PERSONA SELECTION FREQUENCY (Who Gets Chosen)")
print("=" * 80)

persona_selections = []
for i in range(1, 4):
    name_col = f'persona_{i}_name'
    if name_col in df.columns:
        personas = df[df[name_col] != ''][name_col].tolist()
        persona_selections.extend(personas)

persona_counts = Counter(persona_selections)
total_selections = sum(persona_counts.values())

print("\nPersona Selection Counts:")
for persona, count in persona_counts.most_common():
    pct = count / total_selections * 100
    papers = count  # Since we have 10 papers
    print(f"  {persona:20s}: {count:2d} times ({pct:5.1f}%) - appeared in {count} papers")

# =============================================================================
# 2. PERSONA VERDICT DISTRIBUTIONS
# =============================================================================

print("\n" + "=" * 80)
print("2. PERSONA VERDICT DISTRIBUTIONS (Pass/Fail Rates)")
print("=" * 80)

persona_verdicts = {
    'Round 1': {},
    'Round 2C': {}
}

# Collect R1 verdicts per persona
for i in range(1, 4):
    name_col = f'persona_{i}_name'
    r1_col = f'persona_{i}_round1_verdict'
    r2c_col = f'persona_{i}_final_verdict'

    for _, row in df.iterrows():
        persona = row.get(name_col, '')
        if persona:
            r1_verdict = row.get(r1_col, '')
            r2c_verdict = row.get(r2c_col, '')

            if persona not in persona_verdicts['Round 1']:
                persona_verdicts['Round 1'][persona] = []
            if persona not in persona_verdicts['Round 2C']:
                persona_verdicts['Round 2C'][persona] = []

            persona_verdicts['Round 1'][persona].append(r1_verdict)
            persona_verdicts['Round 2C'][persona].append(r2c_verdict)

# Compute distributions for each persona
print("\nRound 1 Verdict Distributions:")
print(f"{'Persona':<20s} {'Total':<6s} {'PASS %':<8s} {'REVISE %':<10s} {'FAIL %':<8s}")
print("-" * 60)

r1_stats = []
for persona in sorted(persona_verdicts['Round 1'].keys()):
    verdicts = persona_verdicts['Round 1'][persona]
    counts = Counter(verdicts)
    total = len(verdicts)

    pass_pct = counts.get('PASS', 0) / total * 100
    revise_pct = counts.get('REVISE', 0) / total * 100
    fail_pct = counts.get('FAIL', 0) / total * 100

    print(f"{persona:<20s} {total:<6d} {pass_pct:6.1f}%  {revise_pct:8.1f}%  {fail_pct:6.1f}%")

    r1_stats.append({
        'persona': persona,
        'total': total,
        'pass_pct': pass_pct,
        'fail_pct': fail_pct
    })

print("\nRound 2C Verdict Distributions:")
print(f"{'Persona':<20s} {'Total':<6s} {'PASS %':<8s} {'REVISE %':<10s} {'FAIL %':<8s}")
print("-" * 60)

r2c_stats = []
for persona in sorted(persona_verdicts['Round 2C'].keys()):
    verdicts = persona_verdicts['Round 2C'][persona]
    counts = Counter(verdicts)
    total = len(verdicts)

    pass_pct = counts.get('PASS', 0) / total * 100
    revise_pct = counts.get('REVISE', 0) / total * 100
    fail_pct = counts.get('FAIL', 0) / total * 100

    print(f"{persona:<20s} {total:<6d} {pass_pct:6.1f}%  {revise_pct:8.1f}%  {fail_pct:6.1f}%")

    r2c_stats.append({
        'persona': persona,
        'total': total,
        'pass_pct': pass_pct,
        'fail_pct': fail_pct
    })

# =============================================================================
# 3. WHICH PERSONAS ARE STRICTEST/MOST LENIENT
# =============================================================================

print("\n" + "=" * 80)
print("3. PERSONA STRICTNESS RANKING (By Final Verdict Fail Rate)")
print("=" * 80)

r2c_df = pd.DataFrame(r2c_stats).sort_values('fail_pct', ascending=False)
print("\nStrictest to Most Lenient:")
for i, row in r2c_df.iterrows():
    print(f"  {row['persona']:<20s}: {row['fail_pct']:5.1f}% FAIL rate")

# =============================================================================
# 4. TRAJECTORY CHANGE METRICS PER PERSONA
# =============================================================================

print("\n" + "=" * 80)
print("4. TRAJECTORY CHANGE PATTERNS PER PERSONA")
print("=" * 80)

persona_directions = {}
persona_trajectories = {}

for i in range(1, 4):
    name_col = f'persona_{i}_name'
    dir_col = f'persona_{i}_direction'
    traj_col = f'persona_{i}_trajectory'

    for _, row in df.iterrows():
        persona = row.get(name_col, '')
        if persona:
            direction = row.get(dir_col, '')
            trajectory = row.get(traj_col, '')

            if persona not in persona_directions:
                persona_directions[persona] = []
            if persona not in persona_trajectories:
                persona_trajectories[persona] = []

            persona_directions[persona].append(direction)
            persona_trajectories[persona].append(trajectory)

print("\nHow Often Does Each Persona Change Their Verdict?")
print(f"{'Persona':<20s} {'Total':<6s} {'Stricter %':<12s} {'Lenient %':<12s} {'Unchanged %':<13s}")
print("-" * 70)

change_stats = []
for persona in sorted(persona_directions.keys()):
    directions = persona_directions[persona]
    counts = Counter(directions)
    total = len(directions)

    stricter_pct = counts.get('STRICTER', 0) / total * 100
    lenient_pct = counts.get('LENIENT', 0) / total * 100
    unchanged_pct = counts.get('UNCHANGED', 0) / total * 100

    print(f"{persona:<20s} {total:<6d} {stricter_pct:10.1f}%  {lenient_pct:10.1f}%  {unchanged_pct:11.1f}%")

    change_stats.append({
        'persona': persona,
        'stricter_pct': stricter_pct,
        'unchanged_pct': unchanged_pct
    })

print("\nMost Common Trajectory Patterns Per Persona:")
for persona in sorted(persona_trajectories.keys()):
    trajs = [t for t in persona_trajectories[persona] if t]
    if trajs:
        traj_counts = Counter(trajs)
        print(f"\n{persona}:")
        for traj, count in traj_counts.most_common(3):
            pct = count / len(trajs) * 100
            print(f"  {traj:15s}: {count} ({pct:.1f}%)")

# =============================================================================
# 5. PERSONA INFLUENCE ON FINAL VERDICT
# =============================================================================

print("\n" + "=" * 80)
print("5. PERSONA INFLUENCE ON FINAL VERDICT")
print("=" * 80)

print("\nAverage Weighted Contribution to Final Consensus (R2C):")
print(f"{'Persona':<20s} {'Avg Weight':<12s} {'Avg Contribution':<18s}")
print("-" * 55)

persona_contributions = {}

for i in range(1, 4):
    name_col = f'persona_{i}_name'
    weight_col = f'persona_{i}_weight'
    contrib_col = f'persona_{i}_contribution_r2c'

    for _, row in df.iterrows():
        persona = row.get(name_col, '')
        if persona:
            weight = row.get(weight_col, 0.0)
            contrib = row.get(contrib_col, 0.0)

            if persona not in persona_contributions:
                persona_contributions[persona] = {'weights': [], 'contribs': []}

            persona_contributions[persona]['weights'].append(weight)
            persona_contributions[persona]['contribs'].append(contrib)

for persona in sorted(persona_contributions.keys()):
    avg_weight = sum(persona_contributions[persona]['weights']) / len(persona_contributions[persona]['weights'])
    avg_contrib = sum(persona_contributions[persona]['contribs']) / len(persona_contributions[persona]['contribs'])

    print(f"{persona:<20s} {avg_weight:10.3f}    {avg_contrib:15.3f}")

# =============================================================================
# 6. PERSONA PERFORMANCE BY TIER
# =============================================================================

print("\n" + "=" * 80)
print("6. PERSONA PERFORMANCE BY TIER")
print("=" * 80)

persona_tier_data = []

for i in range(1, 4):
    name_col = f'persona_{i}_name'
    r2c_col = f'persona_{i}_final_verdict'

    for _, row in df.iterrows():
        persona = row.get(name_col, '')
        if persona:
            tier = row['tier']
            verdict = row.get(r2c_col, '')
            persona_tier_data.append({
                'persona': persona,
                'tier': tier,
                'verdict': verdict
            })

persona_tier_df = pd.DataFrame(persona_tier_data)

print("\nFinal Verdict Distribution by Persona and Tier:")
for persona in sorted(persona_tier_df['persona'].unique()):
    print(f"\n{persona}:")
    persona_data = persona_tier_df[persona_tier_df['persona'] == persona]

    for tier in ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4']:
        tier_data = persona_data[persona_data['tier'] == tier]
        if len(tier_data) > 0:
            verdict_counts = tier_data['verdict'].value_counts()
            verdicts_str = ', '.join([f"{v}: {c}" for v, c in verdict_counts.items()])
            print(f"  {tier}: {verdicts_str} (n={len(tier_data)})")

# =============================================================================
# 7. TRAJECTORY CHANGE METRIC
# =============================================================================

print("\n" + "=" * 80)
print("7. TRAJECTORY CHANGE SCORE (Quantifying Verdict Evolution)")
print("=" * 80)

print("""
The Trajectory Change Score measures how much personas change their verdicts:
- Score = Average absolute change magnitude across all personas
- Range: 0.0 (no change) to 1.0 (maximum change, e.g., PASS→FAIL)
- Interpretation: Higher = more dramatic verdict evolution
""")

# Compute trajectory change score per paper
df['trajectory_change_score'] = 0.0
for idx, row in df.iterrows():
    changes = []
    for i in range(1, 4):
        change_col = f'persona_{i}_change_magnitude'
        if change_col in df.columns:
            changes.append(row[change_col])
    df.at[idx, 'trajectory_change_score'] = sum(changes) / len([c for c in changes if c > 0 or c == 0]) if changes else 0.0

print("\nPapers with Highest Trajectory Change Score:")
high_change = df.nlargest(5, 'trajectory_change_score')[
    ['doc_id', 'tier', 'trajectory_change_score', 'num_personas_changed', 'trajectory_pattern']
]
print(high_change.to_string(index=False))

print("\nAverage Trajectory Change Score by Tier:")
print(df.groupby('tier')['trajectory_change_score'].mean().round(4))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nKey Findings:")
print("- Persona selection frequency shows which personas are chosen most")
print("- Pass/Fail rates reveal which personas are strictest vs. most lenient")
print("- Trajectory change patterns show how often personas revise their opinions")
print("- Influence metrics identify which personas have most impact on final verdict")
print("=" * 80)
