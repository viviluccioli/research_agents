#!/usr/bin/env python3
"""
Retroactive Trajectory Analysis Script

Analyzes existing batch referee report CSV results to extract trajectory metrics
and compare against ground truth tiers.

Works with limited data from existing CSV:
- R1 verdicts, R2C verdicts, final verdict, tier, persona names/weights

Usage:
    python analyze_existing_results.py \
        --results-csv results/referee_batch_results_20260423_160156.csv \
        --output-dir results/analysis_existing
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np


# ==============================================================================
# VERDICT MAPPING AND SCORING
# ==============================================================================

VERDICT_VALUES = {
    'PASS': 1.0,
    'REVISE': 0.5,
    'FAIL': 0.0,
    'REJECT': 0.0,
    'UNKNOWN': 0.0,
    '': 0.0
}

VERDICT_ORDER = {
    'PASS': 0,
    'REVISE': 1,
    'FAIL': 2
}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_consensus_score(verdicts: List[str], weights: List[float]) -> float:
    """
    Compute weighted consensus score from verdicts and weights.

    Args:
        verdicts: List of verdict strings (PASS, REVISE, FAIL)
        weights: List of weights for each persona

    Returns:
        Weighted score from 0.0 to 1.0
    """
    if len(verdicts) != len(weights):
        return 0.0

    score = sum(
        VERDICT_VALUES.get(v.upper().strip(), 0.0) * w
        for v, w in zip(verdicts, weights)
    )
    return round(score, 4)


def compute_verdict_variance(verdicts: List[str]) -> float:
    """
    Compute variance in verdicts (measure of disagreement).

    Returns value from 0.0 (unanimous) to 1.0 (maximum disagreement).
    """
    if not verdicts:
        return 0.0

    values = [VERDICT_VALUES.get(v.upper().strip(), 0.0) for v in verdicts]

    if len(set(values)) == 1:
        return 0.0  # Unanimous

    # Normalize variance to 0-1 range
    # Max variance is when verdicts are split between PASS (1.0) and FAIL (0.0)
    variance = np.var(values)
    max_variance = 0.25  # Variance of [1.0, 0.0] or [1.0, 0.5, 0.0]

    return min(variance / max_variance, 1.0)


def classify_direction(r1_score: float, r2c_score: float, threshold: float = 0.05) -> str:
    """
    Classify consensus direction change.

    Args:
        r1_score: Round 1 consensus score
        r2c_score: Round 2C consensus score
        threshold: Minimum difference to count as change

    Returns:
        'STRICTER', 'LENIENT', or 'UNCHANGED'
    """
    delta = r2c_score - r1_score

    if abs(delta) < threshold:
        return 'UNCHANGED'
    elif delta < 0:
        return 'STRICTER'
    else:
        return 'LENIENT'


def classify_trajectory_pattern(
    r1_verdicts: List[str],
    r2c_verdicts: List[str],
    r1_score: float,
    r2c_score: float
) -> str:
    """
    Classify the trajectory pattern of the debate.
    """
    r1_variance = compute_verdict_variance(r1_verdicts)
    r2c_variance = compute_verdict_variance(r2c_verdicts)

    direction = classify_direction(r1_score, r2c_score)

    # Unanimous patterns (low variance in both rounds)
    if r1_variance < 0.1 and r2c_variance < 0.1:
        if r2c_score >= 0.8:
            return 'UNANIMOUS_LENIENT'
        elif r2c_score <= 0.2:
            return 'UNANIMOUS_STRICT'
        else:
            return 'UNANIMOUS_MODERATE'

    # Convergent patterns (high variance in R1, low variance in R2C)
    if r1_variance > r2c_variance + 0.1:
        if r2c_score <= 0.2:
            return 'CONVERGENT_STRICT'
        elif r2c_score >= 0.8:
            return 'CONVERGENT_LENIENT'
        else:
            return 'CONVERGENT_MODERATE'

    # Divergent patterns (low variance in R1, high variance in R2C)
    if r2c_variance > r1_variance + 0.1:
        if direction == 'STRICTER':
            return 'DIVERGENT_STRICTER'
        else:
            return 'DIVERGENT_LENIENT'

    # Mixed/oscillating (high variance in both, or no clear pattern)
    return 'MIXED'


def get_trajectory_string(r1_verdict: str, r2c_verdict: str) -> str:
    """Create trajectory string like 'PASS→FAIL'."""
    return f"{r1_verdict}→{r2c_verdict}"


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_batch_results(csv_path: str) -> pd.DataFrame:
    """
    Load batch results CSV into pandas DataFrame.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with results
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = [
        'doc_id', 'tier', 'final_verdict',
        'persona_1_name', 'persona_1_weight', 'persona_1_round1_verdict', 'persona_1_final_verdict',
        'persona_2_name', 'persona_2_weight', 'persona_2_round1_verdict', 'persona_2_final_verdict',
        'persona_3_name', 'persona_3_weight', 'persona_3_round1_verdict', 'persona_3_final_verdict'
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


# ==============================================================================
# TRAJECTORY METRICS COMPUTATION
# ==============================================================================

def compute_paper_trajectory_metrics(row: pd.Series) -> Dict:
    """
    Compute trajectory metrics for a single paper.

    Args:
        row: DataFrame row with paper data

    Returns:
        Dictionary with trajectory metrics
    """
    # Extract persona data
    personas = []
    for i in range(1, 4):
        name = row.get(f'persona_{i}_name', '')
        if name:
            personas.append({
                'name': name,
                'weight': row.get(f'persona_{i}_weight', 0.0),
                'r1_verdict': row.get(f'persona_{i}_round1_verdict', ''),
                'r2c_verdict': row.get(f'persona_{i}_final_verdict', '')
            })

    if not personas:
        return {
            'round1_consensus_score': 0.0,
            'round2c_consensus_score': 0.0,
            'consensus_delta': 0.0,
            'consensus_direction': 'UNKNOWN',
            'num_personas_changed': 0,
            'personas_stricter': [],
            'personas_lenient': [],
            'personas_unchanged': [],
            'trajectory_pattern': 'UNKNOWN',
            'round1_verdict_variance': 0.0,
            'round2c_verdict_variance': 0.0,
            'variance_delta': 0.0
        }

    # Compute consensus scores
    r1_verdicts = [p['r1_verdict'] for p in personas]
    r2c_verdicts = [p['r2c_verdict'] for p in personas]
    weights = [p['weight'] for p in personas]

    r1_score = compute_consensus_score(r1_verdicts, weights)
    r2c_score = compute_consensus_score(r2c_verdicts, weights)
    consensus_delta = r2c_score - r1_score

    # Classify personas by verdict change
    personas_stricter = []
    personas_lenient = []
    personas_unchanged = []

    for p in personas:
        r1_val = VERDICT_VALUES.get(p['r1_verdict'].upper().strip(), 0.0)
        r2c_val = VERDICT_VALUES.get(p['r2c_verdict'].upper().strip(), 0.0)

        if r2c_val < r1_val - 0.01:
            personas_stricter.append(p['name'])
        elif r2c_val > r1_val + 0.01:
            personas_lenient.append(p['name'])
        else:
            personas_unchanged.append(p['name'])

    # Compute verdict variance
    r1_variance = compute_verdict_variance(r1_verdicts)
    r2c_variance = compute_verdict_variance(r2c_verdicts)

    # Classify trajectory pattern
    trajectory_pattern = classify_trajectory_pattern(
        r1_verdicts, r2c_verdicts, r1_score, r2c_score
    )

    return {
        'round1_consensus_score': r1_score,
        'round2c_consensus_score': r2c_score,
        'consensus_delta': consensus_delta,
        'consensus_direction': classify_direction(r1_score, r2c_score),
        'num_personas_changed': len(personas_stricter) + len(personas_lenient),
        'personas_stricter': personas_stricter,
        'personas_lenient': personas_lenient,
        'personas_unchanged': personas_unchanged,
        'trajectory_pattern': trajectory_pattern,
        'round1_verdict_variance': r1_variance,
        'round2c_verdict_variance': r2c_variance,
        'variance_delta': r2c_variance - r1_variance
    }


def add_trajectory_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add trajectory metrics to DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with added trajectory metrics columns
    """
    metrics = df.apply(compute_paper_trajectory_metrics, axis=1)

    # Unpack metrics into separate columns
    df['round1_consensus_score'] = [m['round1_consensus_score'] for m in metrics]
    df['round2c_consensus_score'] = [m['round2c_consensus_score'] for m in metrics]
    df['consensus_delta'] = [m['consensus_delta'] for m in metrics]
    df['consensus_direction'] = [m['consensus_direction'] for m in metrics]
    df['num_personas_changed'] = [m['num_personas_changed'] for m in metrics]
    df['trajectory_pattern'] = [m['trajectory_pattern'] for m in metrics]
    df['round1_verdict_variance'] = [m['round1_verdict_variance'] for m in metrics]
    df['round2c_verdict_variance'] = [m['round2c_verdict_variance'] for m in metrics]
    df['variance_delta'] = [m['variance_delta'] for m in metrics]

    return df


# ==============================================================================
# ANALYSIS FUNCTIONS (Priority Metrics)
# ==============================================================================

def compute_tier_verdict_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    **Analyze actual verdict distributions by tier without assumed alignment.**

    Returns DataFrame with verdict frequencies by tier.
    """
    results = []

    for tier in sorted(df['tier'].unique()):
        tier_df = df[df['tier'] == tier]
        n = len(tier_df)

        if n == 0:
            continue

        # R1 verdict distribution
        r1_verdicts = []
        for i in range(1, 4):
            r1_verdicts.extend(tier_df[f'persona_{i}_round1_verdict'].tolist())
        r1_counter = Counter([v for v in r1_verdicts if v])

        # R2C verdict distribution
        r2c_verdicts = []
        for i in range(1, 4):
            r2c_verdicts.extend(tier_df[f'persona_{i}_final_verdict'].tolist())
        r2c_counter = Counter([v for v in r2c_verdicts if v])

        # Final verdict distribution
        final_counter = Counter(tier_df['final_verdict'])

        results.append({
            'tier': tier,
            'n_papers': n,

            # R1 distributions
            'r1_pass_pct': r1_counter.get('PASS', 0) / max(len(r1_verdicts), 1) * 100,
            'r1_revise_pct': r1_counter.get('REVISE', 0) / max(len(r1_verdicts), 1) * 100,
            'r1_fail_pct': r1_counter.get('FAIL', 0) / max(len(r1_verdicts), 1) * 100,

            # R2C distributions
            'r2c_pass_pct': r2c_counter.get('PASS', 0) / max(len(r2c_verdicts), 1) * 100,
            'r2c_revise_pct': r2c_counter.get('REVISE', 0) / max(len(r2c_verdicts), 1) * 100,
            'r2c_fail_pct': r2c_counter.get('FAIL', 0) / max(len(r2c_verdicts), 1) * 100,

            # Final distributions
            'final_pass_pct': final_counter.get('PASS', 0) / n * 100,
            'final_revise_pct': final_counter.get('REVISE', 0) / n * 100,
            'final_fail_pct': final_counter.get('FAIL', 0) / n * 100,
        })

    return pd.DataFrame(results)


def compute_trajectory_patterns_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    **PRIORITY METRIC #1: Tier-specific patterns**

    Group by tier and analyze trajectory patterns.
    """
    results = []

    for tier in sorted(df['tier'].unique()):
        tier_df = df[df['tier'] == tier]
        n = len(tier_df)

        if n == 0:
            continue

        # Direction counts
        direction_counts = tier_df['consensus_direction'].value_counts()

        # Trajectory pattern mode
        pattern_mode = tier_df['trajectory_pattern'].mode()
        pattern_mode_str = pattern_mode[0] if len(pattern_mode) > 0 else 'N/A'

        results.append({
            'tier': tier,
            'n_papers': n,
            'avg_r1_consensus_score': tier_df['round1_consensus_score'].mean(),
            'avg_r2c_consensus_score': tier_df['round2c_consensus_score'].mean(),
            'avg_consensus_delta': tier_df['consensus_delta'].mean(),
            'pct_stricter': direction_counts.get('STRICTER', 0) / n * 100,
            'pct_lenient': direction_counts.get('LENIENT', 0) / n * 100,
            'pct_unchanged': direction_counts.get('UNCHANGED', 0) / n * 100,
            'avg_personas_changed': tier_df['num_personas_changed'].mean(),
            'mode_trajectory_pattern': pattern_mode_str,
            'avg_r1_variance': tier_df['round1_verdict_variance'].mean(),
            'avg_r2c_variance': tier_df['round2c_verdict_variance'].mean(),
            'avg_variance_delta': tier_df['variance_delta'].mean()
        })

    return pd.DataFrame(results)


def compute_trajectory_pattern_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    **PRIORITY METRIC #2: Trajectory patterns**

    Analyze how verdicts evolve through debate.
    """
    # Get all persona trajectory strings
    trajectories = []

    for _, row in df.iterrows():
        tier = row['tier']
        for i in range(1, 4):
            name = row.get(f'persona_{i}_name', '')
            if name:
                r1 = row.get(f'persona_{i}_round1_verdict', '')
                r2c = row.get(f'persona_{i}_final_verdict', '')
                if r1 and r2c:
                    trajectories.append({
                        'trajectory': get_trajectory_string(r1, r2c),
                        'tier': tier
                    })

    traj_df = pd.DataFrame(trajectories)

    # Overall frequencies
    traj_counts = traj_df['trajectory'].value_counts()
    total = len(trajectories)

    results = []
    for traj, count in traj_counts.items():
        traj_tier_df = traj_df[traj_df['trajectory'] == traj]
        tier_mode = traj_tier_df['tier'].mode()
        tier_mode_str = tier_mode[0] if len(tier_mode) > 0 else 'N/A'

        # Tier distribution
        tier_dist = traj_tier_df['tier'].value_counts()
        tier_dist_str = ', '.join([f"{t}: {c}" for t, c in tier_dist.items()])

        results.append({
            'trajectory': traj,
            'count': count,
            'pct_of_total': count / total * 100,
            'most_common_tier': tier_mode_str,
            'tier_distribution': tier_dist_str
        })

    return pd.DataFrame(results).sort_values('count', ascending=False)


def compute_debate_effectiveness_metrics(df: pd.DataFrame) -> Dict:
    """
    **PRIORITY METRIC #3: Debate effectiveness**

    Measure whether debate improves calibration against ground truth.
    """
    # Extract numeric tier values
    tier_map = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3, 'Tier 4': 4}
    df['tier_numeric'] = df['tier'].map(tier_map)

    # Filter out rows with missing data
    valid_df = df.dropna(subset=['tier_numeric', 'round1_consensus_score', 'round2c_consensus_score'])

    if len(valid_df) < 2:
        return {
            'r1_consensus_tier_correlation': 0.0,
            'r2c_consensus_tier_correlation': 0.0,
            'correlation_improvement': 0.0,
            'n_papers': 0,
            'interpretation': 'Insufficient data'
        }

    # Compute correlations (negative correlation expected: lower tier → higher score)
    r1_corr = valid_df[['tier_numeric', 'round1_consensus_score']].corr().iloc[0, 1]
    r2c_corr = valid_df[['tier_numeric', 'round2c_consensus_score']].corr().iloc[0, 1]

    # Improvement in correlation (more negative = better tier discrimination)
    corr_improvement = r2c_corr - r1_corr

    # Analyze by-tier effectiveness
    tier_effectiveness = []
    for tier in sorted(valid_df['tier'].unique()):
        tier_df = valid_df[valid_df['tier'] == tier]

        # Expected direction: Tier 1/2 should become more lenient, Tier 3/4 should become stricter
        tier_num = tier_map.get(tier, 0)
        if tier_num <= 2:
            # High quality: expect lenient/unchanged
            beneficial = tier_df[tier_df['consensus_direction'].isin(['LENIENT', 'UNCHANGED'])]
        else:
            # Low quality: expect stricter/unchanged
            beneficial = tier_df[tier_df['consensus_direction'].isin(['STRICTER', 'UNCHANGED'])]

        tier_effectiveness.append({
            'tier': tier,
            'n_papers': len(tier_df),
            'pct_beneficial_movement': len(beneficial) / len(tier_df) * 100 if len(tier_df) > 0 else 0
        })

    return {
        'r1_consensus_tier_correlation': round(r1_corr, 4),
        'r2c_consensus_tier_correlation': round(r2c_corr, 4),
        'correlation_improvement': round(corr_improvement, 4),
        'n_papers': len(valid_df),
        'interpretation': (
            'Improved' if corr_improvement < -0.05 else
            'Worsened' if corr_improvement > 0.05 else
            'No substantial change'
        ),
        'tier_effectiveness': tier_effectiveness
    }


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

def generate_analysis_report(
    df: pd.DataFrame,
    tier_patterns: pd.DataFrame,
    trajectory_freqs: pd.DataFrame,
    verdict_dists: pd.DataFrame,
    debate_effectiveness: Dict
) -> str:
    """
    Generate human-readable markdown report.
    """
    report = []

    report.append("# Referee Report Trajectory Analysis\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Total Papers Analyzed:** {len(df)}\n")
    report.append("\n---\n")

    # ==== PRIORITY METRIC #1: Tier-Specific Patterns ====
    report.append("\n## Priority Metric #1: Tier-Specific Patterns\n")
    report.append("How do different quality papers behave in the debate process?\n")

    report.append("\n### Trajectory Statistics by Tier\n")
    report.append(tier_patterns.to_markdown(index=False))
    report.append("\n")

    report.append("\n### Key Findings:\n")
    for _, row in tier_patterns.iterrows():
        tier = row['tier']
        direction_summary = (
            f"{row['pct_stricter']:.1f}% stricter, "
            f"{row['pct_lenient']:.1f}% lenient, "
            f"{row['pct_unchanged']:.1f}% unchanged"
        )
        report.append(f"- **{tier}** ({row['n_papers']} papers): {direction_summary}\n")
        report.append(f"  - Consensus: R1={row['avg_r1_consensus_score']:.3f} → R2C={row['avg_r2c_consensus_score']:.3f} (Δ={row['avg_consensus_delta']:.3f})\n")
        report.append(f"  - Most common pattern: {row['mode_trajectory_pattern']}\n")

    report.append("\n---\n")

    # ==== PRIORITY METRIC #2: Trajectory Patterns ====
    report.append("\n## Priority Metric #2: Trajectory Pattern Frequencies\n")
    report.append("What are the most common verdict evolution paths?\n")

    report.append("\n### Top 15 Most Common Trajectory Patterns\n")
    top_patterns = trajectory_freqs.head(15)
    report.append(top_patterns.to_markdown(index=False))
    report.append("\n")

    report.append("\n### Pattern Insights:\n")
    unchanged_count = trajectory_freqs[trajectory_freqs['trajectory'].str.contains('→') &
                                       (trajectory_freqs['trajectory'].str.split('→').str[0] ==
                                        trajectory_freqs['trajectory'].str.split('→').str[1])]['count'].sum()
    total_count = trajectory_freqs['count'].sum()
    report.append(f"- **Unchanged verdicts:** {unchanged_count}/{total_count} ({unchanged_count/total_count*100:.1f}%)\n")
    report.append(f"- **Changed verdicts:** {total_count - unchanged_count}/{total_count} ({(total_count-unchanged_count)/total_count*100:.1f}%)\n")

    report.append("\n---\n")

    # ==== PRIORITY METRIC #3: Debate Effectiveness ====
    report.append("\n## Priority Metric #3: Debate Effectiveness\n")
    report.append("Does the debate process improve calibration against ground truth tiers?\n")

    report.append(f"\n### Correlation Analysis\n")
    report.append(f"- **R1 Consensus-Tier Correlation:** {debate_effectiveness['r1_consensus_tier_correlation']:.4f}\n")
    report.append(f"- **R2C Consensus-Tier Correlation:** {debate_effectiveness['r2c_consensus_tier_correlation']:.4f}\n")
    report.append(f"- **Correlation Improvement:** {debate_effectiveness['correlation_improvement']:.4f}\n")
    report.append(f"- **Interpretation:** {debate_effectiveness['interpretation']}\n")
    report.append(f"- **Papers Analyzed:** {debate_effectiveness['n_papers']}\n")

    report.append("\n### Tier-Specific Effectiveness\n")
    if 'tier_effectiveness' in debate_effectiveness:
        tier_eff_df = pd.DataFrame(debate_effectiveness['tier_effectiveness'])
        report.append(tier_eff_df.to_markdown(index=False))
        report.append("\n")

    report.append("\n### Verdict Distribution by Tier\n")
    report.append(verdict_dists.to_markdown(index=False))
    report.append("\n")

    report.append("\n---\n")
    report.append("\n## Summary\n")
    report.append("This analysis provides quantifiable metrics for understanding:\n")
    report.append("1. How different quality tiers show different debate dynamics\n")
    report.append("2. Which verdict evolution patterns are most common\n")
    report.append("3. Whether debate improves or worsens calibration vs. ground truth\n")

    return ''.join(report)


def export_analysis_results(
    df: pd.DataFrame,
    tier_patterns: pd.DataFrame,
    trajectory_freqs: pd.DataFrame,
    verdict_dists: pd.DataFrame,
    debate_effectiveness: Dict,
    output_dir: Path
):
    """
    Save analysis results to multiple formats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual tables
    tier_patterns.to_csv(output_dir / 'trajectory_patterns_by_tier.csv', index=False)
    trajectory_freqs.to_csv(output_dir / 'trajectory_pattern_frequencies.csv', index=False)
    verdict_dists.to_csv(output_dir / 'verdict_distributions_by_tier.csv', index=False)

    # Save debate effectiveness
    debate_eff_records = [{
        'metric': 'R1 Consensus-Tier Correlation',
        'value': debate_effectiveness['r1_consensus_tier_correlation']
    }, {
        'metric': 'R2C Consensus-Tier Correlation',
        'value': debate_effectiveness['r2c_consensus_tier_correlation']
    }, {
        'metric': 'Correlation Improvement',
        'value': debate_effectiveness['correlation_improvement']
    }]
    pd.DataFrame(debate_eff_records).to_csv(output_dir / 'debate_effectiveness.csv', index=False)

    # Save full summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_papers_analyzed': len(df),
        'tier_patterns': tier_patterns.to_dict(orient='records'),
        'trajectory_frequencies': trajectory_freqs.to_dict(orient='records'),
        'verdict_distributions': verdict_dists.to_dict(orient='records'),
        'debate_effectiveness': debate_effectiveness
    }

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate and save markdown report
    report = generate_analysis_report(
        df, tier_patterns, trajectory_freqs, verdict_dists, debate_effectiveness
    )

    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write(report)

    # Save enhanced CSV with trajectory metrics
    df.to_csv(output_dir / 'enhanced_results.csv', index=False)

    print(f"\n[Output] Analysis results saved to: {output_dir}")
    print(f"  - trajectory_patterns_by_tier.csv")
    print(f"  - trajectory_pattern_frequencies.csv")
    print(f"  - verdict_distributions_by_tier.csv")
    print(f"  - debate_effectiveness.csv")
    print(f"  - analysis_summary.json")
    print(f"  - analysis_report.md")
    print(f"  - enhanced_results.csv")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory patterns in existing referee report batch results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--results-csv',
        type=str,
        required=True,
        help='Path to existing batch results CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/analysis_existing',
        help='Output directory for analysis results (default: ./results/analysis_existing)'
    )

    args = parser.parse_args()

    # Validate inputs
    results_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        print(f"Error: Results CSV not found: {results_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("RETROACTIVE TRAJECTORY ANALYSIS")
    print(f"{'='*80}")
    print(f"Results CSV: {results_path}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n[Step 1/6] Loading batch results...")
    df = load_batch_results(str(results_path))
    print(f"  Loaded {len(df)} papers")

    # Add trajectory metrics
    print("\n[Step 2/6] Computing trajectory metrics...")
    df = add_trajectory_metrics(df)
    print(f"  Added trajectory metrics for {len(df)} papers")

    # Compute tier verdict distributions
    print("\n[Step 3/6] Computing tier verdict distributions...")
    verdict_dists = compute_tier_verdict_distributions(df)
    print(f"  Analyzed {len(verdict_dists)} tiers")

    # Compute tier-specific patterns
    print("\n[Step 4/6] Computing tier-specific trajectory patterns...")
    tier_patterns = compute_trajectory_patterns_by_tier(df)
    print(f"  Computed patterns for {len(tier_patterns)} tiers")

    # Compute trajectory pattern frequencies
    print("\n[Step 5/6] Computing trajectory pattern frequencies...")
    trajectory_freqs = compute_trajectory_pattern_frequencies(df)
    print(f"  Found {len(trajectory_freqs)} unique trajectory patterns")

    # Compute debate effectiveness
    print("\n[Step 6/6] Computing debate effectiveness metrics...")
    debate_effectiveness = compute_debate_effectiveness_metrics(df)
    print(f"  Analyzed debate effectiveness for {debate_effectiveness['n_papers']} papers")
    print(f"  Correlation improvement: {debate_effectiveness['correlation_improvement']:.4f} ({debate_effectiveness['interpretation']})")

    # Export results
    print(f"\n[Export] Saving analysis results...")
    export_analysis_results(
        df, tier_patterns, trajectory_freqs, verdict_dists,
        debate_effectiveness, output_dir
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"View results in: {output_dir}/")
    print(f"Read report: {output_dir}/analysis_report.md")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
