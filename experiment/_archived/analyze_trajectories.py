#!/usr/bin/env python3
"""
Enhanced Trajectory Analysis Script

Analyzes batch referee report CSV results (with pre-computed trajectory metrics)
to compare against ground truth tiers. Compatible with both legacy and enhanced formats.

Works with enhanced batch results containing:
- Pre-computed consensus scores, trajectory patterns, verdict variances
- Individual persona trajectory strings and directions
- Weighted contributions per persona

Usage:
    python analyze_trajectories.py \
        --results-csv results/referee_batch_results_ENHANCED.csv \
        --output-dir results/analysis_enhanced
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Import all analysis functions from the retroactive script
# This allows reuse while adding enhanced features
sys.path.insert(0, str(Path(__file__).parent))
from analyze_existing_results import (
    add_trajectory_metrics,
    compute_tier_verdict_distributions,
    compute_trajectory_patterns_by_tier,
    compute_trajectory_pattern_frequencies,
    compute_debate_effectiveness_metrics,
    generate_analysis_report,
    export_analysis_results
)


def load_batch_results(csv_path: str) -> pd.DataFrame:
    """
    Load batch results CSV - handles both legacy and enhanced formats.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with results
    """
    df = pd.read_csv(csv_path)

    # Check if this is an enhanced format CSV (has pre-computed metrics)
    has_enhanced_metrics = all(col in df.columns for col in [
        'round1_consensus_score',
        'round2c_consensus_score',
        'trajectory_pattern'
    ])

    if has_enhanced_metrics:
        print(f"  Detected enhanced format (pre-computed trajectory metrics)")
        return df
    else:
        print(f"  Detected legacy format - computing trajectory metrics...")
        # Use retroactive analysis functions to add metrics
        return add_trajectory_metrics(df)


def compute_enhanced_persona_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze individual persona performance using enhanced trajectory data.

    Only available for enhanced format CSVs.

    Returns DataFrame with:
    - Persona type
    - Total appearances
    - Average weight
    - % STRICTER trajectories
    - % LENIENT trajectories
    - % UNCHANGED trajectories
    - Average contribution to R1 consensus
    - Average contribution to R2C consensus
    """
    if 'persona_1_direction' not in df.columns:
        print("  [Warning] Enhanced persona metrics not available in this dataset")
        return pd.DataFrame()

    persona_stats = []

    # Collect all persona data
    for i in range(1, 4):
        name_col = f'persona_{i}_name'
        dir_col = f'persona_{i}_direction'
        weight_col = f'persona_{i}_weight'
        contrib_r1_col = f'persona_{i}_contribution_r1'
        contrib_r2c_col = f'persona_{i}_contribution_r2c'

        if all(col in df.columns for col in [name_col, dir_col, weight_col]):
            persona_data = df[df[name_col] != ''][[name_col, dir_col, weight_col, contrib_r1_col, contrib_r2c_col]].copy()
            persona_data.columns = ['name', 'direction', 'weight', 'contrib_r1', 'contrib_r2c']
            persona_stats.append(persona_data)

    if not persona_stats:
        return pd.DataFrame()

    all_personas = pd.concat(persona_stats, ignore_index=True)

    # Group by persona type
    results = []
    for persona_type in sorted(all_personas['name'].unique()):
        persona_df = all_personas[all_personas['name'] == persona_type]
        n = len(persona_df)

        direction_counts = persona_df['direction'].value_counts()

        results.append({
            'persona_type': persona_type,
            'total_appearances': n,
            'avg_weight': persona_df['weight'].mean(),
            'pct_stricter': direction_counts.get('STRICTER', 0) / n * 100,
            'pct_lenient': direction_counts.get('LENIENT', 0) / n * 100,
            'pct_unchanged': direction_counts.get('UNCHANGED', 0) / n * 100,
            'avg_contribution_r1': persona_df['contrib_r1'].mean(),
            'avg_contribution_r2c': persona_df['contrib_r2c'].mean(),
            'avg_contribution_delta': (persona_df['contrib_r2c'] - persona_df['contrib_r1']).mean()
        })

    return pd.DataFrame(results).sort_values('total_appearances', ascending=False)


def generate_enhanced_report(
    df: pd.DataFrame,
    tier_patterns: pd.DataFrame,
    trajectory_freqs: pd.DataFrame,
    verdict_dists: pd.DataFrame,
    debate_effectiveness: dict,
    persona_analysis: pd.DataFrame
) -> str:
    """
    Generate enhanced markdown report with persona-level analysis.
    """
    # Start with base report
    base_report = generate_analysis_report(
        df, tier_patterns, trajectory_freqs, verdict_dists, debate_effectiveness
    )

    # Add enhanced persona analysis if available
    if not persona_analysis.empty:
        enhanced_section = [
            "\n---\n",
            "\n## Enhanced Analysis: Persona Performance\n",
            "Analysis of individual persona behavior patterns across all papers.\n",
            "\n### Persona Trajectory Statistics\n",
            persona_analysis.to_markdown(index=False),
            "\n\n### Key Insights:\n"
        ]

        # Generate insights from persona data
        for _, row in persona_analysis.iterrows():
            persona = row['persona_type']
            enhanced_section.append(
                f"- **{persona}** appeared {row['total_appearances']} times "
                f"(avg weight: {row['avg_weight']:.2f})\n"
            )
            enhanced_section.append(
                f"  - Direction: {row['pct_stricter']:.1f}% stricter, "
                f"{row['pct_lenient']:.1f}% lenient, "
                f"{row['pct_unchanged']:.1f}% unchanged\n"
            )
            enhanced_section.append(
                f"  - Contribution: R1={row['avg_contribution_r1']:.3f} → "
                f"R2C={row['avg_contribution_r2c']:.3f} "
                f"(Δ={row['avg_contribution_delta']:.3f})\n"
            )

        # Insert before summary
        report_parts = base_report.split("\n## Summary\n")
        if len(report_parts) == 2:
            return report_parts[0] + ''.join(enhanced_section) + "\n## Summary\n" + report_parts[1]

    return base_report


def export_enhanced_results(
    df: pd.DataFrame,
    tier_patterns: pd.DataFrame,
    trajectory_freqs: pd.DataFrame,
    verdict_dists: pd.DataFrame,
    debate_effectiveness: dict,
    persona_analysis: pd.DataFrame,
    output_dir: Path
):
    """
    Save analysis results with enhanced persona analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save all standard tables
    tier_patterns.to_csv(output_dir / 'trajectory_patterns_by_tier.csv', index=False)
    trajectory_freqs.to_csv(output_dir / 'trajectory_pattern_frequencies.csv', index=False)
    verdict_dists.to_csv(output_dir / 'verdict_distributions_by_tier.csv', index=False)

    # Save enhanced persona analysis
    if not persona_analysis.empty:
        persona_analysis.to_csv(output_dir / 'persona_performance_analysis.csv', index=False)

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
        'debate_effectiveness': debate_effectiveness,
        'persona_performance': persona_analysis.to_dict(orient='records') if not persona_analysis.empty else []
    }

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate and save enhanced markdown report
    report = generate_enhanced_report(
        df, tier_patterns, trajectory_freqs, verdict_dists,
        debate_effectiveness, persona_analysis
    )

    with open(output_dir / 'analysis_report.md', 'w') as f:
        f.write(report)

    # Save enhanced CSV with all metrics
    df.to_csv(output_dir / 'enhanced_results.csv', index=False)

    print(f"\n[Output] Analysis results saved to: {output_dir}")
    print(f"  - trajectory_patterns_by_tier.csv")
    print(f"  - trajectory_pattern_frequencies.csv")
    print(f"  - verdict_distributions_by_tier.csv")
    if not persona_analysis.empty:
        print(f"  - persona_performance_analysis.csv")
    print(f"  - debate_effectiveness.csv")
    print(f"  - analysis_summary.json")
    print(f"  - analysis_report.md")
    print(f"  - enhanced_results.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory patterns in referee report batch results (enhanced or legacy format)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--results-csv',
        type=str,
        required=True,
        help='Path to batch results CSV file (enhanced or legacy format)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results/analysis_enhanced',
        help='Output directory for analysis results (default: ./results/analysis_enhanced)'
    )

    args = parser.parse_args()

    # Validate inputs
    results_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)

    if not results_path.exists():
        print(f"Error: Results CSV not found: {results_path}")
        sys.exit(1)

    print(f"\n{'='*80}")
    print("ENHANCED TRAJECTORY ANALYSIS")
    print(f"{'='*80}")
    print(f"Results CSV: {results_path}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n[Step 1/7] Loading batch results...")
    df = load_batch_results(str(results_path))
    print(f"  Loaded {len(df)} papers")

    # Compute tier verdict distributions
    print("\n[Step 2/7] Computing tier verdict distributions...")
    verdict_dists = compute_tier_verdict_distributions(df)
    print(f"  Analyzed {len(verdict_dists)} tiers")

    # Compute tier-specific patterns
    print("\n[Step 3/7] Computing tier-specific trajectory patterns...")
    tier_patterns = compute_trajectory_patterns_by_tier(df)
    print(f"  Computed patterns for {len(tier_patterns)} tiers")

    # Compute trajectory pattern frequencies
    print("\n[Step 4/7] Computing trajectory pattern frequencies...")
    trajectory_freqs = compute_trajectory_pattern_frequencies(df)
    print(f"  Found {len(trajectory_freqs)} unique trajectory patterns")

    # Compute debate effectiveness
    print("\n[Step 5/7] Computing debate effectiveness metrics...")
    debate_effectiveness = compute_debate_effectiveness_metrics(df)
    print(f"  Analyzed debate effectiveness for {debate_effectiveness['n_papers']} papers")
    print(f"  Correlation improvement: {debate_effectiveness['correlation_improvement']:.4f} ({debate_effectiveness['interpretation']})")

    # Compute enhanced persona analysis (if available)
    print("\n[Step 6/7] Computing enhanced persona performance analysis...")
    persona_analysis = compute_enhanced_persona_analysis(df)
    if not persona_analysis.empty:
        print(f"  Analyzed {len(persona_analysis)} unique persona types")
    else:
        print(f"  [Skipped] Enhanced metrics not available in this dataset")

    # Export results
    print(f"\n[Step 7/7] Saving analysis results...")
    export_enhanced_results(
        df, tier_patterns, trajectory_freqs, verdict_dists,
        debate_effectiveness, persona_analysis, output_dir
    )

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"View results in: {output_dir}/")
    print(f"Read report: {output_dir}/analysis_report.md")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
