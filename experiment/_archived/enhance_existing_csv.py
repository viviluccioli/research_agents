#!/usr/bin/env python3
"""
Enhance Existing CSV with Trajectory Metrics

Takes an existing referee report CSV and appends trajectory metrics as new columns,
preserving all original columns. Focus on variance metrics to capture debate spread.

Usage:
    python enhance_existing_csv.py \
        --input-csv results/referee_batch_results_20260423_160156.csv \
        --output-csv results/referee_batch_results_20260423_160156_enhanced.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


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


# ==============================================================================
# ENHANCED VARIANCE METRICS
# ==============================================================================

def compute_consensus_score(verdicts: List[str], weights: List[float]) -> float:
    """Compute weighted consensus score from verdicts and weights."""
    if len(verdicts) != len(weights):
        return 0.0
    score = sum(
        VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) * w
        for v, w in zip(verdicts, weights)
    )
    return round(score, 4)


def compute_verdict_variance(verdicts: List[str]) -> float:
    """Compute variance in verdicts (0=unanimous, higher=more disagreement)."""
    if not verdicts:
        return 0.0
    values = [VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) for v in verdicts]
    if len(set(values)) == 1:
        return 0.0
    return round(float(np.var(values)), 4)


def compute_verdict_std_dev(verdicts: List[str]) -> float:
    """Compute standard deviation of verdicts."""
    if not verdicts:
        return 0.0
    values = [VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) for v in verdicts]
    if len(set(values)) == 1:
        return 0.0
    return round(float(np.std(values)), 4)


def compute_verdict_range(verdicts: List[str]) -> float:
    """Compute range of verdicts (max - min)."""
    if not verdicts:
        return 0.0
    values = [VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) for v in verdicts]
    return round(max(values) - min(values), 4)


def compute_verdict_entropy(verdicts: List[str]) -> float:
    """
    Compute information entropy of verdicts (measure of disagreement).
    0 = unanimous, higher = more diverse opinions.
    """
    if not verdicts:
        return 0.0

    # Count each verdict type
    from collections import Counter
    counts = Counter([v.upper().strip() if v else '' for v in verdicts])
    counts.pop('', None)  # Remove empty strings

    if not counts or len(counts) == 1:
        return 0.0

    # Compute entropy
    total = sum(counts.values())
    entropy = -sum((c/total) * np.log2(c/total) for c in counts.values())
    return round(entropy, 4)


def compute_distance_from_unanimous(verdicts: List[str]) -> float:
    """
    Compute how far verdicts are from unanimous agreement.
    0 = unanimous, 1 = maximum disagreement.
    """
    if not verdicts:
        return 0.0

    values = [VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) for v in verdicts]
    unique_values = len(set(values))

    if unique_values == 1:
        return 0.0

    # Normalize: 3 unique values (PASS, REVISE, FAIL) = maximum disagreement
    max_unique = min(3, len(verdicts))
    return round((unique_values - 1) / (max_unique - 1), 4)


def compute_convergence_magnitude(r1_variance: float, r2c_variance: float) -> float:
    """
    Compute absolute change in variance (convergence magnitude).
    Positive = diverging, Negative = converging.
    """
    return round(r2c_variance - r1_variance, 4)


def compute_consensus_shift_magnitude(r1_score: float, r2c_score: float) -> float:
    """Compute absolute change in consensus score."""
    return round(abs(r2c_score - r1_score), 4)


def classify_direction(r1_score: float, r2c_score: float, threshold: float = 0.05) -> str:
    """Classify consensus direction change."""
    delta = r2c_score - r1_score
    if abs(delta) < threshold:
        return 'UNCHANGED'
    elif delta < 0:
        return 'STRICTER'
    else:
        return 'LENIENT'


def classify_convergence_pattern(r1_variance: float, r2c_variance: float, threshold: float = 0.05) -> str:
    """Classify convergence/divergence pattern."""
    delta = r2c_variance - r1_variance
    if abs(delta) < threshold:
        return 'STABLE'
    elif delta < 0:
        return 'CONVERGING'
    else:
        return 'DIVERGING'


def classify_trajectory_pattern(
    r1_verdicts: List[str],
    r2c_verdicts: List[str],
    r1_score: float,
    r2c_score: float,
    r1_variance: float,
    r2c_variance: float
) -> str:
    """Classify the trajectory pattern of the debate."""
    direction = classify_direction(r1_score, r2c_score)

    # Unanimous patterns (low variance in both rounds)
    if r1_variance < 0.05 and r2c_variance < 0.05:
        if r2c_score >= 0.8:
            return 'UNANIMOUS_LENIENT'
        elif r2c_score <= 0.2:
            return 'UNANIMOUS_STRICT'
        else:
            return 'UNANIMOUS_MODERATE'

    # Convergent patterns (high variance in R1, low variance in R2C)
    if r1_variance > r2c_variance + 0.05:
        if r2c_score <= 0.2:
            return 'CONVERGENT_STRICT'
        elif r2c_score >= 0.8:
            return 'CONVERGENT_LENIENT'
        else:
            return 'CONVERGENT_MODERATE'

    # Divergent patterns (low variance in R1, high variance in R2C)
    if r2c_variance > r1_variance + 0.05:
        if direction == 'STRICTER':
            return 'DIVERGENT_STRICTER'
        else:
            return 'DIVERGENT_LENIENT'

    return 'MIXED'


def get_trajectory_string(r1: str, r2c: str) -> str:
    """Create trajectory string like 'PASS→FAIL'."""
    return f"{r1}→{r2c}" if (r1 and r2c) else ""


def get_persona_direction(r1_verdict: str, r2c_verdict: str) -> str:
    """Classify individual persona direction change."""
    r1_val = VERDICT_VALUES.get(r1_verdict.upper().strip() if r1_verdict else '', 0.0)
    r2c_val = VERDICT_VALUES.get(r2c_verdict.upper().strip() if r2c_verdict else '', 0.0)

    if abs(r2c_val - r1_val) < 0.01:
        return 'UNCHANGED'
    elif r2c_val < r1_val:
        return 'STRICTER'
    else:
        return 'LENIENT'


def compute_persona_verdict_change_magnitude(r1_verdict: str, r2c_verdict: str) -> float:
    """Compute absolute magnitude of persona verdict change."""
    r1_val = VERDICT_VALUES.get(r1_verdict.upper().strip() if r1_verdict else '', 0.0)
    r2c_val = VERDICT_VALUES.get(r2c_verdict.upper().strip() if r2c_verdict else '', 0.0)
    return round(abs(r2c_val - r1_val), 4)


# ==============================================================================
# MAIN ENHANCEMENT FUNCTION
# ==============================================================================

def enhance_row_with_trajectory_metrics(row: pd.Series) -> pd.Series:
    """
    Add trajectory metrics to a single row.

    Returns a Series with new metric columns.
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

    # Initialize metrics with defaults
    metrics = {
        # Consensus scores
        'round1_consensus_score': 0.0,
        'round2c_consensus_score': 0.0,
        'consensus_delta': 0.0,
        'consensus_shift_magnitude': 0.0,
        'consensus_direction': '',

        # Variance metrics (ENHANCED)
        'round1_verdict_variance': 0.0,
        'round1_verdict_std_dev': 0.0,
        'round1_verdict_range': 0.0,
        'round1_verdict_entropy': 0.0,
        'round1_distance_from_unanimous': 0.0,

        'round2c_verdict_variance': 0.0,
        'round2c_verdict_std_dev': 0.0,
        'round2c_verdict_range': 0.0,
        'round2c_verdict_entropy': 0.0,
        'round2c_distance_from_unanimous': 0.0,

        'variance_delta': 0.0,
        'convergence_magnitude': 0.0,
        'convergence_pattern': '',

        # Persona changes
        'num_personas_changed': 0,
        'personas_stricter': '',
        'personas_lenient': '',
        'personas_unchanged': '',

        # Pattern classification
        'trajectory_pattern': '',

        # Per-persona metrics
        'persona_1_trajectory': '',
        'persona_1_direction': '',
        'persona_1_change_magnitude': 0.0,
        'persona_1_contribution_r1': 0.0,
        'persona_1_contribution_r2c': 0.0,
        'persona_1_contribution_delta': 0.0,

        'persona_2_trajectory': '',
        'persona_2_direction': '',
        'persona_2_change_magnitude': 0.0,
        'persona_2_contribution_r1': 0.0,
        'persona_2_contribution_r2c': 0.0,
        'persona_2_contribution_delta': 0.0,

        'persona_3_trajectory': '',
        'persona_3_direction': '',
        'persona_3_change_magnitude': 0.0,
        'persona_3_contribution_r1': 0.0,
        'persona_3_contribution_r2c': 0.0,
        'persona_3_contribution_delta': 0.0,
    }

    if not personas:
        return pd.Series(metrics)

    # Compute consensus scores
    r1_verdicts = [p['r1_verdict'] for p in personas]
    r2c_verdicts = [p['r2c_verdict'] for p in personas]
    weights = [p['weight'] for p in personas]

    r1_score = compute_consensus_score(r1_verdicts, weights)
    r2c_score = compute_consensus_score(r2c_verdicts, weights)

    # Compute ENHANCED variance metrics
    r1_variance = compute_verdict_variance(r1_verdicts)
    r1_std_dev = compute_verdict_std_dev(r1_verdicts)
    r1_range = compute_verdict_range(r1_verdicts)
    r1_entropy = compute_verdict_entropy(r1_verdicts)
    r1_dist_unanimous = compute_distance_from_unanimous(r1_verdicts)

    r2c_variance = compute_verdict_variance(r2c_verdicts)
    r2c_std_dev = compute_verdict_std_dev(r2c_verdicts)
    r2c_range = compute_verdict_range(r2c_verdicts)
    r2c_entropy = compute_verdict_entropy(r2c_verdicts)
    r2c_dist_unanimous = compute_distance_from_unanimous(r2c_verdicts)

    # Classify personas by change
    personas_stricter = []
    personas_lenient = []
    personas_unchanged = []

    for p in personas:
        direction = get_persona_direction(p['r1_verdict'], p['r2c_verdict'])
        if direction == 'STRICTER':
            personas_stricter.append(p['name'])
        elif direction == 'LENIENT':
            personas_lenient.append(p['name'])
        else:
            personas_unchanged.append(p['name'])

    # Classify patterns
    trajectory_pattern = classify_trajectory_pattern(
        r1_verdicts, r2c_verdicts, r1_score, r2c_score,
        r1_variance, r2c_variance
    )

    # Update metrics
    metrics.update({
        'round1_consensus_score': r1_score,
        'round2c_consensus_score': r2c_score,
        'consensus_delta': round(r2c_score - r1_score, 4),
        'consensus_shift_magnitude': compute_consensus_shift_magnitude(r1_score, r2c_score),
        'consensus_direction': classify_direction(r1_score, r2c_score),

        'round1_verdict_variance': r1_variance,
        'round1_verdict_std_dev': r1_std_dev,
        'round1_verdict_range': r1_range,
        'round1_verdict_entropy': r1_entropy,
        'round1_distance_from_unanimous': r1_dist_unanimous,

        'round2c_verdict_variance': r2c_variance,
        'round2c_verdict_std_dev': r2c_std_dev,
        'round2c_verdict_range': r2c_range,
        'round2c_verdict_entropy': r2c_entropy,
        'round2c_distance_from_unanimous': r2c_dist_unanimous,

        'variance_delta': round(r2c_variance - r1_variance, 4),
        'convergence_magnitude': compute_convergence_magnitude(r1_variance, r2c_variance),
        'convergence_pattern': classify_convergence_pattern(r1_variance, r2c_variance),

        'num_personas_changed': len(personas_stricter) + len(personas_lenient),
        'personas_stricter': ';'.join(personas_stricter),
        'personas_lenient': ';'.join(personas_lenient),
        'personas_unchanged': ';'.join(personas_unchanged),

        'trajectory_pattern': trajectory_pattern,
    })

    # Compute per-persona metrics
    for i, p in enumerate(personas, start=1):
        traj = get_trajectory_string(p['r1_verdict'], p['r2c_verdict'])
        direction = get_persona_direction(p['r1_verdict'], p['r2c_verdict'])
        change_mag = compute_persona_verdict_change_magnitude(p['r1_verdict'], p['r2c_verdict'])

        r1_contrib = p['weight'] * VERDICT_VALUES.get(p['r1_verdict'].upper().strip() if p['r1_verdict'] else '', 0.0)
        r2c_contrib = p['weight'] * VERDICT_VALUES.get(p['r2c_verdict'].upper().strip() if p['r2c_verdict'] else '', 0.0)

        metrics[f'persona_{i}_trajectory'] = traj
        metrics[f'persona_{i}_direction'] = direction
        metrics[f'persona_{i}_change_magnitude'] = change_mag
        metrics[f'persona_{i}_contribution_r1'] = round(r1_contrib, 4)
        metrics[f'persona_{i}_contribution_r2c'] = round(r2c_contrib, 4)
        metrics[f'persona_{i}_contribution_delta'] = round(r2c_contrib - r1_contrib, 4)

    return pd.Series(metrics)


def enhance_csv_with_trajectory_metrics(input_csv: str, output_csv: str):
    """
    Load existing CSV, add trajectory metrics as new columns, and save.
    """
    print(f"\n{'='*80}")
    print("ENHANCE EXISTING CSV WITH TRAJECTORY METRICS")
    print(f"{'='*80}")
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")

    # Load existing CSV
    print("\n[Step 1/3] Loading existing CSV...")
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Compute trajectory metrics for each row
    print("\n[Step 2/3] Computing trajectory metrics for each paper...")
    trajectory_metrics = df.apply(enhance_row_with_trajectory_metrics, axis=1)
    print(f"  Computed {len(trajectory_metrics.columns)} new metric columns")

    # Concatenate original and new columns
    print("\n[Step 3/3] Appending new columns to original data...")
    enhanced_df = pd.concat([df, trajectory_metrics], axis=1)
    print(f"  Enhanced CSV now has {len(enhanced_df.columns)} total columns")

    # Save enhanced CSV
    enhanced_df.to_csv(output_csv, index=False)
    print(f"\n✓ Enhanced CSV saved to: {output_csv}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Original columns: {len(df.columns)}")
    print(f"New columns added: {len(trajectory_metrics.columns)}")
    print(f"Total columns: {len(enhanced_df.columns)}")

    print(f"\nNew variance metrics added:")
    print(f"  - Round 1: variance, std_dev, range, entropy, distance_from_unanimous")
    print(f"  - Round 2C: variance, std_dev, range, entropy, distance_from_unanimous")
    print(f"  - Convergence: variance_delta, convergence_magnitude, convergence_pattern")

    print(f"\nOther new metrics:")
    print(f"  - Consensus: round1/round2c scores, delta, shift_magnitude, direction")
    print(f"  - Trajectory: pattern classification, persona changes")
    print(f"  - Per-persona: trajectory, direction, change_magnitude, contributions")

    print(f"\n{'='*80}")
    print(f"View enhanced results in Excel or pandas!")
    print(f"{'='*80}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Enhance existing referee report CSV with trajectory metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input-csv',
        type=str,
        required=True,
        help='Path to existing batch results CSV file'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        default=None,
        help='Path for enhanced output CSV (default: input_enhanced.csv)'
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input CSV not found: {input_path}")
        sys.exit(1)

    # Set output path
    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        # Default: add _enhanced before extension
        output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"

    # Run enhancement
    enhance_csv_with_trajectory_metrics(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
