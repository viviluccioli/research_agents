#!/usr/bin/env python3
"""
Add Focused Calibration Metrics to Existing CSV

Adds just 8 essential columns to understand tool calibration:
1. Is the tool calibrated? (scores align with tiers)
2. Is the tool confident? (high agreement = confident)
3. Does debate add value? (improved calibration)
4. Which papers are genuinely borderline?

Usage:
    python add_calibration_metrics.py \
        --input-csv results/referee_batch_results_20260423_160156.csv \
        --output-csv results/referee_batch_results_20260423_160156_calibrated.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================

VERDICT_VALUES = {
    'PASS': 1.0,
    'REVISE': 0.5,
    'FAIL': 0.0,
    'REJECT': 0.0,
    'UNKNOWN': 0.0,
    '': 0.0
}

# Expected consensus scores for each tier (for calibration)
EXPECTED_SCORES = {
    'Tier 1': 0.90,  # Top journal - should get high scores
    'Tier 2': 0.70,  # Good journal - should get moderate-high scores
    'Tier 3': 0.30,  # Mediocre journal - should get moderate-low scores
    'Tier 4': 0.10   # Not accepted - should get low scores
}


# ==============================================================================
# CORE METRIC FUNCTIONS
# ==============================================================================

def compute_consensus_score(verdicts: List[str], weights: List[float]) -> float:
    """Compute weighted consensus score (0-1 scale)."""
    if len(verdicts) != len(weights) or not verdicts:
        return 0.0
    score = sum(
        VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) * w
        for v, w in zip(verdicts, weights)
    )
    return round(score, 4)


def compute_agreement_level(verdicts: List[str]) -> float:
    """
    Compute agreement level (0-1 scale).
    1.0 = unanimous (all same verdict)
    0.0 = maximum disagreement (all different verdicts)
    """
    if not verdicts:
        return 0.0

    # Convert to values
    values = [VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) for v in verdicts]

    # Count unique values
    unique_values = len(set(values))

    if unique_values == 1:
        return 1.0  # Unanimous
    elif unique_values == 2:
        return 0.5  # Two different verdicts
    else:
        return 0.0  # Maximum disagreement (all three different)


def get_expected_score(tier: str) -> float:
    """Get expected consensus score for a given tier."""
    return EXPECTED_SCORES.get(tier, 0.5)


def compute_calibration_error(consensus_score: float, expected_score: float) -> float:
    """Compute absolute calibration error."""
    return round(abs(consensus_score - expected_score), 4)


def is_borderline(agreement_level: float, tier: str, threshold: float = 0.5) -> str:
    """
    Flag borderline papers (genuinely ambiguous vs. tool failures).

    Borderline = low agreement AND middle-tier paper
    """
    if agreement_level < threshold and tier in ['Tier 2', 'Tier 3']:
        return 'Yes'
    return 'No'


def did_debate_improve_calibration(
    r1_score: float,
    r2c_score: float,
    expected_score: float
) -> str:
    """
    Check if debate moved score closer to expected range.
    """
    r1_error = abs(r1_score - expected_score)
    r2c_error = abs(r2c_score - expected_score)

    if r2c_error < r1_error:
        return 'Yes'
    elif r2c_error > r1_error:
        return 'No'
    else:
        return 'No Change'


def compute_verdict_stability(r1_score: float, r2c_score: float, threshold: float = 0.15) -> str:
    """
    Classify verdict stability from R1 to R2C.

    Stable: Final verdict in same direction as R1
    Shifted: Moved but stayed in same category
    Flipped: Changed category (e.g., PASS→FAIL)
    """
    delta = r2c_score - r1_score

    # Determine R1 and R2C categories
    def get_category(score):
        if score >= 0.75:
            return 'PASS'
        elif score >= 0.40:
            return 'REVISE'
        else:
            return 'FAIL'

    r1_cat = get_category(r1_score)
    r2c_cat = get_category(r2c_score)

    if r1_cat == r2c_cat:
        return 'Stable'
    elif abs(delta) < threshold:
        return 'Stable'
    elif (r1_cat == 'PASS' and r2c_cat == 'FAIL') or (r1_cat == 'FAIL' and r2c_cat == 'PASS'):
        return 'Flipped'
    else:
        return 'Shifted'


# ==============================================================================
# MAIN ENHANCEMENT FUNCTION
# ==============================================================================

def add_calibration_metrics_to_row(row: pd.Series) -> pd.Series:
    """
    Add 8 focused calibration metrics to a single row.
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

    # Initialize metrics
    metrics = {
        'consensus_score_r1': 0.0,
        'consensus_score_r2c': 0.0,
        'expected_score_for_tier': 0.0,
        'calibration_error': 0.0,
        'agreement_level_r1': 0.0,
        'agreement_level_r2c': 0.0,
        'borderline_flag': 'No',
        'debate_improved_calibration': 'No',
        'agreement_change': 0.0,
        'verdict_stability': 'Unknown'
    }

    if not personas:
        return pd.Series(metrics)

    # Extract verdicts and weights
    r1_verdicts = [p['r1_verdict'] for p in personas]
    r2c_verdicts = [p['r2c_verdict'] for p in personas]
    weights = [p['weight'] for p in personas]

    # 1. Consensus scores
    r1_score = compute_consensus_score(r1_verdicts, weights)
    r2c_score = compute_consensus_score(r2c_verdicts, weights)

    # 2. Expected score for tier
    tier = row.get('tier', '')
    expected_score = get_expected_score(tier)

    # 3. Calibration error
    calibration_error = compute_calibration_error(r2c_score, expected_score)

    # 4. Agreement levels
    r1_agreement = compute_agreement_level(r1_verdicts)
    r2c_agreement = compute_agreement_level(r2c_verdicts)

    # 5. Borderline flag
    borderline = is_borderline(r2c_agreement, tier)

    # 6. Debate improved calibration
    debate_improved = did_debate_improve_calibration(r1_score, r2c_score, expected_score)

    # 7. Agreement change
    agreement_change = round(r2c_agreement - r1_agreement, 4)

    # 8. Verdict stability
    stability = compute_verdict_stability(r1_score, r2c_score)

    # Update metrics
    metrics.update({
        'consensus_score_r1': r1_score,
        'consensus_score_r2c': r2c_score,
        'expected_score_for_tier': expected_score,
        'calibration_error': calibration_error,
        'agreement_level_r1': r1_agreement,
        'agreement_level_r2c': r2c_agreement,
        'borderline_flag': borderline,
        'debate_improved_calibration': debate_improved,
        'agreement_change': agreement_change,
        'verdict_stability': stability
    })

    return pd.Series(metrics)


def enhance_csv_with_calibration_metrics(input_csv: str, output_csv: str):
    """
    Load existing CSV, add calibration metrics, and save.
    """
    print(f"\n{'='*80}")
    print("ADD FOCUSED CALIBRATION METRICS")
    print(f"{'='*80}")
    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")

    # Load existing CSV
    print("\n[Step 1/3] Loading existing CSV...")
    df = pd.read_csv(input_csv)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    # Compute calibration metrics for each row
    print("\n[Step 2/3] Computing calibration metrics for each paper...")
    calibration_metrics = df.apply(add_calibration_metrics_to_row, axis=1)
    print(f"  Computed {len(calibration_metrics.columns)} new calibration metrics")

    # Concatenate original and new columns
    print("\n[Step 3/3] Appending calibration metrics to original data...")
    enhanced_df = pd.concat([df, calibration_metrics], axis=1)
    print(f"  Enhanced CSV now has {len(enhanced_df.columns)} total columns")

    # Save enhanced CSV
    enhanced_df.to_csv(output_csv, index=False)
    print(f"\n✓ Enhanced CSV saved to: {output_csv}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*80}")

    print("\n1. OVERALL CALIBRATION ERROR BY TIER")
    print("   (Lower = better alignment with expected tier quality)")
    print("-" * 60)
    tier_errors = enhanced_df.groupby('tier')['calibration_error'].agg(['mean', 'std', 'min', 'max'])
    print(tier_errors.round(3))

    print("\n2. AGREEMENT LEVELS BY TIER")
    print("   (Higher = more confident/unanimous verdicts)")
    print("-" * 60)
    tier_agreement = enhanced_df.groupby('tier')[['agreement_level_r1', 'agreement_level_r2c']].mean()
    print(tier_agreement.round(3))

    print("\n3. DEBATE EFFECTIVENESS")
    print("   (Did debate improve calibration?)")
    print("-" * 60)
    debate_effectiveness = enhanced_df['debate_improved_calibration'].value_counts()
    print(debate_effectiveness)

    print("\n4. BORDERLINE PAPERS")
    print("   (Genuinely ambiguous papers)")
    print("-" * 60)
    borderline_count = enhanced_df['borderline_flag'].value_counts()
    print(borderline_count)
    borderline_papers = enhanced_df[enhanced_df['borderline_flag'] == 'Yes'][['doc_id', 'tier', 'calibration_error', 'agreement_level_r2c']]
    if len(borderline_papers) > 0:
        print("\nBorderline papers:")
        print(borderline_papers.to_string(index=False))

    print("\n5. MISCALIBRATION ALERTS")
    print("   (High error + high confidence = wrong but confident)")
    print("-" * 60)
    miscalibrated = enhanced_df[
        (enhanced_df['calibration_error'] > 0.4) &
        (enhanced_df['agreement_level_r2c'] > 0.8)
    ][['doc_id', 'tier', 'consensus_score_r2c', 'expected_score_for_tier', 'calibration_error']]

    if len(miscalibrated) > 0:
        print("\n⚠️  Papers with high confidence but large calibration error:")
        print(miscalibrated.to_string(index=False))
    else:
        print("\n✓ No high-confidence miscalibrations detected")

    print(f"\n{'='*80}")
    print("METRICS ADDED (10 columns)")
    print(f"{'='*80}")
    print("""
CALIBRATION METRICS:
  consensus_score_r1          - R1 weighted score (0-1)
  consensus_score_r2c         - R2C weighted score (0-1)
  expected_score_for_tier     - Expected score for this tier (0-1)
  calibration_error           - abs(consensus - expected)

CONFIDENCE METRICS:
  agreement_level_r1          - R1 unanimity (1=all agree, 0=all differ)
  agreement_level_r2c         - R2C unanimity
  borderline_flag             - Yes/No for ambiguous papers

DEBATE VALUE METRICS:
  debate_improved_calibration - Yes/No (moved closer to expected)
  agreement_change            - Change in unanimity (R2C - R1)
  verdict_stability           - Stable/Shifted/Flipped
""")

    print(f"{'='*80}")
    print("View enhanced results in Excel or pandas!")
    print(f"{'='*80}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Add focused calibration metrics to referee report CSV",
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
        help='Path for enhanced output CSV (default: input_calibrated.csv)'
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
        # Default: add _calibrated before extension
        output_path = input_path.parent / f"{input_path.stem}_calibrated{input_path.suffix}"

    # Run enhancement
    enhance_csv_with_calibration_metrics(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
