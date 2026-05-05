# Hybrid Scoring System - Implementation Guide

## Overview

The referee report system now supports **hybrid scoring**: personas provide both:
1. **Categorical verdict** (PASS/REVISE/FAIL) - preserves interpretability
2. **Confidence score** (1-10) - adds granularity for calibration analysis

## What Changed

### 1. Prompt Updates

**Round 1** now requests:
```
**SCORING REQUIREMENT**: In addition to your categorical verdict, provide a confidence score from 1-10:
- 1-3: FAIL zone (reject)
- 4-7: REVISE zone (needs revision)
- 8-10: PASS zone (accept)

**OUTPUT FORMAT**:
- **Verdict**: [PASS/REVISE/FAIL]
- **Confidence Score**: [X/10]
```

**Round 2C** now requests:
```
- **Final Verdict**: [PASS / REVISE / FAIL]
- **Final Score**: [X/10] (1-3=FAIL, 4-7=REVISE, 8-10=PASS)
- **Final Rationale**: [...]
```

### 2. New Functions (`engine.py`)

**`extract_score_from_report(report: str) -> Optional[float]`**
- Extracts confidence scores from reports
- Looks for patterns like "**Final Score**: 7/10"
- Returns float (1-10) or None if not found

**Updated `calculate_consensus()`**:
- Now returns both categorical and numeric consensus scores
- Categorical: PASS=1.0, REVISE=0.5, FAIL=0.0 (existing method)
- Numeric: Weighted average of scores/10 (new method)

```python
{
    'verdicts': {'Econometrician': 'REVISE', ...},
    'scores': {'Econometrician': 6.0, ...},
    'weighted_score_categorical': 0.425,  # Legacy
    'weighted_score_numeric': 0.550,      # New (6.0+5.0+5.5)/3/10
    'decision': 'REJECT AND RESUBMIT'
}
```

### 3. Enhanced CSV Outputs (`batch_referee_reports.py`)

**New columns per persona:**
- `persona_X_round1_score` - Initial confidence score (1-10)
- `persona_X_final_score` - Final confidence score (1-10)

**New aggregate columns:**
- `round1_consensus_score_numeric` - R1 weighted score (0-1, from numeric scores)
- `round2c_consensus_score_numeric` - R2C weighted score (0-1, from numeric scores)
- `consensus_delta_numeric` - Change in numeric consensus (R2C - R1)

**Backwards compatible:**
- All existing columns remain (categorical consensus scores)
- New columns are `None` if personas don't provide scores

## How to Use

### Running Experiments

```bash
cd experiment

# Run on sample papers
python batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth ../experiment-papers/IFDP_2020/IFDP_2020_tracking_clean.csv \
    --output-dir results/ \
    --limit 3

# Output includes both categorical and numeric scores
```

### Analyzing Results

**1. Check if numeric scores are available:**
```python
import pandas as pd
df = pd.read_csv('results/referee_batch_results_TIMESTAMP.csv')

# Check score availability
score_availability = {
    'persona_1_r1': df['persona_1_round1_score'].notna().sum(),
    'persona_1_r2c': df['persona_1_final_score'].notna().sum(),
    'consensus_numeric': df['round2c_consensus_score_numeric'].notna().sum()
}
print(score_availability)
```

**2. Compare categorical vs numeric consensus:**
```python
# Papers with both score types
both = df[df['round2c_consensus_score_numeric'].notna()].copy()

# Correlation
correlation = both[['round2c_consensus_score', 'round2c_consensus_score_numeric']].corr()
print(correlation)

# Divergence cases (categorical says one thing, numeric says another)
both['score_divergence'] = abs(both['round2c_consensus_score'] - both['round2c_consensus_score_numeric'])
divergent = both.nlargest(5, 'score_divergence')
```

**3. Analyze score granularity:**
```python
# Distribution of numeric scores by tier
import matplotlib.pyplot as plt

for tier in df['tier'].unique():
    tier_df = df[df['tier'] == tier]
    scores = []
    for i in range(1, 4):
        scores.extend(tier_df[f'persona_{i}_final_score'].dropna())

    plt.hist(scores, bins=10, alpha=0.5, label=tier)

plt.xlabel('Confidence Score (1-10)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Score Distribution by Tier')
plt.show()
```

**4. Identify borderline papers:**
```python
# Papers where personas cluster near category boundaries
# E.g., scores 3-4 (FAIL/REVISE boundary) or 7-8 (REVISE/PASS boundary)

def is_borderline_score(score):
    if score is None:
        return False
    return (3 <= score <= 4) or (7 <= score <= 8)

df['has_borderline_scores'] = df.apply(
    lambda row: any(
        is_borderline_score(row[f'persona_{i}_final_score'])
        for i in range(1, 4)
    ),
    axis=1
)

borderline_papers = df[df['has_borderline_scores']]
```

## Benefits of Hybrid Approach

### 1. Higher Granularity
- **Before**: All FAIL verdicts treated equally
- **After**: Can distinguish "borderline fail" (score 3) from "clear reject" (score 1)

### 2. Better Calibration Metrics
- **Before**: Calibration error on 3-point scale (0, 0.5, 1.0)
- **After**: Calibration error on continuous scale (0-1.0)

**Example:**
```python
# Categorical (coarse)
Tier 1 papers: average = 0.18 (expected 0.90)
  → Error: 0.72, but can't tell if consensus was "almost REVISE" or "clearly FAIL"

# Numeric (fine-grained)
Tier 1 papers: average = 0.35 (from scores 3, 4, 3)
  → Error: 0.55, and we know personas were borderline FAIL/REVISE
```

### 3. Debate Dynamics Visible
- **Before**: REVISE→REVISE looks stable even if opinion shifted
- **After**: Can see 6→5 (becoming stricter within REVISE zone)

### 4. Threshold Analysis
Current thresholds (PASS ≥0.75, REVISE 0.40-0.75, FAIL <0.40) may be miscalibrated.

With numeric scores, you can test alternative thresholds:
```python
# Test if moving boundary helps calibration
def test_threshold(df, pass_threshold=0.70, revise_threshold=0.35):
    def classify(score):
        if score >= pass_threshold:
            return 'PASS'
        elif score >= revise_threshold:
            return 'REVISE'
        else:
            return 'FAIL'

    df['alternative_verdict'] = df['round2c_consensus_score_numeric'].apply(classify)
    # Compare against ground truth...
```

## Limitations

### 1. Model Consistency
Personas may not use the 1-10 scale consistently:
- Some may cluster around 5 (middle)
- Others may use full range

**Mitigation**: Analyze per-persona score distributions

### 2. False Precision
Score "6" vs "7" may not have meaningful difference

**Mitigation**: Focus on score ranges (1-3, 4-5, 6-7, 8-10) rather than individual points

### 3. Backwards Compatibility
Old results don't have scores

**Solution**: `add_calibration_metrics.py` still works with categorical scores

## Testing Checklist

Before running large batches:

- [ ] Run on 1-2 test papers
- [ ] Verify scores appear in output CSV
- [ ] Check that scores align with verdicts (PASS papers should have 8-10 scores)
- [ ] Confirm both categorical and numeric consensus are computed
- [ ] Test `add_calibration_metrics.py` on new CSV

## Expected Score Benchmarks

Update these based on numeric data:

| Tier | Expected Categorical | Expected Numeric (suggested) |
|------|---------------------|------------------------------|
| Tier 1 (top journal) | 0.90 | 0.85 (score 8.5/10) |
| Tier 2 (good journal) | 0.70 | 0.65 (score 6.5/10) |
| Tier 3 (mediocre) | 0.30 | 0.35 (score 3.5/10) |
| Tier 4 (not accepted) | 0.10 | 0.20 (score 2.0/10) |

After initial run, refine these based on observed distributions.

## Next Steps

1. **Run pilot** (3-5 papers) to validate score extraction
2. **Analyze score distributions** by persona and tier
3. **Compare calibration** categorical vs numeric
4. **Refine expected scores** if needed
5. **Scale to full batch** if improvements are significant

## Questions Answered by Numeric Scores

✅ **"Are Tier 1 papers borderline FAIL or clearly FAIL?"**
   → Check if scores cluster at 3-4 (borderline) or 1-2 (clear)

✅ **"Does debate cause small refinements or dramatic shifts?"**
   → Compute average score change: 1 point (refinement) vs 4 points (flip)

✅ **"Should verdict thresholds be adjusted?"**
   → Test alternative boundaries against ground truth

✅ **"Which personas are most/least confident?"**
   → Analyze score variance per persona

✅ **"Do unanimous verdicts have high scores or just agreement?"**
   → Compare: all FAIL with scores [3,3,3] vs [1,1,1]

