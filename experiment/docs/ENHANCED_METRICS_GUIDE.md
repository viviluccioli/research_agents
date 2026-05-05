# Enhanced Trajectory Metrics Guide

This document explains all 41 new columns added to the enhanced CSV.

## File Location
- **Enhanced CSV**: `experiment/results/referee_batch_results_20260423_160156_enhanced.csv`
- **Total columns**: 62 (21 original + 41 new)

## Variance Metrics (5 measures × 2 rounds = 10 columns)

### Round 1 Variance Metrics (Initial Disagreement)

| Column | Range | Interpretation |
|--------|-------|----------------|
| `round1_verdict_variance` | 0-0.25 | Statistical variance (0 = unanimous) |
| `round1_verdict_std_dev` | 0-0.5 | Standard deviation of verdicts |
| `round1_verdict_range` | 0-1.0 | Max - Min verdict (0=all same, 1=PASS to FAIL) |
| `round1_verdict_entropy` | 0-1.58 | Information entropy (0=unanimous, higher=more diverse) |
| `round1_distance_from_unanimous` | 0-1.0 | Normalized disagreement (0=unanimous, 1=max disagreement) |

### Round 2C Variance Metrics (Final Disagreement)

| Column | Range | Interpretation |
|--------|-------|----------------|
| `round2c_verdict_variance` | 0-0.25 | Statistical variance after debate |
| `round2c_verdict_std_dev` | 0-0.5 | Standard deviation after debate |
| `round2c_verdict_range` | 0-1.0 | Max - Min verdict after debate |
| `round2c_verdict_entropy` | 0-1.58 | Information entropy after debate |
| `round2c_distance_from_unanimous` | 0-1.0 | Normalized disagreement after debate |

**Key Insight:** Compare R1 vs R2C variance to see if debate increased or decreased disagreement.

## Convergence Metrics (3 columns)

| Column | Values | Interpretation |
|--------|--------|----------------|
| `variance_delta` | -0.25 to +0.25 | Change in variance (R2C - R1). Negative = converging |
| `convergence_magnitude` | 0-0.25 | Absolute change in variance |
| `convergence_pattern` | CONVERGING, DIVERGING, STABLE | Classification of convergence behavior |

**Patterns:**
- **CONVERGING**: Variance decreased by >0.05 (personas moved toward agreement)
- **DIVERGING**: Variance increased by >0.05 (personas moved apart)
- **STABLE**: Variance changed by <0.05 (disagreement level unchanged)

## Consensus Metrics (5 columns)

| Column | Range | Interpretation |
|--------|-------|----------------|
| `round1_consensus_score` | 0-1.0 | Weighted consensus in R1 (0=all FAIL, 0.5=all REVISE, 1=all PASS) |
| `round2c_consensus_score` | 0-1.0 | Weighted consensus in R2C |
| `consensus_delta` | -1.0 to +1.0 | Change in consensus (R2C - R1). Negative = stricter |
| `consensus_shift_magnitude` | 0-1.0 | Absolute magnitude of consensus shift |
| `consensus_direction` | STRICTER, LENIENT, UNCHANGED | Direction of consensus change |

**Consensus Score Thresholds:**
- **>0.75**: Strong pass recommendation
- **0.40-0.75**: Resubmit/revise range
- **<0.40**: Strong reject recommendation

## Trajectory Classification (1 column)

| Column | Values | Interpretation |
|--------|--------|----------------|
| `trajectory_pattern` | See below | Overall debate pattern classification |

**Pattern Types:**

**Unanimous patterns** (low variance R1 & R2C):
- `UNANIMOUS_LENIENT`: All personas agree PASS in both rounds
- `UNANIMOUS_MODERATE`: All personas agree REVISE in both rounds
- `UNANIMOUS_STRICT`: All personas agree FAIL in both rounds

**Convergent patterns** (high variance R1 → low variance R2C):
- `CONVERGENT_STRICT`: Mixed in R1, converge to FAIL/REVISE in R2C
- `CONVERGENT_MODERATE`: Mixed in R1, converge to REVISE in R2C
- `CONVERGENT_LENIENT`: Mixed in R1, converge to PASS/REVISE in R2C

**Divergent patterns** (low variance R1 → high variance R2C):
- `DIVERGENT_STRICTER`: Started agreeing, spread out toward stricter verdicts
- `DIVERGENT_LENIENT`: Started agreeing, spread out toward lenient verdicts

**Other**:
- `MIXED`: No clear convergence/divergence pattern

## Persona Change Tracking (4 columns)

| Column | Format | Interpretation |
|--------|--------|----------------|
| `num_personas_changed` | 0-3 | Count of personas that changed verdict |
| `personas_stricter` | Names separated by `;` | Which personas became stricter (e.g., "Econometrician;ML_Expert") |
| `personas_lenient` | Names separated by `;` | Which personas became more lenient |
| `personas_unchanged` | Names separated by `;` | Which personas kept same verdict |

## Per-Persona Metrics (3 personas × 6 metrics = 18 columns)

For each persona (1, 2, 3):

| Column | Format | Interpretation |
|--------|--------|----------------|
| `persona_X_trajectory` | "R1→R2C" | Verdict evolution (e.g., "PASS→FAIL") |
| `persona_X_direction` | STRICTER, LENIENT, UNCHANGED | Direction of change |
| `persona_X_change_magnitude` | 0-1.0 | Absolute magnitude of verdict change |
| `persona_X_contribution_r1` | 0-1.0 | Weighted contribution to R1 consensus |
| `persona_X_contribution_r2c` | 0-1.0 | Weighted contribution to R2C consensus |
| `persona_X_contribution_delta` | -1.0 to +1.0 | Change in contribution (R2C - R1) |

**Example:**
```
persona_1_name: Econometrician
persona_1_weight: 0.5
persona_1_trajectory: REVISE→FAIL
persona_1_direction: STRICTER
persona_1_change_magnitude: 0.5
persona_1_contribution_r1: 0.25 (0.5 weight × 0.5 REVISE value)
persona_1_contribution_r2c: 0.0  (0.5 weight × 0.0 FAIL value)
persona_1_contribution_delta: -0.25
```

## How to Use These Metrics

### 1. Identify High-Disagreement Papers

Sort by any variance metric (variance, std_dev, range, entropy, distance_from_unanimous):

```python
# Papers with highest initial disagreement
df.sort_values('round1_verdict_entropy', ascending=False)

# Papers with highest final disagreement
df.sort_values('round2c_verdict_variance', ascending=False)
```

### 2. Find Papers Where Debate Converged/Diverged

```python
# Most convergent (debate reduced disagreement)
df.nsmallest(10, 'variance_delta')

# Most divergent (debate increased disagreement)
df.nlargest(10, 'variance_delta')

# Filter by pattern
df[df['convergence_pattern'] == 'CONVERGING']
```

### 3. Analyze by Tier

```python
# Compare variance across tiers
df.groupby('tier')[['round1_verdict_variance', 'round2c_verdict_variance']].mean()

# Which tiers show most convergence?
df.groupby('tier')['variance_delta'].mean()
```

### 4. Track Individual Persona Behavior

```python
# Which personas change most often?
for i in range(1, 4):
    changes = df[df[f'persona_{i}_direction'] != 'UNCHANGED']
    print(f"Persona {i}: {len(changes)} changes")

# Most volatile persona (highest average change magnitude)
df[[f'persona_{i}_change_magnitude' for i in range(1, 4)]].mean()
```

### 5. Identify Trajectory Patterns by Tier

```python
# Most common patterns per tier
df.groupby(['tier', 'trajectory_pattern']).size()

# Papers with unanimous strict pattern (all agreed FAIL from start)
df[df['trajectory_pattern'] == 'UNANIMOUS_STRICT']
```

## Example Analysis Queries

### Question: Which papers had the most dramatic debate?

Look for:
- High `consensus_shift_magnitude` (big change in overall score)
- High `convergence_magnitude` (big change in disagreement)
- High `num_personas_changed` (many personas changed verdict)

### Question: Do Tier 1 papers have lower initial variance?

```python
tier1 = df[df['tier'] == 'Tier 1']['round1_verdict_variance'].mean()
tier4 = df[df['tier'] == 'Tier 4']['round1_verdict_variance'].mean()
print(f"Tier 1 avg variance: {tier1}, Tier 4 avg variance: {tier4}")
```

### Question: Which trajectory patterns are most predictive of final verdict?

```python
df.groupby(['trajectory_pattern', 'final_verdict']).size()
```

## Variance Metrics Comparison

**When to use each metric:**

- **variance**: Best for statistical analysis, can be decomposed mathematically
- **std_dev**: More interpretable scale (same units as verdicts), good for effect sizes
- **range**: Simple, intuitive, but ignores middle values
- **entropy**: Information-theoretic, sensitive to distribution shape
- **distance_from_unanimous**: Normalized 0-1 scale, easiest to interpret

**Recommendation:** Use `variance` for statistical tests, `entropy` for measuring diversity, and `distance_from_unanimous` for easy interpretation.

## Column Order in CSV

Original columns (1-21) → New metrics (22-62)

New metrics appear in this order:
1. Consensus scores (5 cols)
2. Round 1 variance metrics (5 cols)
3. Round 2C variance metrics (5 cols)
4. Convergence metrics (3 cols)
5. Persona changes (4 cols)
6. Trajectory pattern (1 col)
7. Per-persona metrics (18 cols: 6 per persona × 3 personas)

## Quick Reference: Key Columns for Analysis

**For variance analysis:**
- `round1_verdict_entropy` - Initial disagreement
- `round2c_verdict_entropy` - Final disagreement
- `variance_delta` - Change in disagreement
- `convergence_pattern` - Converging/Diverging/Stable

**For consensus analysis:**
- `round1_consensus_score` - Initial position
- `round2c_consensus_score` - Final position
- `consensus_delta` - How much changed
- `consensus_direction` - Stricter/Lenient/Unchanged

**For trajectory analysis:**
- `trajectory_pattern` - Overall pattern classification
- `num_personas_changed` - How many changed
- `persona_X_trajectory` - Individual persona paths

**For tier comparison:**
- Group by `tier` and compare any of the above metrics
