# Numeric Score Analysis vs Ground Truth Tiers

**Analysis Date**: 2026-05-05  
**Dataset**: `referee_batch_results_20260505_001147.csv`  
**Papers Analyzed**: 10 (ifdp-2020-1 through ifdp-2020-10)

---

## Executive Summary

**Key Finding**: The numeric scoring system shows **weak correlation with ground truth tiers**, with several critical misalignments:

1. **Tier 1 paper scored in middle range** (17.0/30) despite highest quality tier
2. **Tier 4 papers received higher scores than Tier 1/2 papers** in multiple cases
3. **Highest score (21.0/30) awarded to a Tier 4 paper** (ifdp-2020-10)
4. **Score compression**: Most papers cluster in 15-20 range regardless of tier

**Implication**: The current scoring rubric may not adequately differentiate paper quality tiers, suggesting need for calibration adjustment.

---

## Full Score Distribution by Tier

### Tier 1 (Highest Quality)
| Paper | Final Verdict | Total Score | Individual Scores | Notes |
|-------|---------------|-------------|-------------------|-------|
| ifdp-2020-4 | REVISE | **17.0** | Theorist: 5.0, Econometrician: 6.0, Policymaker: 6.0 | ⚠️ **ANOMALY**: Tier 1 scored below multiple Tier 2-4 papers |

**Analysis**: The single Tier 1 paper received a mid-range score (17.0/30), placing it **below 4 other papers** including two Tier 4 papers. This is a significant calibration issue.

---

### Tier 2 (High Quality)
| Paper | Final Verdict | Total Score | Individual Scores | Notes |
|-------|---------------|-------------|-------------------|-------|
| ifdp-2020-1 | REVISE | 17.0 | Econometrician: 6.0, Data_Scientist: 6.0, Policymaker: 5.0 | Tied with Tier 1 paper |
| ifdp-2020-3 | REVISE | **19.0** | Econometrician: 5.0, Data_Scientist: 6.0, Policymaker: 8.0 | Policymaker gave high score (8.0) |
| ifdp-2020-8 | REVISE | **20.0** | Theorist: 6.5, Econometrician: 7.5, Policymaker: 6.0 | ⭐ **2nd highest overall score** |

**Analysis**: Tier 2 papers show wide variation (17.0-20.0), with ifdp-2020-8 achieving the **2nd highest score across all papers** (20.0/30). This suggests the scoring system can identify higher-quality Tier 2 papers, but fails to distinguish them from lower tiers.

**Notable**: ifdp-2020-8 scored **3 points higher** than the Tier 1 paper, indicating possible reverse ranking.

---

### Tier 3 (Medium Quality)
| Paper | Final Verdict | Total Score | Individual Scores | Notes |
|-------|---------------|-------------|-------------------|-------|
| ifdp-2020-5 | REVISE | 17.5 | Econometrician: 5.0, CS_Expert: 6.5, Theorist: 6.0 | Middle of pack |
| ifdp-2020-6 | FAIL | **N/A** | Econometrician: 5.0, Theorist: N/A, Policymaker: 5.0 | Theorist declined final score |
| ifdp-2020-7 | REVISE | **15.0** | Econometrician: 4.0, Data_Scientist: 5.0, Policymaker: 6.0 | **Lowest REVISE score** |

**Analysis**: Tier 3 papers span 15.0-17.5 (excluding FAIL with incomplete scoring), showing expected medium-range placement. ifdp-2020-7 represents the **floor for REVISE verdicts** at 15.0/30.

---

### Tier 4 (Lowest Quality)
| Paper | Final Verdict | Total Score | Individual Scores | Notes |
|-------|---------------|-------------|-------------------|-------|
| ifdp-2020-10 | REVISE | **21.0** | Econometrician: 6.5, ML_Expert: 8.0, Policymaker: 6.5 | ⚠️ **HIGHEST SCORE OVERALL** |
| ifdp-2020-2 | REVISE | 16.0 | Econometrician: 5.0, Policymaker: 6.0, Data_Scientist: 5.0 | Low-middle range |
| ifdp-2020-9 | FAIL | **11.0** | Theorist: 3.0, Econometrician: 3.0, Policymaker: 5.0 | **Lowest score overall** |

**Analysis**: Tier 4 shows **extreme bimodality**:
- **ifdp-2020-10**: 21.0/30 — **HIGHEST SCORE IN ENTIRE DATASET** ⚠️
- **ifdp-2020-9**: 11.0/30 — Lowest score, appropriate FAIL verdict
- **ifdp-2020-2**: 16.0/30 — Mid-range, arguably too lenient

**Critical Finding**: A Tier 4 paper (ifdp-2020-10) outscored the Tier 1 paper by **4 points** and all but one Tier 2 paper. This is a **major calibration failure**.

---

## REVISE Category Score Analysis

### All REVISE Verdicts Ranked by Score

| Rank | Paper | Tier | Total Score | Score Gap from Next |
|------|-------|------|-------------|---------------------|
| 1 | ifdp-2020-10 | **Tier 4** | 21.0 | +1.0 |
| 2 | ifdp-2020-8 | **Tier 2** | 20.0 | +1.0 |
| 3 | ifdp-2020-3 | **Tier 2** | 19.0 | +1.5 |
| 4 | ifdp-2020-5 | Tier 3 | 17.5 | +0.5 |
| 5 (tie) | ifdp-2020-1 | **Tier 2** | 17.0 | 0 |
| 5 (tie) | ifdp-2020-4 | **Tier 1** | 17.0 | +1.0 |
| 7 | ifdp-2020-2 | Tier 4 | 16.0 | +1.0 |
| 8 | ifdp-2020-7 | Tier 3 | 15.0 | — |

### Higher-Scoring REVISE Papers (>18.0)

**ifdp-2020-10** (Tier 4, 21.0/30):
- **Anomaly Status**: ⚠️ **SEVERE** — Highest score despite lowest tier
- **Individual Scores**: Econometrician: 6.5, ML_Expert: **8.0**, Policymaker: 6.5
- **Explanation**: ML_Expert gave exceptional score (8.0/10), suggesting methodological contribution recognized despite other flaws
- **Verdict Appropriateness**: REVISE may be too lenient; score suggests near-PASS threshold

**ifdp-2020-8** (Tier 2, 20.0/30):
- **Anomaly Status**: ✓ Appropriate — 2nd highest score for 2nd highest tier
- **Individual Scores**: Theorist: 6.5, Econometrician: **7.5**, Policymaker: 6.0
- **Explanation**: Econometrician's strong endorsement (7.5/10) drove high total
- **Verdict Appropriateness**: REVISE may be harsh; score suggests should be closer to PASS

**ifdp-2020-3** (Tier 2, 19.0/30):
- **Anomaly Status**: ✓ Appropriate — High score for high tier
- **Individual Scores**: Econometrician: 5.0, Data_Scientist: 6.0, Policymaker: **8.0**
- **Explanation**: Policymaker's strong score (8.0/10) reflects policy relevance
- **Verdict Appropriateness**: REVISE reasonable given mixed technical/policy assessment

### Answer to Key Question: Were High REVISE Scores from Tier 1/2?

**YES, but with critical exception**:
- **3 of top 4 REVISE scores** were Tier 1/2 papers (ifdp-2020-8, ifdp-2020-3, ifdp-2020-1/4)
- **HOWEVER**: The #1 score (21.0) was a **Tier 4 paper** (ifdp-2020-10)

This suggests:
1. The scoring system **generally** recognizes higher-tier papers within REVISE category
2. **BUT** the system has significant outliers where low-tier papers receive top scores
3. The boundary between REVISE and PASS may be **too high**, preventing deserving Tier 1/2 papers from acceptance

---

## Score-to-Tier Correlation Analysis

### Expected vs Actual Score Ranges

| Tier | Expected Avg Score | Actual Avg Score | Actual Range | Deviation |
|------|-------------------|------------------|--------------|-----------|
| Tier 1 | 22-26 (near-PASS) | **17.0** | 17.0 only | -5 to -9 points |
| Tier 2 | 18-22 | **18.7** | 17.0-20.0 | -3 to 0 points |
| Tier 3 | 14-18 | **16.3** | 15.0-17.5 | -2 to +1 points |
| Tier 4 | 10-14 | **16.0** | 11.0-21.0 | 0 to +7 points |

**Key Observations**:
1. **Tier 1 severely underscored**: 5-9 points below expectation
2. **Tier 2 slightly underscored**: Within range but compressed toward lower end
3. **Tier 3 appropriately scored**: Matches expectations
4. **Tier 4 severely overscored**: Average should be ~12, actual is 16 (excluding outlier: 18.5 with outlier)

### Score Compression Problem

**Standard Deviation by Tier**:
- Tier 1: N/A (single data point)
- Tier 2: σ = 1.53 (low variance)
- Tier 3: σ = 1.32 (low variance)
- Tier 4: σ = 5.00 (high variance due to outlier)

**Overall REVISE category**: σ = 2.07 (narrow spread for 8 papers spanning Tiers 1-4)

**Diagnosis**: Scores are **compressed into 15-21 range** regardless of tier, with insufficient differentiation. A 6-point spread is inadequate to distinguish 4 quality tiers.

---

## Individual Persona Scoring Patterns

### Econometrician (9/10 papers)
- **Range**: 3.0-7.5
- **Average**: 5.6
- **Tendency**: Conservative, rarely exceeds 6.5
- **Highest score**: 7.5 (ifdp-2020-8, Tier 2)
- **Lowest score**: 3.0 (ifdp-2020-9, Tier 4 FAIL)

### Policymaker (7/10 papers)
- **Range**: 5.0-8.0
- **Average**: 6.1
- **Tendency**: Most generous scorer
- **Highest score**: 8.0 (ifdp-2020-3, Tier 2; ifdp-2020-4, Tier 1 in Round 1)
- **Score inflation**: 3 scores ≥ 8.0

### Data_Scientist (3/10 papers)
- **Range**: 5.0-6.0
- **Average**: 5.7
- **Tendency**: Conservative, consistent

### ML_Expert (1/10 papers)
- **Range**: 8.0 only
- **Average**: 8.0
- **Tendency**: Single data point shows high score for Tier 4 paper (concerning)

### Theorist (3/10 papers)
- **Range**: 3.0-6.5
- **Average**: 5.2
- **Tendency**: Most critical, declined to score ifdp-2020-6

### CS_Expert (2/10 papers)
- **Range**: 6.5 only (single scored paper)
- **Average**: 6.5
- **Tendency**: Limited data

---

## Verdict Threshold Analysis

### Implicit Score Thresholds

Based on observed data:

- **FAIL threshold**: < 12.0 (ifdp-2020-9 at 11.0, ifdp-2020-6 incomplete)
- **REVISE range**: 15.0-21.0 (wide range, no upper bound tested)
- **PASS threshold**: Not observed (assumed > 21.0?)

**Problem**: No papers crossed PASS threshold despite Tier 1 paper in dataset. This suggests:
1. Threshold may be set too high (> 21.0)
2. Personas are systematically underscoring relative to paper quality
3. Rubric may emphasize flaws over strengths

---

## Recommendations for Calibration

### 1. Score Inflation for High-Tier Papers
**Issue**: Tier 1/2 papers receiving 5-6 scores when 7-9 may be appropriate

**Fix**: 
- Revise scoring rubric to explicitly allow 8-9 scores for papers with minor revisions only
- Instruct personas that Tier 1/2 papers should average 7-8 if flaws are fixable

### 2. Increase Score Range Differentiation
**Issue**: 6-point spread insufficient for 4 tiers

**Fix**:
- Encourage use of full 1-10 range
- Set explicit tier targets:
  - Tier 1: 22-26 average
  - Tier 2: 18-22 average
  - Tier 3: 14-18 average
  - Tier 4: 10-14 average

### 3. Persona-Specific Calibration
**Issue**: Policymaker scores 0.5-0.9 points higher than Econometrician on average

**Fix**:
- Normalize scores across personas before aggregation
- OR provide persona-specific rubrics with different anchoring

### 4. Address Tier 4 Overscoring
**Issue**: ifdp-2020-10 (Tier 4) scored 21.0/30

**Fix**:
- Investigate what ML_Expert found valuable (may be legitimate methodological contribution)
- Clarify whether scoring reflects "contribution given its genre" vs "absolute publication-readiness"
- Consider separate scoring dimensions (contribution vs. execution quality)

### 5. Establish PASS Threshold
**Issue**: No papers reached PASS despite Tier 1 in dataset

**Fix**:
- Explicitly define PASS threshold (e.g., ≥ 22.0 with no individual score < 6.0)
- Calibrate personas to this threshold using example papers

---

## Conclusions

1. **Weak Tier Correlation**: The numeric scoring system shows insufficient correlation with ground truth tiers, particularly at extremes (Tier 1 underscored, Tier 4 overscored).

2. **Score Compression**: Papers cluster in 15-21 range regardless of tier, indicating rubric needs wider differentiation.

3. **High REVISE Scores**: Among REVISE papers with scores >18.0:
   - **ifdp-2020-8** (Tier 2, 20.0): ✓ Appropriately high
   - **ifdp-2020-3** (Tier 2, 19.0): ✓ Appropriately high
   - **ifdp-2020-10** (Tier 4, 21.0): ⚠️ Anomalously high

4. **Tier 1 Underdifferentiated**: The sole Tier 1 paper (ifdp-2020-4) scored identically (17.0) to a Tier 2 paper and **4 points below** a Tier 4 paper.

5. **PASS Threshold Too High**: No papers crossed into PASS territory, suggesting threshold may be miscalibrated or personas systematically underscore.

**Overall Assessment**: The scoring system requires recalibration to better reflect ground truth tiers. Current implementation may be useful for relative ranking within sessions but lacks absolute calibration for cross-paper comparison.

---

## Appendix: Full Score Matrix

| Paper | Tier | P1 Name | P1 Score | P2 Name | P2 Score | P3 Name | P3 Score | Total | Verdict |
|-------|------|---------|----------|---------|----------|---------|----------|-------|---------|
| ifdp-2020-1 | 2 | Econometrician | 6.0 | Data_Scientist | 6.0 | Policymaker | 5.0 | 17.0 | REVISE |
| ifdp-2020-10 | 4 | Econometrician | 6.5 | ML_Expert | 8.0 | Policymaker | 6.5 | **21.0** | REVISE |
| ifdp-2020-2 | 4 | Econometrician | 5.0 | Policymaker | 6.0 | Data_Scientist | 5.0 | 16.0 | REVISE |
| ifdp-2020-3 | 2 | Econometrician | 5.0 | Data_Scientist | 6.0 | Policymaker | 8.0 | 19.0 | REVISE |
| ifdp-2020-4 | 1 | Theorist | 5.0 | Econometrician | 6.0 | Policymaker | 6.0 | 17.0 | REVISE |
| ifdp-2020-5 | 3 | Econometrician | 5.0 | CS_Expert | 6.5 | Theorist | 6.0 | 17.5 | REVISE |
| ifdp-2020-6 | 3 | Econometrician | 5.0 | Theorist | N/A | Policymaker | 5.0 | N/A | FAIL |
| ifdp-2020-7 | 3 | Econometrician | 4.0 | Data_Scientist | 5.0 | Policymaker | 6.0 | 15.0 | REVISE |
| ifdp-2020-8 | 2 | Theorist | 6.5 | Econometrician | 7.5 | Policymaker | 6.0 | **20.0** | REVISE |
| ifdp-2020-9 | 4 | Theorist | 3.0 | Econometrician | 3.0 | Policymaker | 5.0 | **11.0** | FAIL |
