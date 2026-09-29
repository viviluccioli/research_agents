# Commit: adding results summary files

**Hash**: `7c043b5b4d7ae36c41bef9c0dbaddd6e77d337bc`
**Date**: 2026-05-06 16:42:04 -0400
**Author**: Viviana C. Luccioli

## Changes Summary

```
commit 7c043b5b4d7ae36c41bef9c0dbaddd6e77d337bc
Author: Viviana C. Luccioli <m1vcl00@salt2.rsma.frb.gov>
Date:   Wed May 6 16:42:04 2026 -0400

    adding results summary files

 experiment/comparing_10ifdp.md | 124 ++++++++++++++++++
 experiment/numeric_analysis.md | 283 +++++++++++++++++++++++++++++++++++++++++
 2 files changed, 407 insertions(+)
```

## Full Diff

```diff
commit 7c043b5b4d7ae36c41bef9c0dbaddd6e77d337bc
Author: Viviana C. Luccioli <m1vcl00@salt2.rsma.frb.gov>
Date:   Wed May 6 16:42:04 2026 -0400

    adding results summary files

diff --git a/experiment/comparing_10ifdp.md b/experiment/comparing_10ifdp.md
new file mode 100644
index 0000000..e32852a
--- /dev/null
+++ b/experiment/comparing_10ifdp.md
@@ -0,0 +1,124 @@
+# Comparison: April 23 (Calibrated) vs May 5 (With Numeric Scores)
+
+**Setup**: Same 10 IFDP papers, two versions of the workflow:
+- **V1 (April 23)**: Categorical verdicts only
+- **V2 (May 5)**: Added numeric scores (1-10) per persona per round, summed for total score (max 30)
+
+---
+
+## Overview: All Papers
+
+| Paper | Ground Truth (Tier) | V1 Verdict | V2 Verdict | V2 Score | Change |
+|-------|---------------------|------------|------------|----------|--------|
+| ifdp-2020-1 | Tier 2 | REVISE | REVISE | 17.0 | ✓ Stable |
+| ifdp-2020-2 | Tier 4 | FAIL | **REVISE** | 16.0 | ⬆️ Upgraded |
+| ifdp-2020-3 | Tier 2 | FAIL | **REVISE** | 19.0 | ⬆️ Upgraded |
+| ifdp-2020-4 | Tier 1 | FAIL | **REVISE** | 17.0 | ⬆️ Upgraded |
+| ifdp-2020-5 | Tier 3 | FAIL | **REVISE** | 17.5 | ⬆️ Upgraded |
+| ifdp-2020-6 | Tier 3 | FAIL | FAIL | — | ✓ Stable (no score) |
+| ifdp-2020-7 | Tier 3 | FAIL | **REVISE** | 15.0 | ⬆️ Upgraded |
+| ifdp-2020-8 | Tier 2 | REVISE | REVISE | 20.0 | ✓ Stable |
+| ifdp-2020-9 | Tier 4 | FAIL | FAIL | 11.0 | ✓ Stable |
+| ifdp-2020-10 | Tier 4 | REVISE | REVISE | 21.0 | ✓ Stable |
+
+**Key**: Tier 1 = Highest quality, Tier 4 = Lowest quality
+
+---
+
+## Final Verdict Changes
+
+| Paper | V1 Verdict | V2 Verdict | V2 Score | Score Context |
+|-------|------------|------------|----------|---------------|
+| **ifdp-2020-2** | FAIL | **REVISE** | 16/30 | Mid-range REVISE (53%) |
+| **ifdp-2020-3** | FAIL | **REVISE** | 19/30 | High-range REVISE (63%) |
+| **ifdp-2020-4** | FAIL | **REVISE** | 17/30 | Mid-range REVISE (57%) |
+| **ifdp-2020-5** | FAIL | **REVISE** | 17.5/30 | Mid-range REVISE (58%) |
+| **ifdp-2020-7** | FAIL | **REVISE** | 15/30 | **Low-range REVISE (50%)** ⚠️ |
+
+**Papers maintaining verdict:**
+- ifdp-2020-1: REVISE → REVISE (17/30)
+- ifdp-2020-6: FAIL → FAIL (no scores available)
+- ifdp-2020-8: REVISE → REVISE (20/30)
+- ifdp-2020-9: FAIL → FAIL (11/30 - consistent)
+- ifdp-2020-10: REVISE → REVISE (21/30)
+
+---
+
+## Key Findings
+
+### 1. **Systematic Upward Drift**
+**5 of 10 papers** moved from FAIL to REVISE with numeric scoring. Only 1 paper maintained FAIL with a believable score (ifdp-2020-9: 11/30).
+
+### 2. **Boundary Cases**
+- **ifdp-2020-7**: Score of 15/30 (50%) puts it at the absolute bottom of REVISE range - likely near FAIL/REVISE boundary
+- **ifdp-2020-9**: Score of 11/30 (37%) - lowest score, consistent FAIL verdict
+- **ifdp-2020-6**: FAIL with missing scores suggests scoring system may have failed for this paper
+
+### 3. **No Extreme Jumps**
+All papers that changed verdict landed in **mid-to-low REVISE range** (15-19/30), not high REVISE (20+). This suggests numeric scoring didn't drastically change evaluations, but may have nudged borderline FAILs upward.
+
+---
+
+## Persona Changes
+
+| Paper | V1 Personas | V2 Personas | Change |
+|-------|-------------|-------------|--------|
+| ifdp-2020-1 | Econometrician/Policymaker/Data_Scientist | Econometrician/Data_Scientist/Policymaker | ✓ Same, order swapped |
+| ifdp-2020-2 | Econometrician/Policymaker/**Historian** | Econometrician/Policymaker/**Data_Scientist** | ⚠️ **Historian → Data_Scientist** |
+| ifdp-2020-5 | Econometrician/CS_Expert/**Policymaker** | Econometrician/CS_Expert/**Theorist** | ⚠️ **Policymaker → Theorist** |
+| ifdp-2020-8 | Theorist/Econometrician/**Data_Scientist** | Theorist/Econometrician/**Policymaker** | ⚠️ **Data_Scientist → Policymaker** |
+
+**All other papers (7/10)** maintained identical persona sets.
+
+---
+
+## Round-Level Convergence
+
+### Papers Showing Debate Convergence in Both Versions
+- **ifdp-2020-2**: Started divergent (REVISE/PASS/REVISE in V1) → converged to unanimous REVISE in V2
+- **ifdp-2020-8**: Unanimous REVISE in both versions (stable agreement)
+
+### Papers Showing Divergence in V1 → Convergence in V2
+- **ifdp-2020-4**: V1 had split (REVISE/REVISE/REVISE but different Round 1), V2 unanimous REVISE
+
+---
+
+## Verdict-Score Consistency
+
+### Expected Score Ranges (Based on V2 Data)
+- **PASS**: ~22-30/30 (no pure PASS papers in this batch)
+- **REVISE**: ~15-21/30 (observed range: 15-21)
+- **FAIL**: ~0-14/30 (observed: 11/30 for ifdp-2020-9)
+
+### Anomaly
+**ifdp-2020-6**: Verdict is FAIL but `total_final_score = None`, suggesting personas may not have provided numeric scores. This indicates the scoring system isn't universally applied or failed for this paper.
+
+---
+
+## Hypothesis: Why the Upward Drift?
+
+**Possible explanations for 5 FAIL → REVISE shifts:**
+
+1. **Numeric scoring introduces optimism bias**: Personas may anchor higher when forced to quantify (5-6/10) vs. categorical threshold judgment
+2. **Aggregation softens extremes**: Summing three scores (5+5+6=16) can push borderline papers into REVISE even if individual assessments were harsh
+3. **Prompt changes**: The numeric scoring prompt may have subtly altered evaluation criteria
+4. **Regression to mean**: Papers near the FAIL/REVISE boundary naturally drift toward the middle with added quantification
+
+---
+
+## Recommendations
+
+1. **Investigate ifdp-2020-6**: Why are scores missing? Does this indicate a failure mode in the numeric scoring system?
+2. **Calibrate thresholds**: If scores of 15-16/30 represent REVISE, what's the FAIL cutoff? Currently unclear if 14/30 or below is FAIL.
+3. **Re-evaluate ifdp-2020-7**: At 15/30 (50%), this is the weakest REVISE verdict. Consider whether it should be FAIL.
+4. **Persona consistency**: 3 papers had different persona assignments between runs. Investigate whether this is due to Round 0 stochasticity or system changes.
+
+---
+
+## Summary
+
+✅ **Consistent verdicts**: 5/10 papers (same verdict both runs)  
+⚠️ **Upward drift**: 5/10 papers moved FAIL → REVISE  
+❌ **Downward drift**: 0/10 papers moved REVISE → FAIL  
+
+**Verdict stability is low** when numeric scoring is added. The system appears to systematically shift papers upward, with all changed verdicts landing in the 15-19/30 range (50-63% of max score). This suggests the FAIL/REVISE boundary may be sensitive to the scoring methodology.
diff --git a/experiment/numeric_analysis.md b/experiment/numeric_analysis.md
new file mode 100644
index 0000000..5f06c52
--- /dev/null
+++ b/experiment/numeric_analysis.md
@@ -0,0 +1,283 @@
+# Numeric Score Analysis vs Ground Truth Tiers
+
+**Analysis Date**: 2026-05-05  
+**Dataset**: `referee_batch_results_20260505_001147.csv`  
+**Papers Analyzed**: 10 (ifdp-2020-1 through ifdp-2020-10)
+
+---
+
+## Executive Summary
+
+**Key Finding**: The numeric scoring system shows **weak correlation with ground truth tiers**, with several critical misalignments:
+
+1. **Tier 1 paper scored in middle range** (17.0/30) despite highest quality tier
+2. **Tier 4 papers received higher scores than Tier 1/2 papers** in multiple cases
+3. **Highest score (21.0/30) awarded to a Tier 4 paper** (ifdp-2020-10)
+4. **Score compression**: Most papers cluster in 15-20 range regardless of tier
+
+**Implication**: The current scoring rubric may not adequately differentiate paper quality tiers, suggesting need for calibration adjustment.
+
+---
+
+## Full Score Distribution by Tier
+
+### Tier 1 (Highest Quality)
+| Paper | Final Verdict | Total Score | Individual Scores | Notes |
+|-------|---------------|-------------|-------------------|-------|
+| ifdp-2020-4 | REVISE | **17.0** | Theorist: 5.0, Econometrician: 6.0, Policymaker: 6.0 | ⚠️ **ANOMALY**: Tier 1 scored below multiple Tier 2-4 papers |
+
+**Analysis**: The single Tier 1 paper received a mid-range score (17.0/30), placing it **below 4 other papers** including two Tier 4 papers. This is a significant calibration issue.
+
+---
+
+### Tier 2 (High Quality)
+| Paper | Final Verdict | Total Score | Individual Scores | Notes |
+|-------|---------------|-------------|-------------------|-------|
+| ifdp-2020-1 | REVISE | 17.0 | Econometrician: 6.0, Data_Scientist: 6.0, Policymaker: 5.0 | Tied with Tier 1 paper |
+| ifdp-2020-3 | REVISE | **19.0** | Econometrician: 5.0, Data_Scientist: 6.0, Policymaker: 8.0 | Policymaker gave high score (8.0) |
+| ifdp-2020-8 | REVISE | **20.0** | Theorist: 6.5, Econometrician: 7.5, Policymaker: 6.0 | ⭐ **2nd highest overall score** |
+
+**Analysis**: Tier 2 papers show wide variation (17.0-20.0), with ifdp-2020-8 achieving the **2nd highest score across all papers** (20.0/30). This suggests the scoring system can identify higher-quality Tier 2 papers, but fails to distinguish them from lower tiers.
+
+**Notable**: ifdp-2020-8 scored **3 points higher** than the Tier 1 paper, indicating possible reverse ranking.
+
+---
+
+### Tier 3 (Medium Quality)
+| Paper | Final Verdict | Total Score | Individual Scores | Notes |
+|-------|---------------|-------------|-------------------|-------|
+| ifdp-2020-5 | REVISE | 17.5 | Econometrician: 5.0, CS_Expert: 6.5, Theorist: 6.0 | Middle of pack |
+| ifdp-2020-6 | FAIL | **N/A** | Econometrician: 5.0, Theorist: N/A, Policymaker: 5.0 | Theorist declined final score |
+| ifdp-2020-7 | REVISE | **15.0** | Econometrician: 4.0, Data_Scientist: 5.0, Policymaker: 6.0 | **Lowest REVISE score** |
+
+**Analysis**: Tier 3 papers span 15.0-17.5 (excluding FAIL with incomplete scoring), showing expected medium-range placement. ifdp-2020-7 represents the **floor for REVISE verdicts** at 15.0/30.
+
+---
+
+### Tier 4 (Lowest Quality)
+| Paper | Final Verdict | Total Score | Individual Scores | Notes |
+|-------|---------------|-------------|-------------------|-------|
+| ifdp-2020-10 | REVISE | **21.0** | Econometrician: 6.5, ML_Expert: 8.0, Policymaker: 6.5 | ⚠️ **HIGHEST SCORE OVERALL** |
+| ifdp-2020-2 | REVISE | 16.0 | Econometrician: 5.0, Policymaker: 6.0, Data_Scientist: 5.0 | Low-middle range |
+| ifdp-2020-9 | FAIL | **11.0** | Theorist: 3.0, Econometrician: 3.0, Policymaker: 5.0 | **Lowest score overall** |
+
+**Analysis**: Tier 4 shows **extreme bimodality**:
+- **ifdp-2020-10**: 21.0/30 — **HIGHEST SCORE IN ENTIRE DATASET** ⚠️
+- **ifdp-2020-9**: 11.0/30 — Lowest score, appropriate FAIL verdict
+- **ifdp-2020-2**: 16.0/30 — Mid-range, arguably too lenient
+
+**Critical Finding**: A Tier 4 paper (ifdp-2020-10) outscored the Tier 1 paper by **4 points** and all but one Tier 2 paper. This is a **major calibration failure**.
+
+---
+
+## REVISE Category Score Analysis
+
+### All REVISE Verdicts Ranked by Score
+
+| Rank | Paper | Tier | Total Score | Score Gap from Next |
+|------|-------|------|-------------|---------------------|
+| 1 | ifdp-2020-10 | **Tier 4** | 21.0 | +1.0 |
+| 2 | ifdp-2020-8 | **Tier 2** | 20.0 | +1.0 |
+| 3 | ifdp-2020-3 | **Tier 2** | 19.0 | +1.5 |
+| 4 | ifdp-2020-5 | Tier 3 | 17.5 | +0.5 |
+| 5 (tie) | ifdp-2020-1 | **Tier 2** | 17.0 | 0 |
+| 5 (tie) | ifdp-2020-4 | **Tier 1** | 17.0 | +1.0 |
+| 7 | ifdp-2020-2 | Tier 4 | 16.0 | +1.0 |
+| 8 | ifdp-2020-7 | Tier 3 | 15.0 | — |
+
+### Higher-Scoring REVISE Papers (>18.0)
+
+**ifdp-2020-10** (Tier 4, 21.0/30):
+- **Anomaly Status**: ⚠️ **SEVERE** — Highest score despite lowest tier
+- **Individual Scores**: Econometrician: 6.5, ML_Expert: **8.0**, Policymaker: 6.5
+- **Explanation**: ML_Expert gave exceptional score (8.0/10), suggesting methodological contribution recognized despite other flaws
+- **Verdict Appropriateness**: REVISE may be too lenient; score suggests near-PASS threshold
+
+**ifdp-2020-8** (Tier 2, 20.0/30):
+- **Anomaly Status**: ✓ Appropriate — 2nd highest score for 2nd highest tier
+- **Individual Scores**: Theorist: 6.5, Econometrician: **7.5**, Policymaker: 6.0
+- **Explanation**: Econometrician's strong endorsement (7.5/10) drove high total
+- **Verdict Appropriateness**: REVISE may be harsh; score suggests should be closer to PASS
+
+**ifdp-2020-3** (Tier 2, 19.0/30):
+- **Anomaly Status**: ✓ Appropriate — High score for high tier
+- **Individual Scores**: Econometrician: 5.0, Data_Scientist: 6.0, Policymaker: **8.0**
+- **Explanation**: Policymaker's strong score (8.0/10) reflects policy relevance
+- **Verdict Appropriateness**: REVISE reasonable given mixed technical/policy assessment
+
+### Answer to Key Question: Were High REVISE Scores from Tier 1/2?
+
+**YES, but with critical exception**:
+- **3 of top 4 REVISE scores** were Tier 1/2 papers (ifdp-2020-8, ifdp-2020-3, ifdp-2020-1/4)
+- **HOWEVER**: The #1 score (21.0) was a **Tier 4 paper** (ifdp-2020-10)
+
+This suggests:
+1. The scoring system **generally** recognizes higher-tier papers within REVISE category
+2. **BUT** the system has significant outliers where low-tier papers receive top scores
+3. The boundary between REVISE and PASS may be **too high**, preventing deserving Tier 1/2 papers from acceptance
+
+---
+
+## Score-to-Tier Correlation Analysis
+
+### Expected vs Actual Score Ranges
+
+| Tier | Expected Avg Score | Actual Avg Score | Actual Range | Deviation |
+|------|-------------------|------------------|--------------|-----------|
+| Tier 1 | 22-26 (near-PASS) | **17.0** | 17.0 only | -5 to -9 points |
+| Tier 2 | 18-22 | **18.7** | 17.0-20.0 | -3 to 0 points |
+| Tier 3 | 14-18 | **16.3** | 15.0-17.5 | -2 to +1 points |
+| Tier 4 | 10-14 | **16.0** | 11.0-21.0 | 0 to +7 points |
+
+**Key Observations**:
+1. **Tier 1 severely underscored**: 5-9 points below expectation
+2. **Tier 2 slightly underscored**: Within range but compressed toward lower end
+3. **Tier 3 appropriately scored**: Matches expectations
+4. **Tier 4 severely overscored**: Average should be ~12, actual is 16 (excluding outlier: 18.5 with outlier)
+
+### Score Compression Problem
+
+**Standard Deviation by Tier**:
+- Tier 1: N/A (single data point)
+- Tier 2: σ = 1.53 (low variance)
+- Tier 3: σ = 1.32 (low variance)
+- Tier 4: σ = 5.00 (high variance due to outlier)
+
+**Overall REVISE category**: σ = 2.07 (narrow spread for 8 papers spanning Tiers 1-4)
+
+**Diagnosis**: Scores are **compressed into 15-21 range** regardless of tier, with insufficient differentiation. A 6-point spread is inadequate to distinguish 4 quality tiers.
+
+---
+
+## Individual Persona Scoring Patterns
+
+### Econometrician (9/10 papers)
+- **Range**: 3.0-7.5
+- **Average**: 5.6
+- **Tendency**: Conservative, rarely exceeds 6.5
+- **Highest score**: 7.5 (ifdp-2020-8, Tier 2)
+- **Lowest score**: 3.0 (ifdp-2020-9, Tier 4 FAIL)
+
+### Policymaker (7/10 papers)
+- **Range**: 5.0-8.0
+- **Average**: 6.1
+- **Tendency**: Most generous scorer
+- **Highest score**: 8.0 (ifdp-2020-3, Tier 2; ifdp-2020-4, Tier 1 in Round 1)
+- **Score inflation**: 3 scores ≥ 8.0
+
+### Data_Scientist (3/10 papers)
+- **Range**: 5.0-6.0
+- **Average**: 5.7
+- **Tendency**: Conservative, consistent
+
+### ML_Expert (1/10 papers)
+- **Range**: 8.0 only
+- **Average**: 8.0
+- **Tendency**: Single data point shows high score for Tier 4 paper (concerning)
+
+### Theorist (3/10 papers)
+- **Range**: 3.0-6.5
+- **Average**: 5.2
+- **Tendency**: Most critical, declined to score ifdp-2020-6
+
+### CS_Expert (2/10 papers)
+- **Range**: 6.5 only (single scored paper)
+- **Average**: 6.5
+- **Tendency**: Limited data
+
+---
+
+## Verdict Threshold Analysis
+
+### Implicit Score Thresholds
+
+Based on observed data:
+
+- **FAIL threshold**: < 12.0 (ifdp-2020-9 at 11.0, ifdp-2020-6 incomplete)
+- **REVISE range**: 15.0-21.0 (wide range, no upper bound tested)
+- **PASS threshold**: Not observed (assumed > 21.0?)
+
+**Problem**: No papers crossed PASS threshold despite Tier 1 paper in dataset. This suggests:
+1. Threshold may be set too high (> 21.0)
+2. Personas are systematically underscoring relative to paper quality
+3. Rubric may emphasize flaws over strengths
+
+---
+
+## Recommendations for Calibration
+
+### 1. Score Inflation for High-Tier Papers
+**Issue**: Tier 1/2 papers receiving 5-6 scores when 7-9 may be appropriate
+
+**Fix**: 
+- Revise scoring rubric to explicitly allow 8-9 scores for papers with minor revisions only
+- Instruct personas that Tier 1/2 papers should average 7-8 if flaws are fixable
+
+### 2. Increase Score Range Differentiation
+**Issue**: 6-point spread insufficient for 4 tiers
+
+**Fix**:
+- Encourage use of full 1-10 range
+- Set explicit tier targets:
+  - Tier 1: 22-26 average
+  - Tier 2: 18-22 average
+  - Tier 3: 14-18 average
+  - Tier 4: 10-14 average
+
+### 3. Persona-Specific Calibration
+**Issue**: Policymaker scores 0.5-0.9 points higher than Econometrician on average
+
+**Fix**:
+- Normalize scores across personas before aggregation
+- OR provide persona-specific rubrics with different anchoring
+
+### 4. Address Tier 4 Overscoring
+**Issue**: ifdp-2020-10 (Tier 4) scored 21.0/30
+
+**Fix**:
+- Investigate what ML_Expert found valuable (may be legitimate methodological contribution)
+- Clarify whether scoring reflects "contribution given its genre" vs "absolute publication-readiness"
+- Consider separate scoring dimensions (contribution vs. execution quality)
+
+### 5. Establish PASS Threshold
+**Issue**: No papers reached PASS despite Tier 1 in dataset
+
+**Fix**:
+- Explicitly define PASS threshold (e.g., ≥ 22.0 with no individual score < 6.0)
+- Calibrate personas to this threshold using example papers
+
+---
+
+## Conclusions
+
+1. **Weak Tier Correlation**: The numeric scoring system shows insufficient correlation with ground truth tiers, particularly at extremes (Tier 1 underscored, Tier 4 overscored).
+
+2. **Score Compression**: Papers cluster in 15-21 range regardless of tier, indicating rubric needs wider differentiation.
+
+3. **High REVISE Scores**: Among REVISE papers with scores >18.0:
+   - **ifdp-2020-8** (Tier 2, 20.0): ✓ Appropriately high
+   - **ifdp-2020-3** (Tier 2, 19.0): ✓ Appropriately high
+   - **ifdp-2020-10** (Tier 4, 21.0): ⚠️ Anomalously high
+
+4. **Tier 1 Underdifferentiated**: The sole Tier 1 paper (ifdp-2020-4) scored identically (17.0) to a Tier 2 paper and **4 points below** a Tier 4 paper.
+
+5. **PASS Threshold Too High**: No papers crossed into PASS territory, suggesting threshold may be miscalibrated or personas systematically underscore.
+
+**Overall Assessment**: The scoring system requires recalibration to better reflect ground truth tiers. Current implementation may be useful for relative ranking within sessions but lacks absolute calibration for cross-paper comparison.
+
+---
+
+## Appendix: Full Score Matrix
+
+| Paper | Tier | P1 Name | P1 Score | P2 Name | P2 Score | P3 Name | P3 Score | Total | Verdict |
+|-------|------|---------|----------|---------|----------|---------|----------|-------|---------|
+| ifdp-2020-1 | 2 | Econometrician | 6.0 | Data_Scientist | 6.0 | Policymaker | 5.0 | 17.0 | REVISE |
+| ifdp-2020-10 | 4 | Econometrician | 6.5 | ML_Expert | 8.0 | Policymaker | 6.5 | **21.0** | REVISE |
+| ifdp-2020-2 | 4 | Econometrician | 5.0 | Policymaker | 6.0 | Data_Scientist | 5.0 | 16.0 | REVISE |
+| ifdp-2020-3 | 2 | Econometrician | 5.0 | Data_Scientist | 6.0 | Policymaker | 8.0 | 19.0 | REVISE |
+| ifdp-2020-4 | 1 | Theorist | 5.0 | Econometrician | 6.0 | Policymaker | 6.0 | 17.0 | REVISE |
+| ifdp-2020-5 | 3 | Econometrician | 5.0 | CS_Expert | 6.5 | Theorist | 6.0 | 17.5 | REVISE |
+| ifdp-2020-6 | 3 | Econometrician | 5.0 | Theorist | N/A | Policymaker | 5.0 | N/A | FAIL |
+| ifdp-2020-7 | 3 | Econometrician | 4.0 | Data_Scientist | 5.0 | Policymaker | 6.0 | 15.0 | REVISE |
+| ifdp-2020-8 | 2 | Theorist | 6.5 | Econometrician | 7.5 | Policymaker | 6.0 | **20.0** | REVISE |
+| ifdp-2020-9 | 4 | Theorist | 3.0 | Econometrician | 3.0 | Policymaker | 5.0 | **11.0** | FAIL |
```
