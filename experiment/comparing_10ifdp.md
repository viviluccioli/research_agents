# Comparison: April 23 (Calibrated) vs May 5 (With Numeric Scores)

**Setup**: Same 10 IFDP papers, two versions of the workflow:
- **V1 (April 23)**: Categorical verdicts only
- **V2 (May 5)**: Added numeric scores (1-10) per persona per round, summed for total score (max 30)

---

## Overview: All Papers

| Paper | Ground Truth (Tier) | V1 Verdict | V2 Verdict | V2 Score | Change |
|-------|---------------------|------------|------------|----------|--------|
| ifdp-2020-1 | Tier 2 | REVISE | REVISE | 17.0 | ✓ Stable |
| ifdp-2020-2 | Tier 4 | FAIL | **REVISE** | 16.0 | ⬆️ Upgraded |
| ifdp-2020-3 | Tier 2 | FAIL | **REVISE** | 19.0 | ⬆️ Upgraded |
| ifdp-2020-4 | Tier 1 | FAIL | **REVISE** | 17.0 | ⬆️ Upgraded |
| ifdp-2020-5 | Tier 3 | FAIL | **REVISE** | 17.5 | ⬆️ Upgraded |
| ifdp-2020-6 | Tier 3 | FAIL | FAIL | — | ✓ Stable (no score) |
| ifdp-2020-7 | Tier 3 | FAIL | **REVISE** | 15.0 | ⬆️ Upgraded |
| ifdp-2020-8 | Tier 2 | REVISE | REVISE | 20.0 | ✓ Stable |
| ifdp-2020-9 | Tier 4 | FAIL | FAIL | 11.0 | ✓ Stable |
| ifdp-2020-10 | Tier 4 | REVISE | REVISE | 21.0 | ✓ Stable |

**Key**: Tier 1 = Highest quality, Tier 4 = Lowest quality

---

## Final Verdict Changes

| Paper | V1 Verdict | V2 Verdict | V2 Score | Score Context |
|-------|------------|------------|----------|---------------|
| **ifdp-2020-2** | FAIL | **REVISE** | 16/30 | Mid-range REVISE (53%) |
| **ifdp-2020-3** | FAIL | **REVISE** | 19/30 | High-range REVISE (63%) |
| **ifdp-2020-4** | FAIL | **REVISE** | 17/30 | Mid-range REVISE (57%) |
| **ifdp-2020-5** | FAIL | **REVISE** | 17.5/30 | Mid-range REVISE (58%) |
| **ifdp-2020-7** | FAIL | **REVISE** | 15/30 | **Low-range REVISE (50%)** ⚠️ |

**Papers maintaining verdict:**
- ifdp-2020-1: REVISE → REVISE (17/30)
- ifdp-2020-6: FAIL → FAIL (no scores available)
- ifdp-2020-8: REVISE → REVISE (20/30)
- ifdp-2020-9: FAIL → FAIL (11/30 - consistent)
- ifdp-2020-10: REVISE → REVISE (21/30)

---

## Key Findings

### 1. **Systematic Upward Drift**
**5 of 10 papers** moved from FAIL to REVISE with numeric scoring. Only 1 paper maintained FAIL with a believable score (ifdp-2020-9: 11/30).

### 2. **Boundary Cases**
- **ifdp-2020-7**: Score of 15/30 (50%) puts it at the absolute bottom of REVISE range - likely near FAIL/REVISE boundary
- **ifdp-2020-9**: Score of 11/30 (37%) - lowest score, consistent FAIL verdict
- **ifdp-2020-6**: FAIL with missing scores suggests scoring system may have failed for this paper

### 3. **No Extreme Jumps**
All papers that changed verdict landed in **mid-to-low REVISE range** (15-19/30), not high REVISE (20+). This suggests numeric scoring didn't drastically change evaluations, but may have nudged borderline FAILs upward.

---

## Persona Changes

| Paper | V1 Personas | V2 Personas | Change |
|-------|-------------|-------------|--------|
| ifdp-2020-1 | Econometrician/Policymaker/Data_Scientist | Econometrician/Data_Scientist/Policymaker | ✓ Same, order swapped |
| ifdp-2020-2 | Econometrician/Policymaker/**Historian** | Econometrician/Policymaker/**Data_Scientist** | ⚠️ **Historian → Data_Scientist** |
| ifdp-2020-5 | Econometrician/CS_Expert/**Policymaker** | Econometrician/CS_Expert/**Theorist** | ⚠️ **Policymaker → Theorist** |
| ifdp-2020-8 | Theorist/Econometrician/**Data_Scientist** | Theorist/Econometrician/**Policymaker** | ⚠️ **Data_Scientist → Policymaker** |

**All other papers (7/10)** maintained identical persona sets.

---

## Round-Level Convergence

### Papers Showing Debate Convergence in Both Versions
- **ifdp-2020-2**: Started divergent (REVISE/PASS/REVISE in V1) → converged to unanimous REVISE in V2
- **ifdp-2020-8**: Unanimous REVISE in both versions (stable agreement)

### Papers Showing Divergence in V1 → Convergence in V2
- **ifdp-2020-4**: V1 had split (REVISE/REVISE/REVISE but different Round 1), V2 unanimous REVISE

---

## Verdict-Score Consistency

### Expected Score Ranges (Based on V2 Data)
- **PASS**: ~22-30/30 (no pure PASS papers in this batch)
- **REVISE**: ~15-21/30 (observed range: 15-21)
- **FAIL**: ~0-14/30 (observed: 11/30 for ifdp-2020-9)

### Anomaly
**ifdp-2020-6**: Verdict is FAIL but `total_final_score = None`, suggesting personas may not have provided numeric scores. This indicates the scoring system isn't universally applied or failed for this paper.

---

## Hypothesis: Why the Upward Drift?

**Possible explanations for 5 FAIL → REVISE shifts:**

1. **Numeric scoring introduces optimism bias**: Personas may anchor higher when forced to quantify (5-6/10) vs. categorical threshold judgment
2. **Aggregation softens extremes**: Summing three scores (5+5+6=16) can push borderline papers into REVISE even if individual assessments were harsh
3. **Prompt changes**: The numeric scoring prompt may have subtly altered evaluation criteria
4. **Regression to mean**: Papers near the FAIL/REVISE boundary naturally drift toward the middle with added quantification

---

## Recommendations

1. **Investigate ifdp-2020-6**: Why are scores missing? Does this indicate a failure mode in the numeric scoring system?
2. **Calibrate thresholds**: If scores of 15-16/30 represent REVISE, what's the FAIL cutoff? Currently unclear if 14/30 or below is FAIL.
3. **Re-evaluate ifdp-2020-7**: At 15/30 (50%), this is the weakest REVISE verdict. Consider whether it should be FAIL.
4. **Persona consistency**: 3 papers had different persona assignments between runs. Investigate whether this is due to Round 0 stochasticity or system changes.

---

## Summary

✅ **Consistent verdicts**: 5/10 papers (same verdict both runs)  
⚠️ **Upward drift**: 5/10 papers moved FAIL → REVISE  
❌ **Downward drift**: 0/10 papers moved REVISE → FAIL  

**Verdict stability is low** when numeric scoring is added. The system appears to systematically shift papers upward, with all changed verdicts landing in the 15-19/30 range (50-63% of max score). This suggests the FAIL/REVISE boundary may be sensitive to the scoring methodology.
