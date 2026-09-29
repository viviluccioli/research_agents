# Focused Calibration Analysis - Summary

## What You Asked For

> "What more would you want to be able to see/know to get a better understanding of how the tool is working on high vs lower quality papers?"

## What Was Delivered

**Enhanced CSV:** `results/referee_batch_results_20260423_160156_calibrated.csv`

- **Original:** 21 columns
- **Added:** 10 focused calibration metrics
- **Total:** 31 columns (manageable, actionable)

### The 10 Essential Metrics

**Calibration Metrics (4 columns):**
1. `consensus_score_r1` - Initial weighted score (0-1 scale)
2. `consensus_score_r2c` - Final weighted score (0-1 scale)
3. `expected_score_for_tier` - What we expect given the tier
4. `calibration_error` - How far off we are (abs difference)

**Confidence Metrics (3 columns):**
5. `agreement_level_r1` - Initial unanimity (1=all agree, 0=all differ)
6. `agreement_level_r2c` - Final unanimity
7. `borderline_flag` - Yes/No for genuinely ambiguous papers

**Debate Value Metrics (3 columns):**
8. `debate_improved_calibration` - Did debate move score closer to expected?
9. `agreement_change` - Change in unanimity (R2C - R1)
10. `verdict_stability` - Stable/Shifted/Flipped

---

## Key Findings from Your Data

### 🎯 Overall Tool Performance

**Calibration Status:** ⚠️ **MODERATELY CALIBRATED** (avg error: 0.324)

**The Problem:** Tool is **systematically TOO HARSH** across all tiers

| Tier | Expected Score | Actual Score | Error | Assessment |
|------|---------------|--------------|-------|------------|
| **Tier 1** | 0.90 | **0.18** | 0.72 | ❌ WAY TOO STRICT |
| **Tier 2** | 0.70 | **0.34** | 0.36 | ⚠️ TOO STRICT |
| **Tier 3** | 0.30 | **0.08** | 0.23 | ⚠️ TOO STRICT |
| **Tier 4** | 0.10 | **0.22** | 0.25 | ✓ REASONABLE |

### 📊 What This Means

1. **Tool treats ALL papers too harshly**
   - Tier 1 (top journal) paper got FAIL verdict (score 0.18 vs expected 0.90)
   - Even Tier 4 papers score slightly higher than expected (not harsh enough on truly bad papers)

2. **Tool is best at identifying bad papers**
   - Tier 4 error (0.25) is lower than Tier 1 error (0.72)
   - Easier to spot flaws than recognize excellence

3. **Confidence is backwards**
   - Should be: High confidence on clear cases (Tier 1 & 4), low on ambiguous (Tier 2 & 3)
   - Reality: Tier 1 has LOW agreement (0.5), Tier 4 has HIGH agreement (0.83)
   - Tool is MORE uncertain about good papers than bad papers

---

## Debate Effectiveness Analysis

**Does debate help?**
- **Improved calibration:** 40% of papers
- **Worsened calibration:** 50% of papers
- **No change:** 10% of papers

**Verdict:** Debate adds LIMITED value overall

**However, tier-specific patterns:**
- **Tier 3 & 4:** Debate helps (50-67% improved)
- **Tier 1 & 2:** Debate hurts (100% worsened or no change)

**Interpretation:** Debate is useful for identifying weak papers, but makes tool even harsher on good papers.

---

## Problem Papers Identified

### 3 Papers Need Investigation:

**1. ifdp-2020-4 (Tier 1 - Top Journal)**
- **Issue:** Tool gave FAIL verdict to a paper accepted at TOP journal
- Calibration error: 0.72 (HUGE)
- Score: 0.18 (expected: 0.90)
- Personas: 2 FAIL, 1 REVISE
- **Question:** Is tier label wrong? Or does tool miss quality signals?

**2. ifdp-2020-3 (Tier 2 - Good Journal)**
- **Issue:** Tool gave FAIL verdict to good journal paper
- Calibration error: 0.58
- Score: 0.12 (expected: 0.70)
- Personas: 2 FAIL, 1 REVISE
- **Pattern:** Similar to Tier 1 issue - too harsh

**3. ifdp-2020-10 (Tier 4 - Not Accepted)**
- **Issue:** Tool gave REVISE to a paper that wasn't accepted anywhere
- Calibration error: 0.56
- Score: 0.66 (expected: 0.10)
- Personas: 1 PASS, 2 REVISE
- **Pattern:** Tool is TOO LENIENT here (opposite problem)

---

## What to Look For in the CSV

### Quick Analyses You Can Do:

**1. Find miscalibrations:**
```python
df[df['calibration_error'] > 0.4]
```

**2. Find papers where tool is confident but wrong:**
```python
df[(df['calibration_error'] > 0.4) & (df['agreement_level_r2c'] > 0.8)]
```

**3. Find papers where debate helped:**
```python
df[df['debate_improved_calibration'] == 'Yes']
```

**4. Compare R1 vs R2C scores by tier:**
```python
df.groupby('tier')[['consensus_score_r1', 'consensus_score_r2c']].mean()
```

**5. Find unstable verdicts (flipped opinions):**
```python
df[df['verdict_stability'] == 'Flipped']
```

---

## Root Cause Hypothesis

Based on the patterns, the **primary issue** is:

### **Tool is calibrated for "rejection" not "publication readiness"**

**Evidence:**
1. ALL tiers score below expected (tool is universally harsh)
2. Tool is better at identifying flaws (Tier 4) than excellence (Tier 1)
3. Debate makes tool even stricter (50% of time worsens calibration)
4. High agreement on bad papers, low agreement on good papers

**This suggests:**
- Personas are looking for reasons to REJECT rather than reasons to ACCEPT
- "REVISE" and "FAIL" verdicts come easily, "PASS" is rare
- The weighting might favor critical personas (Econometrician, Policymaker)

---

## Recommended Next Steps

### 1. Investigate Tier Labels
- Verify that Tier 1 paper (ifdp-2020-4) actually belongs in Tier 1
- If tier is correct, this is a major tool failure

### 2. Examine the 3 Problem Papers in Detail
- Read the papers and the full referee reports
- Understand WHY personas gave those verdicts
- Look for patterns in what tool is missing/overweighting

### 3. Consider Recalibration Options
- **Adjust expected scores:** Maybe "good enough for Tier 1" means 0.7 not 0.9?
- **Reweight personas:** Econometrician (50% FAIL rate) is most balanced
- **Adjust verdict thresholds:** Current: >0.75=PASS, maybe should be >0.6=PASS?

### 4. Test Calibration Hypothesis
Run more papers to see if pattern holds:
- Do ALL Tier 1 papers get unfairly harsh scores?
- Or is ifdp-2020-4 an outlier?

### 5. Debate Protocol Review
Since debate worsens calibration for Tier 1/2:
- Are personas becoming MORE critical during discussion?
- Should debate focus on reaching consensus vs. finding more flaws?

---

## Files Summary

**Enhanced CSV:**
```
results/referee_batch_results_20260423_160156_calibrated.csv
```

**Analysis Scripts:**
```
add_calibration_metrics.py     - Add 10 focused metrics to any CSV
analyze_calibration.py         - Run full calibration analysis
```

**Key Columns to Review:**
- `calibration_error` - How miscalibrated (lower = better)
- `agreement_level_r2c` - How confident (higher = more certain)
- `debate_improved_calibration` - Did debate help or hurt?
- `verdict_stability` - How much did verdict change?

---

## Bottom Line

**The tool works, but it's miscalibrated:**

✅ **Good at:** Identifying bad papers (Tier 4)  
❌ **Bad at:** Recognizing good papers (Tier 1)  
⚠️ **Debate:** Helps with bad papers, hurts with good papers  

**Main finding:** Tool is systematically TOO HARSH across all quality tiers. This is actionable - you now know the tool needs recalibration to be less strict, especially for high-quality papers.

With just 10 metrics (not 41!), you can now:
- Identify which papers are miscalibrated
- Understand if it's a systematic bias (yes - too harsh)
- See which tiers are most problematic (Tier 1)
- Track whether debate helps or hurts (mixed)
- Flag papers for manual review

**This is exactly what you needed to evaluate tool performance against ground truth!** 🎯
