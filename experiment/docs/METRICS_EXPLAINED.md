# Calibration Metrics Explained

## 1. Agreement Level (Confidence/Unanimity)

**What it measures:** How much personas agree with each other

**Scale:** 0.0 to 1.0
- **1.0 = Unanimous** (all 3 personas have same verdict)
- **0.5 = Split** (2 agree, 1 differs)
- **0.0 = Maximum disagreement** (all 3 have different verdicts)

### Why this matters:
High agreement = tool is confident in its assessment
Low agreement = tool is uncertain (might indicate borderline quality)

### Examples from your data:

**Example 1: High Agreement (1.0) - Tool is CONFIDENT**
```
ifdp-2020-2 (Tier 4):
  Persona 1: FAIL
  Persona 2: FAIL
  Persona 3: FAIL
  → agreement_level_r2c = 1.0 (all agree)
  → Tool is confident this is a bad paper
```

**Example 2: Split Agreement (0.5) - Tool is UNCERTAIN**
```
ifdp-2020-4 (Tier 1):
  Persona 1 (Theorist): FAIL
  Persona 2 (Econometrician): REVISE
  Persona 3 (Policymaker): FAIL
  → agreement_level_r2c = 0.5 (2 FAIL, 1 REVISE)
  → Tool has moderate confidence but not unanimous
```

**Example 3: Maximum Disagreement (0.0) - Tool is VERY UNCERTAIN**
```
Hypothetical example:
  Persona 1: PASS
  Persona 2: REVISE
  Persona 3: FAIL
  → agreement_level_r2c = 0.0 (all different)
  → Tool has no confidence, personas completely disagree
```

### Agreement Level R1 vs R2C:

**`agreement_level_r1`**: How much personas agreed INITIALLY (Round 1)
**`agreement_level_r2c`**: How much personas agreed AFTER DEBATE (Round 2C)

**Comparison tells you if debate increased confidence:**
```
If R2C agreement > R1 agreement → Debate increased confidence
If R2C agreement < R1 agreement → Debate created more disagreement
If R2C agreement = R1 agreement → Debate didn't change confidence
```

---

## 2. Borderline Flag

**What it measures:** Papers that are genuinely ambiguous vs. tool failures

**Values:** "Yes" or "No"

### Flagging Logic:
```
Borderline = Yes if:
  - Low agreement (< 0.5) AND
  - Middle-tier paper (Tier 2 or Tier 3)
```

### Why this matters:
**Distinguishes two types of disagreement:**

**Type A: Expected disagreement (borderline paper)**
- Tier 2 or 3 paper (genuinely middle-quality)
- Personas disagree (low agreement)
- **This is NORMAL** - paper quality is genuinely ambiguous
- Flag: "Yes"

**Type B: Unexpected disagreement (tool failure)**
- Tier 1 or 4 paper (should be clear-cut)
- Personas disagree (low agreement)
- **This is PROBLEMATIC** - tool should be confident on clear cases
- Flag: "No" (but still a problem!)

### Examples from your data:

**NO borderline papers in your data because:**
All papers with low agreement are Tier 1 or 4 (clear-cut tiers), so disagreement indicates tool problems, not genuine ambiguity.

**Example of what WOULD be flagged:**
```
Paper: ifdp-2020-X (Tier 2 - good journal)
  agreement_level_r2c: 0.0 (all personas disagree)
  borderline_flag: Yes
  → This is EXPECTED - Tier 2 papers can be borderline
  → Not a tool failure, just genuinely ambiguous quality
```

**Example of tool failure (NOT flagged as borderline):**
```
ifdp-2020-4 (Tier 1 - TOP journal)
  agreement_level_r2c: 0.5 (moderate disagreement)
  borderline_flag: No
  → This is PROBLEMATIC - tool should be confident on Tier 1
  → Tool failure, not genuine ambiguity
```

---

## 3. Debate Improved Calibration

**What it measures:** Did debate move the score closer to what we expect for that tier?

**Values:** "Yes", "No", or "No Change"

### How it works:

**Step 1: Calculate R1 error**
```
R1 error = |consensus_score_r1 - expected_score_for_tier|
```

**Step 2: Calculate R2C error**
```
R2C error = |consensus_score_r2c - expected_score_for_tier|
```

**Step 3: Compare**
```
If R2C error < R1 error  → "Yes" (debate improved)
If R2C error > R1 error  → "No" (debate worsened)
If R2C error = R1 error  → "No Change"
```

### Examples from your data:

**Example 1: Debate IMPROVED calibration**
```
ifdp-2020-2 (Tier 4):
  Expected score: 0.10
  R1 score: 0.53 (error = 0.43)
  R2C score: 0.00 (error = 0.10)
  → debate_improved_calibration = "Yes"
  → Debate made score MORE aligned with expected (0.00 is closer to 0.10 than 0.53 is)
```

**Example 2: Debate WORSENED calibration**
```
ifdp-2020-1 (Tier 2):
  Expected score: 0.70
  R1 score: 0.50 (error = 0.20)
  R2C score: 0.40 (error = 0.30)
  → debate_improved_calibration = "No"
  → Debate made score LESS aligned with expected (0.40 is farther from 0.70 than 0.50 is)
  → Personas became stricter, moving away from expected
```

**Example 3: No change**
```
ifdp-2020-1 hypothetically:
  Expected score: 0.70
  R1 score: 0.60 (error = 0.10)
  R2C score: 0.80 (error = 0.10)
  → debate_improved_calibration = "No Change"
  → Score moved from 0.60 to 0.80, but both are equally far from 0.70
```

### Why this matters:

Shows whether debate is HELPFUL or HARMFUL for calibration:
- **"Yes" a lot** → Debate adds value, keep it
- **"No" a lot** → Debate makes things worse, reconsider protocol
- **Tier-specific patterns** → Maybe debate helps for some tiers but not others

**Your data shows:**
- Debate helps Tier 3 & 4 (identifying weak papers)
- Debate hurts Tier 1 & 2 (makes tool even harsher on good papers)

---

## 4. Verdict Stability (How Much Changed)

**What it measures:** How much the verdict changed from R1 to R2C

**Values:** "Stable", "Shifted", or "Flipped"

### Categories Explained:

#### **STABLE** = Verdict stayed in same category
```
Examples:
  PASS → PASS
  REVISE → REVISE
  FAIL → FAIL
  
Or minor changes within same zone:
  Score moved from 0.80 to 0.85 (both in PASS zone)
  Score moved from 0.45 to 0.50 (both in REVISE zone)
```

#### **SHIFTED** = Verdict moved to adjacent category
```
Examples:
  PASS → REVISE  (moved down one level)
  REVISE → FAIL  (moved down one level)
  FAIL → REVISE  (moved up one level)
  REVISE → PASS  (moved up one level)
```

#### **FLIPPED** = Verdict made a dramatic change (skipped a category)
```
Examples:
  PASS → FAIL  (skipped REVISE, went from best to worst)
  FAIL → PASS  (skipped REVISE, went from worst to best)
```

### How it's calculated:

**Uses consensus score thresholds:**
```
consensus_score >= 0.75  → PASS category
consensus_score 0.40-0.75 → REVISE category  
consensus_score < 0.40   → FAIL category
```

**Then compares R1 category to R2C category:**
```
Same category → Stable
Adjacent category → Shifted
Skipped category → Flipped
```

### Examples from your data:

**Example 1: STABLE**
```
ifdp-2020-9 (Tier 4):
  R1 score: 0.23 → FAIL category
  R2C score: 0.00 → FAIL category
  verdict_stability: "Stable"
  → Verdict stayed FAIL (became more confident in FAIL, but same category)
```

**Example 2: SHIFTED**
```
ifdp-2020-1 (Tier 2):
  R1 score: 0.50 → REVISE category
  R2C score: 0.40 → REVISE category (borderline)
  verdict_stability: "Stable" (stayed in REVISE range)
  
But if it had gone to:
  R2C score: 0.35 → FAIL category
  verdict_stability: "Shifted" (moved from REVISE to FAIL)
```

**Example 3: FLIPPED**
```
ifdp-2020-5 (Tier 3):
  R1 score: 0.83 → PASS category (personas initially lenient)
  R2C score: 0.17 → FAIL category (personas became very strict)
  verdict_stability: "Flipped"
  → Dramatic change - went from PASS all the way to FAIL
  → Skipped REVISE entirely
```

### Why this matters:

**Shows debate volatility:**
- **Lots of "Stable"** → Debate mostly confirms initial opinions
- **Lots of "Shifted"** → Debate causes moderate opinion changes
- **Any "Flipped"** → Debate causes dramatic reversals (investigate these!)

**Your data shows:**
- 40% Stable (debate confirmed initial assessment)
- 50% Shifted (debate caused moderate changes)
- 10% Flipped (1 paper had dramatic reversal)

**The flipped paper (ifdp-2020-5, Tier 3):**
- Started with personas thinking PASS
- After debate, changed to FAIL
- This is GOOD for a Tier 3 paper (mediocre quality)
- Shows debate can catch initial over-optimism

---

## Quick Reference Table

| Metric | Scale | Good Value | Bad Value | What It Tells You |
|--------|-------|------------|-----------|-------------------|
| **agreement_level_r2c** | 0.0-1.0 | 1.0 (unanimous) | 0.0 (all differ) | Tool confidence |
| **borderline_flag** | Yes/No | "Yes" for Tier 2/3 | "Yes" for Tier 1/4 | Expected vs unexpected disagreement |
| **debate_improved_calibration** | Yes/No | "Yes" | "No" | Whether debate helps |
| **verdict_stability** | Categories | Depends | "Flipped" needs investigation | How much opinions changed |

---

## Using These Together

### Pattern 1: High Confidence + Good Calibration = ✅ TRUST THE TOOL
```
Paper: ifdp-2020-2 (Tier 4)
  agreement_level_r2c: 1.0 (all agree)
  calibration_error: 0.10 (low)
  debate_improved: Yes
  verdict_stability: Shifted
  → Tool is confident AND accurate
  → Debate helped
  → Verdict changed moderately but converged to correct assessment
```

### Pattern 2: High Confidence + Bad Calibration = ⚠️ CONFIDENT BUT WRONG
```
Hypothetical:
  agreement_level_r2c: 1.0 (all agree)
  calibration_error: 0.70 (huge)
  → Tool is confident BUT wrong
  → Most concerning pattern - systematic bias
```

### Pattern 3: Low Confidence + Bad Calibration = 🤷 UNCERTAIN AND WRONG
```
Paper: ifdp-2020-4 (Tier 1)
  agreement_level_r2c: 0.5 (split)
  calibration_error: 0.72 (huge)
  borderline_flag: No (shouldn't be ambiguous for Tier 1)
  → Tool is uncertain AND wrong
  → Major tool failure on what should be clear case
```

### Pattern 4: Low Confidence + Borderline Flag = ✅ EXPECTED AMBIGUITY
```
Hypothetical Tier 2 paper:
  agreement_level_r2c: 0.0 (all differ)
  calibration_error: 0.30 (moderate)
  borderline_flag: Yes
  → Tool is uncertain because paper IS genuinely borderline
  → This is NORMAL, not a failure
```

---

## Summary

- **agreement_level**: Tool confidence (1.0 = unanimous, 0.0 = all disagree)
- **borderline_flag**: Distinguishes expected ambiguity from tool failures
- **debate_improved_calibration**: Did debate move score toward expected range?
- **verdict_stability**: Stable (same category) vs Shifted (adjacent) vs Flipped (dramatic)

**These 4 metrics together tell you:**
1. How confident the tool is
2. Whether that confidence is appropriate
3. Whether debate adds value
4. How volatile the debate process is
