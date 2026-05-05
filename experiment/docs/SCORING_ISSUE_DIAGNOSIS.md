# Scoring System Issue Diagnosis & Fix

## Problem Summary

Analysis of `results/referee_batch_results_20260427_182204.csv` revealed:
- ✅ **R1 scores: 73.3% captured** (11/15 personas provided scores)
- ❌ **R2C scores: 0.0% captured** (0/15 personas provided scores)
- ❌ **Numeric consensus: Failed** (all blank due to missing scores)

## Root Cause

**Personas were ignoring the scoring instruction** in Round 2C prompts.

Original prompts were too passive:
```
### OUTPUT FORMAT
- **Final Verdict**: [PASS / REVISE / FAIL]
- **Final Score**: [X/10] (1-3=FAIL, 4-7=REVISE, 8-10=PASS)
```

Problems:
1. No emphasis that scores are **REQUIRED**
2. Buried in long prompt after debate transcript
3. No example format
4. Personas could skip it without consequence

## Fix Applied

Updated prompts in `app_system/referee/engine.py`:

### Round 1 (line ~461):
```python
⚠️ **REQUIRED: You MUST provide both a categorical verdict AND a numeric quality score.**

**MANDATORY OUTPUT FORMAT** (include these exact lines):
```
- **Verdict**: [PASS/REVISE/FAIL]
- **Score**: [X/10]
```

Example:
```
- **Verdict**: REVISE
- **Score**: 6/10
```
```

### Round 2C (line ~206):
```python
⚠️ **CRITICAL: You MUST include BOTH your Final Verdict AND Final Score in the output below.**

### MANDATORY OUTPUT FORMAT (include these exact lines first):
```
- **Final Verdict**: [PASS / REVISE / FAIL]
- **Final Score**: [X/10]
```
```

## Changes Made

1. ✅ Added **⚠️ warning symbol** to draw attention
2. ✅ Used **REQUIRED** / **CRITICAL** / **MANDATORY** language
3. ✅ Moved format to **top** of output instructions
4. ✅ Added **example** format with code block
5. ✅ Repeated scale reminder (1-3=FAIL, 4-7=REVISE, 8-10=PASS)

## Testing & Validation

**Quick test (recommended before full batch):**
```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents/experiment
./test_single_paper_scoring.sh
```

This runs on 1 paper (~3-4 min) and checks:
- R1 score capture rate
- R2C score capture rate
- Numeric consensus calculation

**Expected outcome:**
- R1 scores: 3/3 (100%)
- R2C scores: 3/3 (100%)
- Numeric consensus: calculated successfully

**If test passes:** Run full batch
**If test fails:** Prompts need further strengthening (see "Additional Options" below)

## Additional Options if Fix Doesn't Work

If personas still ignore scores after this fix:

### Option 1: Severity-Based Scoring
Use explicit severity labels first, then score within range:
```
- **Severity**: [CRITICAL/MAJOR/MODERATE/MINOR/NONE]
- **Score**: [X/10] (within severity range)
```

See `SEVERITY_BASED_SCORING.md` for full approach.

### Option 2: System Prompt Update
Add scoring requirement to **persona system prompts** (currently only in round prompts):
```python
SYSTEM_PROMPTS["Econometrician"] = """
ROLE: Econometrician...

⚠️ IMPORTANT: In every report, you MUST provide:
1. A categorical verdict (PASS/REVISE/FAIL)
2. A numeric quality score (1-10)

[rest of prompt]
"""
```

### Option 3: Post-Processing Validation
Add validation check that rejects reports without scores:
```python
def validate_report_has_score(report: str, persona: str, round: str) -> bool:
    score = extract_score_from_report(report)
    if score is None:
        print(f"[WARNING] {persona} {round} report missing score - requesting resubmission")
        return False
    return True
```

## Why This Happens

LLMs trained on academic writing often:
1. Focus on **qualitative argumentation** over **quantitative rating**
2. Treat numeric scores as "optional metadata" vs core content
3. Prioritize **lengthy justification** over **structured output**

Making scores **structurally prominent** and **linguistically mandatory** addresses this bias.

## Long-Term Solution

For production systems, consider:
1. **JSON schema output** with required fields (score cannot be omitted)
2. **Two-stage prompting**: First collect score, then generate report
3. **Validation hooks**: Auto-retry if score missing
4. **Fine-tuned models**: Train on examples that always include scores

## Next Steps

1. ✅ Run `./test_single_paper_scoring.sh` (~3-4 min)
2. ✅ Verify 100% score capture
3. ✅ Run full batch if test passes:
   ```bash
   python3 batch_referee_reports.py \
       --pdf-dir ifdp_sample/ \
       --ground-truth ../experiment-papers/IFDP_2020/IFDP_2020_tracking_clean.csv \
       --output-dir results/ \
       --limit 5
   ```
4. ✅ Check new CSV for complete score capture
5. ✅ Compare categorical vs numeric consensus scores

## Files Modified

- `app_system/referee/engine.py` (lines ~461, ~206)
  - Round 1 user prompt: Added REQUIRED language and example
  - Round 2C template: Added MANDATORY format and warning

## Files Created

- `experiment/test_single_paper_scoring.sh` - Quick validation script
- `experiment/SCORING_ISSUE_DIAGNOSIS.md` - This document
- `experiment/SEVERITY_BASED_SCORING.md` - Alternative approach
