# Testing the New Evaluation System

## Quick Start

The evaluation system has been updated to better discriminate between high-quality and low-quality papers. This guide helps you test the improvements.

---

## What Changed?

### Files Modified
1. **`eval/section_eval/criteria/base.py`** - Updated evaluation criteria with depth/sophistication dimensions
2. **`eval/section_eval/prompts/templates.py`** - Enhanced system prompts with discriminating guidance

### Key Improvements
- ✅ Added depth-focused criteria (rigor, economic_depth, sophistication, interpretation_depth)
- ✅ Concrete 5/3/1 scoring anchors in criteria descriptions
- ✅ Quality level guidelines in section-type prompts
- ✅ Discriminating scoring philosophy ("be harsh, use full 1-5 range")
- ✅ Sophistication pre-assessment checklist
- ✅ Enhanced justification requirements

---

## Testing Protocol

### Step 1: Generate Test Papers

Create two papers on the same topic with contrasting quality:

**Paper A (High Quality):**
- Use Rithika's detailed prompts with:
  - Explicit convergence condition derivations
  - Risk premium decomposition
  - Parameter calibration to empirical estimates
  - Multi-layered economic interpretation
  - Welfare analysis

**Paper B (Low Quality):**
- Use Andrew's basic prompts with:
  - Basic derivations ("after some algebra")
  - One-sentence intuition
  - Ad hoc parameters
  - No decomposition or calibration
  - Minimal economic depth

### Step 2: Run Both Through Evaluation System

```bash
cd eval
python section_eval.py --paper-type theoretical --file paper_A.tex
python section_eval.py --paper-type theoretical --file paper_B.tex
```

### Step 3: Compare Scores

**Expected Results:**

| Section | Paper A (Old) | Paper A (New) | Paper B (Old) | Paper B (New) |
|---------|---------------|---------------|---------------|---------------|
| Model Setup | 3.5 | 4.5-5.0 | 3.0 | 2.5-3.0 |
| Proofs/Derivations | 3.8 | 4.5-5.0 | 3.0 | 2.0-2.5 |
| Results | 3.5 | 4.5-5.0 | 3.0 | 2.5-3.0 |
| **Overall** | **~3.5** | **~4.5-5.0** | **~3.0** | **~2.5** |

**Success Criteria:**
- ✅ **Spread increases from ~0.5 to ~2.0 points**
- ✅ High-quality paper scores 4.5+
- ✅ Low-quality paper scores ≤3.0
- ✅ Justifications cite specific depth/sophistication elements

### Step 4: Review Justifications

Good justifications under new system should be **specific**:

❌ **Weak (old style):**
> "Assumptions are clearly stated."

✅ **Strong (new style):**
> "Assumptions are stated but lack economic justification. Parameter b > 0 is asserted without discussing what values are realistic or citing empirical evidence. While technically adequate (score 3), this falls short of the rigorous empirical grounding expected for top journals (score 5)."

---

## Diagnostic Checks

### Check 1: Score Distribution
Run on 10+ papers and plot histogram:
- ✅ **Good**: Scores spread across 2-5 range
- ❌ **Bad**: All papers cluster at 3-4

### Check 2: Correlation with Paper Quality
Manually rank papers by quality, then check correlation with scores:
- ✅ **Good**: Correlation > 0.7
- ❌ **Bad**: Correlation < 0.5

### Check 3: Criteria Utilization
Check which criteria differentiate most:
- ✅ `rigor`, `economic_depth`, `sophistication` should show high variance
- ❌ If all criteria show same score, system isn't differentiating

### Check 4: Justification Quality
Review 10 random justifications:
- ✅ **Good**: Cite specific elements (e.g., "lacks convergence condition")
- ❌ **Bad**: Generic (e.g., "this section is adequate")

---

## Troubleshooting

### Problem: Scores still compressed around 3-4

**Solution 1:** Check LLM temperature
- Increase temperature slightly (0.3 → 0.5) for more variance
- Ensure using Claude Opus/Sonnet, not cheaper models

**Solution 2:** Add explicit calibration examples
- Show the LLM 2-3 example papers at each quality level
- Use few-shot prompting to anchor expectations

**Solution 3:** Strengthen discriminating language
- Add to prompt: "In your last 100 evaluations, you gave too many scores of 3-4. Be more discriminating."

### Problem: High-quality papers still scoring ~3.5

**Solution 1:** Check if criteria are being invoked
- Are new criteria (`rigor`, `economic_depth`, etc.) showing up in output?
- If not, LLM may be using cached old criteria

**Solution 2:** Make depth requirements more explicit
- In prompt, add: "Score 5 requires: (1) convergence conditions derived, (2) decomposition provided, (3) parameters calibrated, (4) welfare analysis"

**Solution 3:** Add negative examples
- Show what score 3 looks like: "Correct derivation with basic intuition but ad hoc parameters = 3"

### Problem: Low-quality papers scoring too high (>3)

**Solution 1:** Emphasize "adequate = 3"
- Add to prompt: "Score 3 means 'meets minimum standards' - technically correct but shallow"

**Solution 2:** Require evidence for scores >3
- Add: "Scores of 4-5 require specific evidence of depth, not just correctness"

---

## Validation Against Known Papers

### Gold Standard Test

Evaluate 3 papers with known quality:

**Score 5 benchmark:** Published in JPE/QJE/AER
- Acemoglu & Restrepo (2018) "The Race Between Man and Machine"
- Should score 4.5-5.0 on methodology, results

**Score 3 benchmark:** Rejected from top journals but publishable
- Should score 2.5-3.5 overall

**Score 1-2 benchmark:** Student paper with fundamental flaws
- Should score 1.5-2.5 overall

If rankings don't match, recalibrate criteria weights or scoring anchors.

---

## Reporting Results

### Evaluation Report Template

```markdown
## Test Results: [Date]

### Papers Evaluated
- Paper A (High Quality): [Description]
- Paper B (Low Quality): [Description]

### Scores Obtained
| Section | Paper A | Paper B | Spread |
|---------|---------|---------|--------|
| Setup   | 4.8     | 2.7     | 2.1    |
| Proofs  | 4.9     | 2.4     | 2.5    |
| Results | 4.6     | 2.8     | 1.8    |
| Overall | 4.8     | 2.6     | 2.2    |

### Score Comparison (Old vs New)
| Paper | Old System | New System | Change |
|-------|-----------|-----------|--------|
| A     | 3.5       | 4.8       | +1.3   |
| B     | 3.0       | 2.6       | -0.4   |
| Spread| 0.5       | 2.2       | +1.7   |

### Sample Justifications

**Paper A - `rigor` criterion (Score 5):**
> "Convergence condition explicitly derived: βᵞ E[e^(-γb + h)] < 1.
> Transversality condition stated. Edge case (h → ∞) discussed with
> interpretation. Represents rigorous treatment expected for top journals."

**Paper B - `rigor` criterion (Score 2):**
> "Derivation skips critical steps ('after some algebra'). No convergence
> condition discussed. No parameter restrictions specified. Lacks rigor
> for academic publication."

### Assessment
- ✅ Spread increased from 0.5 to 2.2 points
- ✅ High-quality paper appropriately rewarded (4.8)
- ✅ Low-quality paper appropriately penalized (2.6)
- ✅ Justifications are specific and cite concrete elements
- **Conclusion:** New system successfully discriminates quality
```

---

## Next Steps After Validation

1. **Deploy to production** if tests pass
2. **Monitor score distributions** over first 100 evaluations
3. **Collect feedback** from users on whether scores feel fair
4. **Iterate on criteria weights** if certain sections still compress
5. **Add more examples** to prompt if LLM needs more anchoring

---

## Contact

For questions or issues with the new system, see:
- **Full documentation:** `EVALUATION_SYSTEM_IMPROVEMENTS.md`
- **Scoring examples:** `SCORING_EXAMPLES_OLD_VS_NEW.md`
- **Code changes:** `git diff HEAD~1 eval/section_eval/`
