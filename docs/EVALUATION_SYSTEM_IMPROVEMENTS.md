# Evaluation System Improvements: Addressing Score Compression

## Problem Identified

The original evaluation system consistently rated both high-quality and low-quality papers around 3-3.5, failing to discriminate between:
- **Sophisticated work** (e.g., Rithika's prompts): Deep economic reasoning, rigorous derivations, empirical calibration, welfare analysis
- **Basic work** (e.g., Andrew's prompts): Correct but shallow, routine application, ad hoc parameters, minimal intuition

### Root Causes

1. **Binary evaluation**: Criteria checked "Is X present?" not "How well is X done?"
2. **Coarse criteria**: No measurement of depth, sophistication, or rigor beyond basic correctness
3. **Quote-based validation weakness**: Any text discussing a criterion gets credit
4. **LLM leniency bias**: Without concrete anchors, scores compress toward 3-4
5. **Lack of differentiation guidance**: No explicit instruction to use full 1-5 range discriminatingly

---

## Solutions Implemented

### 1. Added Depth-Focused Criteria

#### For Theoretical/Model Sections (`proofs`)

**Before:**
- `correctness` (35%): "No mathematical errors"
- `logical_flow` (25%): "Steps follow logically"
- `completeness` (20%): "No missing steps"
- `intuition` (20%): "Economic interpretation provided"

**After:**
- `correctness` (20%): "No mathematical errors"
- `rigor` (25%): "Technical conditions handled: convergence derived, transversality stated, edge cases addressed (5=complete treatment, 3=basic derivation, 1=hand-waving)"
- `economic_depth` (25%): "Mechanisms explained with depth: decomposition, welfare analysis, multi-layered interpretation (5=profound insights, 3=one-sentence explanation, 1=tautological)"
- `sophistication` (15%): "Analysis advances beyond routine (5=novel insight, 3=competent standard, 1=trivial)"
- `clarity` (15%): "Clear and accessible to experts"

#### For Empirical Sections (`methodology`, `results`)

**Added criteria:**
- `identification`: Now emphasizes **rigor** (5=compelling with falsification tests, 3=standard adequate, 1=weak/circular)
- `robustness_depth`: Now measures **comprehensiveness** (5=extensive pre-specified, 3=basic checks, 1=minimal/cherry-picked)
- `interpretation_depth`: Now requires **calibration** (5=multi-faceted with decomposition, 3=basic magnitude, 1=only significance stars)
- `calibration`: New criterion for results - effect sizes compared to real-world benchmarks

#### For Macro Sections (`calibration`, `simulations`)

**Enhanced criteria:**
- `justification`: Now requires **empirical grounding** (5=micro-founded with citations, 3=reasonable with defense, 1=ad hoc)
- `fit`: Now requires **rigorous evaluation** (5=formal statistical, 3=qualitative comparison, 1=claimed without evidence)
- `welfare_depth`: Now requires **decomposition** (5=distributional analysis, 3=basic calculation, 1=mentioned without quantification)

---

### 2. Concrete Scoring Anchors in Criteria Descriptions

All key criteria now include explicit 5/3/1 examples:

**Example from `economic_depth`:**
> "Economic mechanisms explained with depth beyond surface intuition: provides decomposition, welfare analysis, multi-layered interpretation (5=profound insights, 3=adequate one-sentence explanation, 1=tautological)"

**Example from `identification`:**
> "Causal identification approach rigorously justified (5=compelling argument with falsification tests, 3=standard approach adequately justified, 1=weak or circular reasoning)"

---

### 3. Quality Level Guidelines in Section Prompts

Each major section type now includes concrete quality benchmarks:

#### Model Setup / Proofs
```
**Quality levels:**
- Excellent (5): Rigorous derivation with convergence conditions explicitly derived;
  transversality conditions stated; edge cases handled; multi-layered economic
  interpretation with decomposition or welfare analysis

- Adequate (3): Mathematically correct with standard steps; basic one-sentence
  intuition per result; parameters chosen without empirical grounding

- Poor (1): Mathematical errors; logical gaps; hand-waving instead of rigor;
  tautological interpretation
```

#### Methodology
```
**Quality levels:**
- Excellent (5): Compelling identification argument with institutional detail;
  falsification tests planned; assumptions defended not just stated; comprehensive
  robustness plan pre-specified

- Adequate (3): Standard identification approach adequately explained; basic
  assumptions stated; typical robustness checks mentioned

- Poor (1): Weak or circular identification reasoning; endogeneity unaddressed;
  assumptions hidden
```

---

### 4. Discriminating Scoring Philosophy

Added to main evaluation prompt:

```
## Scoring Philosophy - BE DISCRIMINATING

You are evaluating for TOP ECONOMICS JOURNALS (JPE, QJE, AER, JF, JFE, RFS).
**Use the full 1-5 range.**

- 5 (Excellent): Publication-ready for top journals. Rigorous, insightful, novel.
  Example: Derives convergence conditions explicitly, provides decomposition analysis,
  calibrates to empirical estimates, discusses welfare implications

- 4 (Good): Strong work needing refinement. Publishable in top field journals.
  Example: Rigorous derivation with some depth, missing sophistication elements

- 3 (Adequate): Meets MINIMUM standards but routine/shallow. Technically correct
  but no depth.
  Example: Correct derivation with standard steps, one-sentence intuition, ad hoc parameters

- 2 (Below Average): Significant issues. Major revision needed.
  Example: Derivation has gaps, intuition superficial, identification weak

- 1 (Poor): Fundamental flaws. Not suitable for publication.
  Example: Mathematical errors, circular reasoning, tautological statements

**CRITICAL**: Papers that merely "check boxes" should score 3 at most.
Avoid compression toward 3-4. If work is routine, score it 3. If insightful and
rigorous, score it 4-5. Be honest.
```

---

### 5. Sophistication Pre-Assessment

Before scoring, evaluator must consider:

#### For Theoretical/Model Sections:
1. **Mathematical rigor**: Convergence conditions derived? Transversality stated? Edge cases handled?
2. **Economic depth**: Intuition beyond "X increases with Y"? Mechanisms decomposed? Welfare discussed?
3. **Parameter realism**: Calibrated to data or ad hoc? Ranges economically justified?
4. **Completeness**: Full parameter space explored? Sensitivity analysis? Robustness checks?
5. **Literature integration**: Builds meaningfully on prior results, or just cites papers?

#### For Empirical Sections:
1. **Identification rigor**: Convincingly argued with institutional detail, or merely asserted?
2. **Robustness comprehensiveness**: Extensive pre-specified, or minimal confirmatory?
3. **Economic interpretation**: Effect sizes calibrated to real-world? Or just significance stars?
4. **Honesty**: Null results and weakening cases discussed transparently?
5. **Data appropriateness**: Rigorously justified, or questionable with limitations dismissed?

**Guidance**: If most answers suggest basic/routine work, scores should be 3 or below.

---

### 6. Enhanced Justification Requirements

Evaluators must now provide:

**Weak justification (discouraged):**
> "Assumptions are clearly stated."

**Strong justification (required):**
> "Assumptions are stated but lack economic justification. For example, b > 0 is asserted
> without discussing what values are realistic or citing empirical evidence. This represents
> adequate technical execution but lacks the depth expected for top journals."

---

## Expected Impact

### High-Quality Paper (Rithika's approach)

**Before:** ~3.5 average
- ✓ Has assumptions → 3-4 on "assumptions clarity"
- ✓ Has results → 3-4 on "correctness"
- ✓ Mentions intuition → 3-4 on "intuition"

**After:** ~4.5 average
- ✓ Derives convergence conditions explicitly → **5** on `rigor`
- ✓ Provides decomposition (consumption risk vs disaster risk) → **5** on `economic_depth`
- ✓ Calibrates parameters to empirical estimates → **5** on `justification` (macro)
- ✓ Discusses welfare implications and mechanisms → **5** on `interpretation_depth`
- ✓ Pre-specifies extensive robustness checks → **5** on `robustness_depth`

### Low-Quality Paper (Andrew's approach)

**Before:** ~3.0 average
- ✓ Has assumptions → 3 on "assumptions clarity"
- ✓ Has results → 3 on "correctness"
- ✓ Mentions intuition → 3 on "intuition"

**After:** ~2.5 average
- ✗ No convergence conditions → **2** on `rigor`
- ✗ One-sentence intuition, no decomposition → **3** on `economic_depth`
- ✗ Ad hoc parameters → **2** on `justification`
- ✗ Basic magnitude discussion → **3** on `interpretation_depth`
- ✗ Minimal robustness → **2** on `robustness_depth`

---

## Files Modified

1. **`eval/section_eval/criteria/base.py`**
   - Updated criteria for: `_THEORETICAL`, `_EMPIRICAL`, `_FINANCE`, `_MACRO`
   - Added: `rigor`, `economic_depth`, `sophistication`, `interpretation_depth`, `robustness_depth`, `calibration`
   - Enhanced descriptions with 5/3/1 anchors

2. **`eval/section_eval/prompts/templates.py`**
   - Added discriminating scoring philosophy section
   - Added sophistication pre-assessment checklist
   - Enhanced section-type guidance with quality levels for:
     - `model_setup`, `proofs`, `methodology`, `results`
     - `identification_strategy`, `robustness_checks`
     - `calibration`, `simulations`, `data`, `literature_review`
   - Strengthened justification requirements

---

## Testing Recommendations

1. **Re-evaluate test papers**: Run both Rithika's and Andrew's papers through updated system
2. **Expected outcomes**:
   - Rithika's paper should score 4.0-4.5 (was 3.5)
   - Andrew's paper should score 2.5-3.0 (was 3.0)
   - **Spread should increase from 0.5 to 1.5-2.0 points**

3. **Monitor for**:
   - Score distribution using full 1-5 range
   - Justifications citing specific depth/sophistication elements
   - Clear differentiation between routine and exceptional work

---

## Future Enhancements (Optional)

1. **Multi-dimensional decomposition**: Score each criterion on multiple sub-dimensions
   - E.g., `proofs` → {mathematical_correctness, rigor_completeness, depth_of_reasoning, novelty}

2. **Calibration examples**: Provide 2-3 example papers at each quality level as anchors

3. **Blind comparison**: Periodically evaluate same paper with old vs new system to measure improvement

4. **Score distribution monitoring**: Track if scores are still compressing; adjust anchors if needed
