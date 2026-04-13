# Equation/Table Fixing Feature - User Guide

## Overview

The app now has an **enhanced two-stage workflow** for processing papers with equations and tables:

### 🎯 Stage 1: Fix Equations & Tables (NEW & IMPROVED)
Upload your paper → Automatic detection → Manual review & correction

### 🎯 Stage 2: Run Evaluation
Section detection → Referee report generation

---

## What's Been Fixed/Improved

### ✅ **1. Better Visibility**
- **Before**: Region fixer was hidden by default (checkbox unchecked)
- **After**: Region fixer expands automatically so you can't miss it
- Visual workflow indicator shows which stage you're in

### ✅ **2. Statistics & Breakdown**
- **Before**: Only showed total count
- **After**: Shows breakdown by type:
  - 🔢 Equations count
  - 📊 Tables count
  - 📈 Total regions count
  - Summary of fixes applied (AI, Manual, OCR, Skipped)

### ✅ **3. LaTeX Support**
- **Before**: Manual text box had no LaTeX guidance
- **After**:
  - Clear "Manual edit/LaTeX" button label
  - Expandable LaTeX guide with examples
  - Tips for subscripts, superscripts, fractions, Greek letters
  - Example equations you can copy/paste

### ✅ **4. Two-Stage Workflow**
- **Before**: Workflow wasn't clear
- **After**:
  - Visual progress indicator at top
  - Stage 1 highlighted when fixing equations
  - Stage 2 highlighted when running evaluation
  - Clear "Apply Fixes and Continue to Referee Report" button

### ✅ **5. Better Region Display**
- **Before**: Generic region headers
- **After**:
  - Icons for equations (🔢) and tables (📊)
  - Quality issue percentage shown
  - Type-specific guidance for each region
  - First 5 regions auto-expanded (was 3)
  - Color-coded by type

---

## How to Use

### Step 1: Upload Your Paper
1. In the Section Evaluator tab, select paper type
2. Choose "Upload file (auto-detect sections)"
3. Click **"Scan for Sections"**

### Step 2: Review Detected Issues (NEW STAGE)
The system automatically detects problematic equations/tables and shows:
- **Metrics**: Count of equations, tables, total regions
- **Each Region**: Original extraction + 4 fix options

### Step 3: Fix Each Region
For each detected issue, you have 4 options:

#### Option A: 🤖 Auto-clean with AI
- Click "Auto-clean with AI"
- AI attempts to fix formatting automatically
- Review the cleaned version
- Click "Apply Fixes and Continue" when done

#### Option B: ✏️ Manual edit/LaTeX (RECOMMENDED FOR EQUATIONS)
- Click "Manual edit/LaTeX"
- **Expand the "LaTeX Guide & Examples"** for help
- Paste your corrected equation using LaTeX notation
- Examples:
  ```
  Y_t = β_0 + β_1 X_{t-1} + ε_t

  \frac{∂L}{∂θ} = \sum_{i=1}^{n} (y_i - \hat{y}_i)

  Pr(Y=1|X) = \frac{exp(Xβ)}{1 + exp(Xβ)}
  ```
- Click "💾 Save"

#### Option C: 🖼️ Upload image
- If you have a screenshot of the correct equation
- Click "Upload image"
- Upload PNG/JPG
- Click "Extract Text (OCR)"

#### Option D: ⏭️ Skip
- Keep original extraction
- Use if issue is minor or acceptable

### Step 4: Apply Fixes
- Review the summary showing:
  - 🤖 AI-Cleaned count
  - ✏️ Manual Edits count
  - 🖼️ Image OCR count
  - ⏭️ Skipped count
- Click **"✅ Apply Fixes and Continue to Referee Report"**

### Step 5: Continue to Evaluation
- System detects sections using fixed text
- Proceed with normal referee report workflow

---

## LaTeX Notation Reference

### Subscripts & Superscripts
```
X_t          → X with subscript t
X^2          → X squared
X_{t-1}      → X with subscript (t-1)
X_{i,t}      → X with subscript i,t
X_t^2        → X subscript t, squared
```

### Greek Letters
```
α (alpha)    β (beta)     γ (gamma)    δ (delta)
ε (epsilon)  θ (theta)    λ (lambda)   μ (mu)
σ (sigma)    Σ (Sigma)    π (pi)       Φ (Phi)
```

### Math Operators
```
\frac{a}{b}           → fraction a/b
\sum_{i=1}^{n}        → summation from i=1 to n
\int_{0}^{1}          → integral from 0 to 1
\partial              → partial derivative symbol ∂
\hat{y}               → y with hat
\bar{x}               → x with bar
\log                  → log
\exp                  → exp
```

### Example Equations
```latex
# Simple regression
Y_t = β_0 + β_1 X_{t-1} + ε_t

# Cobb-Douglas production
log(GDP_t) = α + β log(K_t) + γ log(L_t)

# Maximum likelihood
\frac{∂L}{∂θ} = \sum_{i=1}^{n} (y_i - \hat{y}_i)

# Logit probability
Pr(Y=1|X) = \frac{exp(Xβ)}{1 + exp(Xβ)}

# VAR model
Y_t = Φ_1 Y_{t-1} + Φ_2 Y_{t-2} + ε_t

# ARMA specification
y_t = φ_1 y_{t-1} + θ_1 ε_{t-1} + ε_t
```

---

## Tips & Best Practices

### ✅ DO:
- **Review all detected regions** - especially equations
- **Use LaTeX notation** for equations (most accurate)
- **Expand the LaTeX guide** for syntax help
- **Copy/paste from paper source** if you have .tex file
- **Auto-clean tables** - AI is good at restructuring tables
- **Manually fix equations** - you know the correct notation

### ❌ DON'T:
- Skip all regions without reviewing (defeats the purpose)
- Use plain text for complex equations (use LaTeX)
- Forget to click "Save" after manual edits
- Skip the workflow indicator - it shows your progress

---

## Troubleshooting

### "No extraction issues detected"
- ✅ Good! Your PDF extracted cleanly
- System auto-advances to Stage 2

### "I don't see the region fixer"
- Check that you clicked "Scan for Sections"
- Region fixer only appears if issues are detected
- If detection threshold too high, lower `min_confidence` in code

### "Manual edits not saving"
- Make sure to click "💾 Save" button after pasting
- Check for success message: "✅ Saved!"
- Then click "Apply Fixes and Continue" at bottom

### "LaTeX not rendering correctly"
- LaTeX is preserved as text for LLM consumption
- It won't render visually in the UI
- The evaluation agent understands LaTeX notation

### "Too many regions detected"
- Set higher `min_confidence` threshold (currently 0.5)
- Auto-clean less critical ones
- Manually fix only the most important equations

---

## Technical Details

### Detection Algorithm
- Uses heuristics to detect broken math/tables
- Scores based on:
  - Greek letters, math symbols
  - Broken subscripts/superscripts
  - Orphaned operators
  - Table indicators (*, **, parentheses)
  - Column alignment issues

### Confidence Score
- 0.0 - 1.0 scale (higher = more broken)
- Default threshold: 0.5
- Equations: looks for separated subscripts, symbols
- Tables: looks for regression formatting, stars

### Session State
- Fixes stored in session state per manuscript
- Workflow tracks two stages independently
- Fixed text cached for Stage 2

---

## Future Improvements (Potential)

- [ ] Visual LaTeX rendering preview
- [ ] Bulk "auto-clean all" option
- [ ] Equation/table gallery from common papers
- [ ] Import from Overleaf/GitHub
- [ ] Version history of fixes
- [ ] Export cleaned text as .tex file

---

## Questions?

If you encounter issues or have suggestions:
1. Check session state hasn't been corrupted (refresh page)
2. Try lowering confidence threshold
3. Verify LaTeX syntax in examples
4. Report bugs with screenshot of detected region

**Key Files:**
- `region_fixer.py` - Main UI logic
- `math_cleanup.py` - Detection algorithms
- `main.py` - Workflow orchestration
