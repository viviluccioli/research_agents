# Referee Report - Two-Stage Workflow Update

## What Changed

The **Referee Report** tab now has a **two-stage workflow** with equation/table fixing **before** the multi-agent debate runs.

### ❌ OLD Workflow (Before)
1. Upload PDF
2. Click "Run Multi-Agent Evaluation"
3. Text extracted → Debate runs immediately
4. No chance to fix equations/tables

### ✅ NEW Workflow (After)
1. Upload PDF
2. Click **"Step 1: Extract & Review Text"**
3. **Review & fix equations/tables** (NEW STAGE)
4. Click **"Step 2: Run Multi-Agent Evaluation"**
5. Debate runs with fixed text

---

## How to Use the New Workflow

### Step 1: Upload & Select
1. Go to **Referee Report** tab
2. Upload your PDF in the document uploader
3. Select your manuscript from dropdown
4. Select paper type
5. Configure persona settings (Automatic/Manual)

### Step 2: Extract Text
- Click **"🚀 Step 1: Extract & Review Text"**
- System extracts text and detects problematic regions
- Visual indicator shows you're in **STAGE 1: Fix Equations & Tables**

### Step 3: Review Detected Issues
You'll see:
- **Metrics**: Count of equations (🔢), tables (📊), total regions
- **Each Region**:
  - Original extraction (what system sees)
  - Type-specific guidance
  - 4 fixing options

### Step 4: Fix Each Region
For each detected issue, choose:

#### 🤖 **Auto-clean with AI** (Recommended for tables)
- AI attempts to restructure broken tables
- Good for fixing column alignment

#### ✏️ **Manual edit/LaTeX** (Recommended for equations)
- Click to open text editor
- **Expand "LaTeX Guide & Examples"** for help
- Paste your corrected equation:
  ```
  Y_t = β_0 + β_1 X_{t-1} + ε_t
  ```
- Click "💾 Save"

#### 🖼️ **Upload image**
- Upload screenshot of equation
- Run OCR to extract text

#### ⏭️ **Skip**
- Keep original extraction
- Use if minor/acceptable issue

### Step 5: Apply Fixes
- Review summary:
  - 🤖 AI-Cleaned: X
  - ✏️ Manual Edits: Y
  - 🖼️ Image OCR: Z
  - ⏭️ Skipped: W
- Click **"✅ Apply Fixes and Continue to Referee Report"**
- Visual indicator shows **STAGE 1: ✓ Complete** (green)
- Visual indicator shows **STAGE 2: Ready to run** (blue)

### Step 6: Run Debate
- Click **"🚀 Step 2: Run Multi-Agent Evaluation"**
- Multi-agent debate runs with fixed text
- Results displayed as before

### Step 7: Start Over (Optional)
- If you want to redo the process
- Click **"🔄 Start Over"**
- Clears all session state
- Go back to Step 2

---

## Key Features

### Visual Workflow Indicator
Shows your current stage:

**Stage 1 (Active)**:
```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ STAGE 1: Fix Equations      │  →   │ STAGE 2: Run Debate         │
│ Currently here ✓            │      │ (grayed out)                │
└─────────────────────────────┘      └─────────────────────────────┘
```

**Stage 2 (Active)**:
```
┌─────────────────────────────┐      ┌─────────────────────────────┐
│ STAGE 1: Fix Equations      │  →   │ STAGE 2: Run Debate         │
│ ✓ Complete (green)          │      │ Currently here ✓            │
└─────────────────────────────┘      └─────────────────────────────┘
```

### Statistics Dashboard
- **Before**: No breakdown
- **After**:
  - 🔢 Equations: 8
  - 📊 Tables: 3
  - 📈 Total: 11

### LaTeX Support
- Comprehensive LaTeX guide with examples
- Subscripts: `X_t`, `X_{t-1}`
- Superscripts: `X^2`
- Fractions: `\frac{a}{b}`
- Greek letters: α, β, γ, δ, ε, θ, σ
- Full example equations (regression, logit, VAR, ARMA)

### Region Display
- **Icons**: 🔢 for equations, 📊 for tables
- **Quality Score**: Shows % confidence of issue
- **Guidance**: Type-specific tips per region
- **Auto-expand**: First 5 regions open by default

---

## LaTeX Examples

### Common Economics Equations

```latex
# Simple regression
Y_t = β_0 + β_1 X_{t-1} + ε_t

# Cobb-Douglas
log(Y) = α + β log(K) + γ log(L)

# MLE gradient
\frac{∂L}{∂θ} = \sum_{i=1}^{n} (y_i - \hat{y}_i)

# Logit probability
Pr(Y=1|X) = \frac{exp(Xβ)}{1 + exp(Xβ)}

# VAR model
Y_t = Φ_1 Y_{t-1} + Φ_2 Y_{t-2} + ε_t

# ARMA
y_t = φ_1 y_{t-1} + θ_1 ε_{t-1} + ε_t
```

---

## Technical Details

### Session State Keys
- `referee_extraction_done_{filename}`: Tracks if text extracted
- `referee_fixes_applied_{filename}`: Tracks if fixes applied
- `referee_text_{filename}`: Stores original text
- `referee_fixed_text_{filename}`: Stores fixed text
- `referee_region_fixes_{filename}`: Stores user fixes

### Detection Algorithm
- **Confidence threshold**: 0.5 (adjustable)
- **Equations detected by**:
  - Greek letters (α, β, γ, etc.)
  - Subscripts/superscripts on separate lines
  - Broken math operators
  - Math functions (log, exp, max, min)
- **Tables detected by**:
  - Multiple numeric columns
  - Parenthesized numbers (standard errors)
  - Significance stars (*, **, ***)
  - Table keywords (Table, Panel, Column)

### Workflow Reset
- Clears all cached data for the manuscript
- Allows restarting from Step 1
- Useful if wrong file uploaded or major changes needed

---

## Comparison: Section Evaluator vs Referee Report

| Feature | Section Evaluator | Referee Report |
|---------|-------------------|----------------|
| **Purpose** | Section-by-section evaluation | Multi-agent debate evaluation |
| **Equation Fixer** | ✅ Yes (always shown) | ✅ Yes (NEW - added today) |
| **When shown** | After "Scan for Sections" | After "Extract & Review Text" |
| **Workflow** | Extract → Fix → Detect sections → Evaluate | Extract → Fix → Run debate |
| **Cache prefix** | `se_v3` | `referee` |
| **Best for** | Detailed section-level feedback | Overall publication readiness |

---

## Troubleshooting

### "I don't see the equation fixer"
**Issue**: You clicked "Run Multi-Agent Evaluation" (old button)
**Solution**:
1. The button is now **"Step 1: Extract & Review Text"**
2. Refresh the page or click "Start Over"
3. Click the new Step 1 button

### "No extraction issues detected"
**Good news!** Your PDF extracted cleanly. System auto-advances to Stage 2.

### "Fixes not saving"
1. Make sure to click **"💾 Save"** after editing
2. Check for success message
3. Then click **"Apply Fixes and Continue"** at bottom

### "Can't proceed to Stage 2"
1. Make sure you clicked **"Apply Fixes and Continue"** (or "Skip All")
2. Don't click browser back button - use "Start Over" instead
3. Check that Stage 1 shows green checkmark

### "Want to change fixes"
1. Click **"🔄 Start Over"**
2. Re-extract and make different fixes
3. Or uncheck "Review regions" to skip fixing

---

## Files Modified

1. **`referee/workflow.py`**:
   - Added import for `render_region_fixer`
   - Split button into 2-stage flow
   - Added session state tracking
   - Added visual workflow indicator
   - Added "Start Over" button

2. **`section_eval/region_fixer.py`** (enhanced earlier):
   - Statistics dashboard
   - LaTeX guide with examples
   - Improved region display
   - Better button labels

---

## Benefits

✅ **Better accuracy**: Equations are readable by LLM agents
✅ **User control**: Review before debate runs (saves API calls)
✅ **Transparency**: See what system extracted vs what you provided
✅ **Educational**: LaTeX guide helps users learn notation
✅ **Flexibility**: Can auto-clean, manually edit, or skip
✅ **Efficient**: Only fixes problematic regions, not entire document

---

## Future Enhancements

Possible future improvements:
- [ ] Batch "auto-clean all equations"
- [ ] Visual LaTeX rendering preview
- [ ] Version history of fixes
- [ ] Export cleaned text as .tex file
- [ ] Confidence threshold adjuster in UI
- [ ] Side-by-side comparison (original vs fixed)

---

## Questions?

If the equation fixer doesn't appear:
1. Verify you're in the **Referee Report** tab (not Section Evaluator)
2. Click **"Step 1: Extract & Review Text"** (not old button name)
3. Ensure PDF has equations/tables to detect
4. Try lowering confidence threshold if needed
5. Check browser console for errors

**Key Files**:
- Main workflow: `referee/workflow.py`
- Fixer UI: `section_eval/region_fixer.py`
- Detection logic: `section_eval/math_cleanup.py`
