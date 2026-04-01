# Interactive Region Fixer

## Overview

The Interactive Region Fixer is a UI component that gives users **full control** over fixing problematic equations and tables detected during PDF extraction. Instead of automatically cleaning all regions with LLM, users can choose per-region:

- 🤖 **Auto-clean with LLM** - Let AI fix the formatting
- ✏️ **Paste corrected text** - Manually enter the correct content
- 🖼️ **Upload image** - Upload screenshot and extract via OCR
- ⏭️ **Skip** - Keep the original extraction

This is especially valuable for **theoretical papers** where equations are critical, and **empirical papers** where table structure matters.

## User Workflow

### Step 1: Upload and Extract

1. Upload PDF file
2. Click **"Scan for Sections"**
3. System extracts text

### Step 2: Review Detected Issues

After extraction, the system automatically detects problematic regions and shows:

```
⚠️ Detected 5 problematic region(s) in extraction

□ Review 5 region(s)
```

### Step 3: Expand to Review (Optional)

Check the box to see all detected regions:

```
Region #1 — EQUATION (confidence: 0.85)
  📄 Current Extraction:
    exp(?+? Y +X ? )
    1 t?1 t 2
    Pr(Z = 1|Y ,X ) =
    t t?1 t

  🔧 How to fix this region:
  [🤖 Auto-clean] [✏️ Paste text] [🖼️ Upload image] [⏭️ Skip]
```

### Step 4: Fix Each Region

For each region, users can:

#### Option A: Auto-Clean with LLM

1. Click **🤖 Auto-clean**
2. LLM processes the region
3. Cleaned version appears
4. Ready to apply

**Best for**: Tables, simple equations

#### Option B: Paste Corrected Text

1. Click **✏️ Paste text**
2. Text area appears
3. Paste corrected content (e.g., from LaTeX source)
4. Click **💾 Save**

**Best for**: When you have the LaTeX source, complex equations

#### Option C: Upload Image

1. Click **🖼️ Upload image**
2. Upload screenshot of the equation/table
3. Click **🔍 Extract Text (OCR)**
4. OCR extracts text from image

**Best for**: When PDF has equation images, scanned papers

#### Option D: Skip

1. Click **⏭️ Skip**
2. Original extraction is kept

**Best for**: Minor issues, non-critical content

### Step 5: Apply and Continue

After reviewing regions:

1. Click **✅ Apply Fixes and Continue**
2. System applies all fixes
3. Proceeds to section detection

Or click **⏭️ Skip All** to use original extraction.

## Technical Details

### Detection Algorithm

The system detects problematic regions using heuristics:

**Equation indicators**:
- Greek letters (α, β, σ, ...)
- Subscripts/superscripts on separate lines
- Math operators split across lines
- Statistical notation (Pr(), E[], Var())

**Table indicators**:
- Columns of decimal numbers
- Parenthesized standard errors
- Significance stars (*, **, ***)
- Poor column alignment

**Confidence score** (0-1):
- Higher = more confident it's broken
- Default minimum: 0.5 (adjustable)

### Region Types

**Equation regions**:
- LaTeX formulas
- Mathematical expressions
- Statistical models

**Table regions**:
- Regression results
- Summary statistics
- Data tables

### LLM Cleanup

When user clicks "Auto-clean", the system:

1. Builds specialized prompt based on region type
2. Calls `single_query()` (Claude 3.7 Sonnet)
3. Returns cleaned text
4. Shows preview to user

**Equation cleanup prompt**:
- "Restore subscripts and superscripts"
- "Fix fraction notation"
- "Preserve symbols"

**Table cleanup prompt**:
- "Reconstruct column structure"
- "Align coefficients with standard errors"
- "Preserve significance stars"

### OCR Support

Image upload + OCR requires:

```bash
pip install pytesseract Pillow

# Also install tesseract binary (system-level)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

**OCR process**:
1. User uploads image (PNG/JPG)
2. System opens with Pillow
3. Runs pytesseract OCR
4. Extracts text
5. User can edit extracted text before saving

### State Management

The system uses Streamlit session state to track:

```python
# Per manuscript state keys:
f"{cache_prefix}_extraction_done_{manuscript}"     # Extraction complete
f"{cache_prefix}_region_fixes_{manuscript}"        # User's fixes
f"{cache_prefix}_fixes_applied_{manuscript}"       # User clicked "Apply"
f"{cache_prefix}_fixed_text_{manuscript}"          # Modified text
```

**Workflow states**:
1. **Initial**: No extraction
2. **Extracted**: Text extracted, showing region fixer UI
3. **Fixes applied**: User reviewed/fixed, ready for section detection
4. **Sections detected**: Continue with evaluation

## Configuration

### Adjust Detection Sensitivity

Edit `region_fixer.py` call in `main.py`:

```python
render_region_fixer(
    text=paper_text,
    manuscript_name=manuscript,
    cache_prefix=self.CACHE_PREFIX,
    llm_query_fn=single_query,
    min_confidence=0.5  # Lower = detect more regions (0.3 = aggressive)
)
```

### Customize Region Detection

Edit `math_cleanup.py`:

```python
# Line ~165: Equation threshold
if brokenness > 0.3:  # Lower = more aggressive

# Line ~190: Table threshold
if tableness > 0.4:   # Lower = more aggressive
```

### Disable Feature

To skip region fixer entirely, comment out in `main.py`:

```python
# if extraction_done and not fixes_applied:
#     render_region_fixer(...)
#     return
```

## Examples

### Example 1: Theoretical Paper with Equations

**Detected region**:
```
Region #1 — EQUATION (confidence: 0.85)
📄 Current Extraction:
  exp(?+? Y +X ? )
  1 t?1 t 2
  Pr(Z = 1|Y ,X ) =
```

**User action**: Click "🤖 Auto-clean"

**LLM result**:
```
✨ LLM Cleaned Version:
Pr(Z_t = 1|Y_{t-1}, X_t) = exp(α + β₁Y_{t-1} + X_tβ₂) / [1 + exp(α + β₁Y_{t-1} + X_tβ₂)]
```

**User**: Click "✅ Apply Fixes and Continue"

### Example 2: Empirical Paper with Tables

**Detected region**:
```
Region #2 — TABLE (confidence: 0.70)
📄 Current Extraction:
  Lagged Real GDP Growth 0.381∗∗∗ 0.261∗∗∗ 0.220∗∗
  (0.074) (0.076) (0.100)
  News Sentiment 5.225∗∗∗ 3.329∗∗ 3.089∗∗
```

**User action**: Click "🤖 Auto-clean"

**LLM result**:
```
✨ LLM Cleaned Version:
                                    (1)        (2)        (3)
Lagged Real GDP Growth          0.381***   0.261***   0.220**
                                (0.074)    (0.076)    (0.100)

News Sentiment                  5.225***   3.329**    3.089**
                                (1.245)    (1.402)    (1.420)
```

### Example 3: Using LaTeX Source

**User has LaTeX source file, equation is broken in PDF**:

**User action**:
1. Click "✏️ Paste text"
2. Copy from LaTeX source:
   ```latex
   \Pr(Z_t = 1|Y_{t-1}, X_t) = \frac{\exp(\alpha + \beta_1 Y_{t-1} + X_t \beta_2)}{1 + \exp(\alpha + \beta_1 Y_{t-1} + X_t \beta_2)}
   ```
3. Click "💾 Save"

**Result**: LaTeX preserved exactly as intended

### Example 4: Scanned Paper with Image Equations

**PDF has equation rendered as image**:

**User action**:
1. Take screenshot of equation
2. Click "🖼️ Upload image"
3. Upload screenshot
4. Click "🔍 Extract Text (OCR)"
5. OCR extracts: `Pr(Z_t = 1|Y_t-1, X_t) = ...`
6. User can edit if needed
7. Click "💾 Save"

## Performance

### Detection Phase
- **Time**: ~0.5s per document
- **Cost**: Free (no API calls)

### Per-Region Fixes

| Action | Time | API Calls | Cost |
|--------|------|-----------|------|
| Auto-clean | ~5s | 1 | ~$0.01 |
| Paste text | Instant | 0 | $0 |
| Upload image | ~2-5s | 0 | $0 |
| Skip | Instant | 0 | $0 |

### Typical Usage

**Paper with 5 problematic regions**:
- User auto-cleans 2 equations: 2 API calls (~$0.02)
- User pastes 1 equation from LaTeX: 0 API calls
- User skips 2 minor issues: 0 API calls
- **Total**: 2 API calls, ~$0.02, ~10s user time

**Comparison to full auto-cleanup**:
- Old approach: 5 API calls, $0.05, no user control
- New approach: 2 API calls, $0.02, user control over critical content

## Benefits

### 1. **User Control**
- Users decide what needs fixing
- Critical equations get special attention
- Minor issues can be skipped

### 2. **Cost Efficiency**
- Only clean what matters
- Skip non-critical regions
- Use paste instead of LLM for known content

### 3. **Quality**
- Users can paste LaTeX source (100% accurate)
- LLM cleanup for unknown content
- OCR for scanned papers

### 4. **Transparency**
- Users see what's broken
- Preview LLM results before applying
- Understand extraction quality

## Troubleshooting

### "No extraction issues detected" but table looks bad

Lower the detection threshold:
```python
min_confidence=0.3  # Was 0.5
```

### OCR not working

Check tesseract installation:
```bash
tesseract --version
```

If not installed:
```bash
# Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### LLM cleanup takes too long

This is normal - each cleanup is one API call (~5s). Users can skip regions to speed up.

### Want to skip region fixer entirely

Click "⏭️ Skip All" button - proceeds immediately with original extraction.

## Future Enhancements

Potential improvements:

1. **Batch LLM cleanup**: Clean multiple regions in one API call
2. **Smart defaults**: Auto-clean tables, manual-input equations
3. **LaTeX rendering**: Show preview of rendered equations
4. **Caching**: Remember fixes for same document
5. **Export**: Save fixes for reuse on similar papers
6. **Templates**: Pre-defined fixes for common patterns

## Integration with Other Workflows

### Referee System

To add to referee workflow (`referee/workflow.py`):

```python
from section_eval.region_fixer import render_region_fixer
from utils import single_query

# After file upload and extraction
paper_text = decode_file(filename, file_bytes)

# Add region fixer
paper_text = render_region_fixer(
    text=paper_text,
    manuscript_name=filename,
    cache_prefix="referee",
    llm_query_fn=single_query,
    min_confidence=0.5
)
```

### Batch Processing

For processing multiple papers, can bypass UI:

```python
from section_eval.math_cleanup import detect_math_regions, cleanup_math_regions_with_llm
from utils import single_query

# Auto-clean all regions
text_cleaned = cleanup_math_regions_with_llm(text, single_query, max_regions=10)
```

## Summary

The Interactive Region Fixer gives users **full control** over extraction quality:

✅ **Detect** problematic regions automatically
✅ **Preview** what's broken
✅ **Choose** how to fix each region
✅ **Apply** fixes selectively
✅ **Continue** with clean text

This is especially valuable for papers where **accuracy matters** - theoretical papers with complex equations, empirical papers with detailed tables, and scanned papers with image content.
