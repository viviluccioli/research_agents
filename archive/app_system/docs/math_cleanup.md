# LLM-Powered Math and Table Cleanup

## Overview

The math cleanup system detects and repairs poorly-extracted LaTeX equations and tables from PDF files. When PDFs are converted to text, mathematical notation often becomes garbled:

**Before Cleanup (typical PDF extraction)**:
```
exp(?+? Y +X ? )
1 t?1 t 2
Pr(Z = 1|Y ,X ) = (3)
t t?1 t
1+exp(?+? Y +X ? )
1 t?1 t 2
```

**After Cleanup (LLM-reconstructed)**:
```
Pr(Z_t = 1|Y_{t-1}, X_t) = exp(α + β₁Y_{t-1} + X_t β₂) / [1 + exp(α + β₁Y_{t-1} + X_t β₂)]  (3)
```

Similarly, regression tables with broken alignment are reconstructed into readable formats.

## Architecture

The system has three stages:

### 1. Detection (`math_cleanup.py:detect_math_regions()`)

Scans extracted text for regions that contain mathematical notation or tabular data with degraded formatting. Uses heuristics to identify:

- **Equations**: Greek letters, subscripts/superscripts, math operators, broken notation
- **Tables**: Columns of numbers, parenthesized standard errors, significance stars

Returns a list of `MathRegion` objects with:
- `start_idx`, `end_idx`: Position in text
- `content`: The problematic text
- `region_type`: `'equation'` or `'table'`
- `confidence`: 0-1 score (higher = more certain it's broken)

**Detection patterns**:
```python
# Broken subscripts: "Y t-1" instead of "Y_{t-1}"
r'(?:^|\s)[a-z]\s+[0-9](?:\s|$)'

# Orphaned operators on separate lines
r'(?:^|\n)\s*[+\-*/=]\s*(?:\n|$)'

# Greek letters (often indicates equations)
r'[α-ωΑ-Ω]'

# Standard errors (parenthesized numbers in tables)
r'\(\d+\.\d+\)'
```

### 2. Prompt Construction (`math_cleanup.py:build_cleanup_prompt()`)

Generates a specialized prompt for each region type:

- **Equation prompts**: Ask LLM to restore subscripts, fix fractions, preserve symbols
- **Table prompts**: Ask LLM to reconstruct column structure, align values

### 3. LLM Cleanup (`math_cleanup.py:cleanup_math_regions_with_llm()`)

Processes detected regions (highest confidence first, up to `max_regions`):
1. Builds cleanup prompt for region
2. Calls LLM via `llm_query_fn(prompt)`
3. Replaces original text with cleaned response
4. Processes regions in reverse order (by position) to preserve indices

## Usage

### Option 1: Enable in `decode_file()` (Recommended for App)

```python
from section_eval.text_extraction import decode_file
from utils import single_query

# Standard extraction (no cleanup)
text = decode_file('paper.pdf', file_bytes)

# With LLM cleanup (repairs equations and tables)
text = decode_file('paper.pdf', file_bytes,
                  cleanup_math=True,
                  llm_query_fn=single_query)
```

**Cost considerations**:
- Cleanup only runs if problematic regions are detected
- Default limit: 10 regions per document (configurable via `max_regions`)
- Each region = 1 API call
- Uses `single_query()` which calls the primary model (Claude Sonnet 3.7)

### Option 2: Standalone Cleanup

```python
from section_eval.text_extraction import extract_text_from_pdf
from section_eval.math_cleanup import cleanup_math_regions_with_llm
from utils import single_query

# Extract first
with open('paper.pdf', 'rb') as f:
    text = extract_text_from_pdf(f.read())

# Then clean up
text_cleaned = cleanup_math_regions_with_llm(text, single_query, max_regions=5)
```

### Option 3: Detection Only (No API Calls)

```python
from section_eval.math_cleanup import detect_math_regions, add_quality_warnings

regions = detect_math_regions(text)
print(f"Found {len(regions)} problematic regions")

for region in regions:
    print(f"{region.region_type}: confidence {region.confidence:.2f}")
    print(f"Content: {region.content[:100]}...")

# Add warnings to text
text_with_warnings, warnings = add_quality_warnings(text)
```

## Testing

### Test 1: Detect problematic regions (no LLM calls)

```bash
cd app_system
python tests/test_pdf_extraction.py ../papers/sample_paper.pdf --check-math
```

**Output**:
```
🔍 MATH/TABLE QUALITY CHECK
================================================================================

⚠️  Found 3 region(s) with potential formatting issues:
  #1: equation (confidence: 0.80)
  #2: table (confidence: 0.65)
  #3: equation (confidence: 0.52)

💡 These regions may have:
   - Broken LaTeX equations (subscripts on wrong lines)
   - Poorly structured tables (unclear column alignment)
   - Fragmented mathematical notation
```

### Test 2: Full cleanup with LLM

```bash
cd app_system
source ../venv/bin/activate
python tests/test_math_cleanup.py ../papers/sample_paper.pdf --cleanup
```

**Output**:
```
📋 BEFORE (Region #1 - equation):
────────────────────────────────────────────────────────────────────────────
exp(?+? Y +X ? )
1 t?1 t 2
Pr(Z = 1|Y ,X ) = (3)
t t?1 t

✨ AFTER (cleaned by LLM):
────────────────────────────────────────────────────────────────────────────
Pr(Z_t = 1|Y_{t-1}, X_t) = exp(α + β₁Y_{t-1} + X_tβ₂) / [1 + exp(α + β₁Y_{t-1} + X_tβ₂)]  (3)
```

### Test 3: Compare raw vs cleaned extraction

```bash
cd app_system
python tests/test_math_cleanup.py ../papers/sample_paper.pdf --cleanup

# Outputs:
#   sample_paper_raw.txt      - Original extraction
#   sample_paper_cleaned.txt  - After LLM cleanup
```

## Integration with Main App

To enable cleanup in the Section Evaluator or Referee system:

**Edit `app_system/section_eval/main.py`**:

```python
def _load_paper_text(self, files):
    """Load and extract paper text from uploaded file."""
    for filename, file_bytes in files.items():
        # Option 1: Enable for all PDFs
        text = decode_file(filename, file_bytes,
                          warn_fn=st.warning,
                          cleanup_math=True,
                          llm_query_fn=single_query)

        # Option 2: Enable via user toggle
        cleanup_enabled = st.checkbox("Clean up equations/tables (uses LLM, slower)",
                                     value=False,
                                     help="Repairs poorly-extracted math notation")

        text = decode_file(filename, file_bytes,
                          warn_fn=st.warning,
                          cleanup_math=cleanup_enabled,
                          llm_query_fn=single_query if cleanup_enabled else None)
```

**Edit `app_system/referee/workflow.py`**:

```python
# In RefereeWorkflow.render_ui()
for filename, file_bytes in files.items():
    text = decode_file(filename, file_bytes,
                      cleanup_math=True,  # Enable cleanup
                      llm_query_fn=single_query)
```

## Configuration

### Adjusting Detection Sensitivity

Edit `math_cleanup.py` constants:

```python
# In detect_math_regions()
if brokenness > 0.3:  # Lower = more aggressive (detects more equations)
    # Process as equation

if tableness > 0.4:   # Lower = more aggressive (detects more tables)
    # Process as table
```

### Limiting API Calls

```python
# Process only top 5 most-broken regions
text = cleanup_math_regions_with_llm(text, llm_fn, max_regions=5)

# Process all detected regions (expensive!)
text = cleanup_math_regions_with_llm(text, llm_fn, max_regions=999)
```

### Custom LLM Function

```python
def custom_llm(prompt: str) -> str:
    """Your custom LLM wrapper."""
    # Use faster/cheaper model
    response = call_haiku_model(prompt)
    return response.text

text = cleanup_math_regions_with_llm(text, custom_llm)
```

## Troubleshooting

### "Could not import utils.single_query"

Run tests from `app_system/` directory:
```bash
cd app_system
python tests/test_math_cleanup.py paper.pdf
```

### Cleanup takes too long

Reduce `max_regions` or increase confidence threshold:
```python
regions = [r for r in detect_math_regions(text) if r.confidence > 0.5]
```

### False positives (detecting non-math as math)

Increase thresholds in `detect_math_regions()`:
```python
if brokenness > 0.5:  # Was 0.3 (more conservative)
```

### Equations still broken after cleanup

The LLM may need more context. Increase `context_lines` in detection:
```python
regions = detect_math_regions(text, context_lines=5)  # Was 2
```

## Performance

Typical extraction times (measured on 30-page empirical paper):

| Mode | Time | API Calls | Cost (approx) |
|------|------|-----------|---------------|
| Standard extraction | 2s | 0 | $0 |
| + Detection only | 2.5s | 0 | $0 |
| + Cleanup (3 regions) | 15s | 3 | $0.03 |
| + Cleanup (10 regions) | 45s | 10 | $0.10 |

**Recommendation**: Enable cleanup only for papers with heavy math content (detected via `--check-math` flag first).

## Changelog

**2026-04-01**: Initial implementation
- Added `math_cleanup.py` module
- Integrated into `text_extraction.py`
- Created test scripts: `test_math_cleanup.py`
- Updated `test_pdf_extraction.py` with `--check-math` flag

## Future Enhancements

Potential improvements:

1. **Table extraction from images**: Use `pdf2image` + OCR for tables rendered as images
2. **Equation image detection**: Detect equation images, OCR them, clean with LLM
3. **Caching**: Cache cleaned regions by content hash to avoid redundant API calls
4. **Batch processing**: Process multiple regions in single LLM call to reduce latency
5. **Custom prompts per field**: Economics-specific vs physics-specific equation cleanup
