# PyMuPDF-based PDF Extraction

## Overview

The research agents system now supports advanced PDF extraction using **PyMuPDF (fitz)** with automatic fallback to **pdfplumber**. This upgrade enables:

- ✅ **Figure extraction** (embedded images + rendered vector graphics)
- ✅ **Caption parsing** with multi-panel detection
- ✅ **Table extraction** via OCR (pytesseract)
- ✅ **Multi-column layout detection**
- ✅ **Better handling of LaTeX-generated PDFs**
- ✅ **Automatic fallback** to pdfplumber if PyMuPDF fails

## Architecture

### Extractor Module

`app_system/referee/_utils/pdf_extractor_v2.py`

- `extract_pdf_with_figures()` - Main extraction function
- `extract_text_and_figures()` - PyMuPDF implementation
- `extract_with_pdfplumber_fallback()` - Fallback implementation
- `ExtractedContent` - Container for text, figures, tables, metadata
- `Figure` - Dataclass for figure metadata and image data

### Integration

The new extractor is integrated into the referee workflow:

- `app_system/referee/workflow.py` - Updated `extract_text_from_pdf()` method
- Figures stored in `st.session_state['debate_state']['figures']` for future vision integration
- UI displays extraction diagnostics and figure previews

## Configuration

Settings are loaded from `.env` file or environment variables:

```bash
# Enable/disable PyMuPDF (falls back to pdfplumber if disabled or on error)
USE_PYMUPDF=true

# Minimum figure size (pixels) - filters out logos/icons
PYMUPDF_MIN_FIGURE_SIZE=100

# Resolution scale for rendering vector graphics
# 2.0 = ~150 DPI, 3.0 = ~225 DPI
PYMUPDF_RESOLUTION_SCALE=2.0

# Enable table extraction via OCR
PYMUPDF_EXTRACT_TABLES=true
```

Configuration is loaded in `app_system/config.py`:

```python
from config import (
    USE_PYMUPDF,
    PYMUPDF_MIN_FIGURE_SIZE,
    PYMUPDF_RESOLUTION_SCALE,
    PYMUPDF_EXTRACT_TABLES
)
```

## Dependencies

### Required

```bash
pip install PyMuPDF>=1.23.0
```

### Optional (for OCR)

```bash
pip install pytesseract>=0.3.10 Pillow>=10.0.0

# On Ubuntu/Debian
sudo apt-get install tesseract-ocr

# On macOS
brew install tesseract
```

## Features

### 1. Figure Extraction

Extracts figures using two complementary methods:

#### Method A: Embedded Images

Extracts raster images (JPEG, PNG) embedded in the PDF:

```python
# Extracts embedded images
images = page.get_images(full=True)
for img in images:
    base_image = doc.extract_image(xref)
    # Filter by size (min 100px)
    # Find nearby caption
    # Detect if it's a table (aspect ratio heuristic)
```

#### Method B: Caption-Based Rendering

Finds figure captions and renders the region above as PNG:

```python
# Find captions matching "Figure X:" pattern
captions = find_figure_captions(page)

for caption in captions:
    # Estimate figure bbox (above caption)
    figure_bbox = estimate_figure_bbox_from_caption(caption_bbox, page.rect)

    # Render region as high-res PNG (~150 DPI)
    image_bytes = render_figure_region(page, figure_bbox, resolution_scale=2.0)
```

This handles vector graphics in LaTeX PDFs where figures are not embedded as images.

### 2. Caption Parsing

Extracts metadata from figure captions:

- **Figure number**: "1", "3a", "12B"
- **Caption text**: Full caption (multi-line supported)
- **Title**: First sentence of caption
- **Multi-panel detection**: Detects patterns like "(a)", "(b)", "panel a", "subfigure b"
- **Text references**: Finds mentions like "see Figure 1" or "Figure 2 shows"

Example:

```python
caption = "Figure 3a: Treatment effects over time. Panel (a) shows baseline, (b) shows follow-up."

# Extracted:
figure_number = "3a"
title = "Treatment effects over time"
is_multi_panel = True
panels = ["a", "b"]
```

### 3. Table Extraction

Two-stage table detection:

1. **Caption-based**: Look for "Table X" in nearby text
2. **Heuristic**: Unusual aspect ratios (very wide >2.5:1 or tall <0.4:1)

OCR extraction (if `pytesseract` available):

```python
# Convert to grayscale
img = img.convert('L')

# OCR with PSM 6 (uniform block of text)
text = pytesseract.image_to_string(img, config='--psm 6')

# Format as markdown table
table_markdown = format_as_markdown_table(text, confidence)

# Insert into paper text at "Table X" mention
```

Tables are inserted inline in the text with metadata:

```markdown
[TABLE 1 - EXTRACTED VIA OCR - CONFIDENCE: 85%]
Table 1: Summary statistics

| Variable | Mean | SD | N |
| --- | --- | --- | --- |
| Age | 45.2 | 12.3 | 500 |
| Income | 65000 | 15000 | 500 |
[END TABLE]
```

### 4. Multi-Column Layout Detection

Heuristic detection of multi-column layouts:

```python
# Get x-coordinates of text blocks
x_coords = [block_center_x for block in page.get_text("dict")["blocks"]]

# Check if blocks cluster in left and right regions
left_blocks = [x for x in x_coords if x < page_width * 0.4]
right_blocks = [x for x in x_coords if x > page_width * 0.6]

# Multi-column if significant blocks on both sides
is_multi_column = len(left_blocks) > 5 and len(right_blocks) > 5
```

Helps identify papers that may have text ordering issues.

### 5. Deduplication

Prevents duplicate figures from being extracted twice (once as embedded image, once via caption rendering):

- **By figure number + page**: Same figure number on same page
- **By bbox overlap**: Overlapping bounding boxes (>50% overlap)
- **Priority**: Prefers embedded images (higher resolution) over rendered regions

## Usage

### In Referee Workflow

The extractor is automatically used when processing PDFs:

```python
# In RefereeWorkflow.extract_text_from_pdf()
result = extract_pdf_with_figures(
    file_content=file_content,
    use_pymupdf=USE_PYMUPDF,
    min_figure_size=PYMUPDF_MIN_FIGURE_SIZE,
    resolution_scale=PYMUPDF_RESOLUTION_SCALE,
    extract_tables=PYMUPDF_EXTRACT_TABLES
)

# Store figures for future vision integration
st.session_state['debate_state']['figures'] = result.figures
```

### Direct Usage

```python
from referee._utils.pdf_extractor_v2 import extract_pdf_with_figures

with open('paper.pdf', 'rb') as f:
    pdf_bytes = f.read()

result = extract_pdf_with_figures(
    file_content=pdf_bytes,
    use_pymupdf=True,
    min_figure_size=100,
    resolution_scale=2.0,
    extract_tables=True
)

print(f"Extracted {len(result.figures)} figures")
print(f"Extracted {len(result.tables)} tables")
print(f"Text length: {len(result.text)} chars")

# Access figures
for fig in result.figures:
    print(f"{fig.figure_id}: {fig.caption}")
    # Save figure
    with open(f"{fig.figure_id}.{fig.image_format}", 'wb') as f:
        f.write(fig.image_data)
```

## Testing

Run the test script:

```bash
cd app_system
python tests/test_pymupdf_extractor.py ../papers/sample_paper.pdf
```

This will:
- Test PyMuPDF extraction
- Test pdfplumber fallback
- Display extraction diagnostics
- Show figures, tables, and metadata

## Future Vision Integration

Figures are now stored in session state for future vision-based analysis:

```python
# Stored in session_state after extraction
figures = st.session_state['debate_state']['figures']

# Future: Pass figures to vision-enabled personas
for fig in figures:
    if fig.figure_number == "3":
        # Send fig.image_data to vision model
        # Include fig.caption as context
        analysis = analyze_figure_with_vision(fig.image_data, fig.caption)
```

Personas can reference figures by ID:

```python
# In Round 2A/2B/2C, personas can request figure attachments
comment = {
    "issue": "Figure 3 appears to show outliers...",
    "figures_needed": ["figure_3", "figure_4a"]  # Will be attached for vision analysis
}
```

## Troubleshooting

### PyMuPDF not available

```
⚠️ Using pdfplumber (fallback mode). Figure extraction not available.
To enable figure extraction, ensure PyMuPDF is installed and USE_PYMUPDF=true in .env
```

**Solution**: Install PyMuPDF:

```bash
pip install PyMuPDF>=1.23.0
```

### OCR not available

```
[OCR not available - install pytesseract and Pillow]
```

**Solution**: Install OCR dependencies:

```bash
pip install pytesseract Pillow
# Ubuntu/Debian
sudo apt-get install tesseract-ocr
# macOS
brew install tesseract
```

### Low confidence OCR

```
⚠️ Low confidence OCR (35%)
```

Tables extracted with <50% confidence are flagged. This typically means:
- Table has complex formatting
- Image quality is poor
- Table contains non-text elements (charts, symbols)

**Solution**: Use LaTeX source instead of PDF for better table extraction.

### Extraction fails

If PyMuPDF extraction fails, the system automatically falls back to pdfplumber:

```
⚠️ PDF extraction error: [error message]
Falling back to basic pdfplumber extraction...
```

No action needed - fallback is automatic.

## Performance

Typical extraction times on a modern system:

| Paper Length | pdfplumber | PyMuPDF (no figures) | PyMuPDF (with figures) |
|--------------|------------|----------------------|------------------------|
| 10 pages     | 1-2s       | 1-2s                 | 3-5s                   |
| 30 pages     | 3-5s       | 3-5s                 | 10-15s                 |
| 50 pages     | 5-8s       | 5-8s                 | 20-30s                 |

Figure extraction adds overhead due to:
- Image rendering (vector → raster)
- Caption detection and parsing
- OCR for tables
- Deduplication and post-processing

## Comparison with pdfplumber

| Feature | pdfplumber | PyMuPDF |
|---------|-----------|---------|
| Text extraction | ✅ Good | ✅ Good |
| Table extraction | ✅ Structured | ✅ OCR-based |
| Figure extraction | ❌ No | ✅ Yes |
| Caption parsing | ❌ No | ✅ Yes |
| Vector graphics | ❌ No | ✅ Yes (rendered) |
| Multi-column detection | ❌ No | ✅ Yes |
| Speed | Fast | Moderate |
| Dependencies | Minimal | Requires PyMuPDF |

## Related Files

- `app_system/referee/_utils/pdf_extractor_v2.py` - Extractor implementation
- `app_system/referee/workflow.py` - Integration in referee workflow
- `app_system/config.py` - Configuration loading
- `app_system/tests/test_pymupdf_extractor.py` - Test script
- `requirements.txt` - Updated dependencies

## References

- PyMuPDF documentation: https://pymupdf.readthedocs.io/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- MarginalEdit implementation: `../marginaledit/shared/extractors/pdf_extractor.py`
