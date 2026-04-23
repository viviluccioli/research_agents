# PyMuPDF Quick Start Guide

## Installation (1 minute)

```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents

# Install PyMuPDF
pip install PyMuPDF>=1.23.0

# Optional: Install OCR for table extraction
pip install pytesseract Pillow
sudo apt-get install tesseract-ocr
```

## Configuration (30 seconds)

Edit `app_system/.env`:

```bash
USE_PYMUPDF=true
```

That's it! The system will automatically use PyMuPDF for PDF extraction.

## Test It (1 minute)

```bash
cd app_system
python tests/test_pymupdf_extractor.py ../papers/your_paper.pdf
```

## Use It

Run the app normally:

```bash
cd app_system
bash run_app.sh
```

Upload a PDF in the Referee Report tab. You'll see:

```
✅ PDF Extraction Complete (🚀 PyMuPDF)
Pages: 30  |  Figures: 8  |  Tables: 5  |  Characters: 125,430

📊 Extracted 8 Figure(s) [click to expand]
```

## What You Get

### Before (pdfplumber)
- ✅ Text extraction
- ✅ Basic table extraction
- ❌ No figures
- ❌ No captions
- ❌ No layout detection

### After (PyMuPDF)
- ✅ Text extraction (improved)
- ✅ Advanced table extraction (OCR)
- ✅ **Figure extraction** (embedded + vector graphics)
- ✅ **Caption parsing** (titles, multi-panel detection)
- ✅ **Multi-column detection**
- ✅ **Stored for future vision analysis**

## Access Figures in Code

```python
# Figures are stored in session state
figures = st.session_state['debate_state']['figures']

for fig in figures:
    print(f"{fig.figure_id}: {fig.caption}")
    # fig.image_data contains PNG/JPEG bytes
    # fig.figure_number is "1", "3a", etc.
    # fig.is_multi_panel is True if panels detected
    # fig.references_in_text lists mentions in text
```

## Troubleshooting

### "Using pdfplumber (fallback mode)"

**Problem**: PyMuPDF not installed or disabled

**Solution**:
```bash
pip install PyMuPDF>=1.23.0
# Check .env has USE_PYMUPDF=true
```

### "[OCR not available]"

**Problem**: pytesseract not installed

**Solution**:
```bash
pip install pytesseract Pillow
sudo apt-get install tesseract-ocr
```

### Extraction slow?

**Normal**: Figure extraction adds 2-3s per page with figures. This is expected.

**To disable**: Set `PYMUPDF_EXTRACT_TABLES=false` in `.env`

## Configuration Options

```bash
# Enable/disable PyMuPDF (default: true)
USE_PYMUPDF=true

# Minimum figure size to extract (default: 100px)
# Lower = more figures, but may include icons/logos
PYMUPDF_MIN_FIGURE_SIZE=100

# Resolution scale for rendering (default: 2.0 = ~150 DPI)
# Higher = better quality but slower
PYMUPDF_RESOLUTION_SCALE=2.0

# Enable table OCR (default: true)
PYMUPDF_EXTRACT_TABLES=true
```

## Documentation

- **Full docs**: `app_system/docs/pymupdf_extraction.md`
- **Changelog**: `app_system/docs/changelog.md`
- **Summary**: `PYMUPDF_UPGRADE_SUMMARY.md`

## What's Next?

Figures are now stored and ready for **vision-enabled persona analysis**. Future update will allow personas to:
- View and analyze figures
- Provide figure-specific critiques
- Request specific figures for detailed review

For now, figures are extracted and stored - ready when vision integration is implemented!
