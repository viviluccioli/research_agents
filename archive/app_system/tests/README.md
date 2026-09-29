# Testing Utilities

This directory contains test scripts for the research agents system.

## PDF Extraction Tests

### `test_pdf_extraction.py`

Analyzes PDF extraction quality to see what the LLM receives.

**Usage**:
```bash
# Basic extraction test
python test_pdf_extraction.py ../papers/sample_paper.pdf

# With verbose table output
python test_pdf_extraction.py ../papers/sample_paper.pdf -v

# Check for math/table formatting issues
python test_pdf_extraction.py ../papers/sample_paper.pdf --check-math
```

**Output**:
- Extraction statistics (pages, characters, tables)
- Saved text file: `<pdf_name>_extracted.txt`
- Quality warnings
- Math/table issue detection (with `--check-math`)

### `test_math_cleanup.py`

Tests LLM-powered cleanup of poorly-extracted equations and tables.

**Usage**:
```bash
# Detection only (no API calls)
python test_math_cleanup.py ../papers/sample_paper.pdf

# Full cleanup with LLM (makes API calls)
python test_math_cleanup.py ../papers/sample_paper.pdf --cleanup
```

**Output with `--cleanup`**:
- Detected regions with confidence scores
- Before/after comparison for first region
- Saved files:
  - `<pdf_name>_raw.txt` - Original extraction
  - `<pdf_name>_cleaned.txt` - After LLM cleanup

## Other Tests

### `test_prompt_loader.py`

Tests the prompt loading system for multi-agent debate and section evaluator.

**Usage**:
```bash
python test_prompt_loader.py
```

### `test_section_evaluator_prompts.py`

Tests section evaluator prompt generation.

**Usage**:
```bash
python test_section_evaluator_prompts.py
```

### `test_consensus_calculation.py`

Tests the weighted consensus calculation for referee reports.

**Usage**:
```bash
python test_consensus_calculation.py
```

## Running Tests

**From the tests directory**:
```bash
cd app_system/tests
python test_pdf_extraction.py ../../papers/sample.pdf
```

**From app_system (recommended)**:
```bash
cd app_system
python tests/test_pdf_extraction.py ../papers/sample.pdf
```

## Requirements

All tests require the packages in `requirements.txt`:
```bash
cd app_system
source ../venv/bin/activate
pip install -r requirements.txt
```

## Test Data

Place test PDFs in `papers/` directory at repo root:
```
research_agents/
├── papers/
│   ├── sample_paper.pdf
│   └── test_paper.pdf
├── app_system/
│   └── tests/
│       └── test_*.py
```

## Math Cleanup Workflow

1. **Check if cleanup is needed**:
   ```bash
   python tests/test_pdf_extraction.py paper.pdf --check-math
   ```

2. **If issues detected, run cleanup**:
   ```bash
   python tests/test_math_cleanup.py paper.pdf --cleanup
   ```

3. **Compare outputs**:
   ```bash
   diff paper_raw.txt paper_cleaned.txt
   ```

4. **Enable in app**:
   ```python
   from section_eval.text_extraction import decode_file
   from utils import single_query

   text = decode_file(filename, bytes, cleanup_math=True, llm_query_fn=single_query)
   ```

## Troubleshooting

**Import errors**:
- Make sure you're running from `app_system/` directory
- Activate venv: `source ../venv/bin/activate`

**API errors in `test_math_cleanup.py --cleanup`**:
- Check `.env` file has valid API credentials
- Ensure you're in `app_system/` directory so `config.py` can find `.env`

**File not found**:
- Use relative paths from your current directory
- Check PDF exists: `ls -l ../papers/sample.pdf`
