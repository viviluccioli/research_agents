# PDF Processing and Cost Analysis for Multi-Agent Debate System

**Date:** 2026-04-01

## PDF Processing Assessment

### Current Implementation

The referee system uses **extremely basic PDF text extraction** via `pdfplumber`:

```python
# From referee/workflow.py:208-224
def extract_text_from_pdf(self, file_content):
    """Extract text from a PDF file."""
    with pdfplumber.open(temp_file.name) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text
```

### What This Means

**✅ What IS extracted:**
- Plain text from the PDF
- Basic paragraph structure (where pdfplumber can detect it)

**❌ What IS NOT extracted:**
- **Tables**: Not extracted at all
- **Charts/Figures**: Not extracted at all
- **Equations**: May be garbled or missing
- **Multi-column layouts**: Often mangled
- **Footnotes**: May be out of order
- **References**: May be scrambled

### Reliability Assessment

**Grade: D+ (Minimal but functional)**

The extraction is **not reliable** for papers with:
- Complex tables (empirical papers)
- Mathematical equations (theoretical papers)
- Multi-column formatting (most published papers)
- Figures with captions

The LLM receives **raw text only** — it cannot "see" the PDF visually. It only gets what `pdfplumber.extract_text()` returns, which is often incomplete.

### Impact on Evaluation Quality

**High Risk Areas:**
1. **Empirical papers with regression tables** — Tables are missing, so the LLM cannot verify coefficient values, standard errors, or statistical significance
2. **Theoretical papers with proofs** — Equations may be garbled or missing symbols
3. **Papers citing specific figures** — The LLM cannot see the figures at all

**Why this is problematic:**
- The Empiricist persona cannot properly evaluate results if tables are missing
- The Theorist persona cannot verify proofs if equations are broken
- All personas are working from incomplete information

---

## LLM Call Count and Cost Estimation

### Total LLM Calls Per Run

**Main Debate Pipeline: 14 calls**
1. Round 0 (persona selection): **1 call**
2. Round 1 (independent evaluation): **3 calls** (parallel, one per selected persona)
3. Round 2A (cross-examination): **3 calls** (parallel)
4. Round 2B (direct examination): **3 calls** (parallel)
5. Round 2C (final amendments): **3 calls** (parallel)
6. Round 3 (editor synthesis): **1 call**

**UI Summarization: 13 calls**
7. Round 1 summaries: **3 calls** (parallel, ~512 tokens each)
8. Round 2A summaries: **3 calls** (parallel, ~256 tokens each)
9. Round 2B summaries: **3 calls** (parallel, ~256 tokens each)
10. Round 2C summaries: **3 calls** (parallel, ~384 tokens each)
11. Editor summary: **1 call** (~768 tokens)

**Grand Total: 27 LLM calls per run**

### Token Breakdown (Estimated)

Assumptions:
- Average research paper: ~30,000 tokens (20-30 pages)
- Each persona receives the full paper text in each round
- Round 1-2C prompts: ~500-2000 tokens of additional context

**Input Tokens (per call):**
- Round 0: ~30,000 (paper only)
- Round 1: ~30,500 per persona × 3 = ~91,500
- Round 2A: ~32,000 per persona × 3 = ~96,000 (includes Round 1 context)
- Round 2B: ~33,000 per persona × 3 = ~99,000 (includes R2A transcript)
- Round 2C: ~35,000 per persona × 3 = ~105,000 (includes full debate transcript)
- Round 3: ~40,000 (includes all final reports)
- Summarization: ~3,000-8,000 per call × 13 = ~65,000

**Total Input Tokens: ~526,500 tokens per run**

**Output Tokens (estimated):**
- Round 1: ~2,000 per persona × 3 = ~6,000
- Round 2A: ~1,500 per persona × 3 = ~4,500
- Round 2B: ~1,500 per persona × 3 = ~4,500
- Round 2C: ~2,000 per persona × 3 = ~6,000
- Round 3: ~3,000
- Summarization: ~300-500 per call × 13 = ~5,200

**Total Output Tokens: ~29,200 tokens per run**

### Cost Estimation

Using **Claude 3.7 Sonnet** (from config: `MODEL_SECONDARY`):

Anthropic pricing (as of 2025):
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens

**Per Run Cost:**
- Input: 526,500 tokens × $3.00/1M = **$1.58**
- Output: 29,200 tokens × $15.00/1M = **$0.44**
- **Total: ~$2.02 per paper**

If using Claude Sonnet 4.5 (more expensive):
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens
- **Total: ~$2.02 per paper** (same pricing tier)

**Cost scaling:**
- 10 papers: ~$20
- 100 papers: ~$200
- 1,000 papers: ~$2,000

---

## Testing PDF Processing Quality

### Option 1: Quick Test Script

Create `app_system/tests/test_pdf_extraction.py`:

```python
import pdfplumber
from pathlib import Path

def test_pdf_extraction(pdf_path: str):
    """Test what pdfplumber actually extracts from a PDF."""
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        # Extract first page
        page_1_text = pdf.pages[0].extract_text()
        print("\n=== PAGE 1 TEXT ===")
        print(page_1_text[:1000])  # First 1000 chars

        # Check for tables
        tables = pdf.pages[0].extract_tables()
        print(f"\n=== TABLES ON PAGE 1: {len(tables)} ===")
        if tables:
            print(tables[0])  # Print first table

        # Extract all text
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() or ""

        print(f"\n=== TOTAL TEXT LENGTH: {len(full_text)} chars ===")

        # Save to file for inspection
        output_path = Path(pdf_path).stem + "_extracted.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)

        print(f"✓ Saved extracted text to: {output_path}")

if __name__ == "__main__":
    # Test on a sample paper
    test_pdf_extraction("../papers/your_sample_paper.pdf")
```

Run: `python app_system/tests/test_pdf_extraction.py`

### Option 2: Compare with Original

1. Upload a paper you know well
2. Run the referee system
3. Check the Excel export → look at what the personas flagged
4. Manually check the PDF to see if critical tables/figures were missed

### Option 3: Add Extraction Diagnostics

Modify `referee/workflow.py` to add diagnostics:

```python
def extract_text_from_pdf(self, file_content):
    """Extract text from a PDF file."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file.flush()

        text = ""
        table_count = 0
        try:
            with pdfplumber.open(temp_file.name) as pdf:
                for page in pdf.pages:
                    # Extract text
                    text += page.extract_text() or ""

                    # Count tables (diagnostic)
                    tables = page.extract_tables()
                    table_count += len(tables)

                    # Optionally extract tables as text
                    # for table in tables:
                    #     text += "\n[TABLE]\n"
                    #     for row in table:
                    #         text += " | ".join(str(cell) for cell in row) + "\n"
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")

        # Clean up
        os.unlink(temp_file.name)

        # Show diagnostics
        st.info(f"Extracted {len(text)} characters, detected {table_count} tables")

        return text
```

---

## Recommendations

### Immediate Improvements

1. **Extract tables as text** — Use `pdfplumber.extract_tables()` and format as markdown/CSV
2. **Add extraction diagnostics** — Show users what was extracted (char count, table count)
3. **Add cost tracking to metadata** — Calculate and display estimated cost per run

### Medium-term Improvements

1. **Use vision-capable LLM** — Claude 3.5+ can directly "see" PDF pages via image input
2. **Extract figures** — Convert PDF pages to images and pass to vision model
3. **Better equation handling** — Use a LaTeX-aware extractor

### Long-term Improvements

1. **Require LaTeX source** — Get `.tex` files instead of PDFs (section_eval already supports this!)
2. **Multi-modal processing** — Process text + tables + figures separately
3. **Validate extraction quality** — LLM checks if extraction looks reasonable before proceeding

---

## Adding Cost Metadata

### Implementation

Modify `referee/engine.py:execute_debate_pipeline()` to track tokens and costs:

```python
# Add to metadata section (line ~536)
results['metadata'] = {
    # ... existing fields ...
    'token_usage': {
        'input_tokens_estimated': 526500,  # Calculate from actual paper length
        'output_tokens_estimated': 29200,
        'total_llm_calls': 27,
        'debate_calls': 14,
        'summarization_calls': 13
    },
    'cost_estimate_usd': {
        'input_cost': round(526500 * 3.00 / 1_000_000, 2),
        'output_cost': round(29200 * 15.00 / 1_000_000, 2),
        'total_cost': round((526500 * 3.00 + 29200 * 15.00) / 1_000_000, 2)
    }
}
```

Then display in the UI (add to `referee/workflow.py:display_debate_results()`):

```python
# Add after metadata display
cost_info = results['metadata'].get('cost_estimate_usd', {})
st.info(f"💰 **Estimated Cost:** ${cost_info.get('total_cost', 'N/A')} "
        f"({results['metadata']['token_usage']['total_llm_calls']} LLM calls)")
```

---

## Summary

| Metric | Value |
|--------|-------|
| **PDF extraction quality** | Poor (text only, no tables/figures) |
| **Reliability for empirical papers** | Low (missing regression tables) |
| **Reliability for theoretical papers** | Medium (equations often garbled) |
| **Total LLM calls per run** | 27 (14 debate + 13 summarization) |
| **Estimated cost per paper** | ~$2.02 USD |
| **Can access cost metadata?** | Not currently, but easy to add |
| **Biggest risk** | LLM evaluating based on incomplete information |

**Bottom line:** The PDF processing is functional but unreliable for complex papers. The LLM is doing its best with incomplete data. For production use, consider requiring LaTeX source files or implementing vision-based PDF processing.
