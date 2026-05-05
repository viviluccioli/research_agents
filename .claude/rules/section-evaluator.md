# Section Evaluator Context

This rule activates when working on the section evaluation system.

## File Scope
- `app_system/section_eval/`
- `app_system/prompts/section_evaluator/`
- `app_system/app.py` (Tab 2)

## Key Architecture

**5-Stage Pipeline**:
1. **Text Extraction** (`text_extraction.py`): PDF/LaTeX/plain text → raw text
2. **Section Detection** (`section_detection.py`): Two-pass (heuristic + LLM) → section list
3. **Hierarchy Grouping** (`hierarchy.py`): Group subsections under parents
4. **Evaluation** (`evaluator.py`): Score each section against criteria
5. **Scoring** (`scoring.py`): Aggregate + fatal-flaw logic → overall score

**Data Flow**: `bytes → text → sections → hierarchy → evaluations → scores → UI`

## Key Files

- **`main.py`**: `SectionEvaluatorApp` — Streamlit UI entry point
- **`evaluator.py`**: `SectionEvaluator` — core evaluation logic
- **`scoring.py`**: Score computation + fatal-flaw rules
- **`criteria/base.py`**: Paper types + evaluation criteria registry
- **`section_detection.py`**: Heuristic + LLM two-pass detection
- **`text_extraction.py`**: `decode_file()` — PDF/LaTeX parsing
- **`math_cleanup.py`**: LaTeX normalization utilities
- **`region_fixer.py`**: Interactive math fixing UI

## Criteria System

**Structure** (`criteria/base.py`):
```python
PAPER_TYPES = ["empirical", "theoretical", "policy", "finance", "macro", "systematic_review"]
_UNIVERSAL = {...}  # All paper types
_EMPIRICAL = {...}  # Empirical-specific
_ALL_CRITERIA = {
    "universal": _UNIVERSAL,
    "empirical": {**_UNIVERSAL, **_EMPIRICAL},
    ...
}
```

**Adding a New Paper Type**:
1. Add to `PAPER_TYPES` and `PAPER_TYPE_LABELS`
2. Define `_NEWTYPE` criteria dict
3. Merge into `_ALL_CRITERIA`
4. Add section importance weights in `scoring.py:SECTION_IMPORTANCE`
5. Create prompt file: `prompts/section_evaluator/paper_type_contexts/newtype/v1.0.txt`
6. Update `prompts/section_evaluator/config.yaml`

**Adding New Section Alias**:
- Add to `_SECTION_ALIASES` (exact match) or `_KEYWORD_MAP` (fuzzy match)
- Example: `"empirical strategy": "methodology"` → canonical mapping

## Fatal-Flaw Logic

**Rule**: Any criterion with `critical=True` scoring ≤ 1.5 caps section score at 2.5.

**Configuration**:
```python
FATAL_FLAW_SCORE_THRESHOLD = 1.5
FATAL_FLAW_SCORE_CAP = 2.5
```

**Critical Criteria Examples**:
- `identification` (empirical)
- `correctness` (theoretical)
- `instrument_validity` (empirical)

## Prompt Organization

**Subdirectory structure**:
```
prompts/section_evaluator/
├── paper_type_contexts/{type}/v1.0.txt  # Subdirectories
├── section_type_guidance/abstract_v1.0.txt  # Flat files (exception)
└── master_prompts/evaluation_v1.0.txt  # Flat files (exception)
```

**When to use flat vs subdirectories**:
- **Subdirectories**: Few items (5-10), independently versioned (paper types)
- **Flat files**: Many items (20+), tightly coupled (section guidance)

## Caching

**Cache Keys**:
- Prefix: `"se_cache_v3"` (evaluator) or `"se_v3"` (app)
- Key: `sha256(paper_type | section_name | text[:50000] | figures_external)`
- **IMPORTANT**: Bump prefix when changing result schema

**Cache Invalidation**: Change prefix to force re-evaluation after schema changes.

## Model Configuration

**LLM Calls**:
- `safe_query()`: Direct API call, temperature=0.3, NO thinking mode
- `ConversationManager.conv_query()`: Stateful, auto-prunes at 8000 tokens
- **CRITICAL**: Uses `MODEL_PRIMARY` (Claude 4.5 Sonnet)

**Difference from Referee**:
- Section eval: `safe_query()` (stateless, temp 0.3, no thinking)
- Referee: `single_query()` (stateless, temp 1.0, thinking enabled)

## Section Detection

**Two-Pass Algorithm**:

**Pass 1 (Heuristic)**: Score lines by:
- Casing (UPPERCASE > Title Case > lowercase)
- Numbering patterns (1., 1.1, I., A.)
- Word count (2-8 words preferred)
- Surrounding whitespace

**Pass 2 (LLM)**: Confirm candidates with LLM to reduce false positives.

**Filters**: Exclude math symbols, figure labels, citation fragments.

## Text Extraction

**PDF** (`pdfplumber`):
- Multi-column handling via bounding boxes
- Extract figures/tables separately
- Math may need manual cleanup (`region_fixer.py`)

**LaTeX** (`strip_latex()`):
- Regex-based extraction
- Preserve math delimiters for later cleanup
- Remove preamble/bibliography

**Plain Text**: Raw passthrough (UTF-8)

## Testing

```bash
cd app_system
python -m pytest tests/test_section_evaluator_prompts.py
python -m pytest tests/test_math_cleanup.py
python -m pytest tests/test_pdf_extraction.py
```

## Common Pitfalls

❌ **Don't** use `single_query()` — use `safe_query()` or `ConversationManager`
❌ **Don't** forget to bump cache prefix after schema changes
❌ **Don't** add paper types without section importance weights
❌ **Don't** use thinking mode in section eval (breaks at temp 0.3)
✅ **Do** validate quotes in evaluation results
✅ **Do** test with both auto-detect and manual section input modes
✅ **Do** check math cleanup for LaTeX papers
✅ **Do** verify fatal-flaw logic with edge cases

## Output Formats

**Markdown**: Human-readable report
**CSV**: Benchmarking format (section × criteria matrix)
**PDF**: Final report via `fpdf` (latin-1 encoding, replaces unicode)
