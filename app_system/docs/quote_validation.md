# Quote Validation for Referee Reports

## Overview

The quote validation system automatically verifies that quotes in referee persona reports actually exist in the source paper. This helps prevent hallucinations and ensures that referees are accurately citing the paper being reviewed.

## How It Works

### Validation Process

1. **Quote Extraction**: The system extracts quoted text from persona reports using multiple patterns:
   - Text in double quotes: `"quoted text"`
   - Text in single quotes: `'quoted text'`
   - Statement patterns: `"the paper states: quoted text"`
   - Markdown blockquotes: `> quoted text`

2. **Text Normalization**: Both quotes and paper text are normalized:
   - Whitespace collapsed to single spaces
   - Lowercased for comparison
   - Preserves original text for display

3. **Fuzzy Matching**: The system uses fuzzy string matching (via `thefuzz` library) to validate quotes:
   - Allows for minor OCR/formatting differences
   - Handles quotes that span multiple lines
   - Uses partial ratio matching for substring detection

4. **Threshold-Based Validation**:
   - **Mathematical content** (contains Greek letters, LaTeX, symbols): **95% similarity required**
   - **Regular prose**: **85% similarity required**

### Validation Points

Quote validation runs at two key points in the debate pipeline:

- **After Round 1**: Validates quotes in independent evaluations
- **After Round 2C**: Validates quotes in final amended reports

## Configuration

### Enabling/Disabling

Quote validation is **enabled by default**. To disable it:

```bash
# Set environment variable
export DISABLE_QUOTE_VALIDATION=true

# Or in .env file
DISABLE_QUOTE_VALIDATION=true
```

### Thresholds

Default thresholds are defined in `referee/_utils/quote_validator.py`:

```python
DEFAULT_PROSE_THRESHOLD = 85  # 85% similarity for prose
DEFAULT_MATH_THRESHOLD = 95   # 95% similarity for mathematical content
```

To customize, modify these constants or pass them as arguments to validation functions.

## Dependencies

### Required
- No hard dependencies - the system works with basic string matching

### Optional (Recommended)
- `thefuzz`: Provides fuzzy string matching for better validation
- `python-Levenshtein`: Accelerates fuzzy matching (optional but recommended)

```bash
pip install thefuzz python-Levenshtein
```

**Without thefuzz**: The system falls back to exact substring matching, which is less flexible but still effective for exact quotes.

## Output

### UI Display

In the Streamlit UI, quote validation results appear in the **Metadata** section with:

- **Round 1 Validation**: Total quotes, valid count, invalid count, validation rate
- **Round 2C Validation**: Same metrics for final reports
- **Expandable sections** showing unverified quotes with similarity scores

### Excel Export

The Excel export includes a **Quote Validation** sheet with columns:

- **Round**: Round 1 or Round 2C
- **Persona**: Name of the persona who generated the quote
- **Quote Text**: The quoted text (truncated to 200 chars)
- **Is Valid**: Yes/No based on threshold
- **Similarity Score**: Percentage match (0-100%)
- **Is Mathematical**: Whether mathematical content was detected
- **Threshold Used**: The threshold applied (85% or 95%)

### Metadata

Results are included in the `metadata.quote_validation` field:

```python
{
    'enabled': True,
    'round_1': {
        'total_quotes_found': 12,
        'valid_quotes': 10,
        'invalid_quotes': 2,
        'validation_rate': 83.3,
        'invalid_quote_details': [...]
    },
    'round_2c': { ... }
}
```

## Implementation Details

### File Structure

```
app_system/
├── referee/
│   ├── engine.py                    # Integration point (validation after rounds)
│   ├── workflow.py                  # UI display and Excel export
│   └── _utils/
│       ├── quote_validator.py       # Core validation logic
│       └── __init__.py              # Exports validation functions
└── tests/
    └── test_quote_validator.py      # Test suite
```

### Key Functions

#### `validate_quotes_in_report(report_text, paper_text, persona_name)`
Validates all quotes in a single persona report.

**Returns:**
```python
{
    'persona': 'Empiricist',
    'total_quotes': 5,
    'valid_quotes': 4,
    'invalid_quotes': 1,
    'quotes': [
        {
            'text': 'quoted text',
            'is_valid': True,
            'similarity_score': 92.5,
            'is_mathematical': False,
            'matched_text': 'context from paper',
            'threshold_used': 85
        },
        ...
    ]
}
```

#### `validate_quotes_in_reports(reports, paper_text)`
Validates quotes across multiple persona reports.

**Returns:** Dictionary mapping persona names to their validation results.

#### `get_validation_summary(validation_results)`
Aggregates validation statistics across all personas.

**Returns:**
```python
{
    'total_quotes_found': 15,
    'valid_quotes': 13,
    'invalid_quotes': 2,
    'validation_rate': 86.7,
    'invalid_quote_details': [...],
    'fuzzy_matching_available': True
}
```

## Testing

Run the test suite:

```bash
cd app_system
python tests/test_quote_validator.py
```

Tests cover:
- Text normalization
- Mathematical content detection
- Quote extraction from reports
- Single report validation
- Multi-report validation
- Summary generation

## Limitations

1. **Quote Detection**: May not detect all quote formats (e.g., paraphrased content)
2. **Context**: Validation checks text match but not semantic meaning
3. **Performance**: Large papers (>100 pages) may take longer to validate
4. **False Positives**: Very short common phrases may match incorrectly
5. **False Negatives**: Heavily reformatted quotes may fail even if semantically correct

## Future Enhancements

Potential improvements:

- Semantic similarity using embeddings (e.g., sentence transformers)
- Citation context validation (check if quote is used correctly)
- Support for indirect quotes and paraphrasing
- Integration with PDF coordinate extraction for quote highlighting
- Batch validation optimization for large papers
- Configurable thresholds via UI or config file

## Troubleshooting

### Issue: All quotes failing validation

**Possible causes:**
- Paper text extraction failed (check PDF parsing)
- Encoding issues (non-ASCII characters)
- Thresholds too strict

**Solution:**
- Verify paper text was extracted correctly
- Check encoding (should be UTF-8)
- Lower thresholds temporarily for debugging

### Issue: Mathematical quotes failing

**Possible causes:**
- LaTeX rendering differences
- Symbol encoding mismatches

**Solution:**
- Mathematical content uses 95% threshold (stricter)
- Ensure consistent encoding throughout pipeline
- Consider preprocessing LaTeX before validation

### Issue: Performance degradation

**Possible causes:**
- Very large papers (>100k characters)
- Many personas generating many quotes

**Solution:**
- Install `python-Levenshtein` for faster fuzzy matching
- Consider sampling quotes rather than validating all
- Cache paper text preprocessing

## References

- **thefuzz documentation**: https://github.com/seatgeek/thefuzz
- **Levenshtein distance**: https://en.wikipedia.org/wiki/Levenshtein_distance
- **Related code**: `section_eval/evaluator.py` (similar validation for section evaluator)
