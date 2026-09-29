"""
Test script for the quote validator in the referee system.

Run from app_system/:
    python tests/test_quote_validator.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from referee._utils.quote_validator import (
    validate_quotes_in_report,
    validate_quotes_in_reports,
    get_validation_summary,
    _extract_quotes_from_text,
    _is_mathematical_content,
    _normalize_text,
    FUZZY_AVAILABLE
)


def test_text_normalization():
    """Test text normalization."""
    print("Testing text normalization...")

    text = "  This is    a   test\nwith  newlines  "
    normalized = _normalize_text(text)
    assert normalized == "this is a test with newlines"

    print("✓ Text normalization works")


def test_mathematical_content_detection():
    """Test detection of mathematical content."""
    print("\nTesting mathematical content detection...")

    # Mathematical text
    math_text = "The coefficient β is significant at p < 0.05"
    assert _is_mathematical_content(math_text) is True

    # LaTeX patterns
    latex_text = "The model is $y = \\alpha + \\beta x + \\epsilon$"
    assert _is_mathematical_content(latex_text) is True

    # Regular prose
    prose_text = "This is a regular sentence without math"
    assert _is_mathematical_content(prose_text) is False

    print("✓ Mathematical content detection works")


def test_quote_extraction():
    """Test quote extraction from text."""
    print("\nTesting quote extraction...")

    report = """
    The author states: "This is a very important finding that we need to validate here."

    As the paper notes: "Economic growth is driven by innovation and productivity improvements."

    The paper also includes this quote: "The regression coefficient is statistically significant."

    Short quotes like "test" should be ignored.
    """

    quotes = _extract_quotes_from_text(report)

    print(f"  Found {len(quotes)} quotes:")
    for i, quote in enumerate(quotes, 1):
        print(f"    {i}. {quote[:60]}...")

    assert len(quotes) >= 2, f"Expected at least 2 quotes, found {len(quotes)}"

    print("✓ Quote extraction works")


def test_quote_validation():
    """Test quote validation against paper text."""
    print("\nTesting quote validation...")

    paper_text = """
    Economic growth is driven by innovation and productivity improvements.
    In this paper, we examine the relationship between R&D spending and GDP growth.
    Our findings indicate that a 1% increase in R&D investment leads to a 0.5% increase
    in GDP growth over a 5-year period. This result is robust to various model specifications.
    The coefficient β = 0.5 is statistically significant at p < 0.01.
    """

    report_text = """
    The paper states: "Economic growth is driven by innovation and productivity improvements."

    The authors claim: "a 1% increase in R&D investment leads to a 0.5% increase in GDP growth"

    This is a hallucinated quote: "The economy will collapse next year."

    Mathematical result: "The coefficient β = 0.5 is statistically significant at p < 0.01"
    """

    result = validate_quotes_in_report(
        report_text,
        paper_text,
        persona_name="TestPersona"
    )

    print(f"\n  Persona: {result['persona']}")
    print(f"  Total quotes: {result['total_quotes']}")
    print(f"  Valid quotes: {result['valid_quotes']}")
    print(f"  Invalid quotes: {result['invalid_quotes']}")

    print("\n  Quote details:")
    for i, quote in enumerate(result['quotes'], 1):
        status = "✓" if quote['is_valid'] else "✗"
        math = "📐" if quote['is_mathematical'] else "📝"
        print(f"    {status} {math} [{quote['similarity_score']:.1f}%] {quote['text'][:60]}...")

    # We should have found at least 3 quotes
    assert result['total_quotes'] >= 3, f"Expected at least 3 quotes, found {result['total_quotes']}"

    # We should have at least 2 valid quotes
    assert result['valid_quotes'] >= 2, f"Expected at least 2 valid quotes, found {result['valid_quotes']}"

    # We should have at least 1 invalid quote (the hallucinated one)
    assert result['invalid_quotes'] >= 1, f"Expected at least 1 invalid quote, found {result['invalid_quotes']}"

    print("\n✓ Quote validation works")


def test_multi_report_validation():
    """Test validation across multiple persona reports."""
    print("\nTesting multi-report validation...")

    paper_text = """
    This paper examines monetary policy transmission mechanisms.
    We find that interest rate changes affect output with a 2-quarter lag.
    The Phillips curve relationship remains stable in our sample period.
    """

    reports = {
        'Empiricist': """
        The paper states: "interest rate changes affect output with a 2-quarter lag"
        This is well-supported by the data.
        """,
        'Theorist': """
        The authors note: "The Phillips curve relationship remains stable"
        However, they also claim: "inflation will reach 100% next year" which seems incorrect.
        """
    }

    results = validate_quotes_in_reports(reports, paper_text)

    print(f"\n  Validated {len(results)} persona reports:")
    for persona, result in results.items():
        print(f"    {persona}: {result['valid_quotes']}/{result['total_quotes']} quotes valid")

    # Get summary
    summary = get_validation_summary(results)

    print(f"\n  Overall Summary:")
    print(f"    Total quotes: {summary['total_quotes_found']}")
    print(f"    Valid: {summary['valid_quotes']}")
    print(f"    Invalid: {summary['invalid_quotes']}")
    print(f"    Validation rate: {summary['validation_rate']:.1f}%")
    print(f"    Fuzzy matching available: {summary['fuzzy_matching_available']}")

    assert summary['total_quotes_found'] >= 2

    print("\n✓ Multi-report validation works")


def main():
    """Run all tests."""
    print("=" * 70)
    print("QUOTE VALIDATOR TEST SUITE")
    print("=" * 70)

    if not FUZZY_AVAILABLE:
        print("\n⚠️  WARNING: thefuzz library not available")
        print("   Install with: pip install thefuzz python-Levenshtein")
        print("   Tests will use fallback exact matching\n")
    else:
        print("\n✓ thefuzz library available - using fuzzy matching\n")

    try:
        test_text_normalization()
        test_mathematical_content_detection()
        test_quote_extraction()
        test_quote_validation()
        test_multi_report_validation()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
