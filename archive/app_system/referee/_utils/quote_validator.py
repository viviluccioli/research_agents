"""
Quote Validation Module for Referee Reports

Validates quoted text in referee persona reports against the source paper to detect
potential hallucinations. Uses fuzzy string matching with different thresholds for
mathematical content vs. prose.

Usage:
    from referee._utils.quote_validator import validate_quotes_in_reports

    validation_results = validate_quotes_in_reports(
        reports={'Empiricist': report_text, ...},
        paper_text=paper_text
    )
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

# Try to import thefuzz, fall back to simple matching if not available
try:
    from thefuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("thefuzz library not available. Install with: pip install thefuzz python-Levenshtein")

# Configuration
DEFAULT_PROSE_THRESHOLD = 85  # 85% similarity for prose
DEFAULT_MATH_THRESHOLD = 95   # 95% similarity for mathematical content
MIN_QUOTE_LENGTH = 10         # Minimum characters to validate
MAX_QUOTE_LENGTH = 500        # Maximum quote length to extract

# Mathematical symbols that indicate technical content
MATH_SYMBOLS = ['α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'σ', 'ρ', 'π', '∫', '∑', '∏', '∂', '∇', '≈', '≠', '≤', '≥', '∈', '∉', '∀', '∃']


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison by:
    - Collapsing whitespace
    - Removing newlines
    - Stripping leading/trailing space
    - Lowercasing
    """
    if not text:
        return ""
    # Collapse all whitespace (including newlines) to single space
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized.lower()


def _is_mathematical_content(text: str) -> bool:
    """
    Detect if text contains mathematical symbols or formulas.
    Returns True if mathematical content is detected.
    """
    if not text:
        return False

    # Check for LaTeX-like patterns
    latex_patterns = [r'\$.*?\$', r'\\[a-zA-Z]+', r'\^', r'_\{', r'\{.*?\}']
    for pattern in latex_patterns:
        if re.search(pattern, text):
            return True

    # Check for mathematical symbols
    return any(symbol in text for symbol in MATH_SYMBOLS)


def _extract_quotes_from_text(text: str) -> List[str]:
    """
    Extract quoted text from a report.
    Looks for text in double quotes, single quotes, or blockquotes.

    Returns:
        List of extracted quote strings
    """
    if not text:
        return []

    quotes = []

    # Pattern 1: Text in double quotes
    # Match quotes that are reasonably long
    double_quote_pattern = r'"([^"]{' + str(MIN_QUOTE_LENGTH) + r',' + str(MAX_QUOTE_LENGTH) + r'})"'
    quotes.extend(re.findall(double_quote_pattern, text))

    # Pattern 2: Text in single quotes (less common for quotes)
    single_quote_pattern = r"'([^']{" + str(MIN_QUOTE_LENGTH) + r',' + str(MAX_QUOTE_LENGTH) + r"})'"
    quotes.extend(re.findall(single_quote_pattern, text))

    # Pattern 3: Text after "the paper states:", "as stated:", etc.
    statement_pattern = r'(?:the paper states|as stated|the authors? (?:write|state|claim|note)):\s*["\']?([^"\'.]{' + str(MIN_QUOTE_LENGTH) + r',' + str(MAX_QUOTE_LENGTH) + r'})["\']?'
    quotes.extend(re.findall(statement_pattern, text, re.IGNORECASE))

    # Pattern 4: Markdown blockquotes (lines starting with >)
    blockquote_pattern = r'^>\s*(.+)$'
    blockquotes = re.findall(blockquote_pattern, text, re.MULTILINE)
    for quote in blockquotes:
        if MIN_QUOTE_LENGTH <= len(quote) <= MAX_QUOTE_LENGTH:
            quotes.append(quote)

    # Clean and deduplicate
    cleaned_quotes = []
    seen = set()
    for quote in quotes:
        cleaned = quote.strip()
        if cleaned and len(cleaned) >= MIN_QUOTE_LENGTH:
            normalized = _normalize_text(cleaned)
            if normalized not in seen:
                seen.add(normalized)
                cleaned_quotes.append(cleaned)

    return cleaned_quotes


def _validate_quote_fuzzy(
    quote: str,
    paper_text: str,
    prose_threshold: int = DEFAULT_PROSE_THRESHOLD,
    math_threshold: int = DEFAULT_MATH_THRESHOLD
) -> Tuple[bool, float, Optional[str]]:
    """
    Validate a quote against paper text using fuzzy matching.

    Args:
        quote: The quoted text to validate
        paper_text: The full paper text to search in
        prose_threshold: Similarity threshold for prose (0-100)
        math_threshold: Similarity threshold for mathematical content (0-100)

    Returns:
        Tuple of (is_valid, similarity_score, matched_text)
    """
    if not quote or not paper_text:
        return False, 0.0, None

    # Normalize inputs
    norm_quote = _normalize_text(quote)
    norm_paper = _normalize_text(paper_text)

    # If quote is very short, don't validate (likely not a real quote)
    if len(norm_quote) < MIN_QUOTE_LENGTH:
        return False, 0.0, None

    # Determine threshold based on content type
    is_math = _is_mathematical_content(quote)
    threshold = math_threshold if is_math else prose_threshold

    # Try exact match first (fastest)
    if norm_quote in norm_paper:
        # Find the original text from paper
        start_idx = norm_paper.find(norm_quote)
        # Rough estimate of original position
        matched_snippet = paper_text[start_idx:start_idx + len(quote) + 50].strip()
        return True, 100.0, matched_snippet

    # If fuzzy matching is not available, try partial match
    if not FUZZY_AVAILABLE:
        # Fallback: check if first 40 chars match
        partial = norm_quote[:40]
        if len(partial) > 10 and partial in norm_paper:
            start_idx = norm_paper.find(partial)
            matched_snippet = paper_text[start_idx:start_idx + len(quote) + 50].strip()
            return True, 90.0, matched_snippet
        return False, 0.0, None

    # Use fuzzy matching to find best match
    # Split paper into overlapping windows
    quote_len = len(norm_quote)
    window_size = quote_len + 20  # Allow some extra characters
    stride = max(10, quote_len // 4)  # 25% overlap

    best_score = 0.0
    best_match = None
    best_idx = 0

    # Scan through paper in windows
    for i in range(0, len(norm_paper) - window_size + 1, stride):
        window = norm_paper[i:i + window_size]
        # Use partial ratio for substring matching
        score = fuzz.partial_ratio(norm_quote, window)

        if score > best_score:
            best_score = score
            best_idx = i
            best_match = window

    # Also check full document with token_set_ratio (handles word reordering)
    full_score = fuzz.token_set_ratio(norm_quote, norm_paper)
    if full_score > best_score:
        best_score = full_score
        best_match = norm_paper[:len(quote) + 100]  # Get snippet from start

    # Determine if valid based on threshold
    is_valid = best_score >= threshold

    # Extract original text snippet if match found
    matched_snippet = None
    if is_valid and best_match:
        # Try to find approximate location in original paper
        matched_snippet = paper_text[best_idx:best_idx + len(quote) + 50].strip()

    return is_valid, best_score, matched_snippet


def validate_quotes_in_report(
    report_text: str,
    paper_text: str,
    persona_name: str = "Persona",
    prose_threshold: int = DEFAULT_PROSE_THRESHOLD,
    math_threshold: int = DEFAULT_MATH_THRESHOLD
) -> Dict:
    """
    Validate all quotes in a single persona report.

    Args:
        report_text: The persona's report text
        paper_text: The full paper text
        persona_name: Name of the persona (for logging)
        prose_threshold: Similarity threshold for prose
        math_threshold: Similarity threshold for mathematical content

    Returns:
        Dictionary with validation results:
        {
            'persona': str,
            'total_quotes': int,
            'valid_quotes': int,
            'invalid_quotes': int,
            'quotes': [
                {
                    'text': str,
                    'is_valid': bool,
                    'similarity_score': float,
                    'is_mathematical': bool,
                    'matched_text': str or None
                },
                ...
            ]
        }
    """
    # Extract quotes from report
    quotes = _extract_quotes_from_text(report_text)

    # Validate each quote
    quote_results = []
    valid_count = 0

    for quote in quotes:
        is_valid, score, matched = _validate_quote_fuzzy(
            quote, paper_text, prose_threshold, math_threshold
        )

        is_math = _is_mathematical_content(quote)

        quote_results.append({
            'text': quote,
            'is_valid': is_valid,
            'similarity_score': score,
            'is_mathematical': is_math,
            'matched_text': matched,
            'threshold_used': math_threshold if is_math else prose_threshold
        })

        if is_valid:
            valid_count += 1

    return {
        'persona': persona_name,
        'total_quotes': len(quotes),
        'valid_quotes': valid_count,
        'invalid_quotes': len(quotes) - valid_count,
        'quotes': quote_results
    }


def validate_quotes_in_reports(
    reports: Dict[str, str],
    paper_text: str,
    prose_threshold: int = DEFAULT_PROSE_THRESHOLD,
    math_threshold: int = DEFAULT_MATH_THRESHOLD
) -> Dict[str, Dict]:
    """
    Validate quotes in multiple persona reports.

    Args:
        reports: Dictionary mapping persona names to their report text
        paper_text: The full paper text
        prose_threshold: Similarity threshold for prose (0-100)
        math_threshold: Similarity threshold for mathematical content (0-100)

    Returns:
        Dictionary mapping persona names to their validation results
    """
    all_results = {}

    for persona, report in reports.items():
        result = validate_quotes_in_report(
            report,
            paper_text,
            persona,
            prose_threshold,
            math_threshold
        )
        all_results[persona] = result

    return all_results


def get_validation_summary(validation_results: Dict[str, Dict]) -> Dict:
    """
    Generate a summary of quote validation across all personas.

    Args:
        validation_results: Results from validate_quotes_in_reports()

    Returns:
        Summary dictionary with aggregate statistics
    """
    total_quotes = sum(r['total_quotes'] for r in validation_results.values())
    valid_quotes = sum(r['valid_quotes'] for r in validation_results.values())
    invalid_quotes = sum(r['invalid_quotes'] for r in validation_results.values())

    # List all invalid quotes with persona attribution
    invalid_quote_details = []
    for persona, result in validation_results.items():
        for quote in result['quotes']:
            if not quote['is_valid']:
                invalid_quote_details.append({
                    'persona': persona,
                    'quote': quote['text'][:100] + ('...' if len(quote['text']) > 100 else ''),
                    'similarity_score': quote['similarity_score']
                })

    return {
        'total_quotes_found': total_quotes,
        'valid_quotes': valid_quotes,
        'invalid_quotes': invalid_quotes,
        'validation_rate': (valid_quotes / total_quotes * 100) if total_quotes > 0 else 0.0,
        'invalid_quote_details': invalid_quote_details,
        'fuzzy_matching_available': FUZZY_AVAILABLE
    }


def mark_unverified_quotes_in_text(report_text: str, validation_result: Dict) -> str:
    """
    Mark unverified quotes in report text with [UNVERIFIED QUOTE] tag.

    Args:
        report_text: The original report text
        validation_result: Result from validate_quotes_in_report()

    Returns:
        Modified report text with unverified quotes marked
    """
    modified_text = report_text

    # Sort quotes by length (longest first) to avoid substring replacement issues
    invalid_quotes = [
        q['text'] for q in validation_result['quotes']
        if not q['is_valid']
    ]
    invalid_quotes.sort(key=len, reverse=True)

    # Mark each invalid quote
    for quote in invalid_quotes:
        # Try to find the quote in various formats
        patterns = [
            f'"{quote}"',
            f"'{quote}'",
            f'>{quote}',  # blockquote
            quote  # plain
        ]

        for pattern in patterns:
            if pattern in modified_text:
                replacement = f'[UNVERIFIED QUOTE: {pattern}]'
                modified_text = modified_text.replace(pattern, replacement, 1)
                break

    return modified_text
