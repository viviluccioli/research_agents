"""
LLM-powered cleanup for mathematical equations and complex tables extracted from PDFs.

This module detects poorly-extracted LaTeX equations and tables in raw PDF text,
then uses an LLM to reconstruct them into readable form for downstream evaluation.
"""

import re
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MathRegion:
    """Represents a detected mathematical or table region in text."""
    start_idx: int
    end_idx: int
    content: str
    region_type: str  # 'equation', 'table', 'complex_math'
    confidence: float  # 0-1 score for detection confidence


def _contains_math_indicators(text: str) -> bool:
    """
    Check if text contains indicators of mathematical notation.

    Args:
        text: Text snippet to analyze

    Returns:
        True if text likely contains math equations
    """
    # Patterns indicating equations
    patterns = [
        r'[α-ωΑ-Ω]',  # Greek letters
        r'[₀-₉⁰-⁹]',  # Subscripts/superscripts
        r'∑|∫|∂|∇|∈|∀|∃|∞',  # Math operators
        r'exp\(|log\(|ln\(|max\(|min\(',  # Math functions
        r'Pr\(|E\[|Var\(',  # Statistical notation
        r'\+\s*\.\s*\+|\-\s*\.\s*\-',  # Broken operators
        r'(?:^|\s)[a-z]\s+[0-9](?:\s|$)',  # Broken subscripts like "t 1" or "Y 0"
        r'[a-zA-Z]_[a-zA-Z0-9]',  # Underscore notation (X_t, β_1)
        r'[=<>≤≥≠≈∝].*[=<>≤≥≠≈∝]',  # Multiple comparison operators
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def _is_broken_math(text: str) -> float:
    """
    Assess how "broken" mathematical notation appears.

    Args:
        text: Text snippet to analyze

    Returns:
        Score 0-1 indicating brokenness (higher = more broken)
    """
    score = 0.0

    # Broken subscripts/superscripts on separate lines
    if re.search(r'(?:^|\n)\s*[₀-₉t\-+?]\s*(?:\n|$)', text):
        score += 0.3

    # Orphaned operators or numbers
    if re.search(r'(?:^|\n)\s*[+\-*/=]\s*(?:\n|$)', text):
        score += 0.2

    # Fraction indicators (appears when fractions break across lines)
    if re.search(r'(?:^|\n)\s*(?:1\+|[\d]+)\s*(?:\n|$)', text):
        score += 0.2

    # Variables separated from subscripts
    if re.search(r'[A-Z]\s+[₀-₉t₋₊]', text):
        score += 0.3

    # High density of math symbols with low alphanumeric ratio
    math_chars = len(re.findall(r'[α-ωΑ-Ω₀-₉⁰-⁹∑∫∂∇∈∀∃∞=<>≤≥≠≈∝+\-*/()]', text))
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars > 0 and (math_chars / total_chars) > 0.4:
        score += 0.3

    return min(score, 1.0)


def _is_broken_table(text: str) -> float:
    """
    Assess whether text appears to be a poorly-extracted table.

    Args:
        text: Text snippet to analyze

    Returns:
        Score 0-1 indicating table-brokenness (higher = likely broken table)
    """
    score = 0.0

    # Multiple columns of numbers (common in regression tables)
    if re.search(r'(?:\d+\.\d+\s+){3,}', text):
        score += 0.3

    # Parenthesized numbers (standard errors in tables)
    parens = len(re.findall(r'\(\d+\.\d+\)', text))
    if parens >= 3:
        score += 0.2

    # Significance stars (*, **, ***)
    if re.search(r'\*{1,3}(?:\s|$)', text):
        score += 0.2

    # Table keywords
    if re.search(r'(?i)^(?:table|panel|column|specification|model)\s*\d+', text, re.MULTILINE):
        score += 0.3

    # Poor structure: text jammed together without clear columns
    lines = text.split('\n')
    if len(lines) >= 5:
        # Check if lines lack consistent structure
        word_counts = [len(line.split()) for line in lines if line.strip()]
        if word_counts and max(word_counts) > 10 and min(word_counts) < 3:
            score += 0.2

    return min(score, 1.0)


def detect_math_regions(text: str, context_lines: int = 2) -> List[MathRegion]:
    """
    Detect regions in text that likely contain broken equations or tables.

    Args:
        text: Full extracted PDF text
        context_lines: Number of lines of context to include around detected regions

    Returns:
        List of MathRegion objects
    """
    regions = []
    lines = text.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip empty lines
        if not line.strip():
            i += 1
            continue

        # Look ahead window for multi-line expressions
        window_size = 10
        window_text = '\n'.join(lines[i:min(i+window_size, len(lines))])

        # Check for equations
        if _contains_math_indicators(window_text):
            brokenness = _is_broken_math(window_text)

            if brokenness > 0.3:
                # Find extent of math region
                start_line = max(0, i - context_lines)
                end_line = i

                # Expand forward while still looks like math
                j = i + 1
                while j < len(lines) and j < i + window_size:
                    chunk = '\n'.join(lines[i:j+1])
                    if _contains_math_indicators(chunk) and _is_broken_math(chunk) > 0.2:
                        end_line = j
                        j += 1
                    else:
                        break

                end_line = min(len(lines) - 1, end_line + context_lines)

                # Extract region
                region_text = '\n'.join(lines[start_line:end_line+1])
                start_idx = sum(len(lines[k]) + 1 for k in range(start_line))
                end_idx = sum(len(lines[k]) + 1 for k in range(end_line + 1))

                regions.append(MathRegion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    content=region_text,
                    region_type='equation',
                    confidence=brokenness
                ))

                # Skip past this region
                i = end_line + 1
                continue

        # Check for tables
        if _is_broken_table(window_text):
            tableness = _is_broken_table(window_text)

            if tableness > 0.4:
                # Find extent of table region
                start_line = max(0, i - context_lines)
                end_line = i

                # Expand forward while still looks like table
                j = i + 1
                while j < len(lines) and j < i + 15:  # Tables can be longer
                    chunk = '\n'.join(lines[i:j+1])
                    if _is_broken_table(chunk) > 0.3:
                        end_line = j
                        j += 1
                    else:
                        break

                end_line = min(len(lines) - 1, end_line + context_lines)

                # Extract region
                region_text = '\n'.join(lines[start_line:end_line+1])
                start_idx = sum(len(lines[k]) + 1 for k in range(start_line))
                end_idx = sum(len(lines[k]) + 1 for k in range(end_line + 1))

                regions.append(MathRegion(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    content=region_text,
                    region_type='table',
                    confidence=tableness
                ))

                # Skip past this region
                i = end_line + 1
                continue

        i += 1

    # Merge overlapping regions
    regions = _merge_overlapping_regions(regions)

    return regions


def _merge_overlapping_regions(regions: List[MathRegion]) -> List[MathRegion]:
    """Merge overlapping or adjacent regions."""
    if not regions:
        return []

    # Sort by start position
    regions = sorted(regions, key=lambda r: r.start_idx)

    merged = [regions[0]]

    for region in regions[1:]:
        last = merged[-1]

        # If overlapping or very close (within 50 chars)
        if region.start_idx <= last.end_idx + 50:
            # Merge: extend the last region
            merged[-1] = MathRegion(
                start_idx=last.start_idx,
                end_idx=max(last.end_idx, region.end_idx),
                content=last.content + '\n' + region.content,
                region_type=last.region_type if last.confidence >= region.confidence else region.region_type,
                confidence=max(last.confidence, region.confidence)
            )
        else:
            merged.append(region)

    return merged


def build_cleanup_prompt(region: MathRegion) -> str:
    """
    Build an LLM prompt to clean up a detected math/table region.

    Args:
        region: MathRegion to clean

    Returns:
        Prompt string
    """
    if region.region_type == 'equation':
        return f"""The following text appears to be a mathematical equation that was poorly extracted from a PDF. LaTeX formatting was lost, causing subscripts, superscripts, and symbols to be misaligned or placed on separate lines.

Your task: Reconstruct this into a readable mathematical expression that an LLM can understand.

**Instructions**:
- Restore subscripts and superscripts to their proper positions (e.g., "X t-1" → "X_{t-1}")
- Fix fraction notation (numerator/denominator)
- Preserve all Greek letters and mathematical symbols
- Use standard mathematical notation (e.g., β_0, σ², log(), exp())
- If there are multiple equations, separate them clearly
- Keep any surrounding context text unchanged
- Output ONLY the cleaned text, no explanations

**Poorly Extracted Text**:
```
{region.content}
```

**Cleaned Output** (just the fixed text):"""

    else:  # table
        return f"""The following text appears to be a table (likely a regression results table) that was poorly extracted from a PDF. The structure is broken, making it hard to see which values belong to which columns.

Your task: Reconstruct this into a readable format that preserves the data structure.

**Instructions**:
- Identify column headers and row labels
- Align coefficients with their standard errors (usually in parentheses)
- Preserve significance stars (*, **, ***)
- Use a simple text-based table format with clear column separation
- If the structure is too broken, output the values in a list format with clear labels
- Keep any table title or notes
- Output ONLY the cleaned text, no explanations

**Poorly Extracted Text**:
```
{region.content}
```

**Cleaned Output** (just the fixed table/data):"""


def cleanup_math_regions_with_llm(text: str, llm_query_fn, max_regions: int = 10) -> str:
    """
    Detect and clean up math/table regions in PDF-extracted text using an LLM.

    Args:
        text: Raw PDF-extracted text
        llm_query_fn: Function that takes a prompt string and returns LLM response
                      (e.g., utils.single_query or a wrapper around it)
        max_regions: Maximum number of regions to process (to avoid excessive API calls)

    Returns:
        Text with math/table regions cleaned up
    """
    regions = detect_math_regions(text)

    if not regions:
        return text

    # Sort by confidence and take top N
    regions = sorted(regions, key=lambda r: r.confidence, reverse=True)[:max_regions]

    # Sort by position for replacement (process backwards to preserve indices)
    regions = sorted(regions, key=lambda r: r.start_idx, reverse=True)

    # Process each region
    for region in regions:
        try:
            prompt = build_cleanup_prompt(region)
            cleaned = llm_query_fn(prompt).strip()

            # Replace in text
            text = text[:region.start_idx] + cleaned + text[region.end_idx:]

        except Exception as e:
            # If LLM call fails, leave original text
            print(f"Warning: Failed to clean region at position {region.start_idx}: {e}")
            continue

    return text


def add_quality_warnings(text: str) -> Tuple[str, List[str]]:
    """
    Analyze extracted text and generate quality warnings.

    Args:
        text: Extracted PDF text

    Returns:
        Tuple of (text with warnings prepended, list of warning messages)
    """
    warnings = []

    # Check for math content
    has_equations = bool(re.search(r'(?:equation|formula|exp\(|log\(|Pr\()', text, re.IGNORECASE))
    has_broken_math = any(_is_broken_math(chunk) > 0.3
                          for chunk in re.findall(r'.{100}', text))

    if has_equations and has_broken_math:
        warnings.append("⚠️  This document contains mathematical equations with potentially degraded formatting. Subscripts and mathematical notation may be unclear.")

    # Check for tables
    has_tables = bool(re.search(r'(?i)^(?:table|panel)\s*\d+', text, re.MULTILINE))
    has_broken_tables = any(_is_broken_table(chunk) > 0.4
                           for chunk in re.findall(r'.{200}', text))

    if has_tables and has_broken_tables:
        warnings.append("⚠️  This document contains tables with potentially degraded structure. Column alignment and data associations may be unclear.")

    # Prepend warnings to text
    if warnings:
        warning_block = "\n".join(warnings) + "\n\n" + "="*80 + "\n\n"
        text = warning_block + text

    return text, warnings
