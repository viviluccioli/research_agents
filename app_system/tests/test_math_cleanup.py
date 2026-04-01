#!/usr/bin/env python3
"""
Test script for LLM-powered math and table cleanup.

This script tests the detection and cleanup of poorly-extracted equations
and tables from PDF text.

Usage:
    python test_math_cleanup.py <path_to_pdf>

    # Run with math cleanup enabled
    python test_math_cleanup.py ../papers/sample_paper.pdf --cleanup

    # Test detection only (no LLM calls)
    python test_math_cleanup.py ../papers/sample_paper.pdf --detect-only
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from section_eval.text_extraction import extract_text_from_pdf
from section_eval.math_cleanup import (
    detect_math_regions,
    cleanup_math_regions_with_llm,
    add_quality_warnings
)


def mock_llm_query(prompt: str) -> str:
    """
    Mock LLM function for testing detection without API calls.
    """
    return "[CLEANED BY LLM - Mock response]"


def load_real_llm_query():
    """
    Load the real LLM query function from utils.

    Returns:
        single_query function from utils module
    """
    try:
        from utils import single_query
        return single_query
    except ImportError:
        print("⚠️  Warning: Could not import utils.single_query")
        print("   Make sure you're running from app_system/ directory")
        print("   Falling back to mock LLM")
        return mock_llm_query


def test_extraction_with_cleanup(pdf_path: str, use_real_llm: bool = False):
    """
    Test PDF extraction with and without math cleanup.

    Args:
        pdf_path: Path to PDF file
        use_real_llm: If True, use real LLM; if False, use mock
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"📄 Testing extraction on: {pdf_path.name}")
    print("=" * 80)

    # Load PDF
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()

    # Extract WITHOUT cleanup
    print("\n📥 Extracting text (standard extraction)...")
    text_raw = extract_text_from_pdf(file_bytes, warn_fn=print)

    # Detect regions
    print("\n🔍 Detecting math/table regions...")
    regions = detect_math_regions(text_raw)

    print(f"\n✅ Found {len(regions)} problematic regions:")
    print("-" * 80)

    for i, region in enumerate(regions, 1):
        print(f"\nRegion #{i}:")
        print(f"  Type: {region.region_type}")
        print(f"  Confidence: {region.confidence:.2f}")
        print(f"  Position: chars {region.start_idx}-{region.end_idx}")
        print(f"  Length: {len(region.content)} chars")
        print(f"\n  Preview (first 200 chars):")
        print("  " + "-" * 76)
        preview = region.content[:200].replace('\n', '\n  ')
        print(f"  {preview}{'...' if len(region.content) > 200 else ''}")
        print()

    # Test cleanup if requested
    if use_real_llm and regions:
        print("\n" + "=" * 80)
        print("🤖 Running LLM cleanup...")
        print("=" * 80)

        llm_fn = load_real_llm_query()

        # Show before/after for first region
        if regions:
            region = regions[0]
            print(f"\n📋 BEFORE (Region #1 - {region.region_type}):")
            print("-" * 80)
            print(region.content)

            # Clean up entire text
            text_clean = cleanup_math_regions_with_llm(text_raw, llm_fn, max_regions=3)

            # Find the cleaned version (approximate by position)
            print(f"\n✨ AFTER (cleaned by LLM):")
            print("-" * 80)
            # Extract roughly the same region from cleaned text
            cleaned_snippet = text_clean[region.start_idx:region.start_idx+len(region.content)]
            print(cleaned_snippet)

            # Save both versions
            output_dir = pdf_path.parent
            raw_file = output_dir / f"{pdf_path.stem}_raw.txt"
            clean_file = output_dir / f"{pdf_path.stem}_cleaned.txt"

            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(text_raw)
            with open(clean_file, 'w', encoding='utf-8') as f:
                f.write(text_clean)

            print(f"\n💾 Saved outputs:")
            print(f"   Raw:     {raw_file}")
            print(f"   Cleaned: {clean_file}")

    # Quality warnings
    print("\n" + "=" * 80)
    print("⚠️  QUALITY ASSESSMENT")
    print("=" * 80)

    text_with_warnings, warnings = add_quality_warnings(text_raw)

    if warnings:
        for warning in warnings:
            print(f"  {warning}")
    else:
        print("  ✅ No significant extraction issues detected")

    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)

    if regions:
        print(f"  This PDF has {len(regions)} region(s) with degraded math/table formatting.")
        print(f"  Consider using cleanup_math=True in decode_file() for better results.")
        print(f"\n  To enable LLM cleanup:")
        print(f"    from utils import single_query")
        print(f"    text = decode_file(filename, bytes, cleanup_math=True, llm_query_fn=single_query)")
    else:
        print("  ✅ No problematic regions detected - standard extraction is sufficient.")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test math and table cleanup for PDF extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect problematic regions only (no LLM calls)
  python test_math_cleanup.py paper.pdf

  # Run full cleanup with real LLM
  python test_math_cleanup.py paper.pdf --cleanup

  # Detect only
  python test_math_cleanup.py paper.pdf --detect-only
        """
    )
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('--cleanup', action='store_true',
                       help='Run actual LLM cleanup (makes API calls)')
    parser.add_argument('--detect-only', action='store_true',
                       help='Only detect regions, no cleanup (default)')

    args = parser.parse_args()

    # Default to detect-only unless --cleanup specified
    use_llm = args.cleanup and not args.detect_only

    if use_llm:
        print("🔑 Using real LLM for cleanup (will make API calls)")
    else:
        print("🔍 Detection mode only (no API calls)")

    test_extraction_with_cleanup(args.pdf_path, use_real_llm=use_llm)


if __name__ == "__main__":
    main()
