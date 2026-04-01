#!/usr/bin/env python3
"""
PDF Extraction Testing Utility

This script tests PDF text and table extraction to assess what the LLM receives
when evaluating papers in the multi-agent debate system.

Usage:
    python test_pdf_extraction.py <path_to_pdf>
    python test_pdf_extraction.py ../papers/sample_paper.pdf
    python test_pdf_extraction.py ../papers/sample_paper.pdf --check-math

The script will:
1. Extract all text and tables from the PDF
2. Display extraction statistics (pages, tables, characters)
3. Save the extracted content to a text file for manual inspection
4. Show sample table extractions
5. Optionally detect problematic math/table regions (--check-math)
"""

import sys
import pdfplumber
from pathlib import Path
import argparse


def format_table_as_markdown(table):
    """
    Format an extracted table as markdown.

    Args:
        table: List of lists representing table rows

    Returns:
        str: Markdown-formatted table
    """
    if not table or not any(table):
        return "[Empty table]"

    markdown = ""

    # Process table rows
    for i, row in enumerate(table):
        if row is None:
            continue

        # Clean cells (handle None values)
        cells = [str(cell).strip() if cell is not None else "" for cell in row]

        # Create markdown row
        markdown += "| " + " | ".join(cells) + " |\n"

        # Add header separator after first row
        if i == 0:
            markdown += "|" + "|".join(["---" for _ in cells]) + "|\n"

    return markdown


def extract_and_analyze_pdf(pdf_path: str, verbose: bool = False, check_math: bool = False):
    """
    Extract text and tables from a PDF and provide detailed analysis.

    Args:
        pdf_path: Path to the PDF file
        verbose: If True, print detailed output including table contents
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Error: File is not a PDF: {pdf_path}")
        sys.exit(1)

    print(f"📄 Analyzing PDF: {pdf_path.name}")
    print("=" * 80)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            total_tables = 0
            total_chars = 0
            full_text = ""
            table_details = []

            print(f"\n📊 Processing {total_pages} pages...\n")

            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                page_text = page.extract_text() or ""
                full_text += page_text
                page_chars = len(page_text)
                total_chars += page_chars

                # Extract tables
                tables = page.extract_tables()
                page_tables = len(tables)
                total_tables += page_tables

                # Progress indicator
                print(f"  Page {page_num:3d}: {page_chars:6,} chars, {page_tables} table(s)")

                # Process tables
                if tables:
                    for table_num, table in enumerate(tables, 1):
                        global_table_num = len(table_details) + 1
                        table_md = format_table_as_markdown(table)
                        table_details.append({
                            'page': page_num,
                            'table_num': global_table_num,
                            'rows': len(table),
                            'cols': len(table[0]) if table else 0,
                            'markdown': table_md
                        })

                        # Add table to full text
                        full_text += f"\n\n[TABLE {global_table_num} - Page {page_num}]\n"
                        full_text += table_md
                        full_text += "\n[END TABLE]\n\n"

                # Add page break
                full_text += f"\n\n--- PAGE {page_num} ---\n\n"

            # Print summary
            print("\n" + "=" * 80)
            print("📊 EXTRACTION SUMMARY")
            print("=" * 80)
            print(f"Total Pages:      {total_pages}")
            print(f"Total Characters: {total_chars:,}")
            print(f"Total Tables:     {total_tables}")
            print(f"Estimated Tokens: {int(total_chars / 4):,} (rough estimate)")
            print("=" * 80)

            # Table details
            if table_details:
                print("\n📋 TABLE DETAILS")
                print("=" * 80)
                for table_info in table_details:
                    print(f"Table #{table_info['table_num']} (Page {table_info['page']}): "
                          f"{table_info['rows']} rows × {table_info['cols']} cols")

                if verbose:
                    print("\n📝 SAMPLE TABLE EXTRACTIONS (first 3 tables)")
                    print("=" * 80)
                    for i, table_info in enumerate(table_details[:3]):
                        print(f"\nTable #{table_info['table_num']} (Page {table_info['page']}):")
                        print("-" * 80)
                        print(table_info['markdown'])

            else:
                print("\n⚠️  WARNING: No tables detected!")
                print("If this paper contains tables, they may not be extractable,")
                print("or they might be embedded as images rather than structured data.")

            # Save extracted text
            output_path = pdf_path.parent / f"{pdf_path.stem}_extracted.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# PDF Extraction Report\n")
                f.write(f"# Source: {pdf_path.name}\n")
                f.write(f"# Pages: {total_pages} | Characters: {total_chars:,} | Tables: {total_tables}\n")
                f.write(f"# Generated: {Path(__file__).name}\n")
                f.write("\n" + "=" * 80 + "\n\n")
                f.write(full_text)

            print(f"\n✅ Extraction complete!")
            print(f"📁 Full extracted text saved to: {output_path}")
            print(f"\nℹ️  This is what the LLM receives when evaluating the paper.")

            # Quality warnings
            print("\n" + "=" * 80)
            print("⚠️  QUALITY ASSESSMENT")
            print("=" * 80)

            if total_tables == 0:
                print("🔴 NO TABLES: Empirical evaluations may be inaccurate")
            elif total_tables < 5:
                print(f"🟡 FEW TABLES ({total_tables}): Check if all tables were extracted")
            else:
                print(f"🟢 TABLES DETECTED ({total_tables}): Extraction looks reasonable")

            if total_chars < 10000:
                print("🟡 SHORT TEXT: Paper may be incomplete or poorly scanned")
            elif total_chars > 200000:
                print("🟡 VERY LONG TEXT: May exceed token limits in some contexts")
            else:
                print(f"🟢 TEXT LENGTH REASONABLE ({total_chars:,} chars)")

            print("\nℹ️  Manually inspect the extracted text file to assess quality.")

            # Check for math/table issues if requested
            if check_math:
                try:
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from section_eval.math_cleanup import detect_math_regions, add_quality_warnings

                    print("\n" + "=" * 80)
                    print("🔍 MATH/TABLE QUALITY CHECK")
                    print("=" * 80)

                    regions = detect_math_regions(full_text)

                    if regions:
                        print(f"\n⚠️  Found {len(regions)} region(s) with potential formatting issues:")
                        for i, region in enumerate(regions, 1):
                            print(f"  #{i}: {region.region_type} (confidence: {region.confidence:.2f})")

                        print(f"\n💡 These regions may have:")
                        print("   - Broken LaTeX equations (subscripts on wrong lines)")
                        print("   - Poorly structured tables (unclear column alignment)")
                        print("   - Fragmented mathematical notation")

                        print(f"\n🔧 To fix these issues, use:")
                        print("   python test_math_cleanup.py <pdf_path> --cleanup")

                    else:
                        print("✅ No significant math/table formatting issues detected")

                    # Add warnings
                    _, warnings = add_quality_warnings(full_text)
                    if warnings:
                        print("\n⚠️  Warnings:")
                        for warning in warnings:
                            print(f"   {warning}")

                except ImportError:
                    print("\n⚠️  Math cleanup module not available")
                except Exception as e:
                    print(f"\n⚠️  Error during math check: {e}")

            print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test PDF extraction quality for multi-agent debate system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_pdf_extraction.py paper.pdf
  python test_pdf_extraction.py ../papers/sample_paper.pdf --verbose
  python test_pdf_extraction.py paper.pdf -v

This tool shows exactly what the LLM receives when evaluating papers.
        """
    )
    parser.add_argument('pdf_path', help='Path to the PDF file to analyze')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output including sample table contents')
    parser.add_argument('--check-math', action='store_true',
                       help='Check for problematic math/table regions')

    args = parser.parse_args()

    extract_and_analyze_pdf(args.pdf_path, verbose=args.verbose, check_math=args.check_math)


if __name__ == "__main__":
    main()
