"""
Example: Batch Paper Classification

Processes multiple papers from a folder or CSV file and generates
comprehensive results in CSV or Excel format.
"""

import sys
import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from referee import classify_paper, adjust_persona_weights


def load_papers_from_folder(folder_path: str) -> List[Dict]:
    """
    Load all .txt files from a folder.

    Args:
        folder_path: Path to folder containing paper text files

    Returns:
        List of dicts with 'id' and 'text' keys
    """
    papers = []
    folder = Path(folder_path)

    for file_path in folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            papers.append({
                'id': file_path.stem,
                'text': text,
                'source_file': str(file_path)
            })
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    return papers


def load_papers_from_csv(csv_path: str, text_column: str = 'text') -> List[Dict]:
    """
    Load papers from CSV file.

    Args:
        csv_path: Path to CSV file
        text_column: Name of column containing paper text

    Returns:
        List of dicts with paper data
    """
    papers = []

    try:
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")

        for idx, row in df.iterrows():
            papers.append({
                'id': row.get('id', row.get('paper_id', f"paper_{idx}")),
                'text': row[text_column],
                'source_file': csv_path
            })

    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    return papers


def process_papers_batch(
    papers: List[Dict],
    use_llm: bool = True,
    progress: bool = True
) -> List[Dict]:
    """
    Process multiple papers and return classification results.

    Args:
        papers: List of paper dicts with 'id' and 'text' keys
        use_llm: Whether to use LLM classification
        progress: Whether to show progress

    Returns:
        List of result dicts
    """
    results = []

    for i, paper in enumerate(papers):
        if progress:
            print(f"Processing {i+1}/{len(papers)}: {paper['id']}")

        try:
            # Classify paper
            classification = classify_paper(paper['text'], use_llm=use_llm)

            # Adjust persona weights
            weights = adjust_persona_weights(classification)

            # Build result
            result = {
                'paper_id': paper['id'],
                'primary_type': classification.primary_type,
                'math_intensity': classification.math_intensity,
                'data_requirements': classification.data_requirements,
                'econometric_methods': ', '.join(classification.econometric_methods),
                'confidence_primary': classification.confidence_scores.get('primary_type', 0.0),
                'confidence_math': classification.confidence_scores.get('math_intensity', 0.0),
                'confidence_data': classification.confidence_scores.get('data_requirements', 0.0),
                'reasoning': classification.reasoning,
                'selected_personas': ', '.join([p for p, w in weights.items() if w > 0]),
                'persona_weights': ', '.join([f"{p}:{w:.2f}" for p, w in sorted(weights.items(), key=lambda x: x[1], reverse=True) if w > 0])
            }

            results.append(result)

        except Exception as e:
            print(f"  ⚠️  Error processing {paper['id']}: {e}")
            results.append({
                'paper_id': paper['id'],
                'error': str(e)
            })

    return results


def export_results_csv(results: List[Dict], output_path: str):
    """Export results to CSV file."""
    if not results:
        print("No results to export")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")


def export_results_excel(results: List[Dict], output_path: str):
    """Export results to Excel file with multiple sheets."""
    if not results:
        print("No results to export")
        return

    # Main results sheet
    df_main = pd.DataFrame(results)

    # Summary statistics
    summary_data = []
    if 'primary_type' in df_main.columns:
        type_counts = df_main['primary_type'].value_counts()
        for paper_type, count in type_counts.items():
            summary_data.append({
                'Metric': f'{paper_type} Papers',
                'Value': count
            })

    df_summary = pd.DataFrame(summary_data)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_main.to_excel(writer, sheet_name='Classifications', index=False)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process academic papers for classification"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to folder of .txt files or CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Output file path (default: results.csv)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'excel'],
        default='csv',
        help='Output format (default: csv)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='CSV column containing paper text (default: text)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Use keyword-based classification only (no API calls)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    args = parser.parse_args()

    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)

    # Load papers
    print("=" * 60)
    print("LOADING PAPERS")
    print("=" * 60)

    if os.path.isdir(args.input):
        print(f"Loading papers from folder: {args.input}")
        papers = load_papers_from_folder(args.input)
    elif args.input.endswith('.csv'):
        print(f"Loading papers from CSV: {args.input}")
        papers = load_papers_from_csv(args.input, text_column=args.text_column)
    else:
        print("Error: Input must be a folder or CSV file")
        sys.exit(1)

    if not papers:
        print("Error: No papers found")
        sys.exit(1)

    print(f"✅ Loaded {len(papers)} papers\n")

    # Process papers
    print("=" * 60)
    print("PROCESSING PAPERS")
    print("=" * 60)

    use_llm = not args.no_llm
    if not use_llm:
        print("⚠️  Using keyword-based classification only (LLM disabled)\n")

    results = process_papers_batch(
        papers,
        use_llm=use_llm,
        progress=not args.quiet
    )

    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    if args.format == 'csv':
        export_results_csv(results, args.output)
    elif args.format == 'excel':
        if not args.output.endswith('.xlsx'):
            args.output = args.output.replace('.csv', '.xlsx')
        export_results_excel(results, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total papers processed: {len(results)}")

    # Count by type
    type_counts = {}
    for result in results:
        if 'primary_type' in result:
            paper_type = result['primary_type']
            type_counts[paper_type] = type_counts.get(paper_type, 0) + 1

    if type_counts:
        print("\nPapers by type:")
        for paper_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {paper_type:12} {count:3d}")

    print("\n✅ Batch processing complete!")


if __name__ == "__main__":
    main()
