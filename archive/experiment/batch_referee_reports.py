#!/usr/bin/env python3
"""
Batch Referee Report Processor

This script runs the multi-agent referee report system on a directory of PDFs
and generates a CSV and JSON file with results, matched against a ground truth CSV.

Usage:
    python batch_referee_reports.py --pdf-dir PATH --ground-truth PATH [--output-dir PATH]

Example:
    python batch_referee_reports.py \
        --pdf-dir /path/to/pdfs \
        --ground-truth /path/to/tracking.csv \
        --output-dir ./results
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add app_system to path to import referee system
sys.path.insert(0, str(Path(__file__).parent.parent / "app_system"))

from referee.engine import execute_debate_pipeline, extract_verdict_from_report, extract_score_from_report
from section_eval.text_extraction import extract_text_from_pdf


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

VERDICT_VALUES = {
    'PASS': 1.0,
    'REVISE': 0.5,
    'FAIL': 0.0,
    'REJECT': 0.0,
    'UNKNOWN': 0.0,
    '': 0.0
}


def compute_consensus_score(verdicts: List[str], weights: List[float]) -> float:
    """Compute weighted consensus score from verdicts and weights."""
    if len(verdicts) != len(weights):
        return 0.0
    score = sum(
        VERDICT_VALUES.get(v.upper().strip() if v else '', 0.0) * w
        for v, w in zip(verdicts, weights)
    )
    return round(score, 4)


def compute_agreement_level(verdicts: List[str]) -> str:
    """Compute agreement level from verdicts."""
    if not verdicts:
        return 'UNKNOWN'
    unique_verdicts = set(v.upper().strip() for v in verdicts if v)
    if len(unique_verdicts) == 1:
        return 'UNANIMOUS'
    elif len(unique_verdicts) == 2:
        return 'PARTIAL'
    else:
        return 'DIVERGENT'


def load_ground_truth(csv_path: str) -> Dict[str, str]:
    """
    Load ground truth CSV and create a mapping from doc_id to Tier.

    Args:
        csv_path: Path to ground truth CSV file

    Returns:
        Dictionary mapping doc_id to Tier value
    """
    ground_truth = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        if 'doc_id' not in reader.fieldnames or 'Tier' not in reader.fieldnames:
            raise ValueError(f"Ground truth CSV must contain 'doc_id' and 'Tier' columns. Found: {reader.fieldnames}")

        for row in reader:
            doc_id = row['doc_id']
            tier = row['Tier']
            ground_truth[doc_id] = tier

    print(f"[Ground Truth] Loaded {len(ground_truth)} entries from {csv_path}")
    return ground_truth


def extract_doc_id_from_filename(filename: str) -> str:
    """
    Extract doc_id from PDF filename.
    Assumes doc_id is the filename without extension.

    Args:
        filename: PDF filename (e.g., "paper123.pdf")

    Returns:
        doc_id (e.g., "paper123")
    """
    return Path(filename).stem


def extract_round1_verdict(persona_name: str, round1_reports: Dict[str, str]) -> str:
    """
    Extract verdict from Round 1 report for a specific persona.

    Args:
        persona_name: Name of the persona
        round1_reports: Dictionary of Round 1 reports

    Returns:
        Verdict string (PASS/REVISE/FAIL/UNKNOWN)
    """
    report = round1_reports.get(persona_name, "")
    return extract_verdict_from_report(report)


def extract_round1_score(persona_name: str, round1_reports: Dict[str, str]) -> Optional[float]:
    """
    Extract confidence score from Round 1 report for a specific persona.

    Args:
        persona_name: Name of the persona
        round1_reports: Dictionary of Round 1 reports

    Returns:
        Score as float (1-10), or None if not found
    """
    report = round1_reports.get(persona_name, "")
    return extract_score_from_report(report)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_characters(text: str) -> int:
    """Count characters in text."""
    return len(text)


def process_single_pdf(
    pdf_path: Path,
    ground_truth: Dict[str, str],
    output_dir: Path
) -> Optional[Dict]:
    """
    Process a single PDF through the referee report system.

    Args:
        pdf_path: Path to PDF file
        ground_truth: Ground truth mapping
        output_dir: Output directory for results

    Returns:
        Dictionary with results, or None if processing failed
    """
    filename = pdf_path.name
    doc_id = extract_doc_id_from_filename(filename)

    print(f"\n{'='*80}")
    print(f"[Processing] {filename} (doc_id: {doc_id})")
    print(f"{'='*80}")

    # Check if doc_id exists in ground truth
    tier = ground_truth.get(doc_id, "UNKNOWN")
    if tier == "UNKNOWN":
        print(f"[Warning] doc_id '{doc_id}' not found in ground truth CSV")
    else:
        print(f"[Ground Truth] Tier: {tier}")

    # Extract text from PDF
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        print(f"[Extraction] Reading PDF...")
        paper_text = extract_text_from_pdf(pdf_bytes)

        if not paper_text or len(paper_text.strip()) < 100:
            print(f"[Error] PDF extraction failed or text too short: {len(paper_text)} chars")
            return None

        word_count = count_words(paper_text)
        char_count = count_characters(paper_text)
        print(f"[Extraction] Success: {char_count:,} chars, {word_count:,} words")

    except Exception as e:
        print(f"[Error] Failed to extract PDF: {e}")
        return None

    # Run referee report
    try:
        start_time = time.time()

        print(f"[Debate] Starting multi-agent debate pipeline...")
        results = asyncio.run(execute_debate_pipeline(
            paper_text=paper_text,
            progress_callback=None,
            paper_type=None,  # Let LLM determine
            custom_context=None,
            manual_personas=None,
            manual_weights=None,
            enable_quote_validation=False,  # Disable for speed
            use_cache=False,
            force_refresh=True
        ))

        end_time = time.time()
        duration_seconds = end_time - start_time
        duration_formatted = f"{int(duration_seconds // 60):02d}:{int(duration_seconds % 60):02d}"

        print(f"[Debate] Complete in {duration_formatted} ({duration_seconds:.1f}s)")

    except Exception as e:
        print(f"[Error] Debate pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Extract results
    try:
        selection = results['selection']
        selected_personas = selection['selected_personas']
        weights = selection['weights']

        round1_reports = results['round_1']
        round2c_reports = results['round_2c']

        consensus = results['consensus']
        final_decision = consensus['decision']

        # Map decision to verdict
        verdict_map = {
            'ACCEPT': 'PASS',
            'REJECT AND RESUBMIT': 'REVISE',
            'REJECT': 'FAIL'
        }
        final_verdict = verdict_map.get(final_decision, final_decision)

        final_report = results['final_decision']

        # Extract persona data (support up to 3 personas)
        persona_data = []
        for i in range(3):
            if i < len(selected_personas):
                persona_name = selected_personas[i]
                persona_weight = weights.get(persona_name, 0.0)
                round1_verdict = extract_round1_verdict(persona_name, round1_reports)
                round1_score = extract_round1_score(persona_name, round1_reports)
                final_persona_verdict = consensus['verdicts'].get(persona_name, "UNKNOWN")
                final_persona_score = consensus['scores'].get(persona_name)

                persona_data.append({
                    'name': persona_name,
                    'weight': persona_weight,
                    'round1_verdict': round1_verdict,
                    'round1_score': round1_score,
                    'final_verdict': final_persona_verdict,
                    'final_score': final_persona_score
                })
            else:
                # Fill with empty data if fewer than 3 personas
                persona_data.append({
                    'name': '',
                    'weight': 0.0,
                    'round1_verdict': '',
                    'round1_score': None,
                    'final_verdict': '',
                    'final_score': None
                })

        # Compute consensus metrics
        # Get active personas (non-empty names)
        active_personas = [p for p in persona_data if p['name']]

        # Extract verdicts and weights for consensus calculation
        r1_verdicts = [p['round1_verdict'] for p in active_personas]
        r2c_verdicts = [p['final_verdict'] for p in active_personas]
        weights_list = [p['weight'] for p in active_personas]

        # Compute consensus scores (categorical)
        consensus_score_r1 = compute_consensus_score(r1_verdicts, weights_list)
        consensus_score_r2c = compute_consensus_score(r2c_verdicts, weights_list)

        # Compute agreement levels
        agreement_level_r1 = compute_agreement_level(r1_verdicts)
        agreement_level_r2c = compute_agreement_level(r2c_verdicts)

        # Compute consensus scores (numeric, if available)
        r1_scores = [p['round1_score'] for p in active_personas if p['round1_score'] is not None]
        r2c_scores = [p['final_score'] for p in active_personas if p['final_score'] is not None]

        if len(r1_scores) == len(active_personas):
            # All personas provided R1 scores, compute weighted average (normalized to 0-1)
            round1_consensus_score_numeric = sum(s / 10.0 * w for s, w in zip(r1_scores, weights_list))
        else:
            round1_consensus_score_numeric = None

        if len(r2c_scores) == len(active_personas):
            # All personas provided R2C scores, compute weighted average (normalized to 0-1)
            round2c_consensus_score_numeric = sum(s / 10.0 * w for s, w in zip(r2c_scores, weights_list))
        else:
            round2c_consensus_score_numeric = None

        # Compute numeric consensus delta if both available
        if round1_consensus_score_numeric is not None and round2c_consensus_score_numeric is not None:
            consensus_delta_numeric = round2c_consensus_score_numeric - round1_consensus_score_numeric
        else:
            consensus_delta_numeric = None

        # Compute total final score (sum of all persona final scores, max 30)
        total_final_score = sum(
            p['final_score'] for p in active_personas
            if p['final_score'] is not None
        )
        # If not all personas provided scores, set to None
        if len([p for p in active_personas if p['final_score'] is not None]) < len(active_personas):
            total_final_score = None

        # Build result dictionary
        result = {
            # Basic metadata (columns 1-6)
            'doc_id': doc_id,
            'filename': filename,
            'duration_seconds': duration_seconds,
            'duration_formatted': duration_formatted,
            'char_count': char_count,
            'word_count': word_count,

            # Persona 1 (columns 7-12)
            'persona_1_name': persona_data[0]['name'],
            'persona_1_weight': persona_data[0]['weight'],
            'persona_1_round1_verdict': persona_data[0]['round1_verdict'],
            'persona_1_final_verdict': persona_data[0]['final_verdict'],
            'persona_1_round1_score': persona_data[0]['round1_score'],
            'persona_1_final_score': persona_data[0]['final_score'],

            # Persona 2 (columns 13-18)
            'persona_2_name': persona_data[1]['name'],
            'persona_2_weight': persona_data[1]['weight'],
            'persona_2_round1_verdict': persona_data[1]['round1_verdict'],
            'persona_2_final_verdict': persona_data[1]['final_verdict'],
            'persona_2_round1_score': persona_data[1]['round1_score'],
            'persona_2_final_score': persona_data[1]['final_score'],

            # Persona 3 (columns 19-24)
            'persona_3_name': persona_data[2]['name'],
            'persona_3_weight': persona_data[2]['weight'],
            'persona_3_round1_verdict': persona_data[2]['round1_verdict'],
            'persona_3_final_verdict': persona_data[2]['final_verdict'],
            'persona_3_round1_score': persona_data[2]['round1_score'],
            'persona_3_final_score': persona_data[2]['final_score'],

            # Final results (columns 25-28)
            'final_verdict': final_verdict,
            'total_final_score': total_final_score,  # Right next to verdict for easy comparison
            'final_report': final_report,
            'tier': tier,

            # Consensus scores (columns 29-32)
            'consensus_score_r1': consensus_score_r1,
            'consensus_score_r2c': consensus_score_r2c,
            'agreement_level_r1': agreement_level_r1,
            'agreement_level_r2c': agreement_level_r2c,

            # Numeric consensus scores (columns 33-35)
            'round1_consensus_score_numeric': round(round1_consensus_score_numeric, 4) if round1_consensus_score_numeric is not None else None,
            'round2c_consensus_score_numeric': round(round2c_consensus_score_numeric, 4) if round2c_consensus_score_numeric is not None else None,
            'consensus_delta_numeric': round(consensus_delta_numeric, 4) if consensus_delta_numeric is not None else None,
        }

        print(f"[Result] Verdict: {final_verdict} | Personas: {', '.join([p['name'] for p in persona_data if p['name']])} | Duration: {duration_formatted}")

        return result

    except Exception as e:
        print(f"[Error] Failed to extract results: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(results: List[Dict], output_dir: Path):
    """
    Save results to CSV and JSON files.

    Args:
        results: List of result dictionaries
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # CSV output
    csv_path = output_dir / f"referee_batch_results_{timestamp}.csv"

    # Define CSV columns in desired order (35 total columns)
    csv_columns = [
        # Basic metadata (6 columns)
        'doc_id',
        'filename',
        'duration_seconds',
        'duration_formatted',
        'char_count',
        'word_count',

        # Persona 1 (6 columns)
        'persona_1_name',
        'persona_1_weight',
        'persona_1_round1_verdict',
        'persona_1_final_verdict',
        'persona_1_round1_score',
        'persona_1_final_score',

        # Persona 2 (6 columns)
        'persona_2_name',
        'persona_2_weight',
        'persona_2_round1_verdict',
        'persona_2_final_verdict',
        'persona_2_round1_score',
        'persona_2_final_score',

        # Persona 3 (6 columns)
        'persona_3_name',
        'persona_3_weight',
        'persona_3_round1_verdict',
        'persona_3_final_verdict',
        'persona_3_round1_score',
        'persona_3_final_score',

        # Final results (4 columns) - total_final_score moved next to final_verdict
        'final_verdict',
        'total_final_score',  # Right next to verdict for easy comparison
        'final_report',
        'tier',

        # Consensus scores - categorical (4 columns)
        'consensus_score_r1',
        'consensus_score_r2c',
        'agreement_level_r1',
        'agreement_level_r2c',

        # Consensus scores - numeric (3 columns)
        'round1_consensus_score_numeric',
        'round2c_consensus_score_numeric',
        'consensus_delta_numeric',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[Output] CSV saved: {csv_path}")

    # JSON output (includes full final_report text)
    json_path = output_dir / f"referee_batch_results_{timestamp}.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[Output] JSON saved: {json_path}")

    # Summary statistics
    total = len(results)
    if total > 0:
        avg_duration = sum(r['duration_seconds'] for r in results) / total
        total_duration = sum(r['duration_seconds'] for r in results)

        verdict_counts = {}
        for r in results:
            verdict = r['final_verdict']
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total papers processed: {total}")
        print(f"Average duration: {int(avg_duration // 60):02d}:{int(avg_duration % 60):02d}")
        print(f"Total duration: {int(total_duration // 60):02d}:{int(total_duration % 60):02d}")
        print(f"\nVerdict distribution:")
        for verdict, count in sorted(verdict_counts.items()):
            print(f"  {verdict}: {count} ({count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process PDFs through the referee report system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python batch_referee_reports.py \\
      --pdf-dir /path/to/pdfs \\
      --ground-truth /path/to/tracking.csv \\
      --output-dir ./results
        """
    )

    parser.add_argument(
        '--pdf-dir',
        type=str,
        required=True,
        help='Path to directory containing PDF files'
    )

    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Path to ground truth CSV file (must contain doc_id and Tier columns)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Path to output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of PDFs to process (for testing)'
    )

    parser.add_argument(
        '--start-from',
        type=str,
        default=None,
        help='Resume from a specific PDF file (e.g., "ifdp-2020-4.pdf")'
    )

    args = parser.parse_args()

    # Validate inputs
    pdf_dir = Path(args.pdf_dir)
    ground_truth_path = Path(args.ground_truth)
    output_dir = Path(args.output_dir)

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(f"Error: PDF directory not found: {pdf_dir}")
        sys.exit(1)

    if not ground_truth_path.exists():
        print(f"Error: Ground truth CSV not found: {ground_truth_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    try:
        ground_truth = load_ground_truth(str(ground_truth_path))
    except Exception as e:
        print(f"Error loading ground truth CSV: {e}")
        sys.exit(1)

    # Find all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"Error: No PDF files found in {pdf_dir}")
        sys.exit(1)

    # Handle --start-from flag
    if args.start_from:
        start_indices = [i for i, p in enumerate(pdf_files) if p.name == args.start_from]
        if not start_indices:
            print(f"Error: Start file '{args.start_from}' not found in PDF directory")
            print(f"Available files: {', '.join(p.name for p in pdf_files[:10])}{'...' if len(pdf_files) > 10 else ''}")
            sys.exit(1)
        start_idx = start_indices[0]
        pdf_files = pdf_files[start_idx:]
        print(f"\n[Resume] Starting from '{args.start_from}' (skipping {start_idx} files)")

    if args.limit:
        pdf_files = pdf_files[:args.limit]
        print(f"\n[Limit] Processing first {args.limit} PDFs")

    print(f"\n[Start] Found {len(pdf_files)} PDF files to process")
    print(f"[Start] Output directory: {output_dir}")

    # Process each PDF
    results = []
    start_time = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[Progress] {i}/{len(pdf_files)}")

        result = process_single_pdf(pdf_path, ground_truth, output_dir)

        if result:
            results.append(result)
        else:
            # Add failure record
            doc_id = extract_doc_id_from_filename(pdf_path.name)
            results.append({
                'doc_id': doc_id,
                'filename': pdf_path.name,
                'duration_seconds': 0,
                'duration_formatted': '00:00',
                'char_count': 0,
                'word_count': 0,

                # Persona 1
                'persona_1_name': '',
                'persona_1_weight': 0.0,
                'persona_1_round1_verdict': '',
                'persona_1_final_verdict': '',
                'persona_1_round1_score': None,
                'persona_1_final_score': None,

                # Persona 2
                'persona_2_name': '',
                'persona_2_weight': 0.0,
                'persona_2_round1_verdict': '',
                'persona_2_final_verdict': '',
                'persona_2_round1_score': None,
                'persona_2_final_score': None,

                # Persona 3
                'persona_3_name': '',
                'persona_3_weight': 0.0,
                'persona_3_round1_verdict': '',
                'persona_3_final_verdict': '',
                'persona_3_round1_score': None,
                'persona_3_final_score': None,

                # Final results
                'final_verdict': 'ERROR',
                'total_final_score': None,  # Right next to verdict
                'final_report': '',
                'tier': ground_truth.get(doc_id, 'UNKNOWN'),

                # Consensus scores
                'consensus_score_r1': 0.0,
                'consensus_score_r2c': 0.0,
                'agreement_level_r1': 'UNKNOWN',
                'agreement_level_r2c': 'UNKNOWN',

                # Numeric consensus scores
                'round1_consensus_score_numeric': None,
                'round2c_consensus_score_numeric': None,
                'consensus_delta_numeric': None,
            })

    end_time = time.time()
    total_duration = end_time - start_time

    # Save results
    save_results(results, output_dir)

    print(f"\n{'='*80}")
    print(f"COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {int(total_duration // 3600):02d}:{int((total_duration % 3600) // 60):02d}:{int(total_duration % 60):02d}")
    print(f"Successful: {sum(1 for r in results if r.get('success', False))}/{len(results)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
