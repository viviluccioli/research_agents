#!/usr/bin/env python3
"""
Persona Selection Consistency Experiment

This script runs Round 0 (persona selection) from the Experiment 4 referee system
multiple times on the same paper to test consistency. The goal is to determine if
the persona selection is stable and whether configuration parameters need calibration.

Usage:
    python run_persona_selection_experiment.py <paper_file_path> [--runs 10] [--output results.csv]
"""

import sys
import os
import argparse
import asyncio
import json
import csv
import datetime
from pathlib import Path
from typing import Dict, List

# Add app_system to path to import required modules
sys.path.insert(0, str(Path(__file__).parent.parent / "app_system"))

from referee.engine_exp_4 import run_round_0_selection
from config import MODEL_PRIMARY, MODEL_SECONDARY, API_BASE
import tiktoken

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

class ExperimentConfig:
    """Configuration for the persona selection experiment."""

    # Model Configuration
    MODEL = MODEL_SECONDARY  # Claude 4.5 Sonnet (used by referee system)
    TEMPERATURE = 1.0  # Default temperature for thinking mode
    MAX_TOKENS = 4096  # Maximum tokens for response

    # Persona Selection Parameters
    NUM_PERSONAS_TO_SELECT = 3  # Always select 3 personas
    NUM_AVAILABLE_PERSONAS = 10  # Total available personas

    # Experiment Parameters
    DEFAULT_NUM_RUNS = 10  # Number of times to run selection

    # API Configuration
    API_ENDPOINT = API_BASE

    # Thinking Mode (used by referee system)
    THINKING_ENABLED = True
    THINKING_BUDGET_TOKENS = 2048

    # Available personas (must match engine_exp_4.py)
    AVAILABLE_PERSONAS = [
        "Theorist",
        "Econometrician",
        "ML_Expert",
        "Data_Scientist",
        "CS_Expert",
        "Historian",
        "Visionary",
        "Policymaker",
        "Ethicist",
        "Perspective"
    ]

    @classmethod
    def to_dict(cls) -> Dict:
        """Return configuration as dictionary for metadata."""
        return {
            "model": cls.MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "num_personas_to_select": cls.NUM_PERSONAS_TO_SELECT,
            "num_available_personas": cls.NUM_AVAILABLE_PERSONAS,
            "thinking_enabled": cls.THINKING_ENABLED,
            "thinking_budget_tokens": cls.THINKING_BUDGET_TOKENS,
            "api_endpoint": cls.API_ENDPOINT,
            "available_personas": ", ".join(cls.AVAILABLE_PERSONAS)
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_paper(paper_path: str) -> tuple[str, str]:
    """
    Load paper from file path.

    Args:
        paper_path: Path to paper file (PDF, TXT, TEX)

    Returns:
        Tuple of (paper_text, paper_title)
    """
    paper_path = Path(paper_path)

    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")

    # Extract title from filename
    paper_title = paper_path.stem

    # Load based on file type
    if paper_path.suffix.lower() == '.pdf':
        # Use PyMuPDF if available, otherwise pdfplumber
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(paper_path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            paper_text = "\n\n".join(text_parts)
            doc.close()
        except ImportError:
            import pdfplumber
            with pdfplumber.open(paper_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                paper_text = "\n\n".join(text_parts)

    elif paper_path.suffix.lower() in ['.txt', '.tex']:
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_text = f.read()

    else:
        raise ValueError(f"Unsupported file type: {paper_path.suffix}")

    if not paper_text or len(paper_text.strip()) < 100:
        raise ValueError(f"Paper text too short or empty: {len(paper_text)} characters")

    return paper_text, paper_title


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        # Fallback approximation
        return len(text) // 4


async def run_single_selection(
    paper_text: str,
    run_number: int,
    config: ExperimentConfig
) -> Dict:
    """
    Run a single persona selection.

    Args:
        paper_text: The paper text
        run_number: Run number (for logging)
        config: Experiment configuration

    Returns:
        Dictionary with selection results
    """
    print(f"\n{'='*60}")
    print(f"Run {run_number}")
    print(f"{'='*60}")

    start_time = datetime.datetime.now()

    # Run Round 0 selection
    selection_data = await run_round_0_selection(
        paper_text=paper_text,
        N=config.NUM_PERSONAS_TO_SELECT,
        paper_type=None,  # Let LLM decide based on content
        custom_context=None,  # No custom context
        manual_personas=None,  # Fully automatic
        manual_weights=None  # Fully automatic
    )

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Extract results
    selected_personas = selection_data["selected_personas"]
    weights = selection_data["weights"]
    justification = selection_data["justification"]

    print(f"Selected: {', '.join(selected_personas)}")
    print(f"Weights: {weights}")
    print(f"Duration: {duration:.2f}s")

    return {
        "run_number": run_number,
        "selected_personas": selected_personas,
        "weights": weights,
        "justification": justification,
        "timestamp": start_time.isoformat(),
        "duration_seconds": duration
    }


def format_results_for_csv(
    paper_title: str,
    results: List[Dict],
    config: ExperimentConfig
) -> List[Dict]:
    """
    Format results for CSV output.

    Args:
        paper_title: Title of the paper
        results: List of selection results
        config: Experiment configuration

    Returns:
        List of dictionaries ready for CSV writing
    """
    csv_rows = []

    for result in results:
        personas = result["selected_personas"]
        weights = result["weights"]

        # Ensure we have exactly 3 personas
        while len(personas) < 3:
            personas.append("")

        row = {
            "run_number": result["run_number"],
            "paper_title": paper_title,
            "persona_1": personas[0] if len(personas) > 0 else "",
            "persona_1_weight": f"{weights.get(personas[0], 0.0):.4f}" if len(personas) > 0 else "",
            "persona_2": personas[1] if len(personas) > 1 else "",
            "persona_2_weight": f"{weights.get(personas[1], 0.0):.4f}" if len(personas) > 1 else "",
            "persona_3": personas[2] if len(personas) > 2 else "",
            "persona_3_weight": f"{weights.get(personas[2], 0.0):.4f}" if len(personas) > 2 else "",
            "justification": result["justification"],
            "timestamp": result["timestamp"],
            "duration_seconds": f"{result['duration_seconds']:.2f}"
        }

        csv_rows.append(row)

    return csv_rows


def analyze_consistency(results: List[Dict]) -> Dict:
    """
    Analyze consistency of persona selections across runs.

    Args:
        results: List of selection results

    Returns:
        Dictionary with consistency metrics
    """
    from collections import Counter

    # Count frequency of each persona being selected
    persona_counts = Counter()
    persona_as_first = Counter()
    persona_weights_sum = {}

    # Track unique combinations
    combinations = []

    for result in results:
        personas = result["selected_personas"]
        weights = result["weights"]

        # Count persona frequency
        for persona in personas:
            persona_counts[persona] += 1
            if persona not in persona_weights_sum:
                persona_weights_sum[persona] = []
            persona_weights_sum[persona].append(weights.get(persona, 0.0))

        # Count first persona
        if personas:
            persona_as_first[personas[0]] += 1

        # Track combination (sorted for comparison)
        combo = tuple(sorted(personas))
        combinations.append(combo)

    # Calculate metrics
    unique_combinations = len(set(combinations))
    most_common_combo = Counter(combinations).most_common(1)[0] if combinations else (None, 0)

    # Calculate average weights
    avg_weights = {}
    for persona, weight_list in persona_weights_sum.items():
        avg_weights[persona] = sum(weight_list) / len(weight_list)

    return {
        "total_runs": len(results),
        "unique_combinations": unique_combinations,
        "most_common_combination": {
            "personas": list(most_common_combo[0]) if most_common_combo[0] else None,
            "frequency": most_common_combo[1],
            "percentage": (most_common_combo[1] / len(results) * 100) if results else 0
        },
        "persona_selection_frequency": dict(persona_counts),
        "persona_as_first_frequency": dict(persona_as_first),
        "average_weights": avg_weights
    }


def save_metadata(
    output_dir: Path,
    paper_title: str,
    paper_tokens: int,
    config: ExperimentConfig,
    consistency_analysis: Dict,
    total_duration: float,
    paper_filename: str = None
):
    """
    Save experiment metadata to JSON file.

    Args:
        output_dir: Output directory
        paper_title: Paper title
        paper_tokens: Number of tokens in paper
        config: Experiment configuration
        consistency_analysis: Consistency analysis results
        total_duration: Total experiment duration in seconds
        paper_filename: Filename (without extension) to use in metadata filename
    """
    metadata = {
        "experiment": {
            "name": "Persona Selection Consistency",
            "date": datetime.datetime.now().isoformat(),
            "total_duration_seconds": total_duration
        },
        "paper": {
            "title": paper_title,
            "filename": paper_filename or paper_title,
            "tokens": paper_tokens
        },
        "configuration": config.to_dict(),
        "results": {
            "consistency_analysis": consistency_analysis
        }
    }

    # Use paper_filename if provided, otherwise fall back to paper_title
    filename_base = paper_filename if paper_filename else paper_title
    metadata_path = output_dir / f"{filename_base}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Metadata saved to: {metadata_path}")


def print_consistency_summary(analysis: Dict):
    """Print a summary of consistency analysis."""
    print("\n" + "="*60)
    print("CONSISTENCY ANALYSIS")
    print("="*60)

    print(f"\nTotal runs: {analysis['total_runs']}")
    print(f"Unique combinations: {analysis['unique_combinations']}")

    print(f"\nMost common combination:")
    combo = analysis['most_common_combination']
    if combo['personas']:
        print(f"  Personas: {', '.join(combo['personas'])}")
        print(f"  Frequency: {combo['frequency']}/{analysis['total_runs']} ({combo['percentage']:.1f}%)")

    print(f"\nPersona selection frequency:")
    for persona, count in sorted(
        analysis['persona_selection_frequency'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / analysis['total_runs']) * 100
        avg_weight = analysis['average_weights'].get(persona, 0.0)
        print(f"  {persona:20s}: {count:2d}/{analysis['total_runs']} ({percentage:5.1f}%) | Avg weight: {avg_weight:.3f}")

    print(f"\nPersona selected as first (highest weight):")
    for persona, count in sorted(
        analysis['persona_as_first_frequency'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / analysis['total_runs']) * 100
        print(f"  {persona:20s}: {count:2d}/{analysis['total_runs']} ({percentage:5.1f}%)")


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

async def run_experiment(
    paper_path: str,
    num_runs: int,
    output_path: str
):
    """
    Run the persona selection experiment.

    Args:
        paper_path: Path to paper file
        num_runs: Number of runs
        output_path: Output CSV path (will be modified to use timestamped subdirectory)
    """
    config = ExperimentConfig()

    print("\n" + "="*60)
    print("PERSONA SELECTION CONSISTENCY EXPERIMENT")
    print("="*60)

    # Load paper
    print(f"\nLoading paper from: {paper_path}")
    paper_text, paper_title = load_paper(paper_path)
    paper_tokens = count_tokens(paper_text)

    # Create timestamped subdirectory for this run
    # Extract papername from file (e.g., kd.pdf -> kd)
    paper_filename = Path(paper_path).stem  # Gets filename without extension
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{paper_filename}-{timestamp}"

    # Determine results directory structure
    if output_path:
        output_path = Path(output_path)
        # If user provided an output path, use its parent as results base
        results_base = output_path.parent
    else:
        results_base = Path(__file__).parent / "results"

    # Create run-specific subdirectory
    output_dir = results_base / run_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update output_path to use the new subdirectory
    output_path = output_dir / f"{paper_filename}_persona_selection.csv"

    print(f"Results will be saved to: {output_dir}")

    print(f"Paper title: {paper_title}")
    print(f"Paper length: {len(paper_text)} characters, ~{paper_tokens} tokens")

    # Print configuration
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    for key, value in config.to_dict().items():
        print(f"{key:30s}: {value}")

    # Run selections
    print("\n" + "="*60)
    print(f"RUNNING {num_runs} SELECTION ROUNDS")
    print("="*60)

    experiment_start = datetime.datetime.now()
    results = []

    for i in range(1, num_runs + 1):
        try:
            result = await run_single_selection(paper_text, i, config)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error in run {i}: {e}")
            import traceback
            traceback.print_exc()

    experiment_end = datetime.datetime.now()
    total_duration = (experiment_end - experiment_start).total_seconds()

    # Analyze consistency
    print("\n" + "="*60)
    print("ANALYZING RESULTS")
    print("="*60)

    consistency_analysis = analyze_consistency(results)
    print_consistency_summary(consistency_analysis)

    # Save results (output_dir and output_path already set up at start of function)
    # Save CSV
    csv_rows = format_results_for_csv(paper_title, results, config)

    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            "run_number",
            "paper_title",
            "persona_1",
            "persona_1_weight",
            "persona_2",
            "persona_2_weight",
            "persona_3",
            "persona_3_weight",
            "justification",
            "timestamp",
            "duration_seconds"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✓ Results saved to: {output_path}")

    # Save metadata
    save_metadata(
        output_dir,
        paper_title,
        paper_tokens,
        config,
        consistency_analysis,
        total_duration,
        paper_filename
    )

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Average time per run: {total_duration/num_runs:.1f} seconds")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run persona selection consistency experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 times (default)
  python run_persona_selection_experiment.py paper.pdf

  # Run 20 times with custom output
  python run_persona_selection_experiment.py paper.pdf --runs 20 --output results/my_results.csv

  # Run with LaTeX source
  python run_persona_selection_experiment.py paper.tex --runs 15
        """
    )

    parser.add_argument(
        "paper_path",
        help="Path to paper file (PDF, TXT, or TEX)"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=ExperimentConfig.DEFAULT_NUM_RUNS,
        help=f"Number of selection runs (default: {ExperimentConfig.DEFAULT_NUM_RUNS})"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output directory base (default: results/). A timestamped subdirectory will be created."
    )

    args = parser.parse_args()

    # Note: output path handling is done in run_experiment() which creates timestamped subdirectories

    # Run experiment
    try:
        asyncio.run(run_experiment(
            args.paper_path,
            args.runs,
            args.output
        ))
    except KeyboardInterrupt:
        print("\n\n❌ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
