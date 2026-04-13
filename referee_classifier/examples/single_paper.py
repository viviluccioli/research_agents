"""
Example: Single Paper Classification

Demonstrates classifying a single paper and viewing the results.
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from referee import classify_paper, adjust_persona_weights, PaperClassification
from referee.personas import format_persona_selection_summary


def load_paper(file_path: str) -> str:
    """Load paper text from file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Classify a single academic paper"
    )
    parser.add_argument(
        '--paper',
        type=str,
        required=True,
        help='Path to paper text file'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Use keyword-based classification only (no API call)'
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.paper):
        print(f"Error: File not found: {args.paper}")
        sys.exit(1)

    # Load paper
    print(f"Loading paper from: {args.paper}")
    paper_text = load_paper(args.paper)
    print(f"Paper length: {len(paper_text)} characters\n")

    # Classify paper
    print("=" * 60)
    print("CLASSIFYING PAPER")
    print("=" * 60)

    use_llm = not args.no_llm
    if not use_llm:
        print("⚠️  Using keyword-based classification only (LLM disabled)\n")

    classification = classify_paper(paper_text, use_llm=use_llm)

    # Display classification results
    print("\n📊 CLASSIFICATION RESULTS")
    print("-" * 60)
    print(f"Primary Type:        {classification.primary_type}")
    print(f"Math Intensity:      {classification.math_intensity}")
    print(f"Data Requirements:   {classification.data_requirements}")
    print(f"Econometric Methods: {', '.join(classification.econometric_methods) if classification.econometric_methods else 'None'}")

    print("\nConfidence Scores:")
    for dimension, score in classification.confidence_scores.items():
        print(f"  {dimension:20} {score:.2f}")

    print(f"\nReasoning:")
    print(f"  {classification.reasoning}")

    if classification.keyword_hints:
        print("\nKeyword Hints (baseline detection):")
        for paper_type, count in classification.keyword_hints.items():
            print(f"  {paper_type:12} {count:3d} matches")

    # Compute adaptive persona weights
    print("\n" + "=" * 60)
    print("ADAPTIVE PERSONA SELECTION")
    print("=" * 60)

    weights = adjust_persona_weights(classification)

    print("\nSelected Personas & Weights:")
    for persona, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0:
            bar = "█" * int(weight * 40)
            print(f"  {persona:12} {weight:.2f} {bar}")

    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(format_persona_selection_summary(classification, weights))

    print("\n✅ Classification complete!")


if __name__ == "__main__":
    main()
