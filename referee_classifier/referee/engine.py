"""
Modified Multi-Agent Debate Engine with Paper Classification

This is an adapted version of app_system/referee/engine.py that integrates
automatic paper classification into Round 0 persona selection.

Key differences from original:
1. Round 0 now includes paper classification step
2. Persona weights are adjusted based on classification
3. Classification results are stored in debate metadata
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent app_system to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app_system"))

from utils import single_query
from referee._utils.paper_classifier import classify_paper, PaperClassification
from referee.personas import adjust_persona_weights, format_persona_selection_summary


async def run_round_0_with_classification(
    paper_text: str,
    use_classification: bool = True,
    force_manual_personas: list = None,
    force_manual_weights: dict = None
) -> dict:
    """
    Round 0: Paper classification followed by adaptive persona selection.

    Args:
        paper_text: The paper to evaluate
        use_classification: Whether to use automatic classification (default: True)
        force_manual_personas: Optional list of manually selected personas (bypasses classification)
        force_manual_weights: Optional dict of manually specified weights

    Returns:
        Dictionary with:
        - selected_personas: List[str]
        - weights: Dict[str, float]
        - justification: str
        - classification: PaperClassification (if use_classification=True)
    """
    print("[Round 0] Starting paper classification and persona selection...")

    # If manual selection is forced, skip classification
    if force_manual_personas:
        if force_manual_weights:
            print(f"[Round 0] Using manual personas: {force_manual_personas} with manual weights")
            return {
                "selected_personas": force_manual_personas,
                "weights": force_manual_weights,
                "justification": "Manually selected by user with specified weights.",
                "classification": None
            }
        else:
            # Manual personas but automatic weights (unusual, but supported)
            equal_weight = 1.0 / len(force_manual_personas)
            weights = {p: equal_weight for p in force_manual_personas}
            return {
                "selected_personas": force_manual_personas,
                "weights": weights,
                "justification": "Manually selected personas with equal weights.",
                "classification": None
            }

    # Automatic classification-based selection
    if use_classification:
        try:
            # Classify paper
            print("[Round 0] Classifying paper...")
            classification = classify_paper(paper_text, use_llm=True)

            print(f"[Round 0] Classification: {classification.primary_type} "
                  f"(math: {classification.math_intensity}, "
                  f"data: {classification.data_requirements})")

            # Adjust persona weights based on classification
            weights = adjust_persona_weights(classification)

            # Extract selected personas (non-zero weights)
            selected_personas = [p for p, w in weights.items() if w > 0.0]

            justification = format_persona_selection_summary(classification, weights)

            return {
                "selected_personas": selected_personas,
                "weights": weights,
                "justification": justification,
                "classification": classification
            }

        except Exception as e:
            print(f"[Round 0] Classification failed: {e}")
            print("[Round 0] Falling back to balanced weights")

            # Fallback to balanced weights
            fallback_personas = ["Empiricist", "Historian", "Visionary"]
            fallback_weights = {p: 1.0/3.0 for p in fallback_personas}

            return {
                "selected_personas": fallback_personas,
                "weights": fallback_weights,
                "justification": f"Fallback selection due to classification error: {e}",
                "classification": None
            }
    else:
        # Classification disabled, use balanced weights
        default_personas = ["Empiricist", "Historian", "Visionary"]
        default_weights = {p: 1.0/3.0 for p in default_personas}

        return {
            "selected_personas": default_personas,
            "weights": default_weights,
            "justification": "Classification disabled, using default balanced weights.",
            "classification": None
        }


async def execute_debate_pipeline(
    paper_text: str,
    use_classification: bool = True,
    force_manual_personas: list = None,
    force_manual_weights: dict = None,
    progress_callback=None
):
    """
    Execute multi-agent debate pipeline with paper classification.

    This is a simplified version for the classifier experiment.
    For full production pipeline, use app_system/referee/engine.py

    Args:
        paper_text: The paper to evaluate
        use_classification: Whether to use automatic classification
        force_manual_personas: Optional manual persona selection
        force_manual_weights: Optional manual weights
        progress_callback: Optional progress reporting function

    Returns:
        Dictionary with:
        - round_0: Selection data with classification
        - classification: PaperClassification object
        - selected_personas: List of personas
        - weights: Dict of weights
    """
    if progress_callback:
        progress_callback("Round 0: Classifying and selecting personas", 0.1)

    # Run Round 0 with classification
    round_0_results = await run_round_0_with_classification(
        paper_text,
        use_classification=use_classification,
        force_manual_personas=force_manual_personas,
        force_manual_weights=force_manual_weights
    )

    if progress_callback:
        progress_callback("Classification complete", 1.0)

    # Return results (in real pipeline, this would continue with debate rounds)
    return {
        "round_0": round_0_results,
        "classification": round_0_results.get("classification"),
        "selected_personas": round_0_results["selected_personas"],
        "weights": round_0_results["weights"],
        "metadata": {
            "classification_enabled": use_classification,
            "manual_override": force_manual_personas is not None
        }
    }


def sync_execute_debate_pipeline(*args, **kwargs):
    """
    Synchronous wrapper for execute_debate_pipeline.

    Useful for non-async contexts.
    """
    return asyncio.run(execute_debate_pipeline(*args, **kwargs))
