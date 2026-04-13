"""
Adaptive Persona Selection System

Adjusts referee persona weights based on automatic paper classification.

Persona types:
- Theorist: Mathematical rigor, proofs, formal models
- Empiricist: Data quality, econometrics, identification
- Historian: Literature context, historical development
- Visionary: Novelty, impact, future directions
- Policymaker: Real-world application, welfare, policy relevance
"""

from typing import Dict, List
from referee._utils.paper_classifier import PaperClassification


# Available personas
PERSONAS = ["Theorist", "Empiricist", "Historian", "Visionary", "Policymaker"]

# Confidence threshold for using adaptive weights
CONFIDENCE_THRESHOLD = 0.7


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize persona weights to sum to 1.0.

    Args:
        weights: Dictionary of persona weights

    Returns:
        Normalized weights summing to 1.0
    """
    total = sum(weights.values())
    if total == 0:
        # Equal weights if all zero
        return {persona: 1.0 / len(weights) for persona in weights}

    return {persona: weight / total for persona, weight in weights.items()}


def adjust_persona_weights(classification: PaperClassification) -> Dict[str, float]:
    """
    Adjust persona weights based on paper classification.

    Logic:
    - Theory + High Math → Heavy Theorist, Zero/Low Policymaker
    - Empirical + Heavy Data → Heavy Empiricist, High Visionary
    - Survey → Heavy Historian, High Visionary
    - Policy → Balanced Policymaker + Empiricist
    - Low Confidence → Revert to balanced weights (0.2 each)

    Args:
        classification: PaperClassification object

    Returns:
        Dictionary mapping persona names to weights (summing to 1.0)
    """
    # Base weights (all equal)
    weights = {persona: 0.2 for persona in PERSONAS}

    # Check confidence threshold
    primary_type_confidence = classification.confidence_scores.get('primary_type', 0.0)

    if primary_type_confidence < CONFIDENCE_THRESHOLD:
        print(f"[Personas] Low confidence ({primary_type_confidence:.2f}), using balanced weights")
        return weights

    # Adjust based on primary type and other dimensions
    primary_type = classification.primary_type
    math_intensity = classification.math_intensity
    data_requirements = classification.data_requirements

    if primary_type == "Theory":
        if math_intensity == "High":
            # Pure theory with heavy math
            weights["Theorist"] = 0.5
            weights["Historian"] = 0.25
            weights["Visionary"] = 0.25
            weights["Empiricist"] = 0.0
            weights["Policymaker"] = 0.0
        elif math_intensity == "Medium":
            # Theory with moderate math
            weights["Theorist"] = 0.4
            weights["Historian"] = 0.2
            weights["Visionary"] = 0.2
            weights["Empiricist"] = 0.1
            weights["Policymaker"] = 0.1
        else:
            # Conceptual theory
            weights["Theorist"] = 0.3
            weights["Historian"] = 0.25
            weights["Visionary"] = 0.25
            weights["Policymaker"] = 0.2
            weights["Empiricist"] = 0.0

    elif primary_type == "Empirical":
        if data_requirements == "Heavy":
            # Data-intensive empirical work
            weights["Empiricist"] = 0.5
            weights["Visionary"] = 0.25
            weights["Historian"] = 0.15
            weights["Policymaker"] = 0.1
            weights["Theorist"] = 0.0
        elif data_requirements == "Light":
            # Light empirical work
            weights["Empiricist"] = 0.35
            weights["Visionary"] = 0.25
            weights["Historian"] = 0.2
            weights["Policymaker"] = 0.1
            weights["Theorist"] = 0.1
        else:
            # Empirical but no clear data (unusual)
            weights["Empiricist"] = 0.3
            weights["Historian"] = 0.25
            weights["Visionary"] = 0.25
            weights["Theorist"] = 0.1
            weights["Policymaker"] = 0.1

    elif primary_type == "Survey":
        # Literature review / meta-analysis
        weights["Historian"] = 0.6
        weights["Visionary"] = 0.4
        weights["Empiricist"] = 0.0
        weights["Theorist"] = 0.0
        weights["Policymaker"] = 0.0

    elif primary_type == "Policy":
        # Policy analysis
        weights["Policymaker"] = 0.4
        weights["Empiricist"] = 0.3
        weights["Historian"] = 0.2
        weights["Visionary"] = 0.1
        weights["Theorist"] = 0.0

    else:
        # Unknown type, use balanced weights
        print(f"[Personas] Unknown primary type '{primary_type}', using balanced weights")
        return weights

    # Normalize to ensure sum = 1.0
    weights = normalize_weights(weights)

    # Filter out zero-weight personas and select top 3
    non_zero = {p: w for p, w in weights.items() if w > 0.0}

    if len(non_zero) > 3:
        # Select top 3 by weight
        top_3 = sorted(non_zero.items(), key=lambda x: x[1], reverse=True)[:3]
        selected_personas = {p: w for p, w in top_3}
        # Renormalize
        selected_personas = normalize_weights(selected_personas)
    else:
        selected_personas = non_zero

    # If fewer than 3 personas, add Historian as default filler
    if len(selected_personas) < 3:
        remaining = [p for p in PERSONAS if p not in selected_personas]
        # Add Historian first, then others
        if "Historian" in remaining:
            selected_personas["Historian"] = 0.1
            remaining.remove("Historian")

        # Add more if still < 3
        while len(selected_personas) < 3 and remaining:
            selected_personas[remaining.pop(0)] = 0.1

        # Renormalize
        selected_personas = normalize_weights(selected_personas)

    print(f"[Personas] Adjusted weights: {selected_personas}")
    return selected_personas


def get_selected_personas(weights: Dict[str, float]) -> List[str]:
    """
    Extract list of selected personas (non-zero weights).

    Args:
        weights: Dictionary of persona weights

    Returns:
        List of persona names with non-zero weights
    """
    return [persona for persona, weight in weights.items() if weight > 0.0]


def format_persona_selection_summary(
    classification: PaperClassification,
    weights: Dict[str, float]
) -> str:
    """
    Format a readable summary of persona selection.

    Args:
        classification: Paper classification results
        weights: Adjusted persona weights

    Returns:
        Formatted string summary
    """
    selected = get_selected_personas(weights)

    summary = f"""**Paper Classification:**
- Primary Type: {classification.primary_type} (confidence: {classification.confidence_scores.get('primary_type', 0.0):.2f})
- Math Intensity: {classification.math_intensity}
- Data Requirements: {classification.data_requirements}
- Econometric Methods: {', '.join(classification.econometric_methods) if classification.econometric_methods else 'None'}

**Reasoning:** {classification.reasoning}

**Selected Personas:**
"""
    for persona in selected:
        weight = weights[persona]
        summary += f"- {persona}: {weight:.2f}\n"

    return summary
