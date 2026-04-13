# -*- coding: utf-8 -*-
"""
Unit tests for adaptive persona selection.

Tests weight adjustment logic for different paper types.
"""

import pytest
from referee._utils.paper_classifier import PaperClassification
from referee.personas import (
    adjust_persona_weights,
    normalize_weights,
    get_selected_personas
)


def create_mock_classification(
    primary_type: str,
    math_intensity: str = "Medium",
    data_requirements: str = "Light",
    confidence: float = 0.9
) -> PaperClassification:
    """Create a mock classification for testing."""
    return PaperClassification(
        primary_type=primary_type,
        math_intensity=math_intensity,
        data_requirements=data_requirements,
        econometric_methods=[],
        confidence_scores={
            'primary_type': confidence,
            'math_intensity': confidence,
            'data_requirements': confidence
        },
        reasoning="Mock classification for testing"
    )


def test_theory_high_math_weights():
    """Test weights for theory paper with high math."""
    classification = create_mock_classification(
        primary_type="Theory",
        math_intensity="High",
        data_requirements="None"
    )

    weights = adjust_persona_weights(classification)

    # Should heavily weight Theorist
    assert weights['Theorist'] >= 0.4
    # Should minimize/eliminate Empiricist and Policymaker
    assert weights.get('Empiricist', 0.0) == 0.0
    assert weights.get('Policymaker', 0.0) == 0.0
    # Should have Historian and Visionary
    assert weights.get('Historian', 0.0) > 0.0


def test_empirical_heavy_data_weights():
    """Test weights for empirical paper with heavy data."""
    classification = create_mock_classification(
        primary_type="Empirical",
        math_intensity="Low",
        data_requirements="Heavy"
    )

    weights = adjust_persona_weights(classification)

    # Should heavily weight Empiricist
    assert weights['Empiricist'] >= 0.4
    # Should include Visionary
    assert weights.get('Visionary', 0.0) > 0.0
    # Theorist should be minimal or zero
    assert weights.get('Theorist', 0.0) <= 0.1


def test_survey_paper_weights():
    """Test weights for survey/literature review paper."""
    classification = create_mock_classification(
        primary_type="Survey",
        math_intensity="Low",
        data_requirements="None"
    )

    weights = adjust_persona_weights(classification)

    # Should heavily weight Historian
    assert weights['Historian'] >= 0.5
    # Should include Visionary
    assert weights['Visionary'] >= 0.3
    # Should minimize technical roles
    assert weights.get('Empiricist', 0.0) == 0.0
    assert weights.get('Theorist', 0.0) == 0.0


def test_policy_paper_weights():
    """Test weights for policy paper."""
    classification = create_mock_classification(
        primary_type="Policy",
        math_intensity="Low",
        data_requirements="Light"
    )

    weights = adjust_persona_weights(classification)

    # Should emphasize Policymaker
    assert weights['Policymaker'] >= 0.3
    # Should include Empiricist
    assert weights['Empiricist'] >= 0.2


def test_low_confidence_fallback():
    """Test fallback to balanced weights when confidence < 0.7."""
    classification = create_mock_classification(
        primary_type="Theory",
        math_intensity="High",
        confidence=0.5  # Low confidence
    )

    weights = adjust_persona_weights(classification)

    # Should use balanced weights (0.2 each)
    assert all(w == pytest.approx(0.2) for w in weights.values())


def test_weights_sum_to_one():
    """Test that all weight adjustments sum to 1.0."""
    test_cases = [
        ("Theory", "High", "None"),
        ("Empirical", "Low", "Heavy"),
        ("Survey", "Low", "None"),
        ("Policy", "Medium", "Light")
    ]

    for primary, math, data in test_cases:
        classification = create_mock_classification(
            primary_type=primary,
            math_intensity=math,
            data_requirements=data
        )

        weights = adjust_persona_weights(classification)
        total = sum(weights.values())

        assert total == pytest.approx(1.0, rel=1e-6), \
            f"Weights for {primary} don't sum to 1.0: {weights}"


def test_normalize_weights():
    """Test weight normalization."""
    # Test normal case
    weights = {'A': 0.3, 'B': 0.5, 'C': 0.2}
    normalized = normalize_weights(weights)
    assert sum(normalized.values()) == pytest.approx(1.0)

    # Test zero total (should give equal weights)
    weights = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    normalized = normalize_weights(weights)
    assert all(w == pytest.approx(1.0/3.0) for w in normalized.values())


def test_get_selected_personas():
    """Test extracting selected personas from weights."""
    weights = {
        'Theorist': 0.5,
        'Historian': 0.3,
        'Empiricist': 0.2,
        'Visionary': 0.0,
        'Policymaker': 0.0
    }

    selected = get_selected_personas(weights)

    assert len(selected) == 3
    assert 'Theorist' in selected
    assert 'Historian' in selected
    assert 'Empiricist' in selected
    assert 'Visionary' not in selected
    assert 'Policymaker' not in selected


def test_exactly_three_personas_selected():
    """Test that exactly 3 personas are selected in most cases."""
    test_cases = [
        ("Theory", "High", "None"),
        ("Empirical", "Low", "Heavy"),
        ("Policy", "Medium", "Light")
    ]

    for primary, math, data in test_cases:
        classification = create_mock_classification(
            primary_type=primary,
            math_intensity=math,
            data_requirements=data
        )

        weights = adjust_persona_weights(classification)
        selected = get_selected_personas(weights)

        assert len(selected) == 3, \
            f"Expected 3 personas for {primary}, got {len(selected)}: {selected}"


if __name__ == "__main__":
    print("Testing persona weight adjustment...")

    print("\nTest 1: Theory + High Math")
    test_theory_high_math_weights()
    print("✓ Passed")

    print("\nTest 2: Empirical + Heavy Data")
    test_empirical_heavy_data_weights()
    print("✓ Passed")

    print("\nTest 3: Survey Paper")
    test_survey_paper_weights()
    print("✓ Passed")

    print("\nTest 4: Policy Paper")
    test_policy_paper_weights()
    print("✓ Passed")

    print("\nTest 5: Low Confidence Fallback")
    test_low_confidence_fallback()
    print("✓ Passed")

    print("\nTest 6: Weights Sum to 1.0")
    test_weights_sum_to_one()
    print("✓ Passed")

    print("\n✅ All persona tests passed!")
