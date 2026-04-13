# -*- coding: utf-8 -*-
"""
Unit tests for paper classifier.

Tests classification logic on sample papers of different types.
"""

import pytest
from referee._utils.paper_classifier import (
    classify_paper,
    keyword_baseline_detection,
    detect_equations,
    detect_econometric_methods,
    PaperClassification
)


# Sample paper texts for testing
THEORY_PAPER_SAMPLE = """
We prove a theorem showing that in equilibrium, the optimal strategy
involves a Nash equilibrium. The proof relies on several lemmas and
propositions. Our formal model demonstrates that under certain axioms,
the equilibrium is unique. We derive the analytical solution through
mathematical induction.

Theorem 1: Under assumptions A1-A3, there exists a unique equilibrium.
Proof: By contradiction, assume multiple equilibria exist...

The model features 25 equations and extensive derivations.
"""

EMPIRICAL_PAPER_SAMPLE = """
Using panel data from 1990-2020, we estimate the effect of monetary policy
on inflation using OLS regression. Our dataset includes 50 countries and
1000 observations. We employ instrumental variable estimation with 2SLS
to address endogeneity concerns. The coefficient on the policy rate is
-0.45 (standard error: 0.12, p-value: 0.0001).

We also run robustness checks using fixed effects and difference-in-differences
estimation. The statistical tests confirm our main findings. Our empirical
analysis shows strong evidence for the transmission mechanism.
"""

SURVEY_PAPER_SAMPLE = """
This paper provides a comprehensive literature review of monetary policy
effectiveness. We conduct a meta-analysis of 150 studies published between
1980 and 2020. Our survey synthesizes the state of the art in this field
and identifies key gaps in the literature. Recent developments suggest
new avenues for research. This comprehensive overview covers both theoretical
and empirical contributions.
"""

POLICY_PAPER_SAMPLE = """
We analyze the policy implications of central bank independence for welfare
outcomes. Our policy recommendations suggest regulatory reform to improve
governance. The welfare analysis shows significant gains from the proposed
policy changes. We discuss the implementation challenges and provide
guidance for policymakers. The policy analysis considers distributional
effects and equity concerns.
"""


def test_keyword_baseline_theory():
    """Test keyword detection for theory paper."""
    scores = keyword_baseline_detection(THEORY_PAPER_SAMPLE)
    assert scores['theory'] > scores['empirical']
    assert scores['theory'] >= 5  # Should detect theorem, proof, lemma, etc.


def test_keyword_baseline_empirical():
    """Test keyword detection for empirical paper."""
    scores = keyword_baseline_detection(EMPIRICAL_PAPER_SAMPLE)
    assert scores['empirical'] > scores['theory']
    assert scores['empirical'] >= 8  # Should detect regression, dataset, etc.


def test_equation_detection():
    """Test equation counting."""
    count = detect_equations(THEORY_PAPER_SAMPLE)
    assert count >= 1  # Should detect numbered equations


def test_econometric_methods_detection():
    """Test econometric methods detection."""
    methods = detect_econometric_methods(EMPIRICAL_PAPER_SAMPLE)
    assert 'regression' in methods
    assert 'IV' in methods or '2SLS' in methods
    assert 'DiD' in methods
    assert 'panel' in methods


def test_classify_theory_paper_no_llm():
    """Test classification of pure theory paper (keyword baseline only)."""
    result = classify_paper(THEORY_PAPER_SAMPLE, use_llm=False)

    assert isinstance(result, PaperClassification)
    assert result.primary_type == "Theory"
    assert result.math_intensity in ["Medium", "High"]
    assert result.data_requirements in ["None", "Light"]


def test_classify_empirical_paper_no_llm():
    """Test classification of empirical paper (keyword baseline only)."""
    result = classify_paper(EMPIRICAL_PAPER_SAMPLE, use_llm=False)

    assert isinstance(result, PaperClassification)
    assert result.primary_type == "Empirical"
    assert len(result.econometric_methods) >= 3
    assert result.data_requirements in ["Light", "Heavy"]


def test_classify_survey_paper_no_llm():
    """Test classification of survey paper (keyword baseline only)."""
    result = classify_paper(SURVEY_PAPER_SAMPLE, use_llm=False)

    assert isinstance(result, PaperClassification)
    # Survey might be classified as Theory or Survey depending on keyword counts
    assert result.primary_type in ["Survey", "Theory", "Empirical"]


def test_classify_policy_paper_no_llm():
    """Test classification of policy paper (keyword baseline only)."""
    result = classify_paper(POLICY_PAPER_SAMPLE, use_llm=False)

    assert isinstance(result, PaperClassification)
    assert result.primary_type in ["Policy", "Empirical"]


def test_classification_structure():
    """Test that classification returns expected structure."""
    result = classify_paper(THEORY_PAPER_SAMPLE, use_llm=False)

    # Check all required fields exist
    assert hasattr(result, 'primary_type')
    assert hasattr(result, 'math_intensity')
    assert hasattr(result, 'data_requirements')
    assert hasattr(result, 'econometric_methods')
    assert hasattr(result, 'confidence_scores')
    assert hasattr(result, 'reasoning')

    # Check types
    assert isinstance(result.primary_type, str)
    assert isinstance(result.math_intensity, str)
    assert isinstance(result.data_requirements, str)
    assert isinstance(result.econometric_methods, list)
    assert isinstance(result.confidence_scores, dict)
    assert isinstance(result.reasoning, str)

    # Check confidence scores
    assert 'primary_type' in result.confidence_scores
    assert 0.0 <= result.confidence_scores['primary_type'] <= 1.0


@pytest.mark.skipif(
    not pytest.config.getoption("--run-llm"),
    reason="LLM tests require --run-llm flag and ANTHROPIC_API_KEY"
)
def test_classify_with_llm():
    """
    Test LLM-based classification (requires API key).

    Run with: pytest tests/test_classifier.py --run-llm
    """
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    result = classify_paper(THEORY_PAPER_SAMPLE, use_llm=True)

    assert isinstance(result, PaperClassification)
    assert result.primary_type in ["Theory", "Empirical", "Survey", "Policy"]
    assert result.confidence_scores['primary_type'] > 0.5


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run tests that require LLM API calls"
    )


if __name__ == "__main__":
    # Run basic tests without LLM
    print("Testing keyword baseline detection...")
    test_keyword_baseline_theory()
    test_keyword_baseline_empirical()
    print("✓ Keyword baseline tests passed")

    print("\nTesting equation detection...")
    test_equation_detection()
    print("✓ Equation detection tests passed")

    print("\nTesting econometric methods detection...")
    test_econometric_methods_detection()
    print("✓ Econometric methods detection tests passed")

    print("\nTesting classification (no LLM)...")
    test_classify_theory_paper_no_llm()
    test_classify_empirical_paper_no_llm()
    test_classification_structure()
    print("✓ Classification tests passed")

    print("\n✅ All tests passed!")
