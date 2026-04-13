"""
Paper Classification System

Automatically classifies academic papers along multiple dimensions:
- Primary type (Theory/Empirical/Survey/Policy)
- Math intensity (Low/Medium/High)
- Data requirements (None/Light/Heavy)
- Econometric methods detection

Uses a two-stage approach:
1. Keyword-based baseline detection
2. LLM refinement with extended thinking (Claude Sonnet)
"""

import re
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


# Classification Keywords
KEYWORDS = {
    'theory': {
        'keywords': [
            r'\btheorem\b', r'\bproof\b', r'\bproposition\b', r'\blemma\b',
            r'\bequilibrium\b', r'\bformal model\b', r'\baxiom\b', r'\bcorollary\b',
            r'\bmathematical framework\b', r'\bderivation\b', r'\banalytical solution\b'
        ],
        'weight': 1.0
    },
    'empirical': {
        'keywords': [
            r'\bregression\b', r'\bdataset\b', r'\bpanel data\b', r'\bestimation\b',
            r'\bcoefficient\b', r'\bempirical analysis\b', r'\bstatistical test\b',
            r'\bstandard error\b', r'\bp-value\b', r'\bestimator\b', r'\bsample\b'
        ],
        'weight': 1.0
    },
    'survey': {
        'keywords': [
            r'\bliterature review\b', r'\bmeta-analysis\b', r'\bcomprehensive overview\b',
            r'\bsurvey\b', r'\bsynthesis of\b', r'\bstate of the art\b', r'\brecent developments\b'
        ],
        'weight': 0.8
    },
    'policy': {
        'keywords': [
            r'\bpolicy recommendation\b', r'\bregulatory\b', r'\bgovernance\b',
            r'\breform\b', r'\bpolicy implication\b', r'\bwelfare\b', r'\bpolicy analysis\b'
        ],
        'weight': 0.9
    }
}

ECONOMETRIC_METHODS = {
    'regression': r'\b(?:OLS|regression|least squares)\b',
    'VAR': r'\b(?:VAR|vector autoregression)\b',
    'ARIMA': r'\b(?:ARIMA|autoregressive)\b',
    'GMM': r'\b(?:GMM|generalized method of moments)\b',
    'DiD': r'\b(?:difference-in-differences|DiD|diff-in-diff)\b',
    'RDD': r'\b(?:regression discontinuity|RDD)\b',
    'IV': r'\b(?:instrumental variable|2SLS|IV estimation)\b',
    'propensity_score': r'\b(?:propensity score|matching)\b',
    'panel': r'\b(?:panel data|fixed effects|random effects)\b',
    'time_series': r'\b(?:time series|GARCH|cointegration)\b',
    'bayesian': r'\b(?:Bayesian|MCMC|posterior)\b',
    'machine_learning': r'\b(?:random forest|neural network|gradient boosting|machine learning)\b'
}


@dataclass
class PaperClassification:
    """Structured paper classification results."""
    primary_type: str  # "Theory" | "Empirical" | "Survey" | "Policy"
    math_intensity: str  # "Low" | "Medium" | "High"
    data_requirements: str  # "None" | "Light" | "Heavy"
    econometric_methods: List[str]  # Detected methods
    confidence_scores: Dict[str, float]  # Confidence per dimension
    reasoning: str  # LLM explanation
    keyword_hints: Optional[Dict[str, int]] = None  # Baseline detection scores


def detect_equations(text: str) -> int:
    """
    Count mathematical equations in text.

    Looks for:
    - LaTeX equation environments
    - Numbered equations
    - Inline math ($ ... $)
    """
    equation_patterns = [
        r'\\begin\{equation\}',
        r'\\begin\{align\}',
        r'\$\$.*?\$\$',
        r'\\\[.*?\\\]',
        r'\n\s*\(\d+\)\s*$'  # Numbered equations like (1), (2)
    ]

    count = 0
    for pattern in equation_patterns:
        count += len(re.findall(pattern, text, re.DOTALL))

    return count


def keyword_baseline_detection(text: str) -> Dict[str, int]:
    """
    Baseline paper type detection using keyword matching.

    Args:
        text: Paper text (full or abstract)

    Returns:
        Dict mapping paper types to match counts
    """
    text_lower = text.lower()
    scores = {}

    for paper_type, config in KEYWORDS.items():
        count = 0
        for keyword_pattern in config['keywords']:
            matches = len(re.findall(keyword_pattern, text_lower, re.IGNORECASE))
            count += matches

        # Apply weight
        scores[paper_type] = int(count * config['weight'])

    return scores


def detect_econometric_methods(text: str) -> List[str]:
    """
    Detect econometric methods mentioned in paper.

    Args:
        text: Paper text

    Returns:
        List of detected method names
    """
    detected = []

    for method_name, pattern in ECONOMETRIC_METHODS.items():
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(method_name)

    return detected


def classify_with_llm(
    paper_text: str,
    keyword_hints: Dict[str, int],
    equation_count: int,
    detected_methods: List[str]
) -> Dict:
    """
    Use Claude with extended thinking for classification refinement.

    Args:
        paper_text: Full paper or abstract (truncated to 3000 chars)
        keyword_hints: Baseline keyword detection results
        equation_count: Number of equations detected
        detected_methods: Econometric methods detected

    Returns:
        Parsed JSON classification
    """
    import anthropic

    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Truncate paper text to first 3000 chars
    text_sample = paper_text[:3000]

    prompt = f"""You are classifying an academic economics paper. Based on keyword analysis, we detected:

**Keyword Matches:**
{json.dumps(keyword_hints, indent=2)}

**Math Indicators:**
- Equation count: {equation_count}

**Detected Econometric Methods:**
{json.dumps(detected_methods, indent=2)}

**Paper Text (first 3000 chars):**
{text_sample}

---

Classify this paper along these dimensions:

1. **Primary type**: Choose ONE: "Theory", "Empirical", "Survey", or "Policy"
2. **Math intensity**: Choose ONE: "Low", "Medium", or "High"
   - Low: < 5 equations, descriptive statistics only
   - Medium: 5-15 equations, moderate derivations
   - High: > 15 equations, complex proofs
3. **Data requirements**: Choose ONE: "None", "Light", or "Heavy"
   - None: Pure theory, no empirical component
   - Light: Simple datasets, descriptive stats
   - Heavy: Large-scale data, complex econometrics
4. **Econometric methods**: Refine the detected list (add/remove as needed)
5. **Confidence scores**: Provide 0-1 confidence for each dimension

**Output Format (STRICT JSON):**
```json
{{
    "primary_type": "Theory|Empirical|Survey|Policy",
    "math_intensity": "Low|Medium|High",
    "data_requirements": "None|Light|Heavy",
    "econometric_methods": ["method1", "method2", ...],
    "confidence_scores": {{
        "primary_type": 0.0-1.0,
        "math_intensity": 0.0-1.0,
        "data_requirements": 0.0-1.0
    }},
    "reasoning": "2-3 sentence explanation of your classification"
}}
```

Return ONLY valid JSON, no markdown formatting.
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            thinking={
                "type": "enabled",
                "budget_tokens": 2000
            },
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract text content (skip thinking blocks)
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text

        # Parse JSON from response
        # Try to extract JSON from markdown code blocks first
        json_match = re.search(r'```json\s*(.*?)\s*```', text_content, re.DOTALL)
        if json_match:
            classification = json.loads(json_match.group(1))
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', text_content, re.DOTALL)
            if json_match:
                classification = json.loads(json_match.group(0))
            else:
                raise ValueError("Could not extract JSON from LLM response")

        return classification

    except Exception as e:
        print(f"LLM classification failed: {e}")
        # Fallback to keyword-based classification
        return create_fallback_classification(keyword_hints, equation_count, detected_methods)


def create_fallback_classification(
    keyword_hints: Dict[str, int],
    equation_count: int,
    detected_methods: List[str]
) -> Dict:
    """
    Create fallback classification when LLM fails.

    Uses simple heuristics based on keyword counts and equation count.
    """
    # Determine primary type (highest keyword count)
    primary_type = max(keyword_hints, key=keyword_hints.get)
    primary_type = primary_type.capitalize()

    # Determine math intensity
    if equation_count < 5:
        math_intensity = "Low"
    elif equation_count < 15:
        math_intensity = "Medium"
    else:
        math_intensity = "High"

    # Determine data requirements
    if keyword_hints.get('empirical', 0) > 5 or len(detected_methods) > 2:
        data_requirements = "Heavy"
    elif keyword_hints.get('empirical', 0) > 0 or len(detected_methods) > 0:
        data_requirements = "Light"
    else:
        data_requirements = "None"

    return {
        "primary_type": primary_type,
        "math_intensity": math_intensity,
        "data_requirements": data_requirements,
        "econometric_methods": detected_methods,
        "confidence_scores": {
            "primary_type": 0.6,  # Low confidence for fallback
            "math_intensity": 0.7,
            "data_requirements": 0.6
        },
        "reasoning": "Fallback classification based on keyword analysis (LLM classification failed)"
    }


def classify_paper(
    paper_text: str,
    use_llm: bool = True
) -> PaperClassification:
    """
    Classify academic paper along multiple dimensions.

    Two-stage approach:
    1. Keyword-based baseline detection
    2. LLM refinement (if enabled)

    Args:
        paper_text: Full paper text or abstract
        use_llm: Whether to use LLM for refinement (default: True)

    Returns:
        PaperClassification object with structured results
    """
    # Stage 1: Keyword baseline
    keyword_hints = keyword_baseline_detection(paper_text)
    equation_count = detect_equations(paper_text)
    detected_methods = detect_econometric_methods(paper_text)

    # Stage 2: LLM refinement (if enabled)
    if use_llm:
        try:
            llm_result = classify_with_llm(
                paper_text,
                keyword_hints,
                equation_count,
                detected_methods
            )
        except Exception as e:
            print(f"LLM classification failed, using fallback: {e}")
            llm_result = create_fallback_classification(
                keyword_hints,
                equation_count,
                detected_methods
            )
    else:
        llm_result = create_fallback_classification(
            keyword_hints,
            equation_count,
            detected_methods
        )

    # Build classification object
    classification = PaperClassification(
        primary_type=llm_result['primary_type'],
        math_intensity=llm_result['math_intensity'],
        data_requirements=llm_result['data_requirements'],
        econometric_methods=llm_result['econometric_methods'],
        confidence_scores=llm_result['confidence_scores'],
        reasoning=llm_result['reasoning'],
        keyword_hints=keyword_hints
    )

    return classification


def classification_to_dict(classification: PaperClassification) -> Dict:
    """Convert PaperClassification to dictionary."""
    return asdict(classification)


def classification_to_json(classification: PaperClassification) -> str:
    """Convert PaperClassification to JSON string."""
    return json.dumps(asdict(classification), indent=2)
