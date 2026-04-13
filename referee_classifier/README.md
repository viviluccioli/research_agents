# Referee Report Pipeline with Automatic Paper Classification

Experimental evaluation system for academic papers with automatic classification and adaptive persona selection.

## Features

- **Automatic Paper Classification**: Two-stage classification using keyword baseline + LLM refinement
- **Adaptive Persona Selection**: Dynamically adjusts reviewer personas based on paper type
- **Batch Processing**: Process multiple papers from folders or CSV files
- **Comprehensive Output**: CSV/Excel exports with classification metadata
- **Confidence-Based Fallback**: Reverts to balanced weights when confidence < 0.7

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Single Paper Example

```bash
# Classify a single paper
python examples/single_paper.py --paper path/to/paper.txt

# Use keyword-only classification (no API calls)
python examples/single_paper.py --paper path/to/paper.txt --no-llm
```

### Batch Processing Example

```bash
# Process folder of papers
python examples/batch_processing.py --input papers/ --output results.csv

# Process CSV file
python examples/batch_processing.py \
    --input papers.csv \
    --text-column abstract \
    --output results.xlsx \
    --format excel
```

## Project Structure

```
referee_classifier/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── referee/
│   ├── __init__.py                   # Package exports
│   ├── _utils/
│   │   ├── __init__.py
│   │   └── paper_classifier.py       # Main classifier
│   ├── personas.py                    # Adaptive persona selection
│   └── engine.py                      # Modified debate engine
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py            # Classifier unit tests
│   └── test_personas.py              # Persona selection tests
├── examples/
│   ├── single_paper.py               # Single paper example
│   └── batch_processing.py           # Batch processing example
├── config/
│   └── classifier_config.yaml        # Configuration
└── docs/
    └── sample_papers/                # Sample test papers
```

## Classification System

### Dimensions

The classifier analyzes papers along 4 dimensions:

1. **Primary Type**: Theory | Empirical | Survey | Policy
2. **Math Intensity**: Low | Medium | High
3. **Data Requirements**: None | Light | Heavy
4. **Econometric Methods**: List of detected methods

### Two-Stage Approach

**Stage 1: Keyword Baseline**
- Fast regex-based pattern matching
- Counts matches for each paper type
- Detects equations and econometric methods

**Stage 2: LLM Refinement**
- Claude Sonnet with 2k token thinking budget
- Validates and refines baseline classification
- Provides confidence scores and reasoning

### Classification Logic

```python
from referee import classify_paper

# Classify paper
classification = classify_paper(paper_text)

print(f"Type: {classification.primary_type}")
print(f"Confidence: {classification.confidence_scores['primary_type']}")
print(f"Methods: {classification.econometric_methods}")
```

## Adaptive Persona Selection

### Available Personas

- **Theorist**: Mathematical rigor, proofs, formal models
- **Empiricist**: Data quality, econometrics, identification
- **Historian**: Literature context, historical development
- **Visionary**: Novelty, impact, future directions
- **Policymaker**: Real-world application, welfare, policy relevance

### Weight Adjustment Rules

The system adjusts persona weights based on classification:

| Paper Type | Math | Data | Persona Weights |
|------------|------|------|-----------------|
| Theory | High | None | Theorist: 0.5, Historian: 0.25, Visionary: 0.25 |
| Empirical | Low | Heavy | Empiricist: 0.5, Visionary: 0.25, Historian: 0.15 |
| Survey | - | - | Historian: 0.6, Visionary: 0.4 |
| Policy | - | Light | Policymaker: 0.4, Empiricist: 0.3, Historian: 0.2 |

**Confidence Threshold**: If classification confidence < 0.7, the system falls back to balanced weights (0.2 each).

### Usage

```python
from referee import classify_paper, adjust_persona_weights

# Classify and adjust weights
classification = classify_paper(paper_text)
weights = adjust_persona_weights(classification)

print(weights)
# {'Theorist': 0.5, 'Historian': 0.25, 'Visionary': 0.25}
```

## API Reference

### Core Functions

#### `classify_paper(paper_text: str, use_llm: bool = True) -> PaperClassification`

Classify academic paper along multiple dimensions.

**Args:**
- `paper_text`: Full paper text or abstract
- `use_llm`: Whether to use LLM refinement (default: True)

**Returns:**
- `PaperClassification` object with:
  - `primary_type`: str
  - `math_intensity`: str
  - `data_requirements`: str
  - `econometric_methods`: List[str]
  - `confidence_scores`: Dict[str, float]
  - `reasoning`: str

#### `adjust_persona_weights(classification: PaperClassification) -> Dict[str, float]`

Adjust persona weights based on paper classification.

**Args:**
- `classification`: PaperClassification object

**Returns:**
- Dictionary mapping persona names to weights (summing to 1.0)

#### `execute_debate_pipeline(paper_text: str, use_classification: bool = True, ...) -> dict`

Execute debate pipeline with classification (simplified for experiments).

**Args:**
- `paper_text`: Paper to evaluate
- `use_classification`: Use automatic classification (default: True)
- `force_manual_personas`: Optional manual persona override
- `force_manual_weights`: Optional manual weight override

**Returns:**
- Dictionary with classification and selection results

## Input Formats

### 1. Single Paper (Text File)

```
paper.txt containing full paper text
```

### 2. CSV Batch

```csv
paper_id,title,text
1,"Monetary Policy Effects","Full text of paper 1..."
2,"Growth Models","Full text of paper 2..."
```

### 3. Folder of Files

```
papers/
├── paper1.txt
├── paper2.txt
└── paper3.txt
```

## Output Formats

### CSV Output

```csv
paper_id,primary_type,math_intensity,data_requirements,econometric_methods,confidence_primary,selected_personas,persona_weights
1,Theory,High,None,,0.92,"Theorist, Historian, Visionary","Theorist:0.50, Historian:0.25, Visionary:0.25"
```

### Excel Output

**Sheet 1: Classifications**
- Full classification results per paper

**Sheet 2: Summary**
- Paper type distribution
- Aggregate statistics

## Testing

### Run Unit Tests

```bash
# Run all tests (keyword-based only)
pytest tests/

# Run with LLM tests (requires API key)
pytest tests/ --run-llm

# Run specific test file
pytest tests/test_classifier.py -v

# Run specific test
pytest tests/test_classifier.py::test_theory_paper_classification -v
```

### Test Coverage

- `test_classifier.py`: Paper classification logic
- `test_personas.py`: Adaptive weight adjustment

## Configuration

Edit `config/classifier_config.yaml` to customize:

- Model selection (`claude-sonnet-4-20250514`)
- Thinking budget (default: 2000 tokens)
- Confidence threshold (default: 0.7)
- Keyword patterns
- Weight adjustment rules

## Examples

### Example 1: Single Paper

```python
from referee import classify_paper, adjust_persona_weights

# Load paper
with open('paper.txt', 'r') as f:
    paper_text = f.read()

# Classify
classification = classify_paper(paper_text)
print(f"Type: {classification.primary_type}")
print(f"Confidence: {classification.confidence_scores['primary_type']:.2f}")

# Get weights
weights = adjust_persona_weights(classification)
print(f"Weights: {weights}")
```

### Example 2: Batch Processing

```python
from referee import classify_paper, adjust_persona_weights
import pandas as pd

# Load papers
papers = pd.read_csv('papers.csv')

results = []
for _, row in papers.iterrows():
    classification = classify_paper(row['text'])
    weights = adjust_persona_weights(classification)

    results.append({
        'paper_id': row['id'],
        'type': classification.primary_type,
        'confidence': classification.confidence_scores['primary_type'],
        'personas': ', '.join([p for p, w in weights.items() if w > 0])
    })

# Export
pd.DataFrame(results).to_csv('results.csv', index=False)
```

### Example 3: Integration with Full Pipeline

```python
# This classifier can be integrated into the full referee pipeline
# from app_system/referee/engine.py by adding classification to Round 0

from referee import classify_paper, adjust_persona_weights

def run_round_0_with_classification(paper_text):
    # Classify paper
    classification = classify_paper(paper_text)

    # Adjust weights
    weights = adjust_persona_weights(classification)

    # Extract selected personas
    selected_personas = [p for p, w in weights.items() if w > 0]

    return {
        'selected_personas': selected_personas,
        'weights': weights,
        'classification': classification
    }
```

## Cost Estimation

### Token Usage per Paper

- **Classification** (with LLM): ~1,500-2,500 input tokens + 500-800 output tokens
- **Keyword-only**: 0 API tokens

### Pricing (Claude Sonnet 4)

- Input: $3.00 per million tokens
- Output: $15.00 per million tokens
- **Estimated cost per paper**: $0.01-0.02 (with LLM classification)

### Batch Processing (100 papers)

- With LLM: ~$1.00-2.00
- Keyword-only: $0.00

## Troubleshooting

### Common Issues

**1. `ANTHROPIC_API_KEY not set`**
```bash
export ANTHROPIC_API_KEY="your-key"
```

**2. Import errors**
```bash
# Make sure you're in the referee_classifier directory
cd referee_classifier
python examples/single_paper.py --paper test.txt
```

**3. Low confidence warnings**
- The system automatically falls back to balanced weights
- Review the keyword_hints in classification output to understand why

**4. LLM classification fails**
- System automatically uses keyword-based fallback
- Check API key and internet connection

## Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Potential improvements for production deployment:

- [ ] PDF paper support (integrate `pdfplumber`)
- [ ] Caching system (port from `app_system/referee/_utils/cache.py`)
- [ ] Full debate rounds integration
- [ ] REST API endpoints (FastAPI)
- [ ] Web UI (Streamlit)
- [ ] Multi-language support
- [ ] Custom classifier training on labeled dataset
- [ ] Confidence calibration
- [ ] A/B testing framework

## Integration with Main System

To integrate this classifier into the main `app_system/referee/` pipeline:

1. Copy `referee/_utils/paper_classifier.py` to `app_system/referee/_utils/`
2. Copy `referee/personas.py` to `app_system/referee/`
3. Modify `app_system/referee/engine.py:run_round_0_selection()` to call `classify_paper()` before persona selection
4. Use `adjust_persona_weights()` to compute weights based on classification
5. Store classification in `debate_state['classification']`
6. Add classification to Excel metadata sheet

## Contributing

This is an experimental system. To contribute:

1. Run tests: `pytest tests/`
2. Format code: `black referee/ tests/ examples/`
3. Type check: `mypy referee/`

## License

Internal research tool for Federal Reserve use.

## Contact

For questions or issues, contact the AI Lab team.

---

**Version**: 0.1.0
**Last Updated**: 2026-04-13
