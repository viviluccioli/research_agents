# Quick Start Guide

Get up and running with the referee classifier in 5 minutes.

## 1. Setup (2 minutes)

```bash
# Clone/navigate to project
cd referee_classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

## 2. Test with Sample Papers (1 minute)

```bash
# Run tests (no API calls)
python tests/test_classifier.py

# Expected output:
# ✓ Keyword baseline tests passed
# ✓ Equation detection tests passed
# ✓ Econometric methods detection tests passed
# ✓ Classification tests passed
# ✅ All tests passed!
```

## 3. Classify a Paper (1 minute)

```bash
# Try with theory sample
python examples/single_paper.py --paper docs/sample_papers/theory_paper.txt

# Expected output:
# Primary Type:        Theory
# Math Intensity:      High
# Data Requirements:   None
# Selected Personas:   Theorist (0.50), Historian (0.25), Visionary (0.25)
```

## 4. Batch Processing (1 minute)

```bash
# Process all sample papers
python examples/batch_processing.py \
    --input docs/sample_papers/ \
    --output sample_results.csv

# View results
cat sample_results.csv
```

## What's Next?

### Use Your Own Papers

```bash
# Single paper
python examples/single_paper.py --paper /path/to/your/paper.txt

# Folder of papers
python examples/batch_processing.py \
    --input /path/to/papers/ \
    --output results.csv
```

### Run Without API Calls

```bash
# Use keyword-only classification (free, fast)
python examples/single_paper.py --paper paper.txt --no-llm
```

### Export to Excel

```bash
python examples/batch_processing.py \
    --input papers/ \
    --output results.xlsx \
    --format excel
```

## Common Issues

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### "Module not found"
```bash
# Make sure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### "File not found"
```bash
# Use absolute paths or check current directory
pwd
ls docs/sample_papers/
```

## Cost Estimates

- **Keyword-only**: $0 (no API calls)
- **With LLM**: ~$0.01-0.02 per paper
- **100 papers**: ~$1-2

## Next Steps

1. Read the full [README.md](../README.md) for detailed documentation
2. Explore [examples/](../examples/) for more usage patterns
3. Review [config/classifier_config.yaml](../config/classifier_config.yaml) for customization
4. Check [tests/](../tests/) for implementation details

## Support

For issues or questions, see the main README or contact the AI Lab team.
