# Skill: Test Changes

This skill guides running the appropriate tests after making changes to the codebase.

## When to Use
- After modifying referee system code
- After updating section evaluator
- After changing prompts
- After adding utilities
- Before committing changes

## Quick Reference

```bash
# All tests
cd app_system && python -m pytest tests/

# Specific subsystem
cd app_system && python -m pytest tests/test_referee*.py
cd app_system && python -m pytest tests/test_section*.py

# Quick smoke test
cd app_system && python -m pytest tests/test_referee_quick.py -v
```

## Test Categories

### 1. Referee System Tests

**Quick smoke test** (2-3 minutes):
```bash
cd app_system
python -m pytest tests/test_referee_quick.py -v
```

**What it tests**:
- Persona selection (Round 0)
- Report generation (Round 1)
- Consensus calculation
- Basic output format

**When to run**: After any referee code changes

---

**Consensus calculation**:
```bash
cd app_system
python -m pytest tests/test_consensus_calculation.py -v
```

**What it tests**:
- Weighted average computation
- Threshold logic (ACCEPT/RESUBMIT/REJECT)
- Edge cases (ties, extreme weights)

**When to run**: After changing consensus logic in `engine.py`

---

**Experiment 4 personas**:
```bash
cd app_system
python -m pytest tests/test_exp4_personas.py -v
```

**What it tests**:
- All 10 personas load correctly
- Prompt files exist
- System prompts valid

**When to run**: After adding personas to exp_4

---

**Quote validation**:
```bash
cd app_system
python -m pytest tests/test_quote_validator.py -v
```

**What it tests**:
- Quote extraction from reports
- Fuzzy matching accuracy
- Adaptive thresholds (math vs. prose)

**When to run**: After modifying quote validation logic

---

**Deduplication**:
```bash
cd app_system
python -m pytest tests/test_deduplicator.py -v
```

**What it tests**:
- Finding clustering
- Similarity metrics
- Perspective preservation

**When to run**: After changing deduplication algorithm

---

**Caching**:
```bash
cd app_system
python -m pytest tests/test_cache.py -v
```

**What it tests**:
- Cache key computation
- Round-level save/load
- Cache invalidation
- TTL expiration

**When to run**: After modifying caching system

---

**Referee display**:
```bash
cd app_system
python -m pytest tests/test_referee_display.py -v
```

**What it tests**:
- UI rendering components
- Persona card styling
- Output formatting

**When to run**: After UI changes

### 2. Section Evaluator Tests

**Section evaluator prompts**:
```bash
cd app_system
python -m pytest tests/test_section_evaluator_prompts.py -v
```

**What it tests**:
- Prompt loader functionality
- Paper type contexts load
- Section guidance loads
- Config.yaml parsing

**When to run**: After changing prompts or config

---

**PDF extraction**:
```bash
cd app_system
python -m pytest tests/test_pdf_extraction.py -v
python -m pytest tests/test_pymupdf_extractor.py -v
```

**What it tests**:
- PDF text extraction
- Figure extraction
- Multi-column handling
- Fallback to pdfplumber

**When to run**: After modifying text extraction

---

**Math cleanup**:
```bash
cd app_system
python -m pytest tests/test_math_cleanup.py -v
```

**What it tests**:
- LaTeX normalization
- Math symbol handling
- Region fixing

**When to run**: After changing math cleanup logic

---

**Prompt loading**:
```bash
cd app_system
python -m pytest tests/test_prompt_loader.py -v
```

**What it tests**:
- PromptLoader class
- Version substitution
- Fallback handling

**When to run**: After changing prompt loading infrastructure

### 3. Integration Tests

**Full integration** (slow, 5-10 minutes):
```bash
cd app_system
python -m pytest tests/ -v --tb=short
```

**What it tests**: Everything

**When to run**: Before major commits, before PRs

### 4. Experiment Tests

**Classifier tests** (if working on classifier):
```bash
cd referee_classifier
python -m pytest tests/ -v
```

**Batch processing** (manual):
```bash
cd experiment
python test_setup.py
```

## Test Workflow by Change Type

### After Changing Referee Engine

```bash
cd app_system

# 1. Quick smoke test
python -m pytest tests/test_referee_quick.py -v

# 2. Consensus logic (if changed)
python -m pytest tests/test_consensus_calculation.py -v

# 3. Quote validation (always good to check)
python -m pytest tests/test_quote_validator.py -v

# 4. Manual test in UI
streamlit run app.py
# Upload test paper, verify output
```

### After Adding/Updating Persona

```bash
cd app_system

# 1. Prompt loading
python -c "
from referee.engine import SYSTEM_PROMPTS
print('NewPersona' in SYSTEM_PROMPTS)
print(len(SYSTEM_PROMPTS['NewPersona']))
"

# 2. Quick smoke test
python -m pytest tests/test_referee_quick.py -v

# 3. Manual test
streamlit run app.py
# Manually select new persona, verify output
```

### After Updating Prompts

```bash
cd app_system

# 1. Prompt loader tests
python -m pytest tests/test_prompt_loader.py -v
python -m pytest tests/test_section_evaluator_prompts.py -v

# 2. System-specific test
python -m pytest tests/test_referee_quick.py -v  # For MAD prompts
# OR
streamlit run app.py  # Manual section eval test

# 3. A/B comparison (if possible)
# Run same paper with old and new version
# Compare outputs
```

### After Changing Section Evaluator

```bash
cd app_system

# 1. Prompt tests
python -m pytest tests/test_section_evaluator_prompts.py -v

# 2. Math cleanup (if relevant)
python -m pytest tests/test_math_cleanup.py -v

# 3. Manual test
streamlit run app.py
# Go to Section Evaluator tab
# Test on 2-3 papers
```

### After Modifying Utilities

```bash
cd app_system

# Test the specific utility
python -m pytest tests/test_cache.py -v              # Caching
python -m pytest tests/test_deduplicator.py -v       # Deduplication
python -m pytest tests/test_quote_validator.py -v    # Quote validation
python -m pytest tests/test_pymupdf_extractor.py -v  # PDF extraction

# Integration test
python -m pytest tests/test_referee_quick.py -v
```

### Before Committing

```bash
cd app_system

# 1. Run all tests
python -m pytest tests/ -v --tb=short

# 2. Check for warnings
python -m pytest tests/ -v --tb=short -W default

# 3. Manual smoke test
streamlit run app.py
# Quick check both tabs

# 4. Verify no uncommitted debug code
git diff
```

## Test Debugging

### When Tests Fail

**1. Check error message**:
```bash
python -m pytest tests/test_failing.py -v --tb=long
```

**2. Run single test**:
```bash
python -m pytest tests/test_failing.py::test_specific_function -v
```

**3. Add debug prints**:
```python
def test_function():
    result = my_function()
    print(f"DEBUG: result = {result}")  # Add this
    assert result == expected
```

**4. Use pdb**:
```bash
python -m pytest tests/test_failing.py --pdb
```

### Common Test Issues

**Import errors**:
```bash
# Make sure you're in app_system/
cd app_system
# Then run tests
python -m pytest tests/
```

**API key errors**:
```bash
# Check .env exists
ls -la app_system/.env

# Verify key is set
cd app_system
python -c "from config import API_KEY; print(API_KEY[:10])"
```

**Cache interference**:
```bash
# Clear cache if tests behave inconsistently
rm -rf app_system/.referee_cache/
python -m pytest tests/test_cache.py -v
```

**Prompt file not found**:
```bash
# Verify prompt files exist
ls app_system/prompts/multi_agent_debate/personas/*/v*.txt

# Check config.yaml
cat app_system/prompts/multi_agent_debate/config.yaml
```

## Performance Testing

**Measure test execution time**:
```bash
cd app_system
python -m pytest tests/ -v --durations=10
```

**Profile slow tests**:
```bash
cd app_system
python -m pytest tests/test_slow.py -v --profile
```

## Coverage Analysis

**Run with coverage**:
```bash
cd app_system
python -m pytest tests/ --cov=referee --cov=section_eval --cov-report=html

# Open report
open htmlcov/index.html
```

**Check specific module coverage**:
```bash
python -m pytest tests/ --cov=referee.engine --cov-report=term-missing
```

## CI/CD Integration

**Pre-commit hook** (`.git/hooks/pre-commit`):
```bash
#!/bin/bash
cd app_system
python -m pytest tests/test_referee_quick.py -q
if [ $? -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi
```

**GitHub Actions** (example):
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          cd app_system
          python -m pytest tests/ -v
```

## Related Skills
- `/add-persona` - After adding persona, test integration
- `/version-prompt` - After versioning, test prompt loading
- `/add-paper-type` - After adding type, test criteria
