# Skill: Run Batch Experiment

This skill guides running batch experiments with proper setup and validation.

## When to Use
- Evaluating system performance on multiple papers
- Comparing against ground truth
- A/B testing prompt versions
- Benchmarking system changes

## Prerequisites
- Papers ready (PDFs in directory)
- Ground truth CSV (if comparing)
- API credentials configured
- Sufficient API quota

## Experiment Types

### 1. Batch Referee Reports

**Purpose**: Run MAD system on multiple papers, compare to ground truth.

**Location**: `experiment/batch_referee_reports.py`

### 2. Persona Selection Experiments

**Purpose**: Test consistency of persona selection (Round 0).

**Location**: `persona_exp/run_persona_selection_experiment.py`

## Steps for Batch Referee Experiment

### 1. Prepare Papers

**Organize PDFs**:
```bash
mkdir -p experiment/papers_batch1
cp /path/to/papers/*.pdf experiment/papers_batch1/
```

**Verify file names**:
```bash
# File names should be clean doc IDs
ls experiment/papers_batch1/
# Good: paper123.pdf, ifdp_2020_1234.pdf
# Bad: Paper #123 (draft).pdf
```

**Rename if needed**:
```bash
cd experiment/papers_batch1
for f in *.pdf; do
    new_name=$(echo "$f" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    mv "$f" "$new_name"
done
```

### 2. Prepare Ground Truth CSV

**Format** (required columns):
```csv
doc_id,Tier
paper123,Tier1
paper456,Tier2
paper789,Tier3
```

**doc_id rules**:
- Must match PDF filename without extension
- Case-sensitive
- Example: `paper123.pdf` → `paper123`

**Tier values** (example):
- `Tier1` — Top quality (expected: ACCEPT)
- `Tier2` — Good (expected: ACCEPT or RESUBMIT)
- `Tier3` — Needs work (expected: RESUBMIT or REJECT)

**Create ground truth**:
```bash
nano experiment/ground_truth.csv
```

```csv
doc_id,Tier,Notes
paper123,Tier1,Excellent empirical work
paper456,Tier2,Good but needs robustness
paper789,Tier3,Methodological concerns
```

### 3. Verify Setup

**Test API access**:
```bash
cd experiment
python test_setup.py
```

**Expected output**:
```
✓ API_KEY is set
✓ API_BASE is set
✓ MODEL_PRIMARY is set
✓ Can import referee.engine
✓ API connection successful
All checks passed!
```

**Estimate cost**:
```python
# Each paper: ~$1.50-2.00 (5-persona)
# Each paper: ~$2.00-3.00 (10-persona)

num_papers = 10
cost_per_paper = 2.0
total_cost = num_papers * cost_per_paper
print(f"Estimated cost: ${total_cost:.2f}")
```

### 4. Test on Small Sample

**CRITICAL**: Test on 1-2 papers first!

```bash
cd experiment

# Create test directory
mkdir -p test_sample
cp papers_batch1/paper123.pdf test_sample/

# Create test ground truth
echo "doc_id,Tier" > test_ground_truth.csv
echo "paper123,Tier1" >> test_ground_truth.csv

# Run test
python batch_referee_reports.py \
    --pdf-dir test_sample/ \
    --ground-truth test_ground_truth.csv \
    --output-dir test_results/
```

**Verify**:
- Script runs without errors
- Output files created
- CSV format correct
- JSON contains full transcripts

### 5. Run Full Batch

```bash
cd experiment

# Run batch
python batch_referee_reports.py \
    --pdf-dir papers_batch1/ \
    --ground-truth ground_truth.csv \
    --output-dir results/$(date +%Y%m%d_%H%M%S)/ \
    > batch_run.log 2>&1 &

# Monitor progress
tail -f batch_run.log
```

**Or use shell script**:
```bash
cd experiment
./run_experiment.sh papers_batch1/ ground_truth.csv
```

### 6. Monitor Progress

**Check log**:
```bash
tail -f experiment/batch_run.log
```

**Check partial results**:
```bash
# Results are saved incrementally
ls experiment/results/*/
cat experiment/results/*/experiment_results.csv
```

**Estimate completion**:
```python
# ~2-5 minutes per paper (5-persona)
# ~3-7 minutes per paper (10-persona)

num_papers = 10
time_per_paper = 4  # minutes
total_time = num_papers * time_per_paper
print(f"Estimated time: {total_time} minutes ({total_time/60:.1f} hours)")
```

### 7. Analyze Results

**CSV format**:
```csv
doc_id,ground_truth_tier,predicted_verdict,selected_personas,weights,match,timestamp
paper123,Tier1,ACCEPT,"Theorist,Empiricist,Visionary","0.35,0.40,0.25",True,2026-04-27T10:30:00
```

**Load and analyze**:
```python
import pandas as pd

df = pd.read_csv("experiment/results/20260427_103000/experiment_results.csv")

# Accuracy
accuracy = (df['match'] == True).mean()
print(f"Accuracy: {accuracy:.2%}")

# Confusion matrix
import pandas as pd
confusion = pd.crosstab(df['ground_truth_tier'], df['predicted_verdict'])
print(confusion)

# Per-tier performance
for tier in ['Tier1', 'Tier2', 'Tier3']:
    tier_df = df[df['ground_truth_tier'] == tier]
    tier_acc = (tier_df['match'] == True).mean()
    print(f"{tier}: {tier_acc:.2%}")
```

**JSON analysis**:
```python
import json

with open("experiment/results/.../experiment_results.json") as f:
    results = json.load(f)

# Extract persona usage frequency
persona_counts = {}
for paper in results['papers']:
    for persona in paper['selected_personas']:
        persona_counts[persona] = persona_counts.get(persona, 0) + 1

print("Persona usage:")
for persona, count in sorted(persona_counts.items(), key=lambda x: -x[1]):
    print(f"  {persona}: {count}")
```

## Steps for Persona Selection Experiment

### 1. Select Test Paper

```bash
# Use a representative paper
cp papers/sample_empirical.pdf persona_exp/test_paper.pdf
```

### 2. Run Experiment

```bash
cd persona_exp

# Run N trials of persona selection
python run_persona_selection_experiment.py \
    test_paper.pdf \
    --runs 20 \
    --output results/test_paper_$(date +%Y%m%d_%H%M%S).csv
```

**Parameters**:
- `--runs N`: Number of selection trials (default: 10)
- `--output PATH`: Output CSV path

### 3. Analyze Consistency

```python
import pandas as pd

df = pd.read_csv("persona_exp/results/test_paper_20260427_103000.csv")

# Frequency distribution
freq = df['selected_personas'].value_counts()
print("Selection frequency:")
print(freq)

# Consistency metric
most_common_freq = freq.max()
consistency = most_common_freq / len(df)
print(f"\nConsistency: {consistency:.2%}")

# Weight stability
for persona in ['Theorist', 'Empiricist', 'Visionary']:
    weights = df[f'{persona}_weight'].dropna()
    if len(weights) > 0:
        print(f"{persona}: mean={weights.mean():.3f}, std={weights.std():.3f}")
```

## Output Structure

### Batch Referee Results
```
results/20260427_103000/
├── experiment_results.csv          # Summary table
├── experiment_results.json         # Full transcripts
└── metadata.json                   # Run configuration
```

### Persona Selection Results
```
results/test_paper_20260427_103000/
├── selection_results.csv           # Per-trial results
├── frequency_distribution.json     # Aggregated stats
└── experiment_metadata.json        # Configuration
```

## Best Practices

### Before Running

✅ **Do**:
- Test on 1-2 papers first
- Verify ground truth format
- Check API quota
- Estimate cost and time
- Use descriptive output directory names

❌ **Don't**:
- Run large batch without testing
- Use production papers for testing
- Ignore API rate limits
- Commit results to git (add to `.gitignore`)

### During Run

✅ **Do**:
- Monitor logs
- Check partial results
- Have backup plan if API fails

❌ **Don't**:
- Kill process without cleanup
- Restart without checking state
- Modify code during run

### After Run

✅ **Do**:
- Save raw results
- Document findings
- Archive configuration
- Update ground truth if errors found

❌ **Don't**:
- Overwrite results directory
- Delete failed runs (useful for debugging)
- Cherry-pick results

## Troubleshooting

### Import Errors
```bash
# Check path resolution
cd experiment
python -c "import sys; print(sys.path)"
python -c "from referee.engine import execute_debate_pipeline; print('OK')"
```

### API Errors
```bash
# Check configuration
python test_setup.py

# Check quota
# (provider-specific)
```

### Ground Truth Mismatch
```python
# Verify doc_ids match
import pandas as pd
import os

gt = pd.read_csv("ground_truth.csv")
pdfs = [f.replace('.pdf', '') for f in os.listdir("papers_batch1/") if f.endswith('.pdf')]

print("In ground truth but not in PDFs:", set(gt['doc_id']) - set(pdfs))
print("In PDFs but not in ground truth:", set(pdfs) - set(gt['doc_id']))
```

### Slow Execution
```python
# Run with reduced calls (for testing)
# Modify batch_referee_reports.py:
# - Skip rounds 2A, 2B, 2C
# - Use only Round 1 reports for quick testing
```

## Advanced: A/B Testing Prompts

### Setup
```bash
# Create two branches
git checkout -b prompt-v1
# Configure prompts to v1.0
git commit -am "Configure for v1.0"

git checkout main
git checkout -b prompt-v2
# Configure prompts to v2.0
git commit -am "Configure for v2.0"
```

### Run Parallel Experiments
```bash
# Terminal 1: v1.0
git checkout prompt-v1
cd experiment
python batch_referee_reports.py ... --output-dir results/v1.0/

# Terminal 2: v2.0
git checkout prompt-v2
cd experiment
python batch_referee_reports.py ... --output-dir results/v2.0/
```

### Compare Results
```python
import pandas as pd

v1 = pd.read_csv("experiment/results/v1.0/experiment_results.csv")
v2 = pd.read_csv("experiment/results/v2.0/experiment_results.csv")

# Accuracy
print(f"v1.0 accuracy: {(v1['match'] == True).mean():.2%}")
print(f"v2.0 accuracy: {(v2['match'] == True).mean():.2%}")

# Agreement
agreement = (v1['predicted_verdict'] == v2['predicted_verdict']).mean()
print(f"Inter-version agreement: {agreement:.2%}")

# Where do they differ?
diffs = v1[v1['predicted_verdict'] != v2['predicted_verdict']]
print(f"\nDifferences on {len(diffs)} papers:")
print(diffs[['doc_id', 'ground_truth_tier']])
```

## Related Skills
- `/test-changes` - Test before running experiments
- `/version-prompt` - Create prompt versions for A/B testing
