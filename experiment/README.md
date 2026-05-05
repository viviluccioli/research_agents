# Batch Referee Report Experiment System

Automated batch processing and benchmarking system for the multi-agent referee report pipeline.

## Overview

This system runs the MAD (Multi-Agent Debate) referee report system on multiple papers and outputs structured results for analysis against ground truth data. It's designed for **system evaluation and benchmarking**, not end-user application.

**What it does**: Processes batches of PDFs through the 5-round debate pipeline, extracts verdicts/scores from all personas, computes consensus metrics, and compares results to ground truth tiers.

## Quick Start

### Prerequisites

```bash
# From repository root
cd /casl/home/m1vcl00/FS-CASL/research_agents

# Activate virtual environment
source venv/bin/activate

# Verify dependencies
cd experiment
python test_setup.py
```

### Basic Usage

```bash
python batch_referee_reports.py \
    --pdf-dir /path/to/pdfs \
    --ground-truth /path/to/ground_truth.csv \
    --output-dir ./results
```

### Test Run (Recommended First)

```bash
# Process first 3 papers to verify setup
python batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth tracking.csv \
    --limit 3
```

### Using the Shell Wrapper

Edit `run_experiment.sh` to set default paths, then:

```bash
bash run_experiment.sh          # Process all PDFs
bash run_experiment.sh 10       # Process first 10 PDFs
```

## Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--pdf-dir` | Yes | Directory containing PDF files | `./papers/` |
| `--ground-truth` | Yes | Ground truth CSV file | `./tracking.csv` |
| `--output-dir` | No | Output directory (default: `./results`) | `./my_results/` |
| `--limit` | No | Limit number of PDFs (for testing) | `5` |
| `--start-from` | No | Resume from specific file | `paper-042.pdf` |

## Ground Truth CSV Format

**Required columns**:
- `doc_id` — Paper identifier (must match PDF filename without `.pdf`)
- `Tier` — Ground truth category/quality rating

**Example**:
```csv
doc_id,Tier
ifdp-2020-1,Tier1
ifdp-2020-2,Tier3
ifdp-2020-3,Tier2
```

**Important**: PDF filenames must match `doc_id` values exactly:
- `ifdp-2020-1.pdf` → `doc_id = "ifdp-2020-1"` ✅
- `IFDP_2020_1.pdf` → `doc_id = "IFDP_2020_1"` ❌ (won't match)

## Output Files

The script generates two timestamped output files:

### 1. CSV File: `referee_batch_results_YYYYMMDD_HHMMSS.csv`

**34 columns** containing:
- **Basic metadata** (6 cols): doc_id, filename, duration, char/word counts
- **Persona 1 data** (6 cols): name, weight, Round 1 verdict, final verdict, Round 1 score, final score
- **Persona 2 data** (6 cols): same structure as Persona 1
- **Persona 3 data** (6 cols): same structure as Persona 1
- **Final results** (3 cols): final verdict, final report, ground truth tier
- **Consensus metrics - categorical** (4 cols): R1/R2C consensus scores, R1/R2C agreement levels
- **Consensus metrics - numeric** (3 cols): R1/R2C numeric consensus, consensus delta

### 2. JSON File: `referee_batch_results_YYYYMMDD_HHMMSS.json`

Same data as CSV plus full final report text for each paper.

## Output Schema Details

### Verdict Values
- **PASS** — Accept for publication
- **REVISE** — Reject and resubmit (major revision)
- **FAIL** — Reject

### Consensus Score (Categorical)
Weighted average of persona verdicts mapped to 0-1 scale:
- `PASS = 1.0`
- `REVISE = 0.5`
- `FAIL = 0.0`

**Example**: 
- Persona 1 (weight 0.5): REVISE → 0.5
- Persona 2 (weight 0.3): PASS → 1.0
- Persona 3 (weight 0.2): FAIL → 0.0
- **Consensus**: (0.5×0.5) + (1.0×0.3) + (0.0×0.2) = **0.55**

### Agreement Level
Measures consensus among personas (ignores weights):
- **UNANIMOUS** — All 3 personas agree
- **PARTIAL** — 2 personas agree, 1 differs
- **DIVERGENT** — All 3 personas disagree

### Numeric Consensus Score
Weighted average of confidence scores (1-10 scale, normalized to 0-1):
- Only computed if all personas provide numeric scores
- **Consensus delta**: Change from Round 1 to Round 2C (positive = consensus increased)

## Processing Pipeline

For each PDF:
1. Extract `doc_id` from filename
2. Look up ground truth `Tier` from CSV
3. Extract text using `pdfplumber`
4. Count characters and words
5. Run 5-round MAD pipeline:
   - **Round 0**: LLM selects 3 personas + assigns weights
   - **Round 1**: Personas write independent reports
   - **Round 2A**: Cross-examination (questions)
   - **Round 2B**: Direct examination (answers)
   - **Round 2C**: Final amendments
   - **Round 3**: Editor synthesizes consensus
6. Extract verdicts, scores, weights
7. Compute consensus metrics
8. Write results to CSV/JSON

## Performance Estimates

**Timing** (Claude 4.5 Sonnet):
- Short paper (~5K words): 2-3 minutes
- Medium paper (~8K words): 3-5 minutes  
- Long paper (~12K words): 5-8 minutes
- **100 papers**: 4-8 hours

**Cost** (Claude 4.5 Sonnet, approximate):
- Per paper: $1.50-2.50
- 100 papers: $150-250

## Configuration

**Speed optimizations** (hardcoded in script):
- Quote validation: **DISABLED** (saves ~15-20% time)
- Caching: **DISABLED** (ensures independent evaluations)

**To enable quote validation**: Edit line 218 in `batch_referee_reports.py`:
```python
enable_quote_validation=True  # Changed from False
```

**To enable caching**: Edit line 219:
```python
use_cache=True  # Changed from False
```

## Analysis Examples

### Load Results in Python

```python
import pandas as pd

# Load most recent results
df = pd.read_csv('results/referee_batch_results_20260505_001147.csv')
```

### Compute Accuracy

```python
# Verdict accuracy by tier
accuracy_by_tier = df.groupby('tier').apply(
    lambda x: (x['final_verdict'] == 'PASS').mean()
)

# Overall accuracy (if tier maps to verdict)
accuracy = (df['final_verdict'] == df['tier']).mean()
```

### Confusion Matrix

```python
import pandas as pd

confusion = pd.crosstab(
    df['tier'], 
    df['final_verdict'], 
    normalize='index'
)
print(confusion)
```

### Consensus Analysis

```python
# Agreement patterns by tier
agreement = df.groupby('tier')['agreement_level_r2c'].value_counts(normalize=True)

# Consensus score distributions
consensus_stats = df.groupby('tier')['consensus_score_r2c'].describe()
```

### Persona Selection Frequency

```python
# Combine all persona columns
personas = pd.concat([
    df['persona_1_name'],
    df['persona_2_name'],
    df['persona_3_name']
])

# Count frequency
persona_freq = personas.value_counts()
print(persona_freq)
```

### Examine Edge Cases

```python
# Papers with divergent opinions
divergent = df[df['agreement_level_r2c'] == 'DIVERGENT']

# Papers where consensus changed significantly
large_delta = df[abs(df['consensus_delta_numeric']) > 0.3]

# Failed papers
errors = df[df['final_verdict'] == 'ERROR']
```

## Error Handling

If a paper fails:
- Error logged to console with traceback
- Placeholder record added: `final_verdict = "ERROR"`, all metrics = 0/None
- Processing continues with remaining papers

**Common failure causes**:
- PDF extraction failed (corrupted file, scanned image without OCR)
- Text too short (<100 characters)
- API timeout or rate limit
- Malformed LLM output

## Troubleshooting

### ImportError: No module named 'referee'

**Solution**: Ensure you're in the `experiment/` directory and virtual environment is activated:
```bash
cd experiment
source ../venv/bin/activate
python test_setup.py
```

### Ground truth doc_id not found

**Solution**: Verify PDF filenames (without `.pdf`) match `doc_id` values in CSV exactly:
```bash
# List PDF files
ls pdf_dir/*.pdf | xargs basename -s .pdf

# List doc_ids in CSV
cut -d',' -f1 ground_truth.csv | tail -n +2
```

### PDF extraction failed

**Solutions**:
- Verify PDF is valid: `pdfplumber` or Acrobat Reader
- Check if PDF is scanned image (requires OCR)
- Try re-downloading PDF

### Out of memory / API rate limits

**Solutions**:
- Use `--limit` flag to process in smaller batches
- Add delays between API calls (edit script)
- Check API quota/rate limits

## Directory Structure

```
experiment/
├── batch_referee_reports.py    # Main script
├── run_experiment.sh            # Shell wrapper
├── test_setup.py                # Dependency checker
├── README.md                    # This file
├── CLAUDE.md                    # Developer documentation
│
├── docs/                        # Analysis documentation
│   ├── METRICS_EXPLAINED.md     # Metric definitions
│   ├── SCORING_RUBRIC.md        # Scoring system
│   └── *.md                     # Other guides
│
├── _archived/                   # Historical analysis scripts
│   ├── analyze_*.py             # Post-hoc analysis
│   └── test_*.py                # Development tests
│
├── results/                     # Output directory (gitignored)
│   ├── *.csv                    # Structured results
│   └── *.json                   # Full transcripts
│
└── ifdp_sample/                 # Sample PDFs for testing
```

## Best Practices

✅ **Do**:
- Run `python test_setup.py` before large batches
- Test with `--limit 3` first
- Verify ground truth CSV format
- Check disk space (10-50MB per 100 papers)
- Save intermediate results (script auto-timestamps)
- Keep `results/` gitignored (large files)

❌ **Don't**:
- Run from repo root (imports will break)
- Commit results to git
- Enable caching for benchmarking (defeats independence)
- Assume doc_id matching works (verify first)
- Start full runs without testing

## Advanced Usage

### Resume Interrupted Run

```bash
python batch_referee_reports.py \
    --pdf-dir /path/to/pdfs \
    --ground-truth tracking.csv \
    --start-from paper-042.pdf  # Resume from here
```

### Process Specific Subset

```bash
# Create temporary directory with subset
mkdir subset_pdfs
cp pdfs/tier1-*.pdf subset_pdfs/

# Process subset
python batch_referee_reports.py \
    --pdf-dir subset_pdfs/ \
    --ground-truth tracking.csv
```

### Parallel Processing (Manual)

```bash
# Terminal 1: Process papers 1-50
python batch_referee_reports.py ... --limit 50

# Terminal 2: Process papers 51-100
python batch_referee_reports.py ... --start-from paper-051.pdf --limit 50
```

**Note**: Combine results manually after (both CSVs have same schema).

## Related Documentation

- **`CLAUDE.md`** — Developer guide, architecture, extending the system
- **`docs/`** — Historical analysis documentation
- **`../app_system/README.md`** — Main application documentation
- **`../CLAUDE.md`** — Project-wide Claude Code guidance

## Model Configuration

Uses **Claude 4.5 Sonnet** for all LLM calls, configured via `../app_system/.env`:

```bash
MODEL_PRIMARY=anthropic.claude-sonnet-4-5-20250929-v1:0
```

Changes to model configuration in `app_system/` automatically apply to experiments.

## Support

**For questions or issues**:
1. Check this README and `CLAUDE.md`
2. Review `docs/` for analysis guidance
3. Test with `--limit 3` to isolate problems
4. Check `results/*.csv` for error patterns

**Common resources**:
- Ground truth format examples: `docs/`
- Sample PDFs: `ifdp_sample/`
- Dependency verification: `python test_setup.py`
