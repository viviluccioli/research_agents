# CLAUDE.md — Experiment System

This file provides guidance to Claude Code when working with the batch experiment system for evaluating the multi-agent referee report system.

## Purpose

The `experiment/` directory contains tools for **benchmarking and evaluating** the referee report system (`app_system/referee/`) against ground truth data. It runs the MAD (Multi-Agent Debate) system on batches of papers and outputs structured results for analysis.

**Key distinction**: This is an **evaluation/research tool**, not part of the production application. It measures system performance, not end-user functionality.

## System Architecture

```
experiment/
├── batch_referee_reports.py    # Main batch processor
├── run_experiment.sh            # Shell wrapper with defaults
├── test_setup.py                # Dependency verification
├── .gitignore                   # Excludes results/ from git
│
├── docs/                        # Analysis documentation
│   ├── README.md                # Docs index
│   ├── METRICS_EXPLAINED.md     # Metric definitions
│   ├── SCORING_RUBRIC.md        # Scoring system docs
│   └── *.md                     # Other analysis guides
│
├── _archived/                   # One-off analysis scripts
│   ├── analyze_*.py             # Post-hoc analysis tools
│   ├── test_*.py                # Development tests
│   └── README.md                # Archive index
│
├── results/                     # Output directory (gitignored)
│   ├── *.csv                    # Structured batch results
│   └── *.json                   # Full debate transcripts
│
└── ifdp_sample/                 # Sample PDFs for testing
```

## Core Script: batch_referee_reports.py

### What It Does

For each PDF in a directory:
1. Extracts `doc_id` from filename (removes `.pdf` extension)
2. Looks up ground truth `Tier` from CSV
3. Extracts text via `section_eval.text_extraction`
4. Runs full 5-round MAD pipeline (`referee.engine.execute_debate_pipeline`)
5. Extracts verdicts, scores, weights for all personas
6. Computes consensus metrics (categorical + numeric)
7. Saves structured results to CSV + JSON

### Import Pattern

```python
import sys
from pathlib import Path

# Add app_system to path
sys.path.insert(0, str(Path(__file__).parent.parent / "app_system"))

from referee.engine import execute_debate_pipeline, extract_verdict_from_report, extract_score_from_report
from section_eval.text_extraction import extract_text_from_pdf
```

**CRITICAL**: Script must be run from `experiment/` directory so relative path resolves correctly.

### Configuration

**Hardcoded settings** (line 207-219):
```python
results = asyncio.run(execute_debate_pipeline(
    paper_text=paper_text,
    progress_callback=None,
    paper_type=None,              # LLM auto-detects
    custom_context=None,
    manual_personas=None,         # LLM auto-selects
    manual_weights=None,
    enable_quote_validation=False, # Disabled for speed
    use_cache=False,
    force_refresh=True
))
```

**To enable quote validation**: Change line 218 to `enable_quote_validation=True`  
**To enable caching**: Change line 219 to `use_cache=True`

## Usage

### Basic Usage

```bash
cd experiment

python batch_referee_reports.py \
    --pdf-dir /path/to/pdfs \
    --ground-truth /path/to/ground_truth.csv \
    --output-dir ./results
```

### With Limit (Testing)

```bash
python batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth tracking.csv \
    --limit 5
```

### Resume From Specific File

```bash
python batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth tracking.csv \
    --start-from ifdp-2020-5.pdf
```

### Using Shell Wrapper

Edit `run_experiment.sh` to set default paths, then:

```bash
bash run_experiment.sh          # All PDFs
bash run_experiment.sh 10       # First 10 PDFs
```

## Ground Truth Format

**Required CSV columns**:
- `doc_id` — Paper identifier (must match PDF filename without `.pdf`)
- `Tier` — Ground truth category/quality tier

**Example**:
```csv
doc_id,Tier,other_optional_columns
ifdp-2020-1,Tier1,Published
ifdp-2020-2,Tier3,Rejected
ifdp-2020-3,Tier2,Revise
```

**Doc ID Matching**:
- `ifdp-2020-1.pdf` → `doc_id = "ifdp-2020-1"`
- `paper123.pdf` → `doc_id = "paper123"`

## Output Schema (35 Columns)

### Basic Metadata (Columns 1-6)
1. `doc_id` — Document identifier from ground truth
2. `filename` — PDF filename
3. `duration_seconds` — Processing time (seconds)
4. `duration_formatted` — Processing time (MM:SS)
5. `char_count` — Characters in extracted text
6. `word_count` — Words in extracted text

### Persona 1 (Columns 7-12)
7. `persona_1_name` — First persona name
8. `persona_1_weight` — First persona weight (0-1, sum to 1.0)
9. `persona_1_round1_verdict` — Round 1 verdict (PASS/REVISE/FAIL)
10. `persona_1_final_verdict` — Round 2C verdict (PASS/REVISE/FAIL)
11. `persona_1_round1_score` — Round 1 confidence score (1-10)
12. `persona_1_final_score` — Round 2C confidence score (1-10)

### Persona 2 (Columns 13-18)
*Same structure as Persona 1*

### Persona 3 (Columns 19-24)
*Same structure as Persona 1*

### Final Results (Columns 25-28)
25. `final_verdict` — Editor decision (PASS/REVISE/FAIL)
26. `total_final_score` — Sum of all persona final scores (max 30). This is the unweighted sum of `persona_1_final_score + persona_2_final_score + persona_3_final_score`. Provides a simple indicator of overall panel confidence that can help identify boundary cases (e.g., a total of 15 suggests verdicts may be on the edge between categories). Set to `None` if any persona did not provide a score. **Placed next to `final_verdict` for easy comparison.**
27. `final_report` — Full final referee report text
28. `tier` — Ground truth tier from CSV

### Consensus Metrics - Categorical (Columns 29-32)
29. `consensus_score_r1` — Weighted verdict score Round 1 (0-1)
30. `consensus_score_r2c` — Weighted verdict score Round 2C (0-1)
31. `agreement_level_r1` — Agreement level Round 1 (UNANIMOUS/PARTIAL/DIVERGENT)
32. `agreement_level_r2c` — Agreement level Round 2C (UNANIMOUS/PARTIAL/DIVERGENT)

### Consensus Metrics - Numeric (Columns 33-35)
33. `round1_consensus_score_numeric` — Weighted average of confidence scores Round 1 (0-1)
34. `round2c_consensus_score_numeric` — Weighted average of confidence scores Round 2C (0-1)
35. `consensus_delta_numeric` — Change in numeric consensus (R2C - R1)

## Verdict Mapping

**Editor Decision → Final Verdict**:
- `ACCEPT` → `PASS`
- `REJECT AND RESUBMIT` → `REVISE`
- `REJECT` → `FAIL`

## Consensus Metrics Explained

### Categorical Consensus Score (Columns 28-29)

Weighted average of verdicts mapped to numeric values:
- `PASS = 1.0`
- `REVISE = 0.5`
- `FAIL = 0.0`

**Formula**:
```python
consensus_score = sum(VERDICT_VALUES[verdict] * weight for verdict, weight in zip(verdicts, weights))
```

**Example**:
- Persona 1 (weight 0.5): REVISE → 0.5
- Persona 2 (weight 0.3): PASS → 1.0
- Persona 3 (weight 0.2): FAIL → 0.0
- **Consensus**: 0.5×0.5 + 1.0×0.3 + 0.0×0.2 = **0.55**

### Agreement Level (Columns 30-31)

Measures consensus among personas (ignores weights):
- **UNANIMOUS**: All personas have same verdict
- **PARTIAL**: 2 personas agree, 1 differs
- **DIVERGENT**: All 3 personas have different verdicts

### Numeric Consensus Score (Columns 32-34)

Weighted average of **confidence scores** (1-10 scale, normalized to 0-1):

**Formula**:
```python
numeric_consensus = sum((score / 10.0) * weight for score, weight in zip(scores, weights))
```

**Only computed if all personas provide scores** (some personas may not report numeric scores in all cases).

**Consensus Delta** (Column 34): Change from Round 1 to Round 2C, indicating how much consensus shifted during debate.

## Performance Notes

**Speed Optimizations**:
- Quote validation **disabled** by default (saves ~15-20% time)
- Caching **disabled** by default (ensures independent evaluations)

**Timing Estimates** (Claude 4.5 Sonnet):
- Short paper (~5K words): 2-3 minutes
- Medium paper (~8K words): 3-5 minutes
- Long paper (~12K words): 5-8 minutes
- **100 papers**: ~4-8 hours

**Cost Estimates** (Claude 4.5 Sonnet):
- Per paper: ~$1.50-2.50
- 100 papers: ~$150-250

## Error Handling

If a paper fails processing:
- Error logged to console
- Placeholder record added with `final_verdict = "ERROR"`
- All numeric fields set to 0/None
- Processing continues with remaining papers

**Common failure causes**:
- PDF extraction failed (scanned image, corrupted file)
- Text too short (<100 chars)
- API call timeout/failure
- Malformed debate output

## Testing Before Large Runs

**Step 1**: Verify setup
```bash
python test_setup.py
```

**Step 2**: Small batch test
```bash
python batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth tracking.csv \
    --limit 3
```

**Step 3**: Inspect output
```bash
head -20 results/referee_batch_results_*.csv
```

**Step 4**: Full run
```bash
bash run_experiment.sh
```

## Analysis Workflow

**After batch processing completes**:

1. **Load results in Python**:
   ```python
   import pandas as pd
   df = pd.read_csv('results/referee_batch_results_YYYYMMDD_HHMMSS.csv')
   ```

2. **Compare predictions to ground truth**:
   ```python
   # Verdict accuracy
   df['match'] = df['final_verdict'] == df['tier']
   accuracy = df['match'].mean()
   
   # Confusion matrix
   confusion = pd.crosstab(df['tier'], df['final_verdict'])
   ```

3. **Analyze consensus patterns**:
   ```python
   # Agreement by tier
   agreement_by_tier = df.groupby('tier')['agreement_level_r2c'].value_counts(normalize=True)
   
   # Consensus score distributions
   df.groupby('tier')['consensus_score_r2c'].describe()
   ```

4. **Examine persona selections**:
   ```python
   # Persona frequency
   personas = pd.concat([df['persona_1_name'], df['persona_2_name'], df['persona_3_name']])
   persona_freq = personas.value_counts()
   ```

5. **Review failures**:
   ```python
   # Papers with errors
   errors = df[df['final_verdict'] == 'ERROR']
   
   # Low agreement cases
   divergent = df[df['agreement_level_r2c'] == 'DIVERGENT']
   ```

## Documentation

### Active Docs
- **`README.md`** — User-facing usage guide
- **`CLAUDE.md`** — This file (developer guide)
- **`test_setup.py`** — Dependency checker

### Reference Docs (`docs/`)
Historical analysis and development documentation:
- `METRICS_EXPLAINED.md` — Detailed metric definitions
- `SCORING_RUBRIC.md` — Scoring system design
- `CALIBRATION_ANALYSIS_SUMMARY.md` — Calibration analysis
- `TRAJECTORY_ANALYSIS_COMPLETE.md` — Verdict trajectory analysis

**Note**: `docs/` contains historical context, may not reflect current implementation.

### Archived Scripts (`_archived/`)
One-off analysis tools used during development:
- `analyze_existing_results.py` — Post-hoc CSV analysis
- `analyze_personas.py` — Persona frequency analysis
- `test_scoring_system.py` — Scoring logic tests

**Note**: Archived scripts not maintained, use at own risk.

## Model Configuration

**CRITICAL**: Uses same model configuration as `app_system/`:
- **Model**: Claude 4.5 Sonnet (`MODEL_PRIMARY`)
- **Temperature**: 1.0 (with thinking mode enabled)
- **Retries**: 3× with 5s delay
- **API**: Configured via `.env` in `app_system/`

Imports from `app_system/config.py`, so changes to model configuration automatically apply.

## Integration with app_system/

**Import dependencies** (read-only):
```python
from referee.engine import execute_debate_pipeline, extract_verdict_from_report, extract_score_from_report
from section_eval.text_extraction import extract_text_from_pdf
from config import MODEL_PRIMARY
```

**Does NOT modify** `app_system/` code — purely consumes the referee system API.

**To update experiment after app_system/ changes**:
1. No code changes needed if API unchanged
2. May need to update output schema if new fields added to debate results
3. Verify with `python test_setup.py` after major app_system/ changes

## Common Pitfalls

❌ **Don't** run from repo root — imports will break  
❌ **Don't** commit results/ to git — large files, gitignored  
❌ **Don't** enable caching for ground truth evaluation — defeats independence  
❌ **Don't** assume doc_id matching — verify PDF filenames match CSV exactly  
✅ **Do** test with `--limit 3` before full runs  
✅ **Do** verify ground truth CSV format before processing  
✅ **Do** check disk space (results can be 10-50MB per 100 papers)  
✅ **Do** save intermediate results (script creates timestamped files)

## Extending the System

### Adding New Metrics

**Step 1**: Compute metric in `process_single_pdf()` (around line 375)
```python
new_metric = compute_new_metric(results)
result['new_metric'] = new_metric
```

**Step 2**: Add to CSV columns list (around line 400)
```python
csv_columns = [
    # ... existing columns ...
    'new_metric',
]
```

**Step 3**: Test with single paper
```bash
python batch_referee_reports.py --limit 1 ...
head -1 results/*.csv | tr ',' '\n' | grep new_metric
```

### Adding New CLI Flags

**Step 1**: Add argument to parser (around line 530)
```python
parser.add_argument(
    '--new-flag',
    type=str,
    help='Description of new flag'
)
```

**Step 2**: Use in `execute_debate_pipeline()` call (around line 210)
```python
results = asyncio.run(execute_debate_pipeline(
    # ... existing args ...
    new_param=args.new_flag
))
```

### Custom Analysis Scripts

Create in `experiment/` (not `_archived/`):

```python
#!/usr/bin/env python3
"""
Custom analysis script.
"""
import pandas as pd
import sys
from pathlib import Path

# Load most recent results
results_dir = Path(__file__).parent / "results"
csv_files = sorted(results_dir.glob("referee_batch_results_*.csv"))
latest = csv_files[-1]

df = pd.read_csv(latest)

# Your analysis here
print(df.describe())
```

Make executable and run:
```bash
chmod +x my_analysis.py
python my_analysis.py
```

## Relationship to Other Experiments

**`persona_exp/`** — Persona selection consistency experiments (separate codebase)  
**`mad_experiments/`** — Prototype MAD variations (separate codebase)  
**`referee_classifier/`** — Paper classification system (separate subsystem)

**This directory** focuses on **end-to-end system evaluation** with ground truth benchmarking, while others focus on specific subsystem experiments.

## Summary

**Purpose**: Batch evaluation of referee report system against ground truth  
**Input**: Directory of PDFs + ground truth CSV  
**Output**: Structured CSV/JSON with verdicts, scores, consensus metrics  
**Runtime**: ~3-5 min/paper on Claude 4.5 Sonnet  
**Use Case**: Benchmarking, calibration, performance tracking  
**Not For**: Production end-user application (use `app_system/app.py` instead)
