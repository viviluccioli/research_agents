# Experiments Context

This rule activates when working on experiments, batch processing, or research evaluation.

## File Scope
- `experiment/` — Batch processing scripts
- `persona_exp/` — Persona selection experiments
- `mad_experiments/` — Experimental MAD implementations
- `referee_classifier/` — Classifier system (separate subsystem)

## Experiment Types

### 1. Batch Referee Reports (`experiment/`)

**Purpose**: Run MAD system on multiple papers + compare to ground truth.

**Key Files**:
- `batch_referee_reports.py` — Main batch processor
- `run_experiment.sh` — Shell wrapper
- `test_setup.py` — Verify environment before running

**Usage**:
```bash
cd experiment
python batch_referee_reports.py \
    --pdf-dir ifdp_sample/ \
    --ground-truth tracking.csv \
    --output-dir results/
```

**Output**:
- CSV: Results matched to ground truth (doc_id, verdict, tier, match)
- JSON: Full debate transcripts per paper

**Ground Truth Format**:
```csv
doc_id,Tier
paper123,Tier1
paper456,Tier2
```

### 2. Persona Selection Experiments (`persona_exp/`)

**Purpose**: Test consistency of Round 0 persona selection.

**Key Files**:
- `run_persona_selection_experiment.py` — Run N trials of selection
- `results/` — Per-paper experiment results

**Usage**:
```bash
cd persona_exp
python run_persona_selection_experiment.py papers/kd.pdf --runs 10 --output results/kd.csv
```

**Output**: CSV with frequency distribution of persona combinations.

**Configuration** (in script):
```python
class ExperimentConfig:
    DEFAULT_NUM_RUNS = 10
    NUM_PERSONAS_TO_SELECT = 3
    NUM_AVAILABLE_PERSONAS = 10  # Exp 4 only
```

### 3. MAD Experiments (`mad_experiments/`)

**Purpose**: Prototype new MAD variations before integrating into `app_system/`.

**Structure**:
```
mad_experiments/
├── exp-1/  # Original 5-persona
├── exp-2/  # Refinements
├── exp-3/  # Memo system prototype
└── exp_4/  # 10-persona system
```

**Workflow**:
1. Prototype in `mad_experiments/exp_N/`
2. Test in isolation (Jupyter notebooks)
3. Integrate into `app_system/` when stable
4. Document in `EXPERIMENT_N_SUMMARY.md`

**Example (Exp 4)**:
- Prototyped in `mad_experiments/exp_4/MADExpCurrent.ipynb`
- Integrated as `engine_exp_4.py` + `workflow_exp_4.py`
- Documented in `EXPERIMENT_4_SUMMARY.md`

### 4. Referee Classifier (`referee_classifier/`)

**Purpose**: Separate subsystem for classifying papers (not integrated with main app).

**Structure**:
```
referee_classifier/
├── referee/       # Classifier logic
├── tests/         # Classifier tests
├── config/        # Configuration
└── examples/      # Sample inputs
```

**Independent System**: Has its own README, tests, requirements.

## Experiment Workflow

**1. Setup Phase**:
```bash
# Verify environment
python test_setup.py

# Check API configuration
python -c "from config import MODEL_PRIMARY; print(MODEL_PRIMARY)"
```

**2. Run Phase**:
```bash
# Single run (development)
python script.py --paper path/to/paper.pdf

# Batch run (production)
./run_experiment.sh
```

**3. Analysis Phase**:
```bash
# Check results
cat results/experiment_results.csv

# Compute metrics (accuracy, precision, recall)
python analyze_results.py results/
```

## Common Patterns

### Path Resolution
All experiment scripts use:
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "app_system"))
```
This allows importing from `app_system/` without installing as package.

### Configuration Loading
```python
from config import MODEL_PRIMARY, API_BASE
from referee.engine import execute_debate_pipeline
```

### Ground Truth Matching
```python
ground_truth = load_ground_truth("tracking.csv")  # doc_id → Tier
doc_id = extract_doc_id_from_filename("paper123.pdf")
tier = ground_truth.get(doc_id)
```

### Async Execution
```python
results = asyncio.run(execute_debate_pipeline(paper_text))
```

## Output Structure

**Batch Results**:
```
results/
├── experiment_results.csv       # Summary table
├── experiment_results.json      # Full transcripts
└── paper_123/                   # Per-paper outputs
    ├── debate_transcript.txt
    └── persona_reports.json
```

**Persona Selection Results**:
```
results/
└── kd-20260420_152801/
    ├── experiment_metadata.json
    ├── selection_results.csv
    └── frequency_distribution.json
```

## Testing Experiments

**Before Large Batch Runs**:
1. Run on 1-2 papers first
2. Verify ground truth matching
3. Check output format
4. Estimate cost/time

**Cost Estimation**:
```python
# Each paper: ~$1.50-2.00 (5-persona)
# Each paper: ~$2.00-3.00 (10-persona)
# Estimate: num_papers × cost_per_paper
```

## Integration Checklist

**When Integrating Experiment → Production**:

1. ✅ Copy core logic to `app_system/`
2. ✅ Update `__init__.py` exports
3. ✅ Add UI in `workflow.py` or create new workflow
4. ✅ Update `CLAUDE.md` with new features
5. ✅ Add tests in `app_system/tests/`
6. ✅ Document in `docs/` if complex
7. ✅ Update `README.md` with usage
8. ✅ Create experiment summary (e.g., `EXPERIMENT_4_SUMMARY.md`)

## Common Pitfalls

❌ **Don't** run large batches without testing on small sample first
❌ **Don't** forget to check API rate limits for batch jobs
❌ **Don't** hardcode paths — use `Path(__file__).parent`
❌ **Don't** commit experiment results to git (add to `.gitignore`)
✅ **Do** save metadata (model, temperature, timestamp) with results
✅ **Do** validate ground truth CSV format before running
✅ **Do** use async for parallel processing where possible
✅ **Do** document experimental configurations in code comments

## Analysis Tips

**CSV Analysis** (pandas):
```python
df = pd.read_csv("results/experiment_results.csv")
accuracy = (df['prediction'] == df['ground_truth']).mean()
confusion_matrix = pd.crosstab(df['ground_truth'], df['prediction'])
```

**JSON Inspection** (jq):
```bash
cat results/experiment_results.json | jq '.papers[] | {doc_id, verdict, personas}'
```
