# Persona Selection Consistency Experiment

This experiment tests the consistency of the Round 0 persona selection process from Experiment 4 (10 personas). By running the selection multiple times on the same paper, we assess whether the LLM consistently chooses the same personas and assigns similar weights.

## Purpose

The persona selection system uses an LLM to choose 3 personas from a pool of 10 based on paper content. This experiment determines if the selection is stable enough for reliable use, or if we need to adjust temperature, thinking budget, or prompt design.

### Available Personas (Experiment 4)

1. **Theorist** - Mathematical logic, proofs, derivations
2. **Econometrician** - Causal inference, identification strategies
3. **ML_Expert** - Machine learning models, neural architecture
4. **Data_Scientist** - Data pipeline, preprocessing, feature engineering
5. **CS_Expert** - Algorithms, computational complexity
6. **Historian** - Literary history, research narrative context
7. **Visionary** - Paradigm shifts, intellectual novelty
8. **Policymaker** - Real-world applicability, policy implications
9. **Ethicist** - Moral values, privacy, fairness, accountability
10. **Perspective** - Distributional consequences, DEI, algorithmic fairness

## Quick Start

```bash
# Navigate to experiment directory
cd /casl/home/m1vcl00/FS-CASL/research_agents/persona_exp

# Activate virtual environment
source ../venv/bin/activate

# Run experiment (default: 10 runs)
python run_persona_selection_experiment.py /path/to/paper.pdf

# Custom number of runs
python run_persona_selection_experiment.py /path/to/paper.pdf --runs 20

# Batch processing (multiple papers)
./run_batch_experiment.sh ../papers/ 10
```

## Installation

Ensure the virtual environment has all dependencies:

```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents
source venv/bin/activate
```

Required packages (should already be installed):
- `tiktoken` - Token counting
- `requests` - API calls
- `pdfplumber` or `PyMuPDF` - PDF extraction
- All packages from `requirements.txt`

API credentials must be configured in `app_system/.env`.

## Usage

### Supported File Types

- **PDF**: `.pdf` files (uses PyMuPDF or pdfplumber)
- **Text**: `.txt` files (plain text)
- **LaTeX**: `.tex` files (LaTeX source)

### Basic Usage

Run with default settings (10 runs):

```bash
python run_persona_selection_experiment.py /path/to/paper.pdf
```

Specify number of runs:

```bash
python run_persona_selection_experiment.py /path/to/paper.pdf --runs 20
```

### Batch Processing

Process multiple papers at once:

```bash
./run_batch_experiment.sh /path/to/papers/ 15
```

Arguments:
- First: Directory containing papers (PDF/TXT/TEX)
- Second (optional): Number of runs per paper (default: 10)

## Output Structure

Each experiment run creates a **timestamped subdirectory** in `results/` with the format:

```
results/[papername]-[timestamp]/
```

Example:
```
results/
├── kd-20260420_143052/
│   ├── kd_persona_selection.csv
│   └── kd_metadata.json
├── kd-20260420_150234/
│   ├── kd_persona_selection.csv
│   └── kd_metadata.json
└── another_paper-20260420_151122/
    ├── another_paper_persona_selection.csv
    └── another_paper_metadata.json
```

This prevents results from being overridden between runs.

### CSV Results File

**Location**: `results/[papername]-[timestamp]/[papername]_persona_selection.csv`

**Columns**:
- `run_number` - Run number (1, 2, 3, ...)
- `paper_title` - Paper filename (without extension)
- `persona_1`, `persona_1_weight` - First selected persona and weight
- `persona_2`, `persona_2_weight` - Second selected persona and weight
- `persona_3`, `persona_3_weight` - Third selected persona and weight
- `justification` - LLM's explanation for the selection
- `timestamp` - ISO timestamp of the run
- `duration_seconds` - How long the run took

### Metadata JSON File

**Location**: `results/[papername]-[timestamp]/[papername]_metadata.json`

Contains:
- **Experiment details**: Date, total duration
- **Paper info**: Title, filename, token count
- **Configuration**: All technical parameters (model, temperature, max_tokens, thinking mode)
- **Consistency analysis**:
  - Unique combinations found
  - Most common combination and frequency
  - Persona selection frequency across all runs
  - Average weights per persona
  - First-position frequency (highest weight)

## Configuration

All configuration is in the `ExperimentConfig` class in `run_persona_selection_experiment.py`:

```python
class ExperimentConfig:
    # Model Configuration
    MODEL = MODEL_SECONDARY  # Claude 4.5 Sonnet (updated from 3.7)
    TEMPERATURE = 1.0  # Required for thinking mode
    MAX_TOKENS = 4096

    # Thinking Mode
    THINKING_ENABLED = True
    THINKING_BUDGET_TOKENS = 2048

    # Selection Parameters
    NUM_PERSONAS_TO_SELECT = 3
    NUM_AVAILABLE_PERSONAS = 10
```

### Why Temperature = 1.0?

The referee system uses thinking mode, which requires temperature 1.0 (Claude API requirement). This may introduce more variability than lower temperatures.

### To Modify Configuration

Edit the `ExperimentConfig` class:

1. **Change temperature**: Set `TEMPERATURE = 0.7` (also set `THINKING_ENABLED = False`)
2. **Adjust thinking budget**: Set `THINKING_BUDGET_TOKENS = 4096`
3. **Change model**: Modify `MODEL` variable

## Interpreting Results

### Consistency Metrics

The script prints consistency analysis at the end:

```
CONSISTENCY ANALYSIS
Total runs: 10
Unique combinations: 2
Most common combination:
  Personas: Econometrician, ML_Expert, Theorist
  Frequency: 8/10 (80.0%)
```

### Consistency Guidelines

**✅ Good Consistency (Production Ready)**:
- Unique combinations: 1-3 out of 10 runs
- Most common frequency: >70%
- Stable weights (low standard deviation)
- Similar justifications across runs

**⚠️ Moderate Consistency (Acceptable)**:
- Unique combinations: 3-5 out of 10 runs
- Most common frequency: 50-70%
- Moderate weight variation

**❌ Poor Consistency (Needs Calibration)**:
- Unique combinations: 6-10 out of 10 runs
- Most common frequency: <50%
- High weight variation
- Different justifications each time

### If Results Are Inconsistent

Consider these adjustments:

1. **Lower temperature**: Set `TEMPERATURE = 0.7` or `0.5` (disable thinking mode)
2. **More specific prompt**: Add constraints to selection prompt
3. **Increase thinking budget**: Set `THINKING_BUDGET_TOKENS = 4096`
4. **Add paper type**: Pre-specify paper type (empirical/theoretical/policy)
5. **Ensemble approach**: Take majority vote from 3-5 parallel selections

## Example Output

### Console Output

```
============================================================
PERSONA SELECTION CONSISTENCY EXPERIMENT
============================================================

Loading paper from: /path/to/paper.pdf
Paper title: paper
Paper length: 45823 characters, ~12000 tokens
Results will be saved to: results/paper-20260420_143052

============================================================
EXPERIMENT CONFIGURATION
============================================================
model                         : anthropic.claude-sonnet-4-5-20250929-v1:0
temperature                   : 1.0
max_tokens                    : 4096
num_personas_to_select        : 3
thinking_enabled              : True
thinking_budget_tokens        : 2048

============================================================
RUNNING 10 SELECTION ROUNDS
============================================================

Run 1
------------------------------------------------------------
Selected: Econometrician, ML_Expert, Theorist
Weights: {'Econometrician': 0.45, 'ML_Expert': 0.35, 'Theorist': 0.20}
Duration: 8.34s

[... runs 2-10 ...]

============================================================
CONSISTENCY ANALYSIS
============================================================
Total runs: 10
Unique combinations: 2

Most common combination:
  Personas: Econometrician, ML_Expert, Theorist
  Frequency: 8/10 (80.0%)

Persona selection frequency:
  Econometrician      :  9/10 (90.0%) | Avg weight: 0.395
  ML_Expert           :  9/10 (90.0%) | Avg weight: 0.352
  Theorist            :  8/10 (80.0%) | Avg weight: 0.243
  Data_Scientist      :  2/10 (20.0%) | Avg weight: 0.175

✓ Results saved to: results/paper-20260420_143052/paper_persona_selection.csv
✓ Metadata saved to: results/paper-20260420_143052/paper_metadata.json

============================================================
EXPERIMENT COMPLETE
Total duration: 84.3 seconds (1.4 minutes)
Average time per run: 8.4 seconds
============================================================
```

## Troubleshooting

### Import Errors

Make sure you're running from the correct directory:

```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents/persona_exp
source ../venv/bin/activate
python run_persona_selection_experiment.py /path/to/paper.pdf
```

### API Errors

Check your API configuration:

```bash
cd ../app_system
cat .env  # Verify API_KEY and API_BASE are set
```

### PDF Extraction Fails

- Install PyMuPDF for better extraction: `pip install PyMuPDF`
- Or convert to text first: `pdftotext paper.pdf paper.txt`

### Selection Takes Too Long

Each selection takes ~5-10 seconds. For 10 runs, expect ~1-2 minutes total.

## Research Questions

This experiment helps answer:

1. **Is temperature 1.0 too high?** - Compare unique combinations across configurations
2. **Does thinking mode help or hurt consistency?** - Test with/without thinking mode
3. **Do paper types matter?** - Run on empirical vs theoretical vs policy papers
4. **Should we use ensemble selection?** - If single runs unstable, consider majority voting
5. **Is prompt design sufficient?** - Check if justifications vary widely

## Next Steps

1. **Analyze consistency** - Review metrics from initial runs
2. **Compare across papers** - Test on 3-5 different paper types
3. **Calibrate configuration** - Adjust temperature/thinking if needed
4. **Document findings** - Note optimal parameters
5. **Integrate learnings** - Apply calibrated settings to main app

## File Structure

```
persona_exp/
├── run_persona_selection_experiment.py   # Main experiment script
├── run_batch_experiment.sh               # Batch processing
├── README.md                             # This file
├── .gitignore                            # Git ignore rules
└── results/                              # Output directory
    └── [papername]-[timestamp]/          # Per-run subdirectories
        ├── [papername]_persona_selection.csv
        └── [papername]_metadata.json
```

## Related Files

- **Main app**: `../app_system/app_exp_4.py`
- **Debate engine**: `../app_system/referee/engine_exp_4.py`
- **Configuration**: `../app_system/config.py` and `../app_system/.env`
- **Persona definitions**: `../app_system/prompts/multi_agent_debate/personas/`

## Integration with Main System

After finding optimal configuration:

1. **Update engine_exp_4.py** - Apply calibrated settings (temperature, thinking mode)
2. **Document in README_EXP_4.md** - Note expected consistency for users
3. **Consider ensemble approach** - If single-run consistency is poor, implement voting

### Example Integration

If you find temperature 0.7 works better:

```python
# In engine_exp_4.py, modify the single_query call
response = single_query(prompt, temperature=0.7)
```

Document findings:

```markdown
## Configuration Notes (in README_EXP_4.md)

Based on consistency experiments (persona_exp/), optimal configuration:
- Temperature: 0.7 (not 1.0)
- Thinking mode: Disabled for selection round
- Selection prompt: Use paper type hint when available
```
