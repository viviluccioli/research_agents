# Experiment 4 Implementation Summary

## Overview

Successfully created a new version of the referee report system (`app_exp_4.py`) based on the experimental framework from `mad_experiments/exp_4/MADExpCurrent.ipynb`. This version expands the multi-agent debate system from **5 personas** to **10 personas** (still selecting 3).

## Files Created

### 1. Core Engine and Workflow

| File | Description | Size |
|------|-------------|------|
| `referee/engine_exp_4.py` | Debate orchestration engine with 10 personas | 33 KB |
| `referee/workflow_exp_4.py` | Streamlit UI wrapper (copied from workflow.py, updated imports) | 87 KB |
| `app_exp_4.py` | Main entry point with tabbed interface | 18 KB |

### 2. Persona Prompts

Created 6 new persona prompt files in `prompts/multi_agent_debate/personas/`:

| Persona | File | Focus Area |
|---------|------|------------|
| **Econometrician** | `econometrician/v1.0.txt` | Causal inference, identification strategies |
| **ML_Expert** | `ml_expert/v1.0.txt` | Model architecture, hyperparameters, interpretability |
| **Data_Scientist** | `data_scientist/v1.0.txt` | Data pipeline, preprocessing, feature engineering |
| **CS_Expert** | `cs_expert/v1.0.txt` | Algorithms, computational complexity |
| **Ethicist** | `ethicist/v1.0.txt` | Moral values, privacy, fairness |
| **Perspective** | `perspective/v1.0.txt` | DEI, distributional consequences, algorithmic fairness |

### 3. Documentation

| File | Description |
|------|-------------|
| `README_EXP_4.md` | Full documentation of the experiment, usage guide, comparison table |
| `run_app_exp_4.sh` | Shell script to launch the app (disables file watcher) |
| `EXPERIMENT_4_SUMMARY.md` | This file - quick reference |

## Key Differences from Base System

### What Changed

1. **Persona Selection Prompt** (`SELECTION_PROMPT` in engine_exp_4.py):
   - Now lists 10 personas instead of 5
   - Updated descriptions to match exp_4 framework
   - Still selects exactly 3 personas

2. **System Prompts** (`SYSTEM_PROMPTS` in engine_exp_4.py):
   - Added 6 new personas
   - Each loads from versioned .txt files
   - Fallback to hardcoded prompts if file loading fails

3. **Debate Prompts** (`DEBATE_PROMPTS` in engine_exp_4.py):
   - Slightly updated formatting to match exp_4 notebook
   - "Principle of Charity" language in Round 2A
   - More concise output format instructions

4. **UI Updates** (workflow_exp_4.py):
   - Persona cards now display all 10 personas in 2 rows
   - Manual selection interface supports 10 personas
   - Updated CSS classes for new personas

### What Stayed the Same

- ✅ Same 5-round debate structure (R0, R1, R2A, R2B, R2C, R3)
- ✅ Same consensus calculation (weighted average, thresholds)
- ✅ Same PDF extraction (PyMuPDF with figures)
- ✅ Same output formats (Excel, Markdown, PDF)
- ✅ Same quote validation system
- ✅ Same manual persona selection options
- ✅ Same paper type integration
- ✅ Same cost estimation and token tracking

## Running the Experiment

### Quick Start

```bash
cd app_system
bash run_app_exp_4.sh
```

### Manual Start

```bash
cd app_system
source ../venv/bin/activate
streamlit run app_exp_4.py
```

### Access

Open browser to: **http://localhost:8501**

## Testing the New Personas

To see the new personas in action:

1. **Upload a machine learning paper** → Should select ML_Expert, Data_Scientist, and possibly CS_Expert or Econometrician
2. **Upload a fairness/ethics paper** → Should select Ethicist, Perspective, and possibly ML_Expert
3. **Upload a computational paper** → Should select CS_Expert, ML_Expert, and possibly Econometrician or Theorist

## Persona Selection Examples

Based on paper content, the system might select:

| Paper Type | Likely Personas |
|------------|-----------------|
| **ML/NLP paper** | ML_Expert, Data_Scientist, Econometrician |
| **Algorithmic fairness** | Ethicist, Perspective, ML_Expert |
| **Computational economics** | CS_Expert, Econometrician, Theorist |
| **Causal inference** | Econometrician, ML_Expert (if using ML), Policymaker |
| **Traditional econometrics** | Econometrician, Historian, Policymaker |
| **Pure theory** | Theorist, Historian, Visionary |

## Code Structure

```
app_system/
├── app_exp_4.py                    # New entry point
├── run_app_exp_4.sh                # Launch script
├── README_EXP_4.md                 # Full documentation
├── EXPERIMENT_4_SUMMARY.md         # This file
│
├── referee/
│   ├── engine.py                   # Original engine (5 personas)
│   ├── engine_exp_4.py            # ✨ New engine (10 personas)
│   ├── workflow.py                 # Original workflow
│   └── workflow_exp_4.py          # ✨ New workflow (uses engine_exp_4)
│
└── prompts/multi_agent_debate/personas/
    ├── theorist/v1.0.txt           # Original personas
    ├── empiricist/v1.0.txt
    ├── historian/v1.0.txt
    ├── visionary/v1.0.txt
    ├── policymaker/v1.0.txt
    ├── econometrician/v1.0.txt    # ✨ New personas
    ├── ml_expert/v1.0.txt
    ├── data_scientist/v1.0.txt
    ├── cs_expert/v1.0.txt
    ├── ethicist/v1.0.txt
    └── perspective/v1.0.txt
```

## Implementation Notes

1. **Prompt Loading**: The system uses `load_persona_prompt()` to dynamically load persona prompts from files, with fallback to hardcoded prompts.

2. **Error Severity Guide**: All personas receive the same error severity guide via the `{error_severity}` placeholder in their prompt files.

3. **Backwards Compatibility**: The original `app.py` and base engine/workflow remain untouched and fully functional.

4. **Manual Selection**: The UI supports manual selection of any combination of 2-5 personas from the full set of 10.

5. **Async Execution**: All debate rounds use `asyncio.gather()` for parallel execution, maintaining performance with 3 personas.

## Next Steps

### Testing Recommendations

1. Test with papers in different domains (ML, fairness, computational, traditional econ)
2. Compare results between base system and exp_4 on the same paper
3. Try manual persona selection to force specific combinations
4. Monitor persona selection patterns across different paper types

### Potential Improvements

1. **Variable N**: Allow selecting 4-5 personas instead of fixed 3
2. **Persona Interactions**: Track which persona combinations work best together
3. **Specialization Scores**: Measure how specialized vs. generalist persona mix affects quality
4. **Domain Routing**: Auto-route papers to exp_4 if they use ML/fairness keywords

### Future Experiments

- **Exp 5**: Dynamic persona weighting based on debate performance
- **Exp 6**: Hierarchical debate (primary + advisory personas)
- **Exp 7**: Domain-specific persona clusters (finance, macro, etc.)

## Original Experiment Reference

This implementation is based on:
- **Notebook**: `mad_experiments/exp_4/MADExpCurrent.ipynb`
- **Key Changes**: Expanded from 5 to 10 personas, updated prompt formatting
- **Core Logic**: Preserved the serial execution pattern from the notebook but adapted for production use

## Support

For questions or issues:
1. Check `README_EXP_4.md` for full documentation
2. Compare with base system in `referee/engine.py`
3. Review persona prompts in `prompts/multi_agent_debate/personas/`
4. Check experiment notebook: `mad_experiments/exp_4/MADExpCurrent.ipynb`
