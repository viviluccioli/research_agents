# Experiment 4: Multi-Agent Debate with 10 Personas

## Overview

This experimental version (`app_exp_4.py`) implements an expanded multi-agent debate system with **10 persona options** instead of the original 5. The system still selects **3 personas** for each paper evaluation, but now has a wider range of specialized personas to choose from based on paper content.

## Changes from Base Version

### New Personas Added

In addition to the original 5 personas (Theorist, Empiricist, Historian, Visionary, Policymaker), we've added:

6. **Econometrician**: Focuses specifically on causal inference, identification strategies, and robust interpretation (more specific than the original "Empiricist")
7. **ML_Expert**: Specializes in machine learning models, architecture decisions, hyperparameter tuning, and model interpretability
8. **Data_Scientist**: Focuses on data pipeline, cleaning, feature engineering, EDA, and preprocessing biases
9. **CS_Expert**: Evaluates algorithm design, computational complexity, memory efficiency, and hardware constraints
10. **Ethicist**: Examines moral values, privacy, consent, fairness, and accountability
11. **Perspective**: Analyzes distributional consequences, algorithmic fairness, DEI issues, and impact on marginalized groups

### What Stays the Same

- **Debate structure**: Still uses the same 5-round debate process (Round 0: Selection, Round 1: Independent Evaluation, Rounds 2A/2B/2C: Cross-examination and amendments, Round 3: Editor decision)
- **Selection count**: Still selects exactly 3 personas (not all 10)
- **Consensus calculation**: Same weighted consensus system (PASS=1.0, REVISE=0.5, FAIL=0.0)
- **PDF parsing**: Same PyMuPDF extraction with figure support
- **Output structure**: Same Excel/MD/PDF exports with full debate transcripts
- **Quote validation**: Same validation system using fuzzy string matching

## Files Created

The experiment creates these new files while leaving the original system intact:

```
app_system/
├── app_exp_4.py                          # New entry point
├── referee/
│   ├── engine_exp_4.py                   # Debate orchestration with 10 personas
│   └── workflow_exp_4.py                 # UI wrapper (imports from engine_exp_4)
└── prompts/multi_agent_debate/personas/
    ├── econometrician/v1.0.txt           # New persona prompts
    ├── ml_expert/v1.0.txt
    ├── data_scientist/v1.0.txt
    ├── cs_expert/v1.0.txt
    ├── ethicist/v1.0.txt
    └── perspective/v1.0.txt
```

## Running the Experiment

### From the app_system directory:

```bash
cd app_system
source ../venv/bin/activate
streamlit run app_exp_4.py
```

Or use the provided script:

```bash
cd app_system
bash run_app_exp_4.sh  # (Create this if needed)
```

### To run without file watcher (for systems with inotify limits):

```bash
cd app_system
streamlit run app_exp_4.py --server.fileWatcherType none
```

## Using the Experiment

1. **Launch the app** using the command above
2. **Upload your paper** (PDF/LaTeX/text)
3. **Select paper type** (empirical/theoretical/policy) for better persona recommendations
4. **Choose persona selection mode**:
   - 🤖 **Fully Automatic**: LLM selects 3 of 10 personas and assigns weights
   - 🎯 **Manual Selection**: You pick personas, LLM assigns weights
   - ⚙️ **Full Manual**: You pick personas AND assign weights
5. **Run the debate** and review the multi-round evaluation

## Comparison with Base System

| Feature | Base System | Experiment 4 |
|---------|-------------|--------------|
| **Persona Options** | 5 (Theorist, Empiricist, Historian, Visionary, Policymaker) | 10 (adds Econometrician, ML_Expert, Data_Scientist, CS_Expert, Ethicist, Perspective) |
| **Personas Selected** | 3 | 3 (same) |
| **Debate Rounds** | 5 rounds | 5 rounds (same structure) |
| **Consensus System** | Weighted consensus | Weighted consensus (same) |
| **Manual Selection** | Yes | Yes (with 10 options) |
| **Paper Types** | Empirical, Theoretical, Policy | Same |
| **PDF Extraction** | PyMuPDF with figures | Same |
| **Quote Validation** | Yes | Yes |

## Technical Details

### Selection Prompt

The Round 0 selection prompt has been updated to include all 10 personas with descriptions:

```python
SELECTION_PROMPT = """
You are the Chief Editor of an economics journal. You must select exactly {N} expert personas
to review the provided paper.

The available personas are:
1. "Theorist": Rigorous mathematical logic, logically airtight explanations and proof
2. "Econometrician": Compelling causal inference, well-defined identification strategies
3. "ML_Expert": Fundamental ML models, neural architecture, hyperparameter decisions
4. "Data_Scientist": Data cleaning, preprocessing, visualization, interpretation
5. "CS_Expert": Algorithm creation, computational complexity, scale
6. "Historian": Literary history, subject-matter context, research narrative
7. "Visionary": Potential for paradigm shifts; broad intellectual novelty
8. "Policymaker": Real-world applicability, regulatory use, welfare implications
9. "Ethicist": Moral values, privacy, fairness, accountability
10. "Perspective": Distributional consequences, DEI, impact on marginalized groups
...
"""
```

### Persona System Prompts

Each persona has a focused role definition and receives the standard error severity guide. Prompts are loaded from versioned files in `prompts/multi_agent_debate/personas/`.

### Debate Prompts

Debate round prompts (2A, 2B, 2C, Round 3) remain structurally the same but with updated formatting to match the experimental notebook style.

## Use Cases

This experimental version is particularly useful for papers that:

- **Use machine learning methods** → ML_Expert and Data_Scientist provide specialized feedback
- **Involve computational algorithms** → CS_Expert evaluates complexity and implementation
- **Have ethical implications** → Ethicist and Perspective examine fairness and social impact
- **Require strong causal inference** → Econometrician provides targeted identification strategy review

## Notes

- The original `app.py` remains unchanged and functional
- Both versions can coexist and be run independently
- Persona prompts are stored in `prompts/multi_agent_debate/personas/` following the standard versioning convention
- The system automatically loads persona prompts via `load_persona_prompt()` in `engine_exp_4.py`
- If a persona file is not found, it falls back to hardcoded prompts in `FALLBACK_SYSTEM_PROMPTS`

## Future Experiments

Potential extensions:
- **Variable N**: Allow selecting 4 or 5 personas instead of fixed 3
- **Specialized subareas**: Add finance, macro, or systematic review personas
- **Dynamic weighting**: Adjust persona weights based on debate performance
- **Hierarchical debate**: Two-tier system with primary and advisory personas

## References

- Base system: `app_system/referee/engine.py`, `app_system/referee/workflow.py`
- Experimental notebook: `mad_experiments/exp_4/MADExpCurrent.ipynb`
- Original design doc: `app_system/docs/FRAMEWORK.md`
