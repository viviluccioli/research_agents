# App System Migration Guide

## Overview

All production app files have been moved to the `app_system/` subdirectory for better organization.

## What Was Moved

### Moved to `app_system/`:

**Application Files:**
- ✅ `app.py` - Main production app (Referee Report + Section Evaluator)
- ✅ `app_demo.py` - Demo 1 (madoutput1.txt - Adjusted R² issue)
- ✅ `app_demo2.py` - Demo 2 (madouput2.txt - standard errors issue)

**Core Module Files:**
- ✅ `referee.py` - Referee Report Checker workflow
- ✅ `multi_agent_debate.py` - Multi-agent debate orchestration
- ✅ `utils.py` - LLM utilities (single_query, ConversationManager)

**Module Directory:**
- ✅ `section_eval/` - Complete section evaluator module (with all subdirectories)

**Demo Data:**
- ✅ `madoutput1.txt` - Demo 1 debate transcript
- ✅ `madouput2.txt` - Demo 2 debate transcript (note: typo in original filename preserved)

## What Stayed in `eval/`

**Experimental Scripts:**
- `madexp.py` - Original experimental multi-agent debate
- `madexp2.py` - Gemini-based Colab version
- `madexp2_local.py` - Local version of madexp2

**Legacy Section Evaluator Versions:**
- `section_eval.py` - Older version
- `section_eval_new.py` - Intermediate version
- `section_eval_vivi_0223.py` - Historical version
- `section_eval_llm_vivi.py` - Historical version

**Utilities:**
- `routing.py` - Routing utilities (not used by app.py)

**Documentation (All .md files):**
- `ARCHITECTURE.md`
- `CRITERIA_MATRIX.md`
- `CRITERIA_REFERENCE.md`
- `DEMO_README.md`
- `EVALUATION_SYSTEM_IMPROVEMENTS.md`
- `Evaluation_system_prompts.md`
- `QUICK_REFERENCE.md`
- `README_DOCS.md`
- `SCORING_EXAMPLES_OLD_VS_NEW.md`
- `SYSTEM_PROMPTS_DOCUMENTATION.md`
- `TESTING_NEW_SYSTEM.md`

**Other:**
- `changelog/` - Change history directory
- `comparative results/` - Results comparison directory
- `mad1_table.png` - Demo visualization
- `pyproject.toml` - Python project config
- `__pycache__/` - Python cache directory

## Running the Apps

### From the `eval/` directory (recommended):

```bash
cd /ofs/home/m1vcl00/FS-CASL/research_agents-main/eval

# Run main app
streamlit run app_system/app.py

# Run demo apps
streamlit run app_system/app_demo.py
streamlit run app_system/app_demo2.py
```

### From the `app_system/` directory:

```bash
cd /ofs/home/m1vcl00/FS-CASL/research_agents-main/eval/app_system

# Run main app
streamlit run app.py

# Run demo apps
streamlit run app_demo.py
streamlit run app_demo2.py
```

## Import Structure

All imports within `app_system/` are relative and work correctly:

```python
# In app.py
from referee import RefereeReportChecker          # ✓ Same directory
from section_eval import SectionEvaluatorApp      # ✓ Subdirectory
from utils import cm                               # ✓ Same directory

# In referee.py
from utils import cm, single_query                 # ✓ Same directory
from multi_agent_debate import execute_debate_pipeline  # ✓ Same directory

# In multi_agent_debate.py
from utils import single_query                     # ✓ Same directory
```

No import statements needed to be changed.

## Benefits of This Structure

1. **Cleaner Organization**: Production app files are separated from experimental scripts
2. **Easier Maintenance**: All app-related files in one place
3. **Clear Separation**: Documentation stays in parent directory for easy access
4. **Preserved Functionality**: All imports work exactly as before
5. **Scalability**: Easy to add new features to app_system without cluttering parent directory

## Next Steps

1. Run the apps from the new location to verify everything works
2. Update any deployment scripts to reference `app_system/app.py`
3. Update documentation to reference the new structure
4. Consider moving experimental scripts to a separate `experiments/` directory

## Rollback (If Needed)

To revert the migration:

```bash
cd /ofs/home/m1vcl00/FS-CASL/research_agents-main/eval
mv app_system/*.py .
mv app_system/*.txt .
mv app_system/section_eval .
rmdir app_system
```

## File Count Summary

- **Moved to app_system/**: 8 files + 1 directory (section_eval/)
- **Stayed in eval/**: 11 Python files + 11 documentation files + 3 directories
- **Total files before**: 30+
- **Total files after**: Same, but organized

## Migration Date

March 11, 2026
