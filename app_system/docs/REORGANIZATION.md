# App System Reorganization (2026-03-30)

## Overview

The `app_system/` directory has been reorganized to follow Python best practices and improve code maintainability.

## What Changed

### Before (Cluttered Root)
```
app_system/
├── app.py
├── app_summary.py               # Alternate entry point
├── full_output_app.py           # Another alternate entry point
├── referee.py                   # Referee report implementation
├── referee_summarized.py        # Summarized version
├── multi_agent_debate.py        # Debate orchestration
├── debate_summarizer.py         # Summarization utilities
├── test_*.py (5 files)          # Tests scattered in root
├── utils.py
├── section_eval/
├── prompts/
├── demos/
└── docs/
```

### After (Organized)
```
app_system/
├── app.py                       # ✓ Main entry point (clean root)
├── utils.py                     # ✓ Shared utilities
├── run_app.sh                   # ✓ Launch helper
├── README.md                    # ✓ User docs
│
├── referee/                     # ✓ NEW: Referee package
│   ├── __init__.py             # Package exports
│   ├── core.py                 # referee.py → renamed
│   ├── summarized.py           # referee_summarized.py → moved
│   ├── debate.py               # multi_agent_debate.py → moved
│   └── summarizer.py           # debate_summarizer.py → moved
│
├── section_eval/                # ✓ Existing package
├── prompts/                     # ✓ External prompts
│
├── tests/                       # ✓ NEW: All tests
│   ├── __init__.py
│   └── test_*.py (5 files)
│
├── demos/                       # ✓ Demo apps
│   ├── app_full_output.py      # full_output_app.py → moved
│   ├── app_summarized_only.py  # app_summary.py → moved
│   └── app_demo*.py
│
├── docs/                        # ✓ Documentation
│   └── *.md (all docs except README.md)
│
└── results/                     # ✓ Output directory
```

## Benefits

### 1. Standard Python Package Structure
- Clear package boundaries with `__init__.py` files
- Proper module namespacing (`referee.core`, `referee.debate`)
- Easy to understand imports: `from referee import RefereeReportChecker`

### 2. Separation of Concerns
- **Tests** isolated in `tests/` directory
- **Documentation** centralized in `docs/`
- **Demos** separated from production code
- **Main entry point** (`app.py`) clearly identified

### 3. Reduced Root Clutter
- Only essential files in root: `app.py`, `utils.py`, `README.md`, `run_app.sh`
- Everything else properly organized in subdirectories

### 4. Better Discoverability
- Related files grouped together (all referee code in `referee/`)
- Clear distinction between production code and test/demo code
- Documentation easy to find in `docs/`

## Import Changes

### Old Imports (Deprecated)
```python
from referee import RefereeReportChecker
from app_summary import RefereeReportCheckerSummarized
from multi_agent_debate import execute_debate_pipeline
```

### New Imports (Current)
```python
# All referee functionality from one package
from referee import RefereeReportChecker, RefereeReportCheckerSummarized
from referee import execute_debate_pipeline, SELECTION_PROMPT, SYSTEM_PROMPTS
```

## Migration Guide

### Running the App
No changes needed! The main app still works the same way:
```bash
cd app_system
streamlit run app.py
# or
bash run_app.sh
```

### Running Tests
```bash
cd app_system
python -m pytest tests/
# or run individual tests
python -m pytest tests/test_consensus_calculation.py
```

### Running Demo Apps
```bash
cd app_system
streamlit run demos/app_full_output.py        # Full output version
streamlit run demos/app_summarized_only.py    # Summarized only version
```

## File Mapping

| Old Location | New Location | Type |
|-------------|-------------|------|
| `referee.py` | `referee/core.py` | Module |
| `referee_summarized.py` | `referee/summarized.py` | Module |
| `multi_agent_debate.py` | `referee/debate.py` | Module |
| `debate_summarizer.py` | `referee/summarizer.py` | Module |
| `app_summary.py` | `demos/app_summarized_only.py` | Demo |
| `full_output_app.py` | `demos/app_full_output.py` | Demo |
| `test_*.py` (5 files) | `tests/test_*.py` | Tests |

## Documentation Updates

The `CLAUDE.md` file has been updated with:
- New file organization rules
- Package structure documentation
- Updated import paths
- File placement guidelines

## Backwards Compatibility

**Breaking changes**: Old import paths no longer work. If you have external scripts that import from the old locations, update them to use the new `referee` package.

**What still works**:
- Main app entry point (`app.py`)
- All Streamlit workflows
- All existing functionality
- Launch scripts (`run_app.sh`)

## Testing

All imports verified:
```
✓ from referee import RefereeReportChecker, RefereeReportCheckerSummarized
✓ from referee import execute_debate_pipeline, SELECTION_PROMPT, SYSTEM_PROMPTS, DEBATE_PROMPTS
✓ from section_eval import SectionEvaluatorApp
✓ from utils import cm, single_query
✓ All Python files compile successfully
✓ Main app imports successfully
```

## Future Guidelines

When adding new files, follow these rules:
- **Tests**: Put in `tests/` with `test_` prefix
- **Documentation**: Put in `docs/` (except `README.md`)
- **New modules**: Create packages with `__init__.py`
- **Demo apps**: Put in `demos/`
- **Keep root clean**: Only main entry points and core utilities in root

---

**Date**: 2026-03-30
**Status**: Complete ✓
**Breaking Changes**: Import paths only (functionality unchanged)
