# Referee Package Structure (2026-03-30)

## Overview

The `referee/` package has been restructured to clearly separate production code from utilities and archived implementations. This makes it immediately obvious to collaborators which files are part of the main code path.

## Directory Structure

```
referee/
├── __init__.py                      # Package exports
│
├── workflow.py                      # ⭐ MAIN PRODUCTION: UI workflow
├── engine.py                        # ⭐ MAIN PRODUCTION: Debate orchestration
│
├── _utils/                          # 🔧 INTERNAL: Helper utilities
│   ├── __init__.py
│   └── summarizer.py                # LLM summarization functions
│
└── _archived/                       # 📦 ARCHIVED: Alternate implementations
    ├── __init__.py
    └── full_output_ui.py            # Full verbose UI (not main code path)
```

## File Descriptions

### Production Code (Top Level)

**`workflow.py`** - Main Referee Report UI
- **Class**: `RefereeWorkflow`
- **Purpose**: Official production UI used in `app.py`
- **Features**: LLM-powered summarization for clean display
- **Status**: ⭐ PRIMARY CODE PATH

**`engine.py`** - Debate Orchestration Engine
- **Main function**: `execute_debate_pipeline()`
- **Purpose**: Orchestrates 5-round multi-agent debate
- **Features**: Persona selection, debate rounds, consensus calculation, metadata tracking
- **Status**: ⭐ PRIMARY CODE PATH

### Internal Utilities (`_utils/`)

**`_utils/summarizer.py`** - Summarization Utilities
- **Main function**: `summarize_all_rounds()`
- **Purpose**: LLM-powered compression of debate outputs
- **Status**: 🔧 Internal helper (underscore indicates not main API)

### Archived Implementations (`_archived/`)

**`_archived/full_output_ui.py`** - Full Output UI
- **Class**: `RefereeReportChecker`
- **Purpose**: Alternate UI showing full uncompressed outputs
- **Status**: 📦 Archived (kept for reference and special cases)

## Import Patterns

### Main Production Code (Recommended)

```python
# In app.py or your code
from referee import RefereeWorkflow, execute_debate_pipeline

# Use the main workflow
workflow = RefereeWorkflow()
```

### Backward Compatibility (Deprecated)

```python
# Old imports still work but are deprecated
from referee import RefereeReportCheckerSummarized

# This is actually an alias to RefereeWorkflow
# Kept for backward compatibility
```

### Archived Code (For Special Cases)

```python
# If you need the full-output version
from referee._archived import RefereeReportChecker
```

### Internal Utilities (Usually Not Needed)

```python
# Rarely imported directly - used internally by workflow.py
from referee._utils import summarize_all_rounds
```

## Naming Conventions

### Underscore Prefix (`_utils/`, `_archived/`)

The underscore prefix is a Python convention meaning "internal/not main API":
- `_utils/` → Internal implementation details
- `_archived/` → Not part of main code path

This immediately signals to collaborators:
- ✅ Top-level files = production code
- ⚠️ Underscore directories = not main code path

### File Names

- **`workflow.py`** - Clear this is THE main workflow
- **`engine.py`** - Clear this does orchestration/heavy lifting
- Not "summarized.py" or "core.py" which were ambiguous

## Migration from Previous Structure

### Old Structure (Confusing)
```
referee/
├── core.py                 # Sounded main but was alternate
├── summarized.py           # Sounded experimental but was main
├── debate.py               # Unclear name
└── summarizer.py           # Helper utility
```

### New Structure (Clear)
```
referee/
├── workflow.py             # ⭐ Obviously main
├── engine.py               # ⭐ Obviously main
├── _utils/                 # 🔧 Obviously internal
│   └── summarizer.py
└── _archived/              # 📦 Obviously not main
    └── full_output_ui.py
```

## Benefits for Collaboration

1. **Immediate Clarity**: Colleagues see `workflow.py` and `engine.py` at top level → these are the main files
2. **Clear Boundaries**: Underscore directories signal "internal" or "archived"
3. **Professional Structure**: Follows Python best practices
4. **Easy Onboarding**: New team members immediately understand organization
5. **Backward Compatible**: Old imports still work during transition

## Key Exports

From `referee/__init__.py`:

```python
# Main production API
RefereeWorkflow              # Main UI class
execute_debate_pipeline      # Main orchestration function
SELECTION_PROMPT            # Persona selection prompt
SYSTEM_PROMPTS              # Persona system prompts
DEBATE_PROMPTS              # Debate round prompts

# Deprecated aliases (backward compatibility)
RefereeReportCheckerSummarized  # → RefereeWorkflow
RefereeReportChecker            # → RefereeWorkflow
```

## Usage in App

**`app.py`** (main entry point):
```python
from referee import RefereeWorkflow

WORKFLOWS = {
    "Referee Report": RefereeWorkflow,
    "Section Evaluator": SectionEvaluatorApp,
}
```

**Demo apps**:
- `demos/app_full_output.py` - Uses archived full-output UI
- `demos/app_summarized_only.py` - Uses main workflow (same as app.py)

## Documentation

- This file: High-level structure overview
- `docs/REORGANIZATION.md`: Complete reorganization history
- `docs/changelog.md`: Detailed change log
- `CLAUDE.md`: Updated with new structure

---

**Date**: 2026-03-30
**Status**: Complete ✓
**Breaking Changes**: None (backward compatibility maintained)
