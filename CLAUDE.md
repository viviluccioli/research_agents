# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Model Requirements

**CRITICAL**: ALL systems in this repository MUST use **Claude 4.5 Sonnet** (`anthropic.claude-sonnet-4-5-20250929-v1:0`).

- ✅ **Use**: Claude 4.5 Sonnet for ALL API calls
- ❌ **Do NOT use**: Claude 3.7, Claude 3.5, or any older models
- The configuration has been updated so that `MODEL_PRIMARY`, `MODEL_SECONDARY`, and `MODEL_TERTIARY` all default to Claude 4.5
- This applies to:
  - Referee report system (multi-agent debate)
  - Section evaluator
  - All experiments
  - All utility functions

If you see any references to older models (3.7, 3.5) in code or documentation, they are outdated and should be ignored or updated.

## Running the app

The app must be run from inside `app_system/` so that imports resolve correctly (all imports use relative paths like `from utils import cm`, `from section_eval import ...`):

```bash
cd app_system
source ../venv/bin/activate
streamlit run app.py
```

Or use the provided script (disables file watcher to avoid inotify limits):

```bash
cd app_system
bash run_app.sh
```

To run demo apps:

```bash
cd app_system
streamlit run demos/app_demo.py
streamlit run demos/app_demo2.py
```

## Setup

### Initial Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install fpdf  # Additional dependency
```

### API Configuration
**IMPORTANT**: API credentials are managed via `.env` file (gitignored).

```bash
cd app_system

# Copy the template
cp .env.example .env

# Edit with your API credentials
nano .env
```

The `.env` file contains:
- `API_KEY` - Your API key
- `API_BASE` - API endpoint URL
- `MODEL_PRIMARY`, `MODEL_SECONDARY`, `MODEL_TERTIARY` - Model identifiers

Configuration is loaded by `app_system/config.py` which uses `python-dotenv`. The system supports multiple API providers (OpenAI, Anthropic, Gemini, custom endpoints).

**Never commit `.env` to git** - it's in `.gitignore`.

## File Organization Rules

The `app_system/` directory follows standard Python package organization:

```
app_system/
├── app.py                        # Main entry point (tabbed UI)
├── utils.py                      # Shared utilities
├── run_app.sh                    # Launch helper
├── README.md                     # User-facing documentation
│
├── referee/                      # Referee report package
│   ├── __init__.py              # Package exports
│   ├── core.py                  # Full output version (RefereeReportChecker)
│   ├── summarized.py            # Summarized version (RefereeReportCheckerSummarized)
│   ├── debate.py                # Multi-agent debate orchestration
│   └── summarizer.py            # LLM output compression
│
├── section_eval/                 # Section evaluator package
│   ├── __init__.py
│   ├── main.py                  # SectionEvaluatorApp
│   ├── evaluator.py
│   ├── scoring.py
│   ├── criteria/                # Criteria registry
│   └── ...
│
├── prompts/                      # External prompt files (versioned)
│   ├── multi_agent_debate/
│   └── section_evaluator/
│
├── tests/                        # All test files
│   ├── __init__.py
│   └── test_*.py
│
├── demos/                        # Demo/alternate apps
│   ├── app_full_output.py       # Full output version demo
│   ├── app_summarized_only.py   # Summarized-only demo
│   └── app_demo*.py             # Other demos
│
├── docs/                         # Optional documentation
│   ├── changelog.md             # Change history (update when appropriate)
│   ├── FRAMEWORK.md             # High-level system overview
│   └── *.md                     # Technical notes (PDF processing, math cleanup, etc.)
│
└── results/                      # Output directory
```

**File placement rules**:
- **Tests**: All `test_*.py` files go in `tests/`
- **New modules**: Create packages with `__init__.py` (e.g., `referee/`, `section_eval/`)
- **Demo apps**: Alternate entry points go in `demos/`
- **No root clutter**: Keep the `app_system/` root clean — only main entry point, utilities, and directories
- **Documentation**: Avoid creating new `.md` files unless absolutely necessary; update existing docs instead

## Prompt Organization

The `prompts/` directory contains versioned external prompt files for both the referee and section evaluator systems. **All prompts follow a standardized organization pattern**:

### Directory Structure
```
prompts/
├── multi_agent_debate/
│   ├── config.yaml                      # Version control config
│   ├── personas/                        # Persona system prompts
│   │   ├── theorist/
│   │   │   └── v1.0.txt
│   │   ├── empiricist/
│   │   │   └── v1.0.txt
│   │   └── ...
│   ├── debate_rounds/                   # Round-specific prompts
│   │   ├── round_0_selection/
│   │   │   └── v1.0.txt
│   │   └── ...
│   └── paper_type_contexts/             # Paper type guidance
│       ├── empirical/
│       │   └── v1.0.txt
│       ├── theoretical/
│       │   └── v1.0.txt
│       └── policy/
│           └── v1.0.txt
│
└── section_evaluator/
    ├── config.yaml                      # Version control config
    ├── paper_type_contexts/             # Paper type contexts
    │   ├── empirical/
    │   │   └── v1.0.txt
    │   ├── theoretical/
    │   │   └── v1.0.txt
    │   ├── policy/
    │   │   └── v1.0.txt
    │   └── ... (finance/, macro/, systematic_review/)
    ├── section_type_guidance/           # Section-specific guidance
    │   └── ... (flat files, not subdirs)
    └── master_prompts/
        └── ... (flat files, not subdirs)
```

### Organization Rules

**CRITICAL**: When adding or modifying prompts, follow these rules:

1. **Subdirectory Structure**: Each prompt category should have subdirectories for each item:
   - ✅ `paper_type_contexts/empirical/v1.0.txt`
   - ❌ `paper_type_contexts/empirical_v1.0.txt`

2. **Version Naming**: Version files always named `v{MAJOR}.{MINOR}.txt`:
   - ✅ `v1.0.txt`, `v1.1.txt`, `v2.0.txt`
   - ❌ `empirical_v1.0.txt`, `round_1.txt`

3. **Config Management**: Update `config.yaml` when changing versions:
   ```yaml
   paper_type_contexts:
     empirical:
       version: "v1.0"
       file: "paper_type_contexts/empirical/{version}.txt"
   ```
   The `{version}` placeholder is substituted by the prompt loader.

4. **Prompt Loader**: Both systems use a `PromptLoader` class that:
   - Reads `config.yaml` to determine active versions
   - Loads prompts from versioned files
   - Caches loaded prompts for performance
   - Provides `reload_prompts()` for testing changes

5. **Creating New Versions**:
   ```bash
   # Copy existing version
   cp prompts/section_evaluator/paper_type_contexts/empirical/v1.0.txt \
      prompts/section_evaluator/paper_type_contexts/empirical/v1.1.txt

   # Edit the new version
   nano prompts/section_evaluator/paper_type_contexts/empirical/v1.1.txt

   # Update config.yaml to use new version
   # Change: version: "v1.0" → version: "v1.1"
   ```

6. **Exception**: Some categories like `section_type_guidance/` and `master_prompts/` still use flat files (e.g., `abstract_v1.0.txt`) rather than subdirectories. This is acceptable for categories with many small files where subdirectories would add clutter.

**When to use subdirectories vs flat files**:
- **Use subdirectories**: For categories with few items (5-10) where each item may have multiple versions and is conceptually distinct (personas, paper types, debate rounds)
- **Use flat files**: For categories with many items (20+) that are tightly coupled and rarely version independently (section guidance, master prompts)

## Architecture

The entire working application lives in `app_system/`. The repo root contains only setup files; `mad_experiments/` and `papers/` are research scratch folders.

### Entry point and routing

`app_system/app.py` renders a two-tab Streamlit UI. Each tab instantiates and persists a workflow object in `st.session_state`. A shared file uploader at the top of the page passes `files: Dict[str, bytes]` to each workflow's `render_ui(files=...)`.

```
app.py
├── Tab "Referee Report"   → referee.RefereeReportCheckerSummarized
└── Tab "Section Evaluator" → section_eval.SectionEvaluatorApp
```

### LLM infrastructure (`app_system/utils.py`)

All LLM calls go through a single internal API (Federal Reserve MartinAI, OpenAI-compatible). Two call patterns exist:

- **`single_query(prompt)`** — stateless, used by the MAD system. Retries 3× with 5s delay.
- **`ConversationManager.conv_query(prompt)`** — stateful, used by section eval. Automatically prunes/summarizes old messages when tokens exceed 8000.

**Model configuration**: ALL systems now use **Claude 4.5 Sonnet**. The legacy model aliases (`model_selection`, `model_selection3`) both point to `MODEL_PRIMARY` which is Claude 4.5.

### Section evaluator (`app_system/section_eval/`)

A self-contained package. Data flows through these stages in order:

1. **Text extraction** (`text_extraction.py`) — PDF via `pdfplumber`, LaTeX via regex `strip_latex()`, plain text raw. Entry point: `decode_file(filename, bytes)`.

2. **Section detection** (`section_detection.py`) — two-pass: heuristic candidate scoring followed by LLM confirmation. Returns list of `{text, type, line_idx}` dicts. The heuristic scores lines by casing, numbering, word count, and surrounding whitespace, filtering out math symbols, figure labels, and citation fragments.

3. **Hierarchy grouping** (`hierarchy.py`) — groups detected subsections under parent sections by numbering (4.1 under 4), indentation, or casing.

4. **Evaluation** (`evaluator.py:SectionEvaluator`) — for each section:
   - Looks up criteria from registry: `get_criteria(paper_type, section_name)`
   - Calls `build_evaluation_prompt()` from `prompts/templates.py`
   - Calls `safe_query()` (which bypasses ConversationManager and hits the API directly with the strong model at temperature 0.3)
   - Parses JSON, validates quotes, computes score
   - Caches result by `sha256(paper_type | section_name | text[:50000] | figures_external)`

5. **Scoring** (`scoring.py`) — `compute_section_score()` returns a weighted average of criterion scores (1–5), multiplied by a section importance factor per paper type. Fatal-flaw rule: any criterion marked `critical=True` that scores ≤ 1.5 caps the section score at 2.5. `compute_overall_score()` aggregates sections and determines publication readiness.

6. **UI** (`main.py:SectionEvaluatorApp`) — two input modes: auto-detect (upload + scan) and freeform (paste per section). Results include per-section scores, criteria breakdowns, quote validity, improvements, and CSV/MD/PDF download.

### Criteria registry (`section_eval/criteria/base.py`)

This is the core configuration file for the section evaluator. It defines:

- `PAPER_TYPES`, `PAPER_TYPE_LABELS`, `SECTION_DEFAULTS` — the paper type registry
- `_UNIVERSAL`, `_EMPIRICAL`, `_THEORETICAL`, `_POLICY`, `_FINANCE`, `_MACRO`, `_SYSREV` — criteria dicts keyed by canonical section type
- `_ALL_CRITERIA` — merges universal + paper-type-specific
- `_SECTION_ALIASES` + `_KEYWORD_MAP` — maps raw header strings to canonical keys (e.g. `"empirical strategy"` → `"methodology"`)
- `FATAL_FLAW_SCORE_THRESHOLD = 1.5`, `FATAL_FLAW_SCORE_CAP = 2.5` — fatal-flaw floor constants
- `critical: True` on criteria that trigger the fatal-flaw cap (e.g. `identification`, `correctness`, `instrument_validity`)

**To add a new paper type**: add to `PAPER_TYPES`, `PAPER_TYPE_LABELS`, `SECTION_DEFAULTS`, define a `_NEW` criteria dict, merge into `_ALL_CRITERIA`, and add section importance weights in `scoring.py:SECTION_IMPORTANCE`.

**To add a new section alias**: add to `_SECTION_ALIASES` or `_KEYWORD_MAP`.

### Referee report system (`app_system/referee/`)

The referee package contains the multi-agent debate (MAD) system for generating referee reports.

**Structure**:
```
referee/
├── workflow.py          # ⭐ Main production UI (RefereeWorkflow)
├── engine.py            # ⭐ Debate orchestration (execute_debate_pipeline)
├── _utils/              # 🔧 Internal helpers
│   ├── summarizer.py    # LLM summarization utilities
│   └── quote_validator.py  # Quote verification (prevent hallucinations)
└── _archived/           # 📦 Archived alternate implementations
    └── full_output_ui.py  # Full verbose UI (not main code path)
```

**Main components**:

- **`referee.workflow`** (`workflow.py`) — `RefereeWorkflow` is the main production UI class used in `app.py`. Uses LLM-powered summarization for clean display.

- **`referee.engine`** (`engine.py`) — `execute_debate_pipeline(paper_text)` orchestrates 5 rounds:
  - **Round 0**: LLM selects 3 of 5 personas (Theorist, Empiricist, Historian, Visionary, Policymaker) and assigns weights summing to 1.0.
  - **Rounds 1, 2A, 2B, 2C**: `asyncio.gather()` runs all selected personas in parallel per round. Each persona receives only the context appropriate for its round (peer reports, Q&A transcript, full debate transcript).
  - **Round 3**: Editor computes weighted consensus (`PASS=1.0, REVISE=0.5, FAIL=0.0`; thresholds: >0.75 → ACCEPT, 0.40–0.75 → RESUBMIT, <0.40 → REJECT) and writes the final referee report.

**Quote Validation** (`_utils/quote_validator.py`): Automatically validates quotes in persona reports to prevent hallucinations. Validates after Round 1 and Round 2C using fuzzy string matching (thefuzz library). Features:
- Extracts quotes from reports (double/single quotes, blockquotes, statement patterns)
- Uses adaptive thresholds: 95% for mathematical content, 85% for prose
- Results shown in UI metadata section and Excel "Quote Validation" sheet
- **Disable**: Set `DISABLE_QUOTE_VALIDATION=true` in environment or `.env`
- **Dependencies**: `pip install thefuzz python-Levenshtein` (optional but recommended)
- See `docs/quote_validation.md` for full documentation

**To add a new persona**: add to `SYSTEM_PROMPTS` in `referee/engine.py` (include the `_ERROR_SEVERITY_GUIDE` block), update the persona list in `SELECTION_PROMPT`, and add the icon/CSS class in `referee/workflow.py`.

**Import paths**: Use `from referee import RefereeWorkflow, execute_debate_pipeline` to access the main classes and functions. The underscore-prefixed subdirectories (`_utils/`, `_archived/`) contain internal/archived code not part of the main API.

## Changelog

For significant changes to `app_system/section_eval/` or `app_system/referee/`, consider updating `app_system/docs/changelog.md`. Only document major features, fixes, or breaking changes — not minor tweaks or refactors.

## Key gotchas

- **Import paths**: All packages (`section_eval/`, `referee/`) import from the parent `utils.py` via `from utils import ...`. This only works when Streamlit is launched from `app_system/`. Running from the repo root will break imports. Demo apps in `demos/` add the parent directory to sys.path.
- **`safe_query` vs `single_query`**: Both use **Claude 4.5 Sonnet**. `safe_query` (in `section_eval/utils.py`) bypasses `ConversationManager` and calls the API directly at temperature 0.3. `single_query` (in `utils.py`) has thinking budget enabled and uses temperature 1.
- **Thinking mode**: `single_query` sends `"thinking": {"type": "enabled", "budget_tokens": 2048}` — temperature must be 1 when this is enabled. `safe_query` does not use thinking mode (temperature 0.3).
- **Cache prefix**: `SectionEvaluator` uses prefix `"se_cache_v3"`, `SectionEvaluatorApp` uses `"se_v3"`. If you change the result schema, bump these prefixes to avoid stale cache hits.
- **`fpdf` encoding**: PDF generation encodes text as `latin-1` with `replace` error handling. Unicode characters in paper text will be silently substituted.
