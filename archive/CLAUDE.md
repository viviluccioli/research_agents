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

To run demo apps:

```bash
cd app_system
streamlit run demos/app_demo.py
streamlit run demos/app_demo2.py
```

To run the experimental 10-persona system (Experiment 4):

```bash
cd app_system
bash run_app_exp_4.sh  # Disables file watcher to avoid inotify limits
# Or directly:
streamlit run app_exp_4.py
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

## Claude Code Hooks & Commit History

This repository uses Claude Code automation hooks to maintain code quality and documentation.

### Automated Hooks

Two hooks are configured in `.claude/settings.json`:

1. **Test Runner** (`.claude/hooks/run-tests-on-edit.sh`)
   - **Trigger**: Runs automatically when Python files in `app_system/` are edited via Write or Edit tools
   - **Action**: Executes pytest on related test files
   - **Behavior**: Non-blocking (shows output but doesn't halt Claude)
   - **Skips**: Test files themselves, files outside `app_system/`

2. **Commit Documentation Generator** (`.claude/hooks/gen-commit-docs.sh`)
   - **Trigger**: Runs automatically after any `git commit` command
   - **Action**: Generates a markdown file documenting the commit
   - **Output Location**: `commit_history/{short_hash}_{sanitized_title}.md`
   - **Content**: Commit hash, date, author, changes summary, and full diff

### Commit History Archive

**IMPORTANT FOR HANDOFF**: All git commits are automatically documented in the `commit_history/` directory.

**Format**: Each file follows the pattern `{7-char-hash}_{commit-title}.md`

**Contents**:
- Commit metadata (hash, date, author)
- Changes summary (`git show --stat`)
- Full diff (`git show`)

**Example**: `commit_history/de77b56_added_git_commit_hook.md`

**Use Cases**:
- Review what changed in a specific commit without running `git show`
- Understand context behind major changes during handoff
- Quick reference for Claude Code when understanding repository history
- Searchable archive of all development decisions

**Note**: The `commit_history/` directory is tracked in git, so all commit documentation is version-controlled and travels with the repository.

### Hook Configuration

To view or modify hook behavior, edit `.claude/settings.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{"command": "bash .claude/hooks/run-tests-on-edit.sh"}]
      },
      {
        "matcher": "Bash",
        "filter": "git commit",
        "hooks": [{"command": "bash .claude/hooks/gen-commit-docs.sh"}]
      }
    ]
  }
}
```

## File Organization Rules

The `app_system/` directory follows standard Python package organization:

```
app_system/
├── app.py                        # Main entry point (5-persona MAD system)
├── app_exp_4.py                  # Experiment 4 entry point (10-persona MAD)
├── app-memo.py                   # Memo evaluation system (5 memo-specific analysts)
├── utils.py                      # Shared utilities (LLM calls, token counting)
├── config.py                     # API configuration loader (.env)
├── README.md                     # User-facing documentation
├── run_app_exp_4.sh              # Launch script for Experiment 4
│
├── referee/                      # Referee report package (MAD system)
│   ├── __init__.py              # Package exports
│   ├── workflow.py              # Main production UI (RefereeWorkflow)
│   ├── engine.py                # 5-persona debate orchestration
│   ├── engine_exp_4.py          # 10-persona debate orchestration
│   ├── workflow_exp_4.py        # UI for 10-persona system
│   ├── memo_engine.py           # Memo evaluation engine (wraps engine.py)
│   ├── memo_prompts.py          # 5 memo-specific analyst personas
│   ├── _utils/                  # Internal utilities (not main API)
│   │   ├── summarizer.py        # LLM output compression
│   │   ├── quote_validator.py   # Quote hallucination prevention
│   │   ├── cache.py             # Granular per-round caching
│   │   ├── deduplicator.py      # Cross-reference deduplication
│   │   └── pdf_extractor_v2.py  # PyMuPDF-based PDF+figure extraction
│   └── _archived/               # Archived implementations
│       └── full_output_ui.py    # Old full-output UI
│
├── section_eval/                 # Section evaluator package
│   ├── __init__.py
│   ├── main.py                  # SectionEvaluatorApp
│   ├── evaluator.py             # Section evaluation logic
│   ├── scoring.py               # Score computation + fatal-flaw logic
│   ├── section_detection.py     # Two-pass section detection
│   ├── text_extraction.py       # PDF/LaTeX/text parsing
│   ├── hierarchy.py             # Section grouping
│   ├── math_cleanup.py          # LaTeX math normalization
│   ├── region_fixer.py          # Interactive math fixing UI
│   ├── criteria/                # Criteria registry
│   │   └── base.py              # Paper types + evaluation criteria
│   └── prompts/                 # Prompt templates
│
├── prompts/                      # External prompt files (versioned)
│   ├── multi_agent_debate/
│   │   ├── config.yaml
│   │   ├── personas/            # 10 personas (5 base + 6 exp_4)
│   │   ├── debate_rounds/
│   │   └── additional_context/
│   └── section_evaluator/
│       ├── config.yaml
│       ├── paper_type_contexts/
│       ├── section_type_guidance/
│       └── master_prompts/
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
├── examples/                     # Example inputs
│   └── sample_memo.txt
│
├── docs/                         # Technical documentation
│   ├── changelog.md             # Change history
│   ├── FRAMEWORK.md             # High-level system overview
│   ├── quote_validation.md      # Quote validation system
│   ├── deduplication.md         # Cross-reference deduplication
│   ├── caching.md               # Caching system documentation
│   ├── pymupdf_extraction.md    # PyMuPDF PDF extraction
│   ├── math_cleanup.md          # LaTeX math normalization
│   ├── MEMO_EVALUATION_README.md # Memo evaluation system
│   └── *.md                     # Other technical notes
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
├── Tab "Referee Report"   → referee.RefereeWorkflow (5 personas)
└── Tab "Section Evaluator" → section_eval.SectionEvaluatorApp

app_exp_4.py
├── Tab "Referee Report"   → referee.RefereeWorkflow (10 personas, uses engine_exp_4)
└── Tab "Section Evaluator" → section_eval.SectionEvaluatorApp
```

### LLM infrastructure (`app_system/utils.py`)

All LLM calls go through a single internal API (Federal Reserve MartinAI, OpenAI-compatible). Three call patterns exist:

- **`single_query(prompt)`** — stateless, includes generic "research assistant" system prompt. Used by non-referee workflows. Retries 3× with 5s delay.
- **`referee_query(prompt)`** — stateless, NO generic system prompt. Used exclusively by referee system to avoid diluting specialized persona instructions. Accepts optional temperature parameter. Retries 3× with 5s delay.
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
├── engine.py            # ⭐ 10-persona debate orchestration (execute_debate_pipeline)
├── engine_exp_4.py      # 🗄️ Legacy 10-persona experimental version (deprecated)
├── workflow_exp_4.py    # 🗄️ Legacy UI (deprecated)
├── _utils/              # 🔧 Internal helpers (exported via __init__.py)
│   ├── summarizer.py    # LLM summarization utilities
│   ├── quote_validator.py  # Quote verification (prevent hallucinations)
│   ├── cache.py         # Granular per-round SHA256-based caching
│   ├── deduplicator.py  # Cross-reference deduplication (similarity-based)
│   └── pdf_extractor_v2.py  # PyMuPDF-based PDF extraction with figures/tables
└── _archived/           # 📦 Archived alternate implementations
    └── full_output_ui.py  # Full verbose UI (not main code path)
```

**Main components**:

- **`referee.workflow`** (`workflow.py`) — `RefereeWorkflow` is the main production UI class used in `app.py`. Uses LLM-powered summarization for clean display.

- **`referee.engine`** (`engine.py`) — `execute_debate_pipeline(paper_text)` orchestrates 5 rounds with 10 available personas:
  - **Round 0**: LLM selects 3 of 10 personas (Theorist, Econometrician, ML_Expert, Data_Scientist, CS_Expert, Historian, Visionary, Policymaker, Ethicist, Perspective) and assigns weights summing to 1.0.
  - **Rounds 1, 2A, 2B, 2C**: `asyncio.gather()` runs all 3 selected personas in parallel per round. Each persona receives only the context appropriate for its round (peer reports, Q&A transcript, full debate transcript).
  - **Round 3**: Editor computes weighted consensus (`PASS=1.0, REVISE=0.5, FAIL=0.0`; thresholds: >0.75 → ACCEPT, 0.40–0.75 → RESUBMIT, <0.40 → REJECT) and writes the final referee report.

**Per-Round Temperature Control** (since 2026-05-06):

The referee system uses differentiated temperatures for each round to balance consistency with thoughtfulness:

```python
ROUND_TEMPERATURES = {
    'round_0': 0.4,   # Persona selection - needs consistency (same personas for similar papers)
    'round_1': 0.7,   # Independent analysis - needs thoughtful, creative evaluation
    'round_2a': 0.7,  # Cross-examination - needs insightful questions and synthesis
    'round_2b': 0.6,  # Direct answers - focused responses to specific questions
    'round_2c': 0.6,  # Final amendments - refined evaluation after debate
    'round_3': 0.4,   # Editor synthesis - faithful consensus calculation, no new ideas
}
```

**Rationale**: Different rounds require different creativity/consistency balance:
- **Low temp (0.4)**: Selection & synthesis need consistency to avoid random variation
- **Medium temp (0.6)**: Focused responses while maintaining quality reasoning
- **High temp (0.7)**: Analysis & debate benefit from creative, thoughtful evaluation

**Implementation**: All LLM calls use `referee_query()` (no generic system prompt) with round-specific temperatures. Metadata tracks the temperature system and per-round values for reproducibility. Expected improvement: 60-80% reduction in verdict variability while maintaining analysis quality.

**To modify**: Edit `ROUND_TEMPERATURES` dict in `referee/engine.py`. Use `get_round_temperature(round_id)` to retrieve values.

**Internal Utilities** (`_utils/`):

1. **Quote Validation** (`quote_validator.py`): Automatically validates quotes in persona reports to prevent hallucinations. Validates after Round 1 and Round 2C using fuzzy string matching (thefuzz library). Features:
   - Extracts quotes from reports (double/single quotes, blockquotes, statement patterns)
   - Uses adaptive thresholds: 95% for mathematical content, 85% for prose
   - Results shown in UI metadata section and Excel "Quote Validation" sheet
   - **Disable**: Set `DISABLE_QUOTE_VALIDATION=true` in environment or `.env`
   - **Dependencies**: `pip install thefuzz python-Levenshtein` (optional but recommended)
   - See `docs/quote_validation.md` for full documentation

2. **Caching** (`cache.py`): SHA256-based granular caching system with per-round granularity. Cache keys computed from paper text, selected personas, weights, and model config. Can save 50-80% of costs during iterative development.
   - Cache directory: `.referee_cache/`
   - Structure: `{cache_key}/round_{0,1,2a,2b,2c,3}_{name}.json`
   - **Enable**: Checkbox in UI, or set `CACHE_ENABLED=true` in `.env`
   - See `docs/caching.md` for full documentation

3. **Deduplication** (`deduplicator.py`): Identifies and merges duplicate findings across persona reports using quote overlap, semantic similarity (optional embeddings), and keyword matching.
   - **Enable**: Set `ENABLE_DEDUPLICATION=true` in `.env` (default: true)
   - **Configure**: `DEDUP_SIMILARITY_THRESHOLD` (default: 0.8)
   - **Embeddings**: Requires `pip install sentence-transformers` for semantic similarity
   - See `docs/deduplication.md` for full documentation

4. **PDF Extraction** (`pdf_extractor_v2.py`): PyMuPDF-based extraction supporting multi-column layouts, figure/table extraction, OCR, and caption parsing. Falls back to pdfplumber if PyMuPDF unavailable.
   - **Dependencies**: `pip install pymupdf Pillow pytesseract` (OCR optional)
   - See `docs/pymupdf_extraction.md` for full documentation

**10-Persona System Evolution**: The system originally used 5 personas (Theorist, Empiricist, Historian, Visionary, Policymaker) but was expanded to 10 personas to provide better coverage of technical depth (added Econometrician, ML_Expert, Data_Scientist, CS_Expert) and ethical dimensions (added Ethicist, Perspective). The 10-persona system is now the primary production system in `engine.py` and `app.py`. Legacy files (`app_exp_4.py`, `engine_exp_4.py`, `workflow_exp_4.py`) are deprecated. See `EXPERIMENT_4_SUMMARY.md` and `README_EXP_4.md` for historical context.

**Memo Evaluation System**: A parallel system for evaluating policy memos instead of research papers. Uses the same MAD architecture but with memo-specific analyst personas (Policy Analyst, Data Analyst, Stakeholder Analyst, Implementation Analyst, Financial Stability Analyst). Files: `app-memo.py`, `memo_engine.py`, `memo_prompts.py`. See `docs/MEMO_EVALUATION_README.md` and `docs/MEMO_SYSTEM_QUICKSTART.md` for details.

**To add an 11th persona**: (1) add to `load_persona_prompt()` persona_dir_map and `FALLBACK_SYSTEM_PROMPTS` in `referee/engine.py`, (2) add to numbered list in `SELECTION_PROMPT` with clear expertise description, (3) add icon/CSS class in `referee/workflow.py`, (4) create prompt file in `prompts/multi_agent_debate/personas/{name}/v1.0.txt` with `{error_severity}` placeholder.

**Import paths**: Use `from referee import RefereeWorkflow, execute_debate_pipeline` to access the main classes and functions. The underscore-prefixed subdirectories (`_utils/`, `_archived/`) contain internal/archived code not part of the main API. Utilities are exported via `referee._utils.__init__.py`.

## Changelog

For significant changes to `app_system/section_eval/` or `app_system/referee/`, consider updating `app_system/docs/changelog.md`. Only document major features, fixes, or breaking changes — not minor tweaks or refactors.

**Recent Major Changes**:
- **2026-05-06**: Per-round temperature control added to referee system (Phase 2 consistency improvements)
- **2026-05-05**: Removed generic system prompt pollution from referee calls (Phase 1 consistency improvements)
- See `CHANGES_2026-05-05.md` and `CHANGES_2026-05-06.md` for detailed documentation
- See `running-ideas.md` for full problem analysis and future improvement phases

## Key gotchas

- **Import paths**: All packages (`section_eval/`, `referee/`) import from the parent `utils.py` via `from utils import ...`. This only works when Streamlit is launched from `app_system/`. Running from the repo root will break imports. Demo apps in `demos/` add the parent directory to sys.path.
- **Query functions**: All use **Claude 4.5 Sonnet** but with different configurations:
  - `safe_query` (section eval): Temperature 0.3, no thinking mode, bypasses ConversationManager
  - `single_query` (generic): Temperature 0.7, includes generic system prompt, no thinking mode
  - `referee_query` (referee system): Temperature varies by round (0.4-0.7), NO generic system prompt, no thinking mode
- **Thinking mode**: Currently NOT enabled in any system. Was documented as enabled but the API parameter was never sent. Thinking mode requires temperature=1.0, which conflicts with per-round temperature control in the referee system.
- **Referee temperatures**: Per-round temperature control means referee calls use different temps (0.4 for selection/synthesis, 0.6-0.7 for analysis). See `ROUND_TEMPERATURES` in `referee/engine.py`.
- **Cache prefix**: `SectionEvaluator` uses prefix `"se_cache_v3"`, `SectionEvaluatorApp` uses `"se_v3"`. If you change the result schema, bump these prefixes to avoid stale cache hits.
- **`fpdf` encoding**: PDF generation encodes text as `latin-1` with `replace` error handling. Unicode characters in paper text will be silently substituted.
