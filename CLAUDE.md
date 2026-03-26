# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The app also requires `fpdf` (not in requirements.txt): `pip install fpdf`.

API credentials are loaded from environment variables — **do not hardcode them**:
- `MARTINAI_API_KEY` (required)
- `MARTINAI_API_BASE` (required)

Set these before running:
```bash
export MARTINAI_API_KEY="your-key-here"
export MARTINAI_API_BASE="your-internal-base-url"
```

## Architecture

The entire working application lives in `app_system/`. The repo root contains only setup files; `mad_experiments/` and `papers/` are research scratch folders.

### Entry point and routing

`app_system/app.py` renders a two-tab Streamlit UI. Each tab instantiates and persists a workflow object in `st.session_state`. A shared file uploader at the top of the page passes `files: Dict[str, bytes]` to each workflow's `render_ui(files=...)`.

```
app.py
├── Tab "Referee Report"   → referee.RefereeReportChecker
└── Tab "Section Evaluator" → section_eval.SectionEvaluatorApp
```

### LLM infrastructure (`app_system/utils.py`)

All LLM calls go through a single internal API (Federal Reserve MartinAI, OpenAI-compatible). Two call patterns exist:

- **`single_query(prompt)`** — stateless, used by the MAD system. Retries 3× with 5s delay.
- **`ConversationManager.conv_query(prompt)`** — stateful, used by section eval. Automatically prunes/summarizes old messages when tokens exceed 8000.

Two model slots are configured: `model_selection3` (Claude 3.7 Sonnet, used by `single_query`) and `model_selection` (Claude Sonnet 4.5, used by `safe_query` in section eval).

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

### Multi-agent debate (`app_system/multi_agent_debate.py` + `referee.py`)

`execute_debate_pipeline(paper_text)` is an async function that orchestrates 5 rounds:

- **Round 0**: LLM selects 3 of 5 personas (Theorist, Empiricist, Historian, Visionary, Policymaker) and assigns weights summing to 1.0.
- **Rounds 1, 2A, 2B, 2C**: `asyncio.gather()` runs all selected personas in parallel per round. Each persona receives only the context appropriate for its round (peer reports, Q&A transcript, full debate transcript).
- **Round 3**: Editor computes weighted consensus (`PASS=1.0, REVISE=0.5, FAIL=0.0`; thresholds: >0.75 → ACCEPT, 0.40–0.75 → RESUBMIT, <0.40 → REJECT) and writes the final referee report.

`referee.py:RefereeReportChecker` wraps this with the Streamlit UI and calls `asyncio.run(execute_debate_pipeline(...))`.

**To add a new persona**: add to `SYSTEM_PROMPTS` in `multi_agent_debate.py` (include the `_ERROR_SEVERITY_GUIDE` block), update the persona list in `SELECTION_PROMPT`, and add the icon/CSS class in `referee.py`.

## Changelog rule

Every change to `app_system/section_eval/` or `app_system/multi_agent_debate.py` must be documented in `app_system/docs/changelog.md`. Format: date header → category (Fix/Feature/Refactor/Performance/UI) → Changed/Added/Removed/Fixed sub-sections.

## Key gotchas

- **Import paths**: `section_eval/utils.py` imports from the parent `utils.py` via `from utils import ...`. This only works when Streamlit is launched from `app_system/`. Running from the repo root will break imports.
- **`safe_query` vs `single_query`**: `safe_query` (in `section_eval/utils.py`) bypasses `ConversationManager` and calls the API directly with `model_selection` at temperature 0.3. `single_query` (in `utils.py`) uses `model_selection3` with thinking budget enabled at temperature 1.
- **Thinking mode**: `single_query` sends `"thinking": {"type": "enabled", "budget_tokens": 2048}` — temperature must be 1 when this is enabled. `safe_query` does not use thinking mode (temperature 0.3).
- **Cache prefix**: `SectionEvaluator` uses prefix `"se_cache_v3"`, `SectionEvaluatorApp` uses `"se_v3"`. If you change the result schema, bump these prefixes to avoid stale cache hits.
- **`fpdf` encoding**: PDF generation encodes text as `latin-1` with `replace` error handling. Unicode characters in paper text will be silently substituted.
