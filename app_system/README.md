# App System Directory

This directory contains all files needed to run the Streamlit evaluation apps.

## Key Features

### Referee Report System
- **Dual output modes**: Choose between full output (14 calls) or with LLM summarization (+10-15 calls)
- **Automatic cost tracking**: Real-time token usage and cost estimation
- **Enhanced PDF extraction**: Automatic table extraction formatted as markdown
- **Configurable personas**: 5 specialized AI reviewers (3 selected per paper)
- **Weighted consensus**: Mathematical aggregation of persona verdicts

### Section Evaluator
- **Auto-detection**: Two-pass section detection (heuristic + LLM)
- **Paper-type-specific criteria**: Customized evaluation for empirical/theoretical/policy papers
- **Fatal-flaw scoring**: Critical criteria cap section scores at 2.5 if ≤ 1.5
- **Multi-format export**: Markdown, PDF, and CSV (for benchmarking)
- **Quote validation**: Verifies all LLM quotes exist in source text

---

## Quick Setup

### 1. Configure API Credentials
```bash
# Copy the example configuration
cp .env.example .env

# Edit .env with your API credentials
nano .env
```

**Important**: Never commit the `.env` file - it contains your API keys and is gitignored.

### 2. Install Dependencies
```bash
cd ..  # Go to repo root
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install fpdf  # Additional dependency
```

### 3. Run the App
```bash
cd app_system
streamlit run app.py
```

---

## Repository Structure

```
app_system/
├── app.py                        # Main entry point (tabbed UI)
├── utils.py                      # Shared utilities (LLM API wrappers)
├── config.py                     # Configuration loader (.env file)
├── run_app.sh                    # Launch helper script
├── README.md                     # This file
│
├── referee/                      # Referee report package (multi-agent debate)
│   ├── __init__.py              # Package exports
│   ├── workflow.py              # ⭐ Main production UI (RefereeWorkflow)
│   ├── engine.py                # ⭐ Debate orchestration (execute_debate_pipeline)
│   ├── _utils/                  # 🔧 Internal helpers
│   │   └── summarizer.py        # LLM summarization utilities
│   ├── _archived/               # 📦 Archived alternate implementations
│   │   └── full_output_ui.py    # Full verbose UI (not main code path)
│   └── REFEREE_PACKAGE_STRUCTURE.md  # Package documentation
│
├── section_eval/                 # Section evaluator package
│   ├── __init__.py              # Package exports
│   ├── main.py                  # SectionEvaluatorApp (main UI class)
│   ├── evaluator.py             # Core evaluation logic
│   ├── scoring.py               # Score calculation and aggregation
│   ├── text_extraction.py       # PDF/LaTeX/text extraction
│   ├── section_detection.py     # Section detection (heuristic + LLM)
│   ├── hierarchy.py             # Section hierarchy grouping
│   ├── utils.py                 # Section eval utilities (safe_query)
│   ├── criteria/                # Criteria registry
│   │   ├── __init__.py
│   │   └── base.py              # Paper types and evaluation criteria
│   └── prompts/                 # Prompt templates
│       ├── __init__.py
│       └── templates.py         # Evaluation prompt builders
│
├── prompts/                      # External prompt files (versioned)
│   ├── multi_agent_debate/      # Referee system prompts
│   └── section_evaluator/       # Section evaluator prompts
│
├── tests/                        # All test files
│   ├── __init__.py
│   ├── test_consensus_calculation.py
│   ├── test_prompt_loader.py
│   ├── test_referee_display.py
│   ├── test_referee_quick.py
│   └── test_section_evaluator_prompts.py
│
├── demos/                        # Demo/alternate apps
│   ├── app_demo.py              # Demo 1: Shows madoutput1.txt results
│   ├── app_demo2.py             # Demo 2: Shows madoutput2.txt results
│   ├── app_demo3.py             # Demo 3
│   ├── app_full_output.py       # Full output version demo
│   └── app_summarized_only.py   # Summarized-only demo
│
├── docs/                         # Documentation files
│   ├── architecture.md          # System architecture
│   ├── changelog.md             # Change log
│   ├── FRAMEWORK.md             # Framework documentation
│   ├── API_CONFIGURATION.md     # API setup guide
│   ├── PROMPT_MANAGEMENT.md     # Prompt system docs
│   ├── PROMPT_QUICKREF.md       # Prompt quick reference
│   ├── PROMPT_SYSTEM_COMPLETE.md # Complete prompt system docs
│   ├── REORGANIZATION.md        # Reorganization notes
│   └── RESTRUCTURING_SUMMARY.md # Restructuring summary
│
└── results/                      # Output directory (generated reports)
```

## Running the Apps

**IMPORTANT**: The app must be run from inside `app_system/` so that imports resolve correctly (all packages import from the parent `utils.py` via `from utils import ...`).

### Main app:

```bash
cd app_system
streamlit run app.py
```

Or use the provided script (disables file watcher to avoid inotify limits):

```bash
cd app_system
bash run_app.sh
```

### Demo apps:

```bash
cd app_system
streamlit run demos/app_demo.py      # Demo 1: madoutput1.txt
streamlit run demos/app_demo2.py     # Demo 2: madoutput2.txt
streamlit run demos/app_demo3.py     # Demo 3
streamlit run demos/app_full_output.py       # Full verbose output version
streamlit run demos/app_summarized_only.py   # Summarized-only version
```

## Key Dependencies

### Multi-Agent Debate Flow:
```
app.py (Tab: "Referee Report")
  └─ referee/workflow.py (RefereeWorkflow)
       └─ referee/engine.py (execute_debate_pipeline)
            ├─ utils.py (single_query, count_tokens - stateless LLM calls)
            ├─ referee/_utils/summarizer.py (optional LLM summarization)
            └─ calculate_token_usage_and_cost() (automatic cost estimation)
```

### Section Evaluator Flow:
```
app.py (Tab: "Section Evaluator")
  └─ section_eval/main.py (SectionEvaluatorApp)
       ├─ section_eval/text_extraction.py (decode_file)
       ├─ section_eval/section_detection.py (detect sections)
       ├─ section_eval/hierarchy.py (group into hierarchy)
       ├─ section_eval/evaluator.py (SectionEvaluator)
       │    ├─ section_eval/criteria/base.py (get_criteria)
       │    ├─ section_eval/prompts/templates.py (build_evaluation_prompt)
       │    └─ section_eval/utils.py (safe_query - direct API calls)
       └─ section_eval/scoring.py (compute scores)
```

## Configuration

Configuration is managed via `.env` file (loaded by `config.py`):

### API Settings (.env file):
- `API_KEY` - Your API key
- `API_BASE` - API endpoint URL (supports OpenAI, Anthropic, Gemini, custom)
- `MODEL_PRIMARY` - Primary model (all systems) - default: **Claude 4.5 Sonnet**
- `MODEL_SECONDARY` - Secondary model (now also Claude 4.5) - formerly 3.7
- `MODEL_TERTIARY` - Tertiary model (now also Claude 4.5) - legacy/backup

See `.env.example` for configuration templates for different API providers.

### Temperature Settings:
- **Referee debate**: Temperature 1.0 (with thinking mode enabled)
- **Section evaluator**: Temperature 0.3 (more conservative for scoring)

### Referee Output Modes:
The referee system supports two output modes (configurable via UI checkbox):

1. **Full Output Only** (default)
   - 14 API calls total (1 selection + 12 persona rounds + 1 editor)
   - Most cost-efficient option
   - Shows complete debate outputs in expandable sections

2. **With LLM Summarization**
   - 24-29 total API calls (14 debate + 10-15 summarization)
   - Cleaner display with executive summaries
   - Full outputs still available in expandable sections
   - Adds ~10-15 compression calls via `summarize_all_rounds()`

### Token Usage and Cost Tracking:
The referee system automatically calculates and displays:
- Input/output token counts (separate for debate vs summarization)
- Number of LLM API calls
- Estimated cost in USD (based on Claude 4.5 Sonnet pricing: $3/M input, $15/M output)
- Breakdown by round and persona

## Notes

### Import Path Requirements
- **Must run from `app_system/`**: All packages (`section_eval/`, `referee/`) import from the parent `utils.py` via `from utils import ...`. Running from the repo root will break imports.
- Demo apps in `demos/` add the parent directory to `sys.path` to resolve imports correctly.

### Architecture
- The `section_eval/` and `referee/` directories are self-contained packages with `__init__.py` exports.
- `app.py` is the main entry point with a two-tab UI (Referee Report + Section Evaluator).
- The entire working application lives in `app_system/`. The repo root contains only setup files; `mad_experiments/` and `papers/` are research scratch folders.

### LLM Call Patterns
- **`single_query()`** (in `utils.py`): Stateless calls used by the MAD system. Uses `model_selection3` (**Claude 4.5 Sonnet**, formerly 3.7) at temperature 1.0 with thinking mode enabled.
- **`safe_query()`** (in `section_eval/utils.py`): Direct API calls used by section evaluator. Uses `model_selection` (**Claude 4.5 Sonnet**) at temperature 0.3, bypasses ConversationManager.
- **`ConversationManager.conv_query()`**: Stateful conversation management with automatic pruning when tokens exceed 8000.

### PDF Processing
- **Table extraction**: Referee workflow's `extract_text_from_pdf()` automatically extracts tables and formats them as markdown
- **Encoding**: Tables use pipe-delimited markdown format (e.g., `| Header 1 | Header 2 |`)
- **Page tracking**: Displays total pages, tables extracted, and character count
- **Token estimation**: Uses `count_tokens()` to estimate token usage before API calls

### Documentation
- Architecture and framework docs live in `app_system/docs/`
- See `docs/architecture.md` for detailed system design
- See `docs/changelog.md` for change history
- See `referee/REFEREE_PACKAGE_STRUCTURE.md` for referee package details
- See `docs/PROMPT_MANAGEMENT.md` for prompt system documentation
