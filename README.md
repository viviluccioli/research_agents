# Research Agents — Evaluation Agent for Economics Papers

A Streamlit application that evaluates academic economics papers using two complementary workflows:

1. **Section Evaluator** — Criteria-based evaluation of individual paper sections, weighted by paper type (empirical, theoretical, policy) and section importance.
2. **Multi-Agent Referee** — Simulated peer review via structured debate between specialized AI personas (Theorist, Empiricist, Historian, Visionary, Policymaker).

The full application lives in `app_system/`.

---

## Key Features

### Referee Report System
- ⚙️ **Dual output modes**: Full output (14 API calls) or with LLM summarization (+10-15 calls for cleaner display)
- 💰 **Automatic cost tracking**: Real-time token usage and cost estimation (input/output tokens, estimated USD)
- 📊 **Enhanced PDF extraction**: Automatic table extraction and markdown formatting
- 👥 **Configurable personas**: 5 specialized AI reviewers (Theorist, Empiricist, Historian, Visionary, Policymaker) — 3 selected per paper
- 🎯 **Weighted consensus**: Mathematical aggregation using importance weights
- 📦 **Multiple export formats**: Markdown transcript and ZIP package downloads

### Section Evaluator
- 🔍 **Auto-detection**: Two-pass section detection (heuristic scoring + LLM confirmation)
- 📝 **Paper-type-specific criteria**: Customized evaluation for empirical/theoretical/policy papers
- ⚠️ **Fatal-flaw scoring**: Critical criteria (e.g., causal ID, proof correctness) cap scores at 2.5 if ≤ 1.5
- 📤 **Multi-format export**: Markdown, PDF, and CSV (useful for benchmarking across papers)
- ✅ **Quote validation**: Verifies all LLM quotes exist in source text
- 💾 **Smart caching**: SHA256-based result caching to avoid redundant API calls

---

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install fpdf               # required for PDF report export
```

### 3. Configure API credentials

API credentials are managed via a `.env` file (gitignored for security).

```bash
cd app_system

# Copy the template
cp .env.example .env

# Edit with your API credentials
nano .env  # or use your preferred editor
```

The `.env` file contains:
- `API_KEY` — Your API key
- `API_BASE` — API endpoint URL
- `MODEL_PRIMARY`, `MODEL_SECONDARY`, `MODEL_TERTIARY` — Model identifiers

Configuration is loaded by `app_system/config.py` using `python-dotenv`. The system supports multiple API providers (OpenAI, Anthropic, Gemini, custom endpoints).

**Important**: Never commit `.env` to git — it's already in `.gitignore`.

---

## Running the App

The app **must be launched from inside `app_system/`** so that relative imports resolve correctly:

```bash
cd app_system
streamlit run app.py
```

Or use the provided script (disables Streamlit's file watcher to avoid system limits):

```bash
cd app_system
bash run_app.sh
```

Demo apps (show pre-generated debate transcripts):

```bash
cd app_system
streamlit run demos/app_demo.py    # Demo 1: Adjusted R² issue
streamlit run demos/app_demo2.py   # Demo 2: Standard errors issue
```

---

## How It Works

### Section Evaluator

1. Upload a PDF, `.tex`, or `.txt` file, or paste text directly.
2. Select paper type (Empirical / Theoretical / Policy).
3. The app auto-detects section boundaries using a two-pass pipeline (heuristic scoring + LLM confirmation).
4. Each section is evaluated against paper-type-specific criteria with weighted scoring (1–5 per criterion).
5. A fatal-flaw rule caps any section at 2.5 if a critical criterion (e.g. causal identification, proof correctness) scores ≤ 1.5.
6. Results include qualitative assessment, per-criterion scores with quote validation, priority improvements, and an overall publication readiness rating.
7. Export as Markdown, PDF, or CSV (one row per criterion — useful for benchmarking across papers).

### Multi-Agent Referee

1. Upload a manuscript PDF (with automatic table extraction).
2. Choose output mode: full output only (14 API calls) or with LLM summarization (+10-15 calls for cleaner display).
3. Round 0: LLM selects the 3 most relevant personas for this paper and assigns importance weights.
4. Round 1: All 3 personas evaluate independently in parallel.
5. Rounds 2A/2B/2C: Structured cross-examination (questions → answers → final amendments).
6. Round 3: Senior Editor computes weighted consensus and writes a formal referee report.
7. View detailed token usage and cost estimates (input/output tokens, API calls, estimated cost).
8. Export full debate transcript as Markdown or downloadable ZIP package.

---

## Repository Structure

```
research_agents/
├── app_system/                        # All application code
│   ├── app.py                         # Main Streamlit entry point (tabbed UI)
│   ├── utils.py                       # LLM infrastructure (single_query, ConversationManager)
│   ├── config.py                      # API configuration loader (reads .env)
│   ├── run_app.sh                     # Launch script
│   ├── .env.example                   # Template for API credentials
│   ├── README.md                      # User-facing documentation
│   │
│   ├── referee/                       # ⭐ Referee report package
│   │   ├── __init__.py               # Package exports
│   │   ├── workflow.py               # Main production UI (RefereeWorkflow)
│   │   ├── engine.py                 # Debate orchestration (execute_debate_pipeline)
│   │   ├── _utils/                   # Internal utilities
│   │   │   └── summarizer.py         # LLM summarization helpers
│   │   └── _archived/                # Archived implementations
│   │       └── full_output_ui.py     # Full verbose UI (not main code path)
│   │
│   ├── section_eval/                  # ⭐ Section evaluator package
│   │   ├── __init__.py
│   │   ├── main.py                   # SectionEvaluatorApp (UI entry point)
│   │   ├── evaluator.py              # Core evaluation logic
│   │   ├── scoring.py                # Weighted scoring + fatal-flaw floor
│   │   ├── section_detection.py      # Heuristic + LLM section detection
│   │   ├── text_extraction.py        # PDF/LaTeX/TXT extraction
│   │   ├── hierarchy.py              # Subsection grouping
│   │   ├── utils.py                  # Section eval utilities
│   │   ├── criteria/                 # Criteria registry
│   │   │   ├── __init__.py
│   │   │   └── base.py               # Paper-type × section criteria
│   │   └── prompts/                  # Section eval prompt templates
│   │
│   ├── prompts/                       # External prompt files (versioned)
│   │   ├── multi_agent_debate/       # MAD system prompts
│   │   └── section_evaluator/        # Section eval prompts
│   │
│   ├── tests/                         # All test files
│   │   ├── __init__.py
│   │   ├── test_consensus_calculation.py
│   │   ├── test_prompt_loader.py
│   │   ├── test_referee_display.py
│   │   ├── test_referee_quick.py
│   │   └── test_section_evaluator_prompts.py
│   │
│   ├── demos/                         # Demo/alternate apps
│   │   ├── app_demo.py               # Demo 1: Adjusted R² issue
│   │   ├── app_demo2.py              # Demo 2: Standard errors issue
│   │   ├── app_demo3.py              # Demo 3
│   │   ├── app_full_output.py        # Full output version demo
│   │   └── app_summarized_only.py    # Summarized-only demo
│   │
│   ├── docs/                          # Documentation files
│   │   ├── architecture.md           # System architecture
│   │   ├── changelog.md              # Change history
│   │   ├── API_CONFIGURATION.md      # API setup guide
│   │   ├── FRAMEWORK.md              # Evaluation framework
│   │   ├── PROMPT_MANAGEMENT.md      # Prompt versioning
│   │   └── ...                       # Other docs
│   │
│   └── results/                       # Output directory
│
├── mad_experiments/                   # Research scratch experiments
│   ├── exp-1/
│   └── exp-2/
│
├── papers/                            # Sample papers for testing
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── CLAUDE.md                          # Guidance for Claude Code
```

---

## Key Configuration

| Setting | Location | Default |
|---|---|---|
| API credentials | `app_system/.env` | (copy from `.env.example`) |
| API endpoint | `app_system/config.py` | Loaded from `.env:API_BASE` |
| MAD model | `app_system/utils.py:model_selection3` | Claude 3.7 Sonnet |
| Section eval model | `app_system/utils.py:model_selection` | Claude Sonnet 4.5 |
| Referee personas | `app_system/referee/engine.py:SYSTEM_PROMPTS` | 5 personas (3 selected per paper) |
| Referee output mode | UI toggle in app | Full output (14 calls) or with summarization (+10-15 calls) |
| Token/cost tracking | `app_system/referee/engine.py` | Automatic estimation with detailed breakdown |
| Fatal-flaw threshold | `app_system/section_eval/criteria/base.py` | score ≤ 1.5 → cap at 2.5 |
| Cache prefix | `app_system/section_eval/evaluator.py:CACHE_PREFIX` | `"se_cache_v3"` |
