# Research Agents — Evaluation Agent for Economics Papers

A Streamlit application that evaluates academic economics papers using two complementary workflows:

1. **Section Evaluator** — Criteria-based evaluation of individual paper sections, weighted by paper type (empirical, theoretical, policy) and section importance.
2. **Multi-Agent Referee** — Simulated peer review via structured debate between specialized AI personas (Theorist, Empiricist, Historian, Visionary, Policymaker).

The full application lives in `app_system/`.

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

1. Upload a manuscript PDF.
2. Round 0: LLM selects the 3 most relevant personas for this paper and assigns importance weights.
3. Round 1: All 3 personas evaluate independently in parallel.
4. Rounds 2A/2B/2C: Structured cross-examination (questions → answers → final amendments).
5. Round 3: Senior Editor computes weighted consensus and writes a formal referee report.
6. Export full debate transcript as Markdown.

---

## Repository Structure

```
research_agents/
├── app_system/                  # All application code
│   ├── app.py                   # Streamlit entry point
│   ├── utils.py                 # LLM infrastructure (single_query, ConversationManager)
│   ├── referee.py               # Referee Report UI workflow
│   ├── multi_agent_debate.py    # MAD pipeline orchestration (Rounds 0–3)
│   ├── run_app.sh               # Launch script
│   ├── section_eval/            # Section evaluator package
│   │   ├── evaluator.py         # Core evaluation logic
│   │   ├── main.py              # Section evaluator UI
│   │   ├── scoring.py           # Weighted scoring + fatal-flaw floor
│   │   ├── section_detection.py # Heuristic + LLM section detection
│   │   ├── text_extraction.py   # PDF/LaTeX/TXT extraction
│   │   ├── hierarchy.py         # Subsection grouping
│   │   ├── criteria/base.py     # Criteria registry (paper-type × section)
│   │   └── prompts/templates.py # Evaluation prompt builder
│   ├── demos/                   # Pre-generated demo apps
│   └── docs/                    # Architecture docs and changelog
├── mad_experiments/             # Research scratch experiments
├── papers/                      # Sample papers for testing
├── requirements.txt
└── CLAUDE.md                    # Guidance for Claude Code
```

---

## Key Configuration

| Setting | Location | Default |
|---|---|---|
| API endpoint | `app_system/utils.py:API_BASE` | Federal Reserve MartinAI |
| MAD model | `app_system/utils.py:model_selection3` | Claude 3.7 Sonnet |
| Section eval model | `app_system/utils.py:model_selection` | Claude Sonnet 4.5 |
| Fatal-flaw threshold | `section_eval/criteria/base.py` | score ≤ 1.5 → cap at 2.5 |
| Cache prefix | `section_eval/evaluator.py:CACHE_PREFIX` | `"se_cache_v3"` |
