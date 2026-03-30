# App System Directory

This directory contains all files needed to run the Streamlit evaluation apps.

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

## Structure

```
app_system/
├── app.py                    # Main production app (2 tabs: Referee Report + Section Evaluator)
├── demos/
│   ├── app_demo.py          # Demo 1: Shows madoutput1.txt results (Adjusted R² issue)
│   └── app_demo2.py         # Demo 2: Shows madoutput2.txt results (standard errors issue)
├── referee.py               # Referee Report Checker workflow (multi-agent debate UI)
├── multi_agent_debate.py    # Multi-agent debate orchestration (Rounds 0-3)
├── utils.py                 # LLM utilities (single_query, ConversationManager)
├── docs/
│   ├── FRAMEWORK.md
│   └── architecture.md
├── section_eval/            # Section evaluator module
│   ├── main.py             # Section evaluator main class
│   ├── criteria/           # Evaluation criteria definitions
│   ├── prompts/            # Prompt templates
│   ├── scoring.py          # Scoring calculations
│   └── utils.py            # Section eval utilities
├── madoutput1.txt          # Demo 1 multi-agent debate transcript
└── madouput2.txt           # Demo 2 multi-agent debate transcript (note: typo in filename)
```

## Running the Apps

### From the app_system directory:

```bash
cd app_system
streamlit run app.py          # Main app
streamlit run demos/app_demo.py     # Demo 1
streamlit run demos/app_demo2.py    # Demo 2
```

### From the repository root:

```bash
streamlit run app_system/app.py          # Main app
streamlit run app_system/demos/app_demo.py     # Demo 1
streamlit run app_system/demos/app_demo2.py    # Demo 2
```

## Key Dependencies

### Multi-Agent Debate Flow:
```
app.py
  └─ referee.py
       └─ multi_agent_debate.py
            └─ utils.py (single_query)
```

### Section Evaluator Flow:
```
app.py
  └─ section_eval/
       ├─ main.py (SectionEvaluatorApp)
       ├─ criteria/base.py (evaluation criteria)
       ├─ prompts/templates.py (prompt templates)
       └─ utils.py (LLM calls)
```

## Configuration

Configuration is managed via `.env` file (loaded by `config.py`):

### API Settings (.env file):
- `API_KEY` - Your API key
- `API_BASE` - API endpoint URL (supports OpenAI, Anthropic, Gemini, custom)
- `MODEL_PRIMARY` - Primary model (Section Evaluator) - default: Claude Sonnet 4.5
- `MODEL_SECONDARY` - Secondary model (Referee Debate) - default: Claude 3.7 Sonnet
- `MODEL_TERTIARY` - Tertiary model (Legacy/Backup)

See `.env.example` for configuration templates for different API providers.

### Temperature Settings:
- **Referee debate**: Temperature 1.0 (with thinking mode enabled)
- **Section evaluator**: Temperature 0.3 (more conservative for scoring)

## Notes

- Demo apps in `demos/` add the parent `app_system/` directory to `sys.path` so shared imports resolve correctly
- The `section_eval/` directory is a self-contained module
- Demo apps load pre-generated debate transcripts from `madoutput1.txt` and `madouput2.txt`
- For development, edit files in this directory
- Architecture and framework docs live in `app_system/docs/`
