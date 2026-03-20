# App System Directory

This directory contains all files needed to run the Streamlit evaluation apps.

## Structure

```
app_system/
├── app.py                    # Main production app (2 tabs: Referee Report + Section Evaluator)
├── app_demo.py              # Demo 1: Shows madoutput1.txt results (Adjusted R² issue)
├── app_demo2.py             # Demo 2: Shows madoutput2.txt results (standard errors issue)
├── referee.py               # Referee Report Checker workflow (multi-agent debate UI)
├── multi_agent_debate.py    # Multi-agent debate orchestration (Rounds 0-3)
├── utils.py                 # LLM utilities (single_query, ConversationManager)
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
streamlit run app_demo.py     # Demo 1
streamlit run app_demo2.py    # Demo 2
```

### From the parent eval directory:

```bash
streamlit run app_system/app.py          # Main app
streamlit run app_system/app_demo.py     # Demo 1
streamlit run app_system/app_demo2.py    # Demo 2
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

### LLM Settings (utils.py):
- **Model**: `model_selection3` = Claude 3.7 Sonnet
- **Temperature**: 0.5 (multi-agent debate)
- **API Endpoint**: Federal Reserve internal API

### Section Evaluator Settings (section_eval/utils.py):
- **Model**: `model_selection` = Claude Sonnet 4.5
- **Temperature**: 0.3 (more conservative for scoring)

## What Stayed in Parent Directory

The following files remain in the parent `eval/` directory as they are experimental, legacy, or documentation:
- `madexp.py`, `madexp2.py`, `madexp2_local.py` - Experimental/Colab scripts
- `section_eval.py`, `section_eval_new.py`, `section_eval_vivi_0223.py`, `section_eval_llm_vivi.py` - Legacy section evaluator versions
- `routing.py` - Routing utilities
- All `.md` documentation files (CRITERIA_MATRIX.md, ARCHITECTURE.md, etc.)
- `changelog/` and `comparative results/` directories

## Notes

- All imports within `app_system/` are relative and should work correctly
- The `section_eval/` directory is a self-contained module
- Demo apps load pre-generated debate transcripts from `madoutput1.txt` and `madouput2.txt`
- For development, edit files in this directory
- For documentation updates, edit markdown files in parent `eval/` directory
