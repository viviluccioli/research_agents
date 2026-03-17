# Evaluation Agent App System

AI-powered evaluation system for economics research papers using multi-agent debate and section-level assessment.

## Overview

This application provides two complementary evaluation approaches:

1. **Multi-Agent Debate (MAD)**: Simulates peer review through structured debates between specialized AI personas, producing holistic accept/reject recommendations with detailed referee reports.

2. **Section Evaluator**: Delivers granular, criteria-based scoring of individual manuscript sections with evidence-backed feedback for iterative improvement.

Both systems are paper-type aware (empirical/theoretical/policy), require textual evidence for all claims, and use proportional error weighting to avoid over-penalizing minor issues.

## Quick Start

### Prerequisites

- Python 3.8+
- Access to Federal Reserve internal API endpoint
- API key for Claude models

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd research_agents-main/eval
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy requests tqdm pdfplumber tiktoken
   ```

   Or use the pyproject.toml:
   ```bash
   pip install -e .
   ```

3. **Configure API credentials**:

   The API configuration is set in `utils.py`:
   - `API_KEY`: Your Federal Reserve API key
   - `API_BASE`: API endpoint URL
   - `model_selection`: Claude model version

   **Important**: Never commit real API keys to version control!

### Running the Application

#### Main Production App

```bash
# From the app_system directory
cd app_system
./run_app.sh

# Or manually
streamlit run app_demo2.py --server.fileWatcherType none --server.port 8501
```

The app will be available at `http://localhost:8501`

#### Demo Versions

Two demo apps are included that showcase pre-generated evaluation results:

```bash
# Demo 1: Shows evaluation of paper with Adjusted R² issue
streamlit run app_demo.py

# Demo 2: Shows evaluation of paper with standard errors issue
streamlit run app_demo2.py
```

### Alternative Ports

If port 8501 is already in use:
```bash
streamlit run app_demo2.py --server.port 8502
```

## Application Structure

```
app_system/
├── app.py                    # Main production app (legacy, 2-tab interface)
├── app_demo.py              # Demo 1: Pre-loaded results (Adjusted R² case)
├── app_demo2.py             # Demo 2: Pre-loaded results (standard errors case)
├── run_app.sh               # Launch script with optimal settings
│
├── referee.py               # Referee Report UI workflow
├── multi_agent_debate.py    # Multi-agent debate orchestration (Rounds 0-3)
├── utils.py                 # Core LLM utilities (API calls, conversation manager)
│
├── section_eval/            # Section evaluator module
│   ├── main.py             # SectionEvaluatorApp main class
│   ├── evaluator.py        # Evaluation engine
│   ├── scoring.py          # Scoring calculations and aggregation
│   ├── text_extraction.py  # PDF/LaTeX document processing
│   ├── section_detection.py # Automatic section identification
│   ├── hierarchy.py        # Section hierarchy management
│   ├── utils.py            # Section eval LLM utilities
│   ├── criteria/           # Evaluation criteria definitions
│   │   ├── __init__.py
│   │   └── base.py         # Paper-type-specific criteria
│   └── prompts/            # Prompt templates
│       ├── __init__.py
│       └── templates.py    # Structured prompts for evaluation
│
├── madoutput1.txt          # Demo 1 debate transcript
└── madouput2.txt           # Demo 2 debate transcript (note: typo in filename)
```

## Features

### Multi-Agent Debate System

- **Endogenous Persona Selection**: AI automatically selects the 3 most relevant reviewers from a pool of 5 specialized personas based on paper content
- **Structured Debate Protocol**: Four-round evaluation process:
  - R0: Persona selection and weight assignment
  - R1: Independent evaluations (PASS/REVISE/FAIL verdicts)
  - R2A-C: Cross-examination, responses, and verdict updates
  - R3: Weighted consensus and final decision
- **Available Personas**: Theorist, Empiricist, Historian, Visionary, Policymaker
- **Weighted Voting**: Final decision uses expertise-weighted consensus
- **Full Transparency**: Complete debate transcript and reasoning chain

### Section Evaluator

- **Paper-Type Awareness**: Automatically detects empirical/theoretical/policy papers and applies appropriate criteria
- **Evidence-Backed Scoring**: Requires 2 verbatim quotes per criterion, algorithmically validated
- **Hierarchical Weighting**:
  - Criterion importance within sections (sum to 100%)
  - Section multipliers across paper (e.g., methodology 1.3×, abstract 0.7×)
- **Minimum Gates**: Prevents strong sections from masking weak methodology
- **Actionable Feedback**: Prioritized improvement suggestions with specific evidence

### Document Processing

- **PDF Support**: Uses pdfplumber with OCR error tolerance
- **LaTeX Support**: Regex-based stripping that preserves mathematical notation
- **Automatic Section Detection**: Heuristic + LLM-confirmed identification of paper structure

## Configuration

### LLM Models

The system uses different Claude models optimized for each task:

**Multi-Agent Debate** (`utils.py`):
- Model: Claude 3.7 Sonnet (`anthropic.claude-3-7-sonnet-20250219-v1:0`)
- Temperature: 0.5 (balanced reasoning + creativity)
- Extended thinking: 2048 token budget for internal reasoning

**Section Evaluator** (`section_eval/utils.py`):
- Model: Claude Sonnet 4.5 (`anthropic.claude-sonnet-4-5-20250929-v1:0`)
- Temperature: 0.3 (conservative for consistent scoring)
- Extended thinking: 2048 token budget

### Customizing Settings

Edit `utils.py` to modify:
- API endpoints and keys
- Model selection
- Temperature settings
- Retry logic
- Token budgets

## Usage Guide

### Multi-Agent Debate Workflow

1. Upload paper (PDF or text)
2. System automatically:
   - Selects 3 relevant personas
   - Assigns expertise weights
   - Conducts 4-round debate
   - Generates weighted consensus
3. Review:
   - Referee report with final decision
   - Individual persona evaluations
   - Full debate transcript
   - Weighted scoring breakdown

**Best for**: Complete manuscripts, submission decisions, holistic assessment

**Runtime**: 3-5 minutes (13-16 LLM calls)

### Section Evaluator Workflow

1. Upload paper (PDF, LaTeX, or text)
2. System automatically:
   - Detects paper type (empirical/theoretical/policy)
   - Identifies sections (abstract, intro, methodology, etc.)
   - Loads appropriate criteria
   - Evaluates each section with evidence quotes
   - Calculates weighted aggregate score
3. Review:
   - Section-by-section scores
   - Criterion breakdowns with evidence
   - Publication readiness assessment
   - Prioritized improvement recommendations

**Best for**: Works in progress, targeted revisions, training, iterative feedback

**Runtime**: 1-2 minutes per paper

### Complementary Workflow

1. **Early drafting**: Use Section Evaluator for iterative feedback
2. **Pre-submission**: Run Multi-Agent Debate for holistic check
3. **Post-review**: Use Section Evaluator to target referee concerns

## Troubleshooting

### Common Issues

**Port already in use**:
```bash
# Try alternative port
streamlit run app_demo2.py --server.port 8502
```

**File watcher errors**:
```bash
# Disable file watcher (already done in run_app.sh)
streamlit run app_demo2.py --server.fileWatcherType none
```

**API connection errors**:
- Verify API_KEY in `utils.py`
- Check VPN/network access to Federal Reserve internal API
- Confirm API endpoint URL is correct

**Import errors**:
```bash
# Ensure you're in the correct directory
cd app_system
python -c "from section_eval import SectionEvaluatorApp"
```

**PDF processing issues**:
- Ensure pdfplumber is installed: `pip install pdfplumber>=0.9.0`
- Try converting PDF to text externally if OCR issues persist

### Debugging

Enable debug mode in `utils.py`:
```python
def single_query(prompt: str, debug_flag=True, retries=3):
    # ... will print request/response details
```

Check Streamlit logs:
```bash
# Run with verbose logging
streamlit run app_demo2.py --logger.level=debug
```

## Architecture

### Key Dependencies

**Multi-Agent Debate Flow**:
```
app_demo2.py
  └─ referee.py
       └─ multi_agent_debate.py
            └─ utils.py (single_query, ConversationManager)
```

**Section Evaluator Flow**:
```
app_demo2.py
  └─ section_eval/
       ├─ main.py (SectionEvaluatorApp)
       ├─ evaluator.py (evaluation engine)
       ├─ criteria/base.py (paper-type-specific criteria)
       ├─ prompts/templates.py (prompt templates)
       └─ utils.py (LLM API calls)
```

### Design Principles

1. **Modular Design**: Clear separation between UI (app files), orchestration (referee/multi_agent_debate), and utilities
2. **Self-Contained Modules**: `section_eval/` is fully independent and can be imported elsewhere
3. **Evidence Requirements**: All assessments must provide textual proof
4. **Proportional Weighting**: Error severity scaled by impact on core claims
5. **Human-in-the-Loop**: AI recommends, humans decide

## Documentation

Full documentation is available in `../docs/`:

- `FRAMEWORK.md` - System overview and design philosophy
- `ARCHITECTURE.md` - Technical architecture deep dive
- `CRITERIA_MATRIX.md` - Complete criteria definitions
- `SYSTEM_PROMPTS_DOCUMENTATION.md` - Prompt engineering guide
- `DEMO_README.md` - Demo walkthrough
- `QUICK_REFERENCE.md` - Quick command reference

## Development

### Adding New Personas

Edit `multi_agent_debate.py` to add personas to the pool:
```python
personas = ["Theorist", "Empiricist", "Historian", "Visionary", "Policymaker", "YourNewPersona"]
```

Then define the persona's instructions in the prompt template.

### Modifying Evaluation Criteria

Edit `section_eval/criteria/base.py`:
1. Add new criteria to the appropriate section dictionary
2. Specify criterion weight (must sum to 100% per section)
3. Define evaluation guidelines

### Changing Section Multipliers

Edit `section_eval/scoring.py` to adjust importance weights:
```python
section_multipliers = {
    "methodology": 1.3,  # Increase for more weight
    "abstract": 0.7,     # Decrease for less weight
}
```

## Limitations

### Current Constraints

1. **No proof verification**: Cannot rigorously check mathematical derivations (flags suspicious steps)
2. **No code execution**: Evaluates reported methodology, not actual replication
3. **Training cutoff**: Very recent papers may not be in knowledge base
4. **Subjective trade-offs**: Surfaces tensions (novelty vs. rigor) but humans decide

### Ethical Guardrails

- **Transparency**: Users know feedback is AI-generated
- **Human authority**: AI recommends, doesn't decide
- **Bias mitigation**: Blind evaluation, multi-persona debate
- **Privacy**: Internal API, no model training on uploads
- **Accountability**: Feedback loops for flagging bad advice

## Support

For issues, questions, or feature requests:
- Check `../docs/QUICK_REFERENCE.md` for common solutions
- Review debate transcripts in `madoutput*.txt` for examples
- Contact: research-agents@federalreserve.gov

## Version

Current Version: 3.0
Last Updated: March 2026
Status: Production-ready, active use in Federal Reserve System

## License

Federal Reserve System Internal Use
