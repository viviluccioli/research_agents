# Research Evaluation Agents

AI-powered evaluation system for economics research papers using multi-agent debate and section-level assessment.

## Overview

This repository contains a dual-system framework for evaluating economics research:

1. **Multi-Agent Debate (MAD)**: Simulates peer review through debates between specialized AI personas
2. **Section Evaluator**: Provides detailed, criteria-based scoring of individual sections

Both systems are paper-type aware (empirical/theoretical/policy), require textual evidence for all claims, and use proportional error weighting.

## Repository Structure

```
research_agents-main/
├── app_system/              # Main evaluation application
│   ├── README.md           # Detailed setup and usage guide
│   ├── app_demo2.py        # Streamlit app (main interface)
│   ├── run_app.sh          # Launch script
│   ├── multi_agent_debate.py  # MAD orchestration
│   ├── referee.py          # Referee report workflow
│   ├── utils.py            # LLM utilities
│   └── section_eval/       # Section evaluator module
│       ├── main.py
│       ├── criteria/       # Evaluation criteria
│       └── prompts/        # Prompt templates
├── docs/                    # Documentation
│   ├── FRAMEWORK.md        # System overview
│   ├── ARCHITECTURE.md     # Technical architecture
│   ├── CRITERIA_MATRIX.md  # Evaluation criteria
│   └── ...                 # Additional documentation
├── pyproject.toml          # Python dependencies
└── *.py                    # Experimental/legacy scripts
```

## Quick Start

### Prerequisites

- Python 3.8+
- Access to Federal Reserve internal API endpoint
- API key for Claude models

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd research_agents-main
   ```

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy requests tqdm pdfplumber tiktoken
   ```

   Or use pyproject.toml:
   ```bash
   pip install -e .
   ```

3. **Configure API credentials**:

   Edit `app_system/utils.py` to set:
   - `API_KEY`: Your Federal Reserve API key
   - `API_BASE`: API endpoint URL
   - `model_selection`: Claude model version

   **Important**: Never commit real API keys to version control!

### Running the Application

```bash
cd app_system
./run_app.sh
```

Or manually:
```bash
streamlit run app_demo2.py --server.fileWatcherType none --server.port 8501
```

The app will be available at `http://localhost:8501`

## Documentation

Full documentation is available in the `docs/` directory:

- **[FRAMEWORK.md](docs/FRAMEWORK.md)** - System overview and design philosophy
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture deep dive
- **[CRITERIA_MATRIX.md](docs/CRITERIA_MATRIX.md)** - Complete evaluation criteria
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Quick command reference
- **[DEMO_README.md](docs/DEMO_README.md)** - Demo walkthrough

For detailed setup instructions, see **[app_system/README.md](app_system/README.md)**

## Key Features

### Multi-Agent Debate
- Endogenous persona selection (AI picks 3 most relevant reviewers)
- Structured 4-round debate protocol
- Weighted consensus voting
- Full transparency with debate transcripts

### Section Evaluator
- Paper-type aware criteria (empirical/theoretical/policy)
- Evidence-backed scoring with verbatim quotes
- Hierarchical importance weighting
- Actionable improvement recommendations

## Use Cases

1. **Pre-submission review**: Get feedback before journal submission
2. **Editorial screening**: Triage borderline papers for review
3. **PhD training**: Provide detailed feedback to students
4. **Internal policy papers**: Assess Fed analysis before senior review
5. **Replication studies**: Identify robustness gaps in published work

## Development

### Repository Organization

- **app_system/**: Production-ready evaluation application
- **docs/**: Comprehensive documentation
- **changelog/**: Version history and changes
- **comparative results/**: Benchmark evaluations
- **rithika_experiments/**: Experimental features
- ***.py** (root level): Legacy/experimental scripts

### Key Files

- `section_eval.py`, `section_eval_new.py` - Legacy section evaluator versions
- `madexp*.py` - Experimental MAD variations
- `routing.py` - Routing utilities

## Support

For issues, questions, or feature requests:
- Check the [app_system README](app_system/README.md) for detailed troubleshooting
- Review documentation in `docs/`
- Contact: research-agents@federalreserve.gov

## Version

Current Version: 3.0
Last Updated: March 2026
Status: Production-ready, active use in Federal Reserve System

## License

Federal Reserve System Internal Use
