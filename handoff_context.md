# Research Agents Project: Handoff Documentation

## Project Overview

This repository contains an **AI-powered peer review system** for economics research papers, built on a **Multi-Agent Debate (MAD)** architecture. The system simulates an academic referee panel where specialized AI personas evaluate papers through structured debate rounds, ultimately producing referee reports and publication recommendations.

The project has two primary evaluation systems:
1. **Referee Report System (MAD)**: Multi-agent debate generating comprehensive referee reports with publication verdicts (ACCEPT/REVISE/REJECT)
2. **Section Evaluator**: Granular section-by-section analysis scoring papers on specific criteria by paper type

## Repository Structure

### 1. `mad_experiments/` - Development History

This directory contains **Rithika's experimental iterations** (former intern) showing the evolution of the MAD system:

- **exp-1/**: Original 5-persona system (Theorist, Empiricist, Historian, Visionary, Policymaker)
- **exp-2/**: Refinements to debate structure and prompts
- **exp-3/**: Prototype for memo evaluation (policy memos vs. research papers)
- **exp_4/**: **10-persona system expansion** - added technical depth (Econometrician, ML Expert, Data Scientist, CS Expert) and ethical dimensions (Ethicist, Perspective)

Each experiment folder contains Jupyter notebooks showing the prototyping process. **Once an experiment proved successful, it was integrated into `app_system/`**. This directory serves as an **archive of design decisions and iterative improvements**.

### 2. `app_system/` - Production Application

This is the **deployed system** - a Streamlit web application providing both evaluation systems.

#### Architecture

**Entry Points**:
- `app.py`: Main application (5-persona MAD + Section Evaluator)
- `app_exp_4.py`: Experimental 10-persona version
- `app-memo.py`: Policy memo evaluation variant

**Core Systems**:

##### Referee Report System (`app_system/referee/`)

Implements the Multi-Agent Debate architecture:

**5-Round Debate Structure**:
1. **Round 0**: LLM selects 3 of 10 available personas based on paper content, assigns weighted importance to each
2. **Round 1**: Selected personas write independent critical analyses in parallel
3. **Round 2A**: Cross-examination - personas generate questions for peers
4. **Round 2B**: Direct examination - personas answer questions
5. **Round 2C**: Amended evaluations incorporating debate insights
6. **Round 3**: Editor synthesizes weighted consensus into final verdict

**Key Features**:
- **Parallel execution**: Uses `asyncio.gather()` to run personas simultaneously
- **Context isolation**: Each round receives only appropriate context (prevents information overload)
- **Per-round temperature control**: Different creativity/consistency balance per round (0.4 for selection/synthesis, 0.7 for analysis)
- **Weighted consensus**: Editor computes verdict from persona recommendations weighted by Round 0 assignments

**10 Available Personas**:
- **Technical**: Theorist, Econometrician, ML Expert, Data Scientist, CS Expert
- **Contextual**: Historian, Visionary, Policymaker
- **Critical**: Ethicist, Perspective (critical lens on social implications)

**Internal Utilities** (`referee/_utils/`):
- **Quote Validation**: Prevents hallucinations by verifying quotes exist in source paper
- **Caching**: SHA256-based per-round caching (50-80% cost savings during development)
- **Deduplication**: Identifies duplicate findings across persona reports
- **PDF Extraction**: PyMuPDF-based extraction with figure/table support

**📄 Key Documentation**: 
- `app_system/referee/description.md`: Architecture deep-dive
- `docs/quote_validation.md`: Quote verification system
- `docs/caching.md`: Caching implementation
- `docs/deduplication.md`: Cross-reference deduplication
- `CHANGES_2026-05-05.md`: Removed generic system prompt pollution (Phase 1 consistency)
- `CHANGES_2026-05-06.md`: Per-round temperature control (Phase 2 consistency)

##### Section Evaluator (`app_system/section_eval/`)

**5-Stage Pipeline**:
1. **Text Extraction**: PDF/LaTeX/plain text parsing
2. **Section Detection**: Two-pass (heuristic + LLM confirmation)
3. **Hierarchy Grouping**: Group subsections under parents
4. **Evaluation**: Score each section against paper-type-specific criteria
5. **Scoring**: Aggregate with fatal-flaw logic (critical criteria failures cap scores)

**Paper Types Supported**: Empirical, Theoretical, Policy, Finance, Macro, Systematic Review

**Fatal-Flaw Logic**: Any criterion marked `critical=True` scoring ≤1.5 caps the entire section score at 2.5 (e.g., identification strategy in empirical papers)

**📄 Key Documentation**: 
- `app_system/docs/FRAMEWORK.md`: High-level system overview
- `docs/math_cleanup.md`: LaTeX normalization

#### Shared Infrastructure (`app_system/utils.py`, `config.py`)

- **Model Configuration**: ALL systems use **Claude 4.5 Sonnet** (`anthropic.claude-sonnet-4-5-20250929-v1:0`)
- **API Management**: Federal Reserve MartinAI (OpenAI-compatible endpoint), configured via `.env` file
- **LLM Call Patterns**:
  - `referee_query()`: Referee system (no generic system prompt, per-round temperatures)
  - `safe_query()`: Section evaluator (temperature 0.3, no thinking mode)
  - `ConversationManager.conv_query()`: Stateful conversations with auto-pruning

#### Prompt Organization (`app_system/prompts/`)

**Versioned external prompt files** organized by system:
- `multi_agent_debate/`: Persona prompts, debate round prompts, paper type contexts
- `section_evaluator/`: Paper type contexts, section guidance, master prompts

**Version Control**: `config.yaml` files specify active versions, prompt files follow `v{MAJOR}.{MINOR}.txt` naming

### 3. `experiment/` - Ground Truth Validation

**Vision**: Validate the MAD system's accuracy by comparing its verdicts against real-world editorial decisions.

**Ground Truth Dataset**:
- ✅ **Positive class**: Papers accepted to **top-tier journals** (AER, QJE, JPE, Econometrica, etc.)
- ❌ **Negative class**: Papers published **only as FEDS Notes or IFDP notes** (Federal Reserve internal publications, not externally peer-reviewed)

**Hypothesis**: If the MAD system is calibrated well, it should:
- **ACCEPT** papers that were accepted to top journals
- **REJECT/REVISE** papers that remained internal-only

**Batch Processing** (`batch_referee_reports.py`):
- Runs MAD system on multiple papers
- Matches verdicts to ground truth from `tracking.csv` (doc_id → Tier)
- Outputs:
  - **CSV**: Summary results with accuracy metrics
  - **JSON**: Full debate transcripts for qualitative analysis

**📄 Key Files**:
- `experiment/tracking.csv`: Ground truth labels (Tier 1 = top journal, Tier 2/3 = internal only)
- `experiment/run_experiment.sh`: Shell wrapper for batch runs

**Limitations**: This is a **rough proxy** for accuracy because:
- Top journal acceptance ≠ perfect quality (editorial decisions involve many factors)
- FEDS/IFDP notes may be high-quality but unsuitable for external publication (timeliness, policy focus)
- The MAD system evaluates different criteria than human editors

### 4. Additional Subsystems

- **`persona_exp/`**: Consistency experiments for Round 0 persona selection (testing if same paper → same personas)
- **`referee_classifier/`**: Separate classification system (not integrated with main app)
- **`commit_history/`**: Auto-generated documentation of every git commit (via Claude Code hook)

## Key Technical Achievements

1. **Consistency Improvements** (2026-05-05 to 2026-05-06):
   - Removed generic system prompt pollution from referee calls (Phase 1)
   - Implemented per-round temperature control (Phase 2)
   - Expected: 60-80% reduction in verdict variability

2. **Quote Validation System**: Prevents LLM hallucinations by fuzzy-matching quotes against source paper (95% threshold for math, 85% for prose)

3. **Granular Caching**: Per-round SHA256-based caching saves 50-80% of costs during iterative development

4. **Parallel Execution**: `asyncio.gather()` runs personas simultaneously (4× speedup for 4 personas)

5. **10-Persona Expansion**: Broader expertise coverage improved technical depth and ethical evaluation

## Important Documentation to Review

**Start Here**:
1. `CLAUDE.md`: Comprehensive system documentation (architecture, file organization, setup)
2. `README.md`: User-facing quickstart guide
3. `app_system/docs/FRAMEWORK.md`: High-level conceptual overview

**Deep Dives**:
- `app_system/referee/description.md`: MAD architecture details
- `EXPERIMENT_4_SUMMARY.md`: 10-persona system evolution
- `docs/quote_validation.md`: Quote verification implementation
- `docs/caching.md`: Caching system documentation
- `running-ideas.md`: Full problem analysis + future improvement phases

**Recent Changes**:
- `CHANGES_2026-05-05.md`: Phase 1 consistency improvements
- `CHANGES_2026-05-06.md`: Phase 2 per-round temperature control
- `commit_history/`: Detailed record of every code change

## Current State & Future Work

**Production Ready**: `app_system/app.py` is the stable production system

**Ongoing Research**:
- Validating accuracy against ground truth (experiment module)
- Further consistency improvements (see `running-ideas.md` for roadmap)
- Potential expansion to 11+ personas or additional paper types

**Known Limitations**:
- Thinking mode NOT currently enabled (conflicts with per-round temperature control)
- Verdict variability still present (Phase 3+ improvements planned)
- Cost: ~$1.50-2.00 per paper (5-persona), ~$2.00-3.00 (10-persona)

---

This system represents a significant advance in automated peer review, combining structured debate, specialized expertise, and rigorous validation to produce human-quality referee reports at scale.
