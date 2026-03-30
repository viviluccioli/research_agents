# Architecture Documentation

**Evaluation Agent System for Academic Economics Research**

Version: 3.0
Last Updated: March 12, 2026

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Principles](#architecture-principles)
3. [System Architecture](#system-architecture)
4. [Multi-Agent Debate System](#multi-agent-debate-system)
5. [Section Evaluation System](#section-evaluation-system)
6. [Shared Infrastructure](#shared-infrastructure)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Technology Stack](#technology-stack)
9. [Extension Points](#extension-points)
10. [Performance Considerations](#performance-considerations)

---

## System Overview

The Evaluation Agent is a comprehensive system for evaluating academic economics papers using two complementary approaches:

1. **Multi-Agent Debate (MAD)**: Simulates peer review through structured debates between specialized AI personas
2. **Section Evaluator**: Provides granular, criteria-based evaluation of individual paper sections

### Key Features

- **Paper-type awareness**: Adapts evaluation criteria to empirical, theoretical, and policy papers
- **Endogenous persona selection**: Dynamically selects the most relevant reviewers for each paper
- **Evidence-based assessment**: Requires textual quotes to support all evaluations
- **Proportional error weighting**: Contextualizes flaws based on their severity and scope
- **Hierarchical section detection**: Automatically identifies paper structure
- **Weighted scoring system**: Reflects the relative importance of different sections

---

## Architecture Principles

### 1. **Separation of Concerns**
- **Presentation Layer** (`app.py`): Streamlit UI, file handling, session management
- **Orchestration Layer** (`referee.py`, `section_eval/main.py`): Workflow coordination
- **Domain Logic** (`multi_agent_debate.py`, `section_eval/evaluator.py`): Core evaluation algorithms
- **Infrastructure** (`utils.py`): Shared LLM clients, token management

### 2. **Modularity**
Each major component is self-contained:
- Multi-agent debate system: `multi_agent_debate.py` + `referee.py`
- Section evaluator: `section_eval/` package
- Shared utilities: `utils.py`

### 3. **Extensibility**
- **Persona system**: Easy to add new reviewer personas with system prompts
- **Criteria registry**: Paper-type and section-specific evaluation criteria in `criteria/base.py`
- **Prompt templates**: Centralized in `prompts/templates.py`

### 4. **Robustness**
- **Fallback mechanisms**: Heuristic fallbacks when LLM parsing fails
- **Quote validation**: Verifies that extracted quotes actually appear in source text
- **Caching**: Prevents redundant LLM calls for identical content
- **Error handling**: Graceful degradation at multiple layers

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          app.py (Main Entry)                     │
│                      Streamlit Multi-Tab Interface               │
│                                                                  │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐ │
│  │  Tab 1: Referee Report      │  │  Tab 2: Section Evaluator  │ │
│  │  (Multi-Agent Debate)       │  │  (Criteria-Based Analysis) │ │
│  └─────────────────────────────┘  └────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                    │                           │
        ┌───────────┴─────────┐    ┌───────────┴──────────┐
        ▼                     ▼    ▼                      ▼
┌─────────────────┐  ┌──────────────────────┐   ┌─────────────────────┐
│   referee.py    │  │ multi_agent_debate   │   │ section_eval/       │
│  (MAD UI/Flow)  │  │ .py (MAD Engine)     │   │ main.py (SE UI)     │
└─────────────────┘  └──────────────────────┘   └─────────────────────┘
        │                     │                            │
        └─────────────────────┴────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │     utils.py       │
                    │ (LLM Infrastructure)│
                    │                    │
                    │ • ConversationMgr  │
                    │ • single_query()   │
                    │ • Token counting   │
                    └────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Federal Reserve   │
                    │  API (MartinAI)    │
                    │  Claude Models     │
                    └────────────────────┘
```

### Directory Structure

```
app_system/
├── app.py                          # Main Streamlit entry point
├── referee.py                      # Multi-agent debate UI workflow
├── multi_agent_debate.py           # MAD core orchestration
├── utils.py                        # Shared LLM infrastructure
├── README.md                       # Usage documentation
│
├── section_eval/                   # Section evaluator package
│   ├── __init__.py                # Package exports
│   ├── main.py                    # Section evaluator UI
│   ├── evaluator.py               # Core evaluation logic
│   ├── text_extraction.py         # PDF/LaTeX text extraction
│   ├── section_detection.py       # Section header detection
│   ├── hierarchy.py               # Subsection grouping
│   ├── scoring.py                 # Weighted scoring system
│   ├── utils.py                   # Section eval utilities
│   │
│   ├── criteria/                  # Evaluation criteria
│   │   ├── __init__.py
│   │   └── base.py               # Criteria registry
│   │
│   └── prompts/                   # Prompt templates
│       ├── __init__.py
│       └── templates.py          # Evaluation prompt builders
│
├── app_demo.py                    # Demo 1: Adjusted R² issue
├── app_demo2.py                   # Demo 2: Standard errors issue
├── madoutput1.txt                 # Demo 1 transcript
└── madouput2.txt                  # Demo 2 transcript (note: typo)
```

---

## Multi-Agent Debate System

### Overview

The Multi-Agent Debate (MAD) system simulates peer review through structured debates between specialized AI personas. It addresses the "wisdom of crowds" problem in automated review by forcing agents to cross-examine each other's assessments.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Debate Pipeline                  │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐          ┌────▼────┐         ┌────▼────┐
    │ Round 0 │          │ Round 1 │         │ Round 2 │
    │ Persona │          │ Independent│       │ Debate  │
    │Selection│          │Evaluation │       │Rounds   │
    └─────────┘          └──────────┘        └─────────┘
         │                    │                    │
         │                    │               ┌────┴─────┐
         │                    │          ┌────▼────┐┌────▼────┐┌────▼────┐
         │                    │          │  R2A:   ││  R2B:   ││  R2C:   │
         │                    │          │  Cross  ││ Answer  ││  Final  │
         │                    │          │  Exam   ││Questions││Amendment│
         │                    │          └─────────┘└─────────┘└─────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                         ┌────▼────┐
                         │ Round 3 │
                         │ Editor  │
                         │Decision │
                         └─────────┘
```

### Components

#### 1. Persona Selection (Round 0)

**File**: `multi_agent_debate.py:run_round_0_selection()`

**Purpose**: Dynamically select the 3 most relevant personas from a pool of 5 specialists.

**Available Personas**:

| Persona | Focus | Checks |
|---------|-------|--------|
| **Theorist** | Mathematical logic, formal proofs, model insight | Derivation correctness, assumption validity |
| **Empiricist** | Data, econometrics, identification strategy | Statistical validity, empirical soundness |
| **Historian** | Literature context, citation lineage | Gap analysis, novelty claims |
| **Visionary** | Innovation, paradigm-shifting potential | Novelty, creativity, intellectual impact |
| **Policymaker** | Real-world application, policy relevance | Welfare implications, actionable insights |

**Process**:
1. LLM reads paper content (first 8000 chars)
2. Selects exactly 3 personas most relevant to the paper
3. Assigns importance weights (sum = 1.0) to each selected persona
4. Provides justification for selection

**Output Format**:
```json
{
  "selected_personas": ["Empiricist", "Historian", "Policymaker"],
  "weights": {
    "Empiricist": 0.4,
    "Historian": 0.3,
    "Policymaker": 0.3
  },
  "justification": "One sentence explaining the choice and weights."
}
```

#### 2. Independent Evaluation (Round 1)

**File**: `multi_agent_debate.py:run_round_1()`

**Process**:
- Each selected persona independently evaluates the full paper
- Uses persona-specific system prompts from `SYSTEM_PROMPTS` dict
- Applies **proportional error weighting**: contextualizes flaws by severity
- Requires **textual evidence**: verbatim quotes, equation numbers, table numbers

**Output Format** (per persona):
```
- **Domain-Specific Audit**: [Critique based on expertise]
- **Proportional Error Analysis**: [Error severity relative to whole paper]
- **Source Evidence**: [MANDATORY: verbatim quotes/citations]
- **Verdict**: [PASS/REVISE/FAIL]
```

**Parallelization**: All 3 personas evaluate concurrently using `asyncio.gather()`

#### 3. Cross-Examination (Round 2A)

**File**: `multi_agent_debate.py:run_round_2a()`

**Purpose**: Force agents to engage critically with each other's assessments.

**Process**:
- Each persona receives the Round 1 reports from their 2 peers
- Identifies cross-domain conflicts (e.g., Empiricist challenges Historian's gap claim)
- Poses specific clarification questions to each peer

**Output Format**:
```
- **Cross-Domain Insights**: [How peer views change/validate your perspective]
- **Constructive Pushback**: [Clashes between your domain and theirs]
- **Clarification Requests**:
    - To [Peer 1]: [1 specific question]
    - To [Peer 2]: [1 specific question]
```

#### 4. Direct Examination (Round 2B)

**File**: `multi_agent_debate.py:run_round_2b()`

**Purpose**: Answer specific questions posed by peers with evidence.

**Process**:
- Each persona reads the R2A transcript
- Identifies questions directed at them
- Provides direct answers with textual evidence
- Concedes flaws or defends their position

**Output Format**:
```
- **Response to [Peer 1]**: [Direct answer with evidence]
- **Response to [Peer 2]**: [Direct answer with evidence]
- **Concession or Defense**: [Acknowledge flaw or defend ground]
```

#### 5. Final Amendments (Round 2C)

**File**: `multi_agent_debate.py:run_round_2c()`

**Purpose**: Submit final verdict after integrating all debate context.

**Process**:
- Each persona receives full debate transcript (R1, R2A, R2B)
- Updates beliefs based on valid peer critiques
- Submits final verdict with updated rationale

**Output Format**:
```
- **Insights Absorbed**: [How debate changed evaluation]
- **Final Verdict**: [PASS / REVISE / FAIL]
- **Final Rationale**: [3 sentences incorporating debate context]
```

#### 6. Editor Decision (Round 3)

**File**: `multi_agent_debate.py:run_round_3()`

**Purpose**: Calculate weighted consensus and synthesize final referee report.

**Weighting System**:
```
Verdict Values:
  PASS   = 1.0
  REVISE = 0.5
  FAIL   = 0.0

Final Score = Σ(verdict_value × persona_weight)

Decision Thresholds:
  Score > 0.75      → ACCEPT
  0.40 ≤ Score ≤ 0.75 → REJECT AND RESUBMIT
  Score < 0.40      → REJECT
```

**Output Format**:
```
- **Weight Calculation**: [Show math explicitly]
- **Debate Synthesis**: [2-3 sentences summarizing panel alignment]
- **Final Decision**: [ACCEPT / REJECT AND RESUBMIT / REJECT]
- **Official Referee Report**: [Synthesized letter to authors with evidence]
```

### Key Design Decisions

1. **Asynchronous Execution**: All rounds use `asyncio` to parallelize LLM calls
2. **Proportional Error Weighting**: Prevents over-penalization of minor flaws
3. **Evidence Requirement**: Forces grounding in actual paper content
4. **Structured Debate**: Multi-round Q&A ensures thorough examination
5. **Mathematical Consensus**: Democratic but expertise-calibrated decision

---

## Section Evaluation System

### Overview

The Section Evaluator provides granular, criteria-based assessment of individual paper sections. It adapts evaluation criteria based on paper type (empirical/theoretical/policy) and section type.

### Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                   Section Evaluation Pipeline                 │
└───────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐      ┌─────▼─────┐     ┌─────▼──────┐
   │  Text    │      │ Section   │     │  Section   │
   │Extraction│      │ Detection │     │ Extraction │
   └──────────┘      └───────────┘     └────────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                    ┌──────▼────────┐
                    │   Evaluator   │
                    │ Per Section:  │
                    │ • Get Criteria│
                    │ • LLM Query   │
                    │ • Parse JSON  │
                    │ • Validate    │
                    │ • Score       │
                    └───────────────┘
                           │
                    ┌──────▼────────┐
                    │   Scoring     │
                    │ • Section Wgt │
                    │ • Overall Agg │
                    │ • Readiness   │
                    └───────────────┘
```

### Components

#### 1. Text Extraction

**File**: `section_eval/text_extraction.py`

**Purpose**: Extract clean text from PDF, LaTeX, and plain text sources.

**Key Functions**:

- `extract_text_from_pdf(file_bytes)`:
  - Uses `pdfplumber` with word-level extraction fallback
  - Detects and fixes missing spaces (common OCR issue)
  - Regex post-processing for glued words

- `strip_latex(text)`:
  - Pure regex LaTeX parser (no external dependencies)
  - Removes preamble, comments, figure/table environments
  - Preserves math expressions and section headers
  - Unwraps formatting commands (`\textbf`, `\emph`, etc.)

- `decode_file(filename, file_bytes)`:
  - Routes to appropriate extractor based on file extension
  - Supports: `.pdf`, `.tex`, `.txt`, `.docx` (via binary PDF fallback)

**Challenges Addressed**:
- **Missing spaces in PDFs**: "sectionPresents" → "section Presents"
- **LaTeX complexity**: Nested commands, custom macros, multiple environments
- **Encoding issues**: UTF-8 with error replacement

#### 2. Section Detection

**File**: `section_eval/section_detection.py`

**Purpose**: Automatically identify paper section boundaries.

**Architecture**:
```
Heuristic Candidate Scoring
          │
          ├──> High-confidence candidates (score ≥ 3)
          │       │
          │       ▼
          │   LLM Confirmation (filters false positives)
          │       │
          │       ▼
          │   Validated Headers
          │
          └──> Low confidence
                  │
                  ▼
              LLM-only Detection (fallback)
```

**Heuristic Scoring** (`_heuristic_candidate_headers`):
```python
Score criteria (cumulative):
  +3: ALL UPPERCASE + 2+ words
  +1: Title Case + 2+ words
  +2: Numbered prefix (1., 2.1, etc.)
  +2: Roman numeral prefix (I., IV., etc.)
  +1: 2-5 word count
  +1: 2-3 word count (extra bonus)
  +1: Preceded by blank line
  +1: Next line is much longer

Filters (reject if any):
  - Math symbols: +, =, %, ×, ≤, ∑, ∫, etc.
  - Figure/table labels: "Figure 3", "Table A1"
  - Citation fragments: "et al., 2024)"
  - Table metadata: "Parameter", "Value", "Accuracy"
  - Chart artifacts: "Balanced ac-", "curacy curacy"
  - Mathematical variables: Unicode math, subscripts
  - Single-character sequences: "t t t", "s s s"
```

**LLM Confirmation** (`detect_sections`):
- Receives candidate list + paper context
- Classifies each as genuine header or false positive
- Maps to canonical section types: `abstract`, `introduction`, `literature_review`, etc.

**Section Type Mapping** (`criteria/base.py:_canonical_section_type`):
```python
Normalization pipeline:
  1. Strip numbering prefix: "3.1 Methodology" → "methodology"
  2. Lowercase and trim
  3. Exact match in alias dictionary
  4. Substring match (longest alias wins)
  5. Keyword scan in priority order
  6. Fallback: "other"

Example aliases:
  "literature review" → "literature_review"
  "related work" → "literature_review"
  "empirical strategy" → "methodology"
  "identification strategy" → "identification_strategy"
```

**References Detection** (`find_references_start`):
- Regex match: `^\s*(?:references|bibliography|works cited|appendix)`
- Stops section extraction at references boundary

#### 3. Section Extraction

**File**: `section_eval/section_detection.py:extract_sections_from_text()`

**Process**:
1. Sort detected headers by line index
2. Slice text between consecutive headers
3. Stop at references section if detected
4. Map to desired section names
5. Fallback: return full text if structure unclear

#### 4. Hierarchy Grouping

**File**: `section_eval/hierarchy.py`

**Purpose**: Group subsections under parent sections for UI display.

**Example**:
```
Detected:
  4. Results
  4.1 Main Estimates
  4.2 Robustness Checks
  5. Discussion

Grouped:
  Results
    ├─ Main Estimates
    └─ Robustness Checks
  Discussion
```

**Algorithm**:
- Numbering-based (4.1, 4.2 under 4)
- Indentation-based (whitespace prefix)
- Case-based (lowercase after uppercase)

#### 5. Criteria Registry

**File**: `section_eval/criteria/base.py`

**Purpose**: Define paper-type and section-specific evaluation criteria.

**Data Structure**:
```python
_ALL_CRITERIA = {
    "empirical": {
        "data": [
            {"name": "source", "weight": 0.15, "description": "..."},
            {"name": "appropriateness", "weight": 0.25, "description": "..."},
            {"name": "limitations", "weight": 0.30, "description": "..."},
            ...
        ],
        "methodology": [...],
        "results": [...],
        ...
    },
    "theoretical": {
        "model_setup": [...],
        "proofs": [...],
        ...
    },
    "policy": {
        "policy_context": [...],
        "recommendations": [...],
        ...
    }
}
```

**Key Criteria Examples**:

**Empirical Papers**:
- **Data**: source, appropriateness, limitations (30% weight!), sample, variables
- **Methodology**: specification, identification (30%), assumptions, robustness_depth, replicability
- **Results**: alignment, interpretation_depth (30%), honesty, calibration, presentation

**Theoretical Papers**:
- **Model Setup**: assumptions, notation, motivation (25%), sophistication (25%), relation_to_lit
- **Proofs**: correctness, rigor (25%), economic_depth (25%), sophistication, clarity

**Policy Papers**:
- **Policy Context**: institutional (30%), historical, stakeholders, current_debate
- **Recommendations**: evidence_basis (30%), feasibility, tradeoffs, specificity

**Universal Sections** (all paper types):
- **Abstract**: completeness, clarity, accuracy, conciseness (equal 0.25 weights)
- **Introduction**: territory, niche (20%), contribution (20%), thesis (20%), roadmap, scope
- **Literature Review**: coverage, organization, synthesis (25%), gap, recency
- **Conclusion**: consistency (25%), contribution, limitations, future_research, implications

**Weight Philosophy**:
- Weights within a section sum to 1.0
- Higher weights (30%+) on critical dimensions that distinguish quality
- Emphasis on depth over surface features (e.g., "interpretation_depth" > "presentation")

#### 6. Evaluation Orchestration

**File**: `section_eval/evaluator.py:SectionEvaluator`

**Process**:

```python
def evaluate_section(section_name, section_text, paper_type, ...):
    # 1. Check cache
    cache_key = hash_text(f"{paper_type}|{section_name}|{section_text[:50000]}|{figures_external}")
    if cache_key in cache:
        return cache[cache_key]

    # 2. Get paper-type + section-specific criteria
    criteria = get_criteria(paper_type, section_name)

    # 3. Adjust criteria if figures are external
    if figures_external:
        criteria = _adjust_criteria_for_external_figures(criteria)

    # 4. Build evaluation prompt
    prompt = build_evaluation_prompt(
        paper_type, paper_context, section_name, section_type,
        section_text, criteria, figures_external
    )

    # 5. Query LLM (max 16000 chars)
    raw_response = safe_query(llm, prompt, max_chars=16000)

    # 6. Parse JSON response
    parsed = parse_json_from_text(raw_response)

    # 7. Sanitize and validate fields
    criteria_evals = _sanitize_criteria_evals(parsed["criteria_evaluations"], criteria)
    improvements = _sanitize_improvements(parsed["improvements"])

    # 8. Validate quotes against source text
    criteria_evals = _mark_quotes(criteria_evals, section_text)

    # 9. Compute weighted score
    section_score = compute_section_score(criteria_evals, section_name, paper_type)

    # 10. Cache and return
    result = {
        "qualitative_assessment": parsed["qualitative_assessment"],
        "criteria_evaluations": criteria_evals,
        "improvements": improvements,
        "section_score": section_score,
        ...
    }
    cache[cache_key] = result
    return result
```

**Fallback Mechanism**:
- If JSON parsing fails, build minimal result with heuristic improvements
- Default all criteria scores to 3.0
- Extract improvement suggestions using phrase search

#### 7. Quote Validation

**File**: `section_eval/evaluator.py:_validate_quote()`

**Purpose**: Verify that LLM-extracted quotes actually appear in source text.

**Algorithm**:
```python
def _validate_quote(quote_text, section_text):
    # Normalize: collapse whitespace, lowercase
    norm_quote = re.sub(r'\s+', ' ', quote_text.strip().lower())
    norm_section = re.sub(r'\s+', ' ', section_text.lower())

    # Exact match
    if norm_quote in norm_section:
        return True

    # Partial match: first 40 chars of quote
    partial = norm_quote[:40]
    return len(partial) > 10 and partial in norm_section
```

**Result**: Each quote gets a `valid: true/false` flag displayed in UI.

#### 8. Prompt Templates

**File**: `section_eval/prompts/templates.py:build_evaluation_prompt()`

**Structure**:
```markdown
# ROLE
You are a senior editor at a top economics journal.

# TASK
Evaluate the [section_name] section of [paper_type] paper.

# PAPER CONTEXT
[3-5 sentence summary of full paper]

# SECTION TEXT
[section content]

# EVALUATION CRITERIA (paper-type aware)
[List of criteria with weights and descriptions]

# INSTRUCTIONS
1. Qualitative assessment (3-5 sentences)
2. Score each criterion (1-5 scale)
3. Extract 2 verbatim quotes per criterion
4. Provide 3-4 improvement suggestions

# OUTPUT FORMAT
{
  "qualitative_assessment": "...",
  "criteria_evaluations": [
    {
      "criterion": "clarity",
      "score": 4,
      "justification": "...",
      "quote_1": {"text": "...", "supports_assessment": true},
      "quote_2": {"text": "...", "supports_assessment": false}
    },
    ...
  ],
  "improvements": [
    {"priority": 1, "suggestion": "...", "rationale": "..."},
    ...
  ]
}
```

**Special Handling**:
- **Figures external**: Instructs evaluator to ignore presentation criterion
- **Paper context**: Provides full-paper summary to avoid local misinterpretation
- **Score semantics**: Explicitly defines 5 = excellent, 3 = adequate, 1 = poor

#### 9. Scoring System

**File**: `section_eval/scoring.py`

**Section Score** (`compute_section_score`):
```python
# Raw score: weighted average of criteria (1-5 scale)
raw_score = Σ(criterion_score × criterion_weight) / Σ(criterion_weight)

# Adjusted score: accounts for section importance
section_importance = SECTION_IMPORTANCE[paper_type][section_type]
adjusted_score = raw_score × section_importance

# Example: Methodology in empirical paper
# raw_score = 4.2, importance = 1.3
# adjusted_score = 4.2 × 1.3 = 5.46
```

**Section Importance Multipliers**:
```python
SECTION_IMPORTANCE = {
    "empirical": {
        "methodology": 1.3,      # Critical
        "data": 1.2,
        "results": 1.2,
        "robustness_checks": 1.1,
        "introduction": 1.0,     # Baseline
        "literature_review": 0.9,
        "conclusion": 0.8,
        "abstract": 0.7,         # Less critical
    },
    "theoretical": {
        "proofs": 1.4,           # Most critical
        "model_setup": 1.3,
        "extensions": 1.1,
        ...
    },
    "policy": {
        "recommendations": 1.3,
        "policy_context": 1.2,
        ...
    }
}
```

**Overall Paper Score** (`compute_overall_score`):
```python
# Weighted average of section raw scores
overall_score = Σ(section_raw_score × section_importance) / Σ(section_importance)

# Min score gate: lowest section score
min_score = min(section_raw_scores)

# Publication readiness (both conditions must be met)
if overall_score > 4.5 and min_score > 3.5:
    readiness = "Ready for submission"
elif overall_score > 4.0 and min_score > 3.0:
    readiness = "Minor revisions needed"
elif overall_score > 3.0 and min_score > 2.5:
    readiness = "Major revisions needed"
else:
    readiness = "Substantial work required"
```

**Rationale**:
- **Section importance**: Reflects reality that methodology errors are more fatal than abstract issues
- **Min score gate**: Prevents high overall score from masking one critically flawed section
- **Dual thresholds**: Ensures both overall quality and minimum section quality

---

## Shared Infrastructure

### LLM Utilities (`utils.py`)

#### 1. API Configuration

```python
# API Configuration now loaded from .env file via config.py
# See docs/API_CONFIGURATION.md for setup instructions
API_BASE = os.getenv("API_BASE")  # Loaded from .env
API_KEY = os.getenv("API_KEY")    # Loaded from .env

# Model selection
model_selection3 = "anthropic.claude-3-7-sonnet-20250219-v1:0"  # MAD system
model_selection = "anthropic.claude-sonnet-4-5-20250929-v1:0"    # Section eval
```

#### 2. Single Query (`single_query`)

**Purpose**: Stateless LLM query for one-off requests.

**Features**:
- Automatic retry on failure (3 attempts)
- 5-second delay between retries
- Temperature: 0 (deterministic)
- No conversation history

**Usage**: MAD system (each round is independent)

#### 3. Conversation Manager (`ConversationManager`)

**Purpose**: Stateful conversation with automatic context management.

**Features**:
- **Token-aware pruning**: Keeps recent messages within context window
- **Automatic summarization**: Condenses old messages when limit approached
- **Message tracking**: Role, text, tokens, timestamp, is_summary flag
- **System prompt persistence**: System message never pruned

**Architecture**:
```python
Conversation Structure:
  [system_prompt]  # Never pruned
  [old_message_1]  # Summarized into...
  [old_message_2]  # ...one summary message
  ...
  [summary]        # Condensed representation
  [recent_msg_1]   # Preserved verbatim
  [recent_msg_2]   # Preserved verbatim
  [user_message]   # Current query
```

**Pruning Algorithm**:
```python
def prune_if_needed():
    if total_tokens <= (total_limit - reply_reserved):
        return  # No pruning needed

    # 1. Preserve recent messages (default: last 1000 tokens)
    recent = []
    for msg in reversed(conversation):
        if tokens(recent) + msg.tokens <= recent_budget:
            recent.append(msg)

    # 2. Segment older messages
    older = [m for m in conversation if m not in recent]
    segments = chunk(older, size=segment_size_msgs)  # Default: 6 msgs/segment

    # 3. Summarize each segment
    for segment in segments:
        summary = summarize_fn(segment)
        replace_segment_with(summary)
```

**Token Counting**:
- Uses `tiktoken` (cl100k_base encoding) if available
- Fallback: 1.3 tokens per word approximation

**Usage**: Section Evaluator (multi-turn interactions in UI)

#### 4. Model Temperature Settings

| System | Model | Temperature | Rationale |
|--------|-------|-------------|-----------|
| MAD | Claude 3.7 Sonnet | 0.5 | Balanced creativity and consistency for debate |
| Section Eval | Claude Sonnet 4.5 | 0.3 | Conservative for scoring reliability |
| Single Query | Any | 0.0 | Deterministic for reproducibility |

---

## Data Flow Diagrams

### Multi-Agent Debate Flow

```
User Upload (PDF/LaTeX)
        │
        ▼
  Extract Text (pdfplumber)
        │
        ▼
┌───────────────────────────────────┐
│ Round 0: Persona Selection        │
│ Input: Paper text (8000 chars)    │
│ LLM: single_query()               │
│ Output: 3 personas + weights      │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Round 1: Independent Evaluation   │
│ Parallel: 3 × call_llm_async()   │
│ Each persona: PASS/REVISE/FAIL   │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Round 2A: Cross-Examination       │
│ Input: R1 reports from peers      │
│ Parallel: 3 × call_llm_async()   │
│ Output: Questions to peers        │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Round 2B: Answer Questions        │
│ Input: R2A transcript             │
│ Parallel: 3 × call_llm_async()   │
│ Output: Answers + concessions     │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Round 2C: Final Amendments        │
│ Input: Full debate transcript     │
│ Parallel: 3 × call_llm_async()   │
│ Output: Final verdicts            │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Round 3: Editor Decision          │
│ Input: R2C verdicts + weights     │
│ LLM: single_query()               │
│ Calculate: Weighted consensus     │
│ Output: ACCEPT/REJECT/RESUBMIT    │
└───────────────────────────────────┘
        │
        ▼
  Display Results + Download
```

### Section Evaluator Flow

```
User Upload (PDF/LaTeX/TXT)
        │
        ▼
┌────────────────────────────────┐
│ decode_file()                  │
│ • PDF → pdfplumber + regex     │
│ • LaTeX → strip_latex()        │
│ • TXT → raw                    │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ detect_sections()              │
│ 1. Heuristic scoring           │
│ 2. LLM confirmation            │
│ 3. Type mapping                │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ extract_sections_from_text()   │
│ Slice text between headers     │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ generate_context_summary()     │
│ 3-5 sentence paper overview    │
└────────────────────────────────┘
        │
        ▼
    For each section:
        │
        ▼
┌────────────────────────────────┐
│ get_criteria(paper_type,       │
│              section_name)      │
│ Lookup in criteria registry    │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ build_evaluation_prompt()      │
│ • Paper context                │
│ • Section text                 │
│ • Criteria list                │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ safe_query(llm, prompt)        │
│ ConversationManager.conv_query()│
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ parse_json_from_text()         │
│ Extract JSON from response     │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ _sanitize_criteria_evals()     │
│ Ensure all fields present      │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ _mark_quotes()                 │
│ Validate quotes in source text │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ compute_section_score()        │
│ Weighted avg × importance      │
└────────────────────────────────┘
        │
        ▼
    End for each section
        │
        ▼
┌────────────────────────────────┐
│ compute_overall_score()        │
│ Aggregate section scores       │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│ generate_overall_assessment()  │
│ Strengths/weaknesses/priorities│
└────────────────────────────────┘
        │
        ▼
  Display Results + Download PDF
```

---

## Technology Stack

### Core Technologies

| Component | Technology | Version/Config |
|-----------|-----------|----------------|
| **UI Framework** | Streamlit | Tab-based interface, file uploaders |
| **Language Models** | Claude (Anthropic) | 3.7 Sonnet (MAD), 4.5 Sonnet (SE) |
| **API Endpoint** | Configured via .env file | See API_CONFIGURATION.md |
| **PDF Extraction** | pdfplumber | Word-level extraction fallback |
| **Async Execution** | asyncio | Python 3.7+ native |
| **Token Counting** | tiktoken | cl100k_base encoding |
| **JSON Parsing** | json (stdlib) | Regex pre-processing for robustness |
| **PDF Generation** | fpdf | Downloadable reports |

### External Dependencies

```
streamlit
pdfplumber
tiktoken
requests
fpdf
```

### Python Version

Requires Python 3.7+ (for `asyncio` native support)

---

## Extension Points

### 1. Adding New Personas (MAD System)

**File**: `multi_agent_debate.py`

**Steps**:
1. Add system prompt to `SYSTEM_PROMPTS` dict:
```python
SYSTEM_PROMPTS["DataScientist"] = """
### ROLE
You are a Data Scientist specializing in ML/AI applications in economics.

### OBJECTIVE
1. Data Quality: Assess preprocessing, feature engineering, validation splits
2. Model Selection: Evaluate appropriateness of ML methods vs. traditional econometrics

### OUTPUT FORMAT
- **Data Pipeline Audit**: [...]
- **Model Justification**: [...]
- **Source Evidence**: [MANDATORY: verbatim quotes]
- **Verdict**: [PASS/REVISE/FAIL]
"""
```

2. Update persona pool in `SELECTION_PROMPT`:
```python
SELECTION_PROMPT = """
...
The available personas are:
1. "Theorist": ...
2. "Empiricist": ...
3. "Historian": ...
4. "Visionary": ...
5. "Policymaker": ...
6. "DataScientist": Focuses on ML/AI methods, data preprocessing, validation
...
"""
```

3. Add icon/color to UI (`referee.py:display_debate_results`):
```python
icon_map["DataScientist"] = "🤖"
# Add CSS class in CUSTOM_CSS
```

### 2. Adding New Paper Types (Section Eval)

**File**: `section_eval/criteria/base.py`

**Steps**:
1. Add to `PAPER_TYPES` list:
```python
PAPER_TYPES = ["empirical", "theoretical", "policy", "survey"]
```

2. Add to `PAPER_TYPE_LABELS` dict:
```python
PAPER_TYPE_LABELS["survey"] = "Systematic Review / Survey"
```

3. Add to `SECTION_DEFAULTS` dict:
```python
SECTION_DEFAULTS["survey"] = [
    "Abstract", "Introduction", "Search Methodology",
    "Inclusion Criteria", "Synthesis", "Discussion", "Conclusion"
]
```

4. Define paper-type-specific criteria:
```python
_SURVEY = {
    "search_methodology": [
        {"name": "databases", "weight": 0.25, "description": "..."},
        {"name": "search_terms", "weight": 0.30, "description": "..."},
        ...
    ],
    ...
}
```

5. Add to master registry:
```python
_ALL_CRITERIA["survey"] = {**_UNIVERSAL, **_SURVEY}
```

6. Define section importance weights (`section_eval/scoring.py`):
```python
SECTION_IMPORTANCE["survey"] = {
    "search_methodology": 1.3,
    "synthesis": 1.2,
    ...
}
```

### 3. Adding New Evaluation Criteria

**File**: `section_eval/criteria/base.py`

**Example**: Add "reproducibility" criterion to empirical methodology

```python
_EMPIRICAL["methodology"] = [
    {"name": "specification", "weight": 0.20, "description": "..."},
    {"name": "identification", "weight": 0.30, "description": "..."},
    {"name": "assumptions", "weight": 0.20, "description": "..."},
    {"name": "robustness_depth", "weight": 0.20, "description": "..."},

    # NEW CRITERION
    {"name": "reproducibility", "weight": 0.10, "description":
     "Code and data availability statement, sufficient detail for replication"},
]
```

**Note**: Ensure weights sum to 1.0 for each section.

### 4. Customizing Prompt Templates

**File**: `section_eval/prompts/templates.py`

**Structure**:
```python
def build_evaluation_prompt(paper_type, section_name, criteria, ...):
    return f"""
# ROLE
You are a senior editor at a {journal_tier[paper_type]} journal.

# TASK
Evaluate the {section_name} section...

# CRITERIA
{format_criteria(criteria)}

# INSTRUCTIONS
[Customize evaluation approach here]

# OUTPUT FORMAT
{output_schema}
"""
```

**Customization Points**:
- **Role description**: Adjust expertise level/journal tier
- **Evaluation philosophy**: Add domain-specific guidelines
- **Output format**: Change schema (requires updating parser)
- **Score semantics**: Redefine what 1-5 means

### 5. Adding New File Formats

**File**: `section_eval/text_extraction.py:decode_file()`

**Example**: Add DOCX support

```python
def extract_text_from_docx(file_bytes):
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n\n".join([para.text for para in doc.paragraphs])

def decode_file(filename, file_bytes, warn_fn=None):
    if filename.endswith('.tex'):
        return strip_latex(file_bytes.decode('utf-8', errors='replace'))
    elif filename.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='replace')
    elif filename.endswith('.docx'):  # NEW
        return extract_text_from_docx(file_bytes)
    else:
        return extract_text_from_pdf(file_bytes, warn_fn=warn_fn)
```

### 6. Customizing Weighted Consensus

**File**: `multi_agent_debate.py:Round_3_Editor` prompt

**Current**:
```python
PASS = 1.0, REVISE = 0.5, FAIL = 0.0
Score > 0.75 → ACCEPT
0.40 ≤ Score ≤ 0.75 → REJECT AND RESUBMIT
Score < 0.40 → REJECT
```

**Alternatives**:
- **Veto power**: "Any FAIL → automatic REJECT"
- **Supermajority**: "Require 2/3 PASS for ACCEPT"
- **Non-linear weights**: "Square persona weights to amplify expertise differences"

**Implementation**: Update prompt instructions in `DEBATE_PROMPTS["Round_3_Editor"]`

---

## Performance Considerations

### 1. Caching Strategy

**Section Evaluator**:
- **Cache Key**: `hash(paper_type | section_name | section_text[:50000] | figures_external)`
- **Storage**: `st.session_state` (in-memory, per-session)
- **Benefits**: Prevents redundant LLM calls during UI interaction
- **Limitations**: Cache cleared on session restart

**Multi-Agent Debate**:
- **No caching**: Each debate is unique due to dynamic persona selection
- **Cost**: 13-16 LLM calls per paper (Round 0 + 3 personas × 4 rounds + Round 3)

### 2. Parallelization

**MAD System**:
- All personas evaluate concurrently in each round
- Uses `asyncio.gather()` for parallel API calls
- **Speedup**: ~3x compared to sequential execution

**Section Evaluator**:
- Sections evaluated sequentially (state dependencies)
- Future optimization: Parallel section evaluation if state isolated

### 3. Token Management

**ConversationManager**:
- **Total limit**: 8000 tokens (context window reservation)
- **Recent budget**: 1000 tokens (preserved verbatim)
- **Reply reserved**: 256 tokens (space for LLM response)
- **Segment size**: 6 messages (summarization granularity)

**Pruning Trigger**:
```
total_tokens > (total_limit - reply_reserved)
8000 > (8000 - 256) = 7744 tokens
```

### 4. Rate Limiting

**Federal Reserve API**:
- No explicit rate limits documented
- Retry strategy: 3 attempts with 5-second delays
- Future: Add exponential backoff

### 5. Memory Usage

**Large Documents**:
- **Full text storage**: Kept in session state (can exceed 1MB)
- **Section extraction**: Creates copies of text slices
- **Mitigation**: Use generators for large document processing (future)

**Streamlit Session State**:
- Grows unbounded during session
- **Mitigation**: Clear cache when switching tabs (implemented)

### 6. LLM Response Times

**Empirical Measurements** (approximate):
- **Single query**: 2-5 seconds (depends on prompt length)
- **MAD full pipeline**: 3-5 minutes (13-16 sequential/parallel calls)
- **Section eval (5 sections)**: 1-2 minutes (5 sequential calls)

**Bottlenecks**:
1. LLM latency (API round-trip)
2. JSON parsing (regex pre-processing)
3. Quote validation (string search)

### 7. Scalability Limits

**Current Design**:
- **Sessions**: Independent, no cross-session state
- **Concurrency**: Limited by Streamlit single-threaded model
- **Throughput**: ~10 papers/hour (MAD) or ~30 papers/hour (Section Eval)

**For High-Volume Use**:
- Replace Streamlit with FastAPI backend
- Add Redis/Postgres for shared caching
- Implement request queue with Celery
- Add load balancing across multiple LLM API keys

---

## Troubleshooting & Debugging

### Common Issues

**1. JSON Parsing Failures**

*Symptom*: "Evaluation could not be parsed" fallback message

*Causes*:
- LLM returns markdown code blocks (```json...```)
- LLM includes explanatory text before/after JSON
- Malformed JSON (missing brackets, trailing commas)

*Solution* (`section_eval/utils.py:parse_json_from_text`):
```python
# Extract JSON from markdown
json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
if json_match:
    return json.loads(json_match.group(1))

# Find first { to last } in text
start = text.find('{')
end = text.rfind('}')
if start != -1 and end != -1:
    return json.loads(text[start:end+1])
```

**2. Missing Sections**

*Symptom*: Expected section not detected

*Diagnosis*:
- Check heuristic candidate scores (add logging to `_heuristic_candidate_headers`)
- Verify LLM classification in `detect_sections` response

*Solution*:
- Use "Search for missing section" button in UI
- Manually specify section boundaries (future feature)
- Adjust heuristic scoring thresholds

**3. Quote Validation Failures**

*Symptom*: All quotes marked invalid despite being present

*Causes*:
- Whitespace differences (tabs vs. spaces)
- OCR errors in PDF extraction
- Quote from figure/table (not in main text)

*Solution*:
- Lower validation threshold (currently 40 char partial match)
- Improve PDF extraction (word-level fallback)
- Allow user to override validation

**4. Slow Performance**

*Symptom*: Evaluation takes >10 minutes

*Diagnosis*:
- Check API latency (add timing logs)
- Verify parallelization working (`asyncio` errors?)
- Look for repeated cache misses

*Solution*:
- Verify cache hits (log cache_key lookups)
- Check for session state clearing between queries
- Reduce max_chars in prompts if hitting token limits

---

## Future Enhancements

### Short-Term (1-3 months)

1. **Export Formats**
   - LaTeX output for direct manuscript integration
   - Structured JSON for programmatic analysis
   - Annotated PDF with inline comments

2. **User Preferences**
   - Adjustable strictness (score thresholds)
   - Custom criteria weights
   - Persona selection override (MAD)

3. **Batch Processing**
   - Upload multiple papers
   - Generate comparative analysis
   - Export summary spreadsheet

### Mid-Term (3-6 months)

4. **Collaborative Features**
   - Share evaluation links
   - Multi-user feedback
   - Version tracking

5. **Learning from Feedback**
   - User ratings of evaluations
   - Fine-tune prompts based on disagreements
   - Build calibration dataset

6. **Advanced Section Detection**
   - LaTeX structure parsing (\section commands)
   - PDF bookmark extraction
   - Machine learning classifier

### Long-Term (6-12 months)

7. **Multimodal Evaluation**
   - Figure/table quality assessment
   - Equation correctness checking
   - Citation graph analysis

8. **Workflow Integration**
   - Overleaf plugin
   - LaTeX package for continuous evaluation
   - GitHub Actions for preprint checks

9. **Meta-Analysis**
   - Aggregate statistics across papers
   - Field-specific benchmarks
   - Evolution of research standards over time

---

## Conclusion

This architecture document provides a comprehensive overview of the Evaluation Agent system. The design emphasizes:

- **Modularity**: Clear separation between MAD and Section Eval subsystems
- **Extensibility**: Easy to add new personas, criteria, paper types
- **Robustness**: Multiple fallback mechanisms for parsing and detection failures
- **Performance**: Caching and parallelization where appropriate
- **User Experience**: Streamlit interface with rich visualizations

For implementation details, refer to inline documentation in source files. For usage instructions, see `README.md`.

**Maintained by**: Federal Reserve Research Team
**Last Review**: March 2026
