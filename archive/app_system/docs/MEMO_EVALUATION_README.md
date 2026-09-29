# Memo Evaluation System

## Overview

The Memo Evaluation System is a specialized version of the Multi-Agent Debate (MAD) framework designed specifically for evaluating policy memos rather than academic research papers. It uses domain-specific analyst personas that focus on policy-relevant criteria.

## Quick Start

### Running the Memo Evaluation App

```bash
cd app_system
source ../venv/bin/activate
streamlit run app-memo.py
```

The app will launch in your browser at `http://localhost:8501`.

## Architecture

### Files

```
app_system/
├── app-memo.py                    # Main Streamlit app for memo evaluation
├── referee/
│   ├── memo_prompts.py           # Memo-specific analyst prompts
│   ├── memo_engine.py            # Memo evaluation engine (uses same debate logic)
│   ├── engine.py                 # Original debate engine (shared infrastructure)
│   └── workflow.py               # Original workflow (shared UI components)
```

### Key Differences from Paper Evaluation

| Aspect | Paper Evaluation | Memo Evaluation |
|--------|------------------|-----------------|
| **Personas** | Theorist, Empiricist, Historian, Visionary, Policymaker | Policy Analyst, Data Analyst, Stakeholder Analyst, Implementation Analyst, Financial Stability Analyst |
| **Severity Classification** | ERROR SEVERITY (for academic errors) | ISSUE SEVERITY (for policy memo issues) |
| **Document Types** | Empirical, Theoretical, Policy papers | Policy Recommendation, Analytical Briefing, Decision Memo |
| **Focus Areas** | Academic rigor, novelty, methodology | Policy soundness, feasibility, stakeholder impact, fiscal implications |

## Analyst Personas

### 1. 🏛️ Policy Analyst
**Focus:** Policy logic, recommendation soundness, and problem diagnosis
- Evaluates whether recommendations are well-justified and aligned with objectives
- Assesses problem definition and policy framework
- Checks logical consistency of policy recommendations

### 2. 📊 Data Analyst
**Focus:** Evidence quality, data sources, and support for claims
- Evaluates credibility and relevance of evidence
- Assesses whether statistics are properly contextualized
- Checks if claims are adequately backed by data

### 3. 👥 Stakeholder Analyst
**Focus:** Stakeholder identification, impact analysis, and equity considerations
- Identifies all affected parties
- Assesses whether interests and concerns are adequately analyzed
- Evaluates mitigation strategies for adverse effects

### 4. ⚙️ Implementation Analyst
**Focus:** Feasibility, action items, timelines, and practical execution
- Evaluates clarity and actionability of recommendations
- Assesses resource requirements and potential obstacles
- Checks coordination needs across agencies/stakeholders

### 5. 💰 Financial Stability Analyst
**Focus:** Costs, fiscal impact, economic risks, and financial stability
- Evaluates cost-benefit analysis
- Assesses fiscal implications and funding sources
- Identifies economic and systemic risks

## Issue Severity Classification

All findings are labeled with one of three severity levels:

- **[FATAL]**: Invalidates the core policy recommendation or renders memo unusable
  - Examples: Fabricated evidence, fundamental misunderstanding, illegal/infeasible recommendations
  - Verdict impact: Any FATAL flaw → FAIL

- **[MAJOR]**: Requires substantial revision but doesn't auto-justify failure
  - Examples: Insufficient evidence, missing impact analysis, unclear action items
  - Verdict impact: Two or more MAJOR flaws → REVISE

- **[MINOR]**: Improves the memo but doesn't block its use
  - Examples: Formatting issues, minor citation gaps, supplementary data
  - Verdict impact: Only MINOR flaws → PASS

## Memo Types

The system recognizes three memo types for analyst selection guidance:

### Policy Recommendation Memo
Proposes specific policy actions and recommendations.
- **Typical analysts**: Policy Analyst (0.45) + Implementation Analyst (0.30) + Data Analyst (0.25)

### Analytical Briefing Memo
Provides analysis without specific recommendations.
- **Typical analysts**: Data Analyst (0.45) + Policy Analyst (0.30) + Financial Stability Analyst (0.25)

### Decision Memo
Presents options and recommends a course of action.
- **Typical analysts**: Policy Analyst (0.40) + Implementation Analyst (0.35) + Financial Stability Analyst (0.25)

## Workflow

The evaluation follows the same 5-round Multi-Agent Debate structure:

1. **Round 0 - Analyst Selection**:
   - Auto-select 3 analysts based on memo content and type, OR
   - Manually select 2-5 analysts
   - Assign importance weights (automatically or manually)

2. **Round 1 - Independent Evaluation**:
   - Each analyst evaluates independently
   - Issues labeled with severity ([FATAL], [MAJOR], [MINOR])
   - Initial verdict: PASS/REVISE/FAIL

3. **Round 2A - Cross-Examination**:
   - Analysts read peer evaluations
   - Engage in cross-domain synthesis
   - Ask clarification questions

4. **Round 2B - Direct Examination**:
   - Analysts respond to questions
   - Provide evidence and take positions (CONCEDE or DEFEND)

5. **Round 2C - Final Amendments**:
   - Review full debate transcript
   - Submit final amended verdict with justification

6. **Round 3 - Senior Reviewer Decision**:
   - Compute weighted consensus (PASS=1.0, REVISE=0.5, FAIL=0.0)
   - Apply decision thresholds:
     - Score > 0.75: ACCEPT
     - Score 0.40-0.75: REJECT AND RESUBMIT
     - Score < 0.40: REJECT
   - Write official evaluation report

## Usage Guide

### Basic Usage

1. Launch the app: `streamlit run app-memo.py`
2. Upload or paste your memo
3. (Optional) Select memo type for analyst selection guidance
4. (Optional) Manually select analysts or let LLM auto-select
5. (Optional) Provide custom evaluation context
6. Click "Run Multi-Agent Evaluation"
7. Review results and download reports

### Advanced Options

#### Manual Analyst Selection
- Check "Manually select analysts"
- Select 2-5 analysts from the list
- Optionally specify exact weights (must sum to 1.0)
- If weights not specified, LLM assigns them automatically

#### Custom Evaluation Context
Examples of custom context:
- "Focus on political feasibility and stakeholder buy-in"
- "Evaluate urgency and timeline realism"
- "Check readiness for senior leadership review"
- "Assess fiscal impact and budget implications"

### Output

The system provides:
- **Summarized Reports**: LLM-compressed summaries of each round
- **Full Reports**: Complete analyst evaluations (expandable)
- **Consensus Calculation**: Weighted scores and individual verdicts
- **Final Decision**: ACCEPT / REJECT AND RESUBMIT / REJECT
- **Senior Reviewer Report**: Synthesized evaluation for memo author
- **Downloads**: JSON (full results) and Markdown (summary report)

## Technical Implementation

### Shared Infrastructure

The memo evaluation system reuses core infrastructure from the paper evaluation system:

- **Debate orchestration**: `referee/engine.py` (async rounds, consensus calculation)
- **Round prompts**: `DEBATE_PROMPTS` (domain-agnostic debate instructions)
- **Summarization**: `referee/_utils/summarizer.py` (LLM-powered compression)
- **UI helpers**: `referee/_archived/full_output_ui.py` (formatting functions)

### Memo-Specific Components

Only the following components are memo-specific:

- **Analyst prompts**: `referee/memo_prompts.py`
  - `MEMO_SYSTEM_PROMPTS`: 5 analyst personas
  - `MEMO_SELECTION_PROMPT`: Analyst selection instructions
  - `MEMO_TYPE_CONTEXTS`: Guidance for each memo type
  - `ISSUE_SEVERITY_GUIDE`: Severity classification rules

- **Memo engine**: `referee/memo_engine.py`
  - Wraps standard engine with memo prompts
  - Uses memo-specific terminology (memo_text vs paper_text)
  - Otherwise identical to paper engine

- **Memo app**: `app-memo.py`
  - Streamlit UI for memo evaluation
  - Memo-specific text and descriptions
  - Same UI/UX patterns as main app

## Extending the System

### Adding New Analysts

To add a new analyst persona:

1. Add prompt to `referee/memo_prompts.py`:
```python
NEW_ANALYST_PROMPT = """### ROLE
You are a [Role]. You focus on [focus areas].

### OBJECTIVE
[Evaluation objectives]

""" + ISSUE_SEVERITY_GUIDE + """

### OUTPUT FORMAT (MANDATORY STRUCTURE)
- **[Section Name]**: [Brief overview]
- **Severity-Labeled Findings**: For EACH finding, use this exact structure:
    [SEVERITY_LABEL] Finding description in one sentence.
    **Source Evidence**: "Verbatim quote from memo"
- **Verdict**: [PASS/REVISE/FAIL — must be consistent with severity labels above]
"""
```

2. Add to `MEMO_SYSTEM_PROMPTS` dictionary:
```python
MEMO_SYSTEM_PROMPTS = {
    ...
    "New Analyst": NEW_ANALYST_PROMPT
}
```

3. Update analyst descriptions in `app-memo.py`

### Adding New Memo Types

To add a new memo type for selection guidance:

1. Add context to `MEMO_TYPE_CONTEXTS` in `referee/memo_prompts.py`:
```python
MEMO_TYPE_CONTEXTS = {
    ...
    "new_memo_type": """MEMO TYPE CONTEXT: NEW MEMO TYPE

[Description]

ANALYST SELECTION GUIDANCE:
[Guidance for each analyst]

TYPICAL ANALYST COMBINATIONS:
[Example combinations with weights]
"""
}
```

2. Add option to memo type selector in `app-memo.py`

## Cost Estimates

Typical costs per evaluation (Claude 3.7 Sonnet pricing):
- **Small memo** (1-2 pages): $0.10 - $0.25
- **Medium memo** (3-5 pages): $0.25 - $0.60
- **Large memo** (6-10 pages): $0.60 - $1.50

Costs scale with:
- Memo length (tokens)
- Number of analysts (3-5)
- Number of debate rounds (fixed at 5)
- Summarization passes (5-6 additional LLM calls)

## Comparison with Paper Evaluation

### Similarities
- Same multi-agent debate architecture (5 rounds)
- Same consensus calculation method
- Same severity classification structure
- Same UI/UX patterns and workflows

### Differences
- Domain-specific personas (policy vs academic)
- Document types (memos vs papers)
- Evaluation criteria (policy-relevant vs academic rigor)
- Terminology (memo_text, analysts vs paper_text, personas)

## Future Enhancements

Potential improvements:
- [ ] Integration into main `app.py` as a third tab
- [ ] Versioned prompt files (like paper system)
- [ ] Memo-specific criteria registry (like section evaluator)
- [ ] Domain-specific memo types (financial, regulatory, legislative)
- [ ] Batch evaluation for multiple memos
- [ ] Comparison mode (evaluate multiple versions)
- [ ] Executive summary extraction
- [ ] Action item tracking

## Troubleshooting

### Common Issues

**Issue**: Analysts not appearing in selection
- **Solution**: Check that `MEMO_SYSTEM_PROMPTS` includes all analysts

**Issue**: Consensus calculation shows UNKNOWN verdicts
- **Solution**: Check that analysts output PASS/REVISE/FAIL in their final reports

**Issue**: Summarization fails
- **Solution**: Check API credentials and rate limits; summaries are optional

**Issue**: Cost higher than expected
- **Solution**: Reduce number of analysts or use shorter memos for testing

## Contact & Support

For questions or issues:
1. Check CLAUDE.md for general system documentation
2. Review main `app_system/README.md` for setup instructions
3. Check `app_system/docs/changelog.md` for recent changes
