# Memo Evaluation System - Quick Start Guide

## What Was Created

A complete memo evaluation system parallel to the paper evaluation system, featuring:

### 1. **Memo-Specific Analyst Personas** (`referee/memo_prompts.py`)
- 🏛️ **Policy Analyst**: Evaluates policy logic and recommendations
- 📊 **Data Analyst**: Evaluates evidence quality and support
- 👥 **Stakeholder Analyst**: Evaluates stakeholder impacts
- ⚙️ **Implementation Analyst**: Evaluates feasibility and execution
- 💰 **Financial Stability Analyst**: Evaluates fiscal and economic risks

### 2. **Memo Evaluation Engine** (`referee/memo_engine.py`)
- Wraps standard debate engine with memo-specific prompts
- Same 5-round Multi-Agent Debate architecture
- Memo-specific terminology and guidance

### 3. **Streamlit App** (`app-memo.py`)
- Standalone memo evaluation interface
- Upload/paste memos for evaluation
- Manual or automatic analyst selection
- Custom evaluation context support
- Download results as JSON or Markdown

### 4. **Documentation**
- `MEMO_EVALUATION_README.md`: Complete system documentation
- `examples/sample_memo.txt`: Example memo for testing

## How to Run

### Step 1: Launch the App

```bash
cd app_system
source ../venv/bin/activate
streamlit run app-memo.py
```

### Step 2: Upload or Paste a Memo

Try the sample memo:
```bash
# The sample is located at:
app_system/examples/sample_memo.txt
```

Or paste your own memo text directly in the app.

### Step 3: Configure Evaluation (Optional)

- **Memo Type**: Select from Policy Recommendation, Analytical Briefing, or Decision Memo
- **Analyst Selection**: Let LLM auto-select or manually choose 2-5 analysts
- **Custom Context**: Add specific evaluation priorities (e.g., "Focus on implementation feasibility")

### Step 4: Run Evaluation

Click "🚀 Run Multi-Agent Evaluation" and wait 2-5 minutes for results.

### Step 5: Review Results

- View summarized and full reports for each round
- See weighted consensus and individual verdicts
- Read final senior reviewer decision
- Download JSON (full) or Markdown (summary) reports

## Key Features

### Automatic Analyst Selection
The system intelligently selects 3 analysts based on:
- Memo content and type
- User-specified evaluation priorities
- Recommended analyst combinations for each memo type

### Severity-Based Evaluation
All findings are labeled:
- **[FATAL]**: Invalidates core recommendations → FAIL
- **[MAJOR]**: Requires revision → REVISE (if 2+)
- **[MINOR]**: Improvements only → PASS

### Weighted Consensus
Final decision uses weighted scoring:
- PASS = 1.0, REVISE = 0.5, FAIL = 0.0
- Score > 0.75: **ACCEPT**
- Score 0.40-0.75: **REJECT AND RESUBMIT**
- Score < 0.40: **REJECT**

## Architecture Overview

```
app-memo.py (UI)
    ↓
referee/memo_engine.py (Memo-specific orchestration)
    ↓
referee/engine.py (Shared debate infrastructure)
    ↓
referee/memo_prompts.py (Analyst personas)
    ↓
utils.py (LLM calls via single_query)
```

## Example Usage

### Example 1: Policy Recommendation Memo

**Memo Type**: Policy Recommendation
**Auto-Selected Analysts**:
- Policy Analyst (0.45)
- Implementation Analyst (0.30)
- Financial Stability Analyst (0.25)

**Custom Context**: "Focus on political feasibility and timeline realism"

### Example 2: Analytical Briefing

**Memo Type**: Analytical Briefing
**Manual Selection**:
- Data Analyst
- Policy Analyst
- Financial Stability Analyst

**Custom Context**: "Evaluate evidence quality and data sources"

### Example 3: Decision Memo

**Memo Type**: Decision Memo
**Auto-Selected Analysts** (with manual weights):
- Policy Analyst (0.40)
- Implementation Analyst (0.35)
- Stakeholder Analyst (0.25)

**Custom Context**: "Check readiness for Board review"

## Integration with Main App

### Current Status
- **Standalone app**: `app-memo.py` runs independently
- **Shared infrastructure**: Uses same debate engine and UI helpers
- **Separate prompts**: Memo-specific analysts in `referee/memo_prompts.py`

### Future Integration Options

#### Option 1: Add as Third Tab to `app.py`
```python
WORKFLOWS = {
    "Referee Report": RefereeWorkflow,
    "Section Evaluator": SectionEvaluatorApp,
    "Memo Evaluator": MemoRefereeWorkflow,  # New
}
```

#### Option 2: Unified Workflow with Mode Switch
```python
class UnifiedRefereeWorkflow:
    def __init__(self, mode="paper"):  # or "memo"
        self.mode = mode
        if mode == "paper":
            self.engine = paper_engine
            self.prompts = paper_prompts
        else:
            self.engine = memo_engine
            self.prompts = memo_prompts
```

#### Option 3: Document Type Detector
```python
# Auto-detect if uploaded document is a paper or memo
doc_type = detect_document_type(text)
if doc_type == "memo":
    use_memo_personas()
else:
    use_paper_personas()
```

## Testing Checklist

- [ ] App launches without errors
- [ ] Can upload PDF/TXT files
- [ ] Can paste text directly
- [ ] Memo type selection works
- [ ] Auto analyst selection works
- [ ] Manual analyst selection works (2-5 analysts)
- [ ] Manual weights work (sum to 1.0)
- [ ] Custom context is applied
- [ ] Evaluation completes successfully
- [ ] All 5 rounds display correctly
- [ ] Summaries generate properly
- [ ] Consensus calculation is correct
- [ ] Final decision displays
- [ ] JSON download works
- [ ] Markdown download works
- [ ] Token/cost tracking works

## Troubleshooting

### "No module named 'referee.memo_prompts'"
**Solution**: Make sure you're running from `app_system/` directory

### "Failed to parse selection"
**Solution**: Check that memo text is not empty or too short

### "LLM weights invalid"
**Solution**: System will fall back to equal weights automatically

### "Error generating summaries"
**Solution**: Summaries are optional; full reports still available

### High costs
**Solution**: Use shorter memos for testing; typical cost is $0.10-$1.50 per evaluation

## Next Steps

### Immediate
1. Test with sample memo: `examples/sample_memo.txt`
2. Try different memo types and analyst combinations
3. Experiment with custom evaluation contexts

### Short-term
1. Collect user feedback on analyst personas
2. Refine prompts based on evaluation quality
3. Add more example memos

### Long-term
1. Integrate into main `app.py` (choose integration option)
2. Create versioned prompt files (like paper system)
3. Add memo-specific criteria registry
4. Support batch evaluation
5. Add memo comparison mode

## Cost Estimates

Based on Claude 3.7 Sonnet pricing ($3/M input, $15/M output):

| Memo Length | Est. Tokens | Est. Cost |
|-------------|-------------|-----------|
| 1-2 pages | ~3,000 | $0.10-$0.25 |
| 3-5 pages | ~7,500 | $0.25-$0.60 |
| 6-10 pages | ~15,000 | $0.60-$1.50 |

Costs include:
- 5 debate rounds
- 3 analysts (parallel calls)
- Summarization passes (5-6 extra calls)

## Resources

- **Full Documentation**: `MEMO_EVALUATION_README.md`
- **Main App Setup**: `README.md`
- **Project Guidelines**: `../CLAUDE.md`
- **Sample Memo**: `examples/sample_memo.txt`

## Questions?

Check the main documentation or review the code:
- `referee/memo_prompts.py` - Analyst personas and prompts
- `referee/memo_engine.py` - Evaluation orchestration
- `app-memo.py` - Streamlit UI

---

**Ready to evaluate your first memo? Run `streamlit run app-memo.py` and get started!** 🚀
