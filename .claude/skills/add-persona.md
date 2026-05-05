# Skill: Add New Persona to MAD System

This skill guides adding a new persona to the multi-agent debate system.

## When to Use
- Adding a persona to the 5-persona base system (`engine.py`)
- Adding a persona to the 10-persona Experiment 4 system (`engine_exp_4.py`)
- Creating a memo-specific analyst (`memo_prompts.py`)

## Prerequisites
- Decide: base system (5), exp_4 (10), or memo system
- Choose persona name (e.g., "Statistician", "ML_Expert")
- Define expertise area and evaluation focus

## Steps

### 1. Create Prompt File

**Location**: `app_system/prompts/multi_agent_debate/personas/{name}/v1.0.txt`

```bash
# Create directory
mkdir -p app_system/prompts/multi_agent_debate/personas/statistician

# Create prompt file
nano app_system/prompts/multi_agent_debate/personas/statistician/v1.0.txt
```

**Template**:
```
You are the **Statistician** in a multi-agent referee panel evaluating a research paper.

Your role is to assess the statistical methodology, inference validity, and data analysis techniques.

## EXPERTISE
- Statistical inference and hypothesis testing
- Experimental design and power analysis
- Model specification and diagnostics
- Multiple testing corrections
- Causal inference methods

## EVALUATION CRITERIA
1. **Statistical Rigor**: Are methods statistically sound?
2. **Inference Validity**: Are conclusions supported by the analysis?
3. **Power Analysis**: Is the sample size adequate?
4. **Robustness**: Are results stable across specifications?
5. **Transparency**: Are limitations clearly stated?

{_ERROR_SEVERITY_GUIDE}

## OUTPUT FORMAT
Provide your evaluation in the following structure:

### Statistical Assessment
[Overall assessment of statistical methods]

### Key Findings
1. **[Finding 1 Title]**
   - **Severity**: [CRITICAL/MAJOR/MODERATE/MINOR]
   - **Evidence**: [Quote from paper or describe location]
   - **Explanation**: [Why this matters]
   - **Recommendation**: [How to address]

### Statistical Verdict
**PASS** / **REVISE** / **FAIL**

**Justification**: [Brief explanation of verdict]
```

### 2. Update config.yaml

**File**: `app_system/prompts/multi_agent_debate/config.yaml`

```yaml
personas:
  # ... existing personas ...
  
  statistician:
    version: "v1.0"
    file: "personas/statistician/{version}.txt"
```

### 3. Update Engine System Prompts

**File**: `app_system/referee/engine.py` (or `engine_exp_4.py`)

**Add to SYSTEM_PROMPTS dict** (around line 50-100):

```python
# Load Statistician prompt
SYSTEM_PROMPTS["Statistician"] = load_persona_prompt("statistician") or f"""
You are the **Statistician** in a multi-agent referee panel...
[fallback prompt if file not found]

{_ERROR_SEVERITY_GUIDE}
"""
```

**CRITICAL**: Include `{_ERROR_SEVERITY_GUIDE}` placeholder!

### 4. Update Selection Prompt

**File**: `app_system/referee/engine.py` (around line 150-250)

**Update SELECTION_PROMPT**:

```python
SELECTION_PROMPT = f"""
You are the Editor coordinating a multi-agent referee panel for an academic paper.

You have access to the following **{NUM_AVAILABLE_PERSONAS} expert reviewers**:

1. **Theorist**: Evaluates theoretical rigor, formalization, and mathematical soundness
2. **Empiricist**: Assesses empirical strategy, identification, and causal inference
3. **Historian**: Examines literature review, contextualization, and historical accuracy
4. **Visionary**: Evaluates novelty, impact, and forward-looking contributions
5. **Policymaker**: Assesses policy relevance, practical implications, and feasibility
6. **Statistician**: NEW - Evaluates statistical methods, inference, and data analysis  # ADD THIS

Your task:
- Select exactly {NUM_PERSONAS_TO_SELECT} reviewers most relevant to this paper
- Assign weights (summing to 1.0) reflecting their relative importance
...
"""
```

### 5. Update UI Styling

**File**: `app_system/referee/workflow.py`

**A. Add CSS class** (in `_inject_custom_css()` method):

```python
.persona-statistician {
    border-left: 4px solid #9c27b0;  /* Purple */
    background: linear-gradient(135deg, #f3e5f5 0%, #ffffff 100%);
}
```

**B. Add icon mapping** (in persona card rendering):

```python
PERSONA_ICONS = {
    "Theorist": "🎓",
    "Empiricist": "🔬", 
    "Historian": "📚",
    "Visionary": "🔮",
    "Policymaker": "⚖️",
    "Statistician": "📊",  # ADD THIS
    # ...
}
```

**C. Update manual selection UI** (if present):

```python
personas_available = [
    "Theorist",
    "Empiricist",
    "Historian",
    "Visionary",
    "Policymaker",
    "Statistician",  # ADD THIS
]
```

### 6. Test the Integration

**A. Test prompt loading**:
```bash
cd app_system
python -c "
from referee.engine import SYSTEM_PROMPTS
print('Statistician' in SYSTEM_PROMPTS)
print(len(SYSTEM_PROMPTS['Statistician']))
"
```

**B. Test selection**:
```bash
cd app_system
python -m pytest tests/test_referee_quick.py -v
```

**C. Test in UI**:
```bash
cd app_system
streamlit run app.py
# Upload paper, manually select "Statistician", verify output
```

### 7. Document the Change

**Update** `app_system/docs/changelog.md`:

```markdown
## [2026-04-27] - Added Statistician Persona

### Added
- New Statistician persona focusing on statistical methodology
- Evaluates inference validity, power analysis, and robustness
- Added to 5-persona base system
- CSS styling (purple gradient)
- Icon: 📊

### Files Modified
- `referee/engine.py`: Added SYSTEM_PROMPTS entry
- `referee/workflow.py`: Added UI styling and icon
- `prompts/multi_agent_debate/config.yaml`: Added configuration
- `prompts/multi_agent_debate/personas/statistician/v1.0.txt`: New prompt
```

## Verification Checklist

Before committing:

- ✅ Prompt file created with correct structure
- ✅ config.yaml updated
- ✅ SYSTEM_PROMPTS dict includes new persona
- ✅ SELECTION_PROMPT mentions new persona
- ✅ UI styling added (CSS + icon)
- ✅ Manual selection UI updated
- ✅ Tests pass
- ✅ Tested in UI with real paper
- ✅ Changelog updated

## Common Issues

**Persona not appearing in selection**:
- Check SELECTION_PROMPT includes the persona
- Verify NUM_AVAILABLE_PERSONAS is correct

**Prompt not loading**:
- Check file path matches config.yaml pattern
- Verify fallback prompt in SYSTEM_PROMPTS
- Check file permissions

**UI styling not applying**:
- Check CSS class name matches: `.persona-{lowercase_name}`
- Clear browser cache
- Restart Streamlit

**Errors in output**:
- Verify _ERROR_SEVERITY_GUIDE placeholder exists
- Check output format matches expected structure
- Test prompt independently before integration

## Advanced: Adding to Multiple Systems

**To add to both base and exp_4**:
1. Add to `engine.py` (5-persona system)
2. Add to `engine_exp_4.py` (10-persona system)
3. Update both SELECTION_PROMPTs
4. Update NUM_AVAILABLE_PERSONAS in both

**To create memo-specific analyst**:
1. Create in `referee/memo_prompts.py` instead
2. Use memo-specific terminology (policy evaluation, stakeholder analysis)
3. Update `app-memo.py` UI accordingly

## Related Skills
- `/version-prompt` - Update persona prompt to new version
- `/test-changes` - Run comprehensive test suite
