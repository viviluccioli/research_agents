# Skill: Add New Paper Type to Section Evaluator

This skill guides adding a new paper type to the section evaluation system.

## When to Use
- Adding support for a new research domain (e.g., "finance", "macroeconomics")
- Creating specialized evaluation criteria for a subdiscipline
- Customizing section importance weights

## Prerequisites
- Identify common sections for this paper type
- Define evaluation criteria specific to the domain
- Determine which criteria are critical (fatal-flaw)
- Decide section importance weights

## Steps

### 1. Define Paper Type Metadata

**File**: `app_system/section_eval/criteria/base.py`

**A. Add to PAPER_TYPES list** (around line 10):

```python
PAPER_TYPES = [
    "empirical",
    "theoretical",
    "policy",
    "finance",      # ADD THIS
    "macro",
    "systematic_review"
]
```

**B. Add to PAPER_TYPE_LABELS dict** (around line 15):

```python
PAPER_TYPE_LABELS = {
    "empirical": "Empirical Paper",
    "theoretical": "Theoretical Paper",
    "policy": "Policy Analysis",
    "finance": "Finance Paper",  # ADD THIS
    "macro": "Macroeconomics Paper",
    "systematic_review": "Systematic Review"
}
```

**C. Add section defaults** (around line 25):

```python
SECTION_DEFAULTS = {
    "empirical": ["abstract", "introduction", "literature_review", ...],
    "theoretical": ["abstract", "introduction", "model", ...],
    "finance": [  # ADD THIS
        "abstract",
        "introduction",
        "literature_review",
        "data",
        "methodology",
        "results",
        "discussion",
        "conclusion"
    ],
}
```

### 2. Define Evaluation Criteria

**File**: `app_system/section_eval/criteria/base.py` (around line 100+)

**Create criteria dictionary**:

```python
# Finance-specific criteria
_FINANCE = {
    "data": [
        {
            "name": "data_quality",
            "description": "Quality and appropriateness of financial data",
            "weight": 0.3,
            "critical": True,  # Fatal-flaw if ≤ 1.5
            "questions": [
                "Is the data source reliable and appropriate?",
                "Are data limitations acknowledged?",
                "Is the sample period justified?"
            ]
        },
        {
            "name": "asset_pricing_theory",
            "description": "Theoretical foundation for asset pricing",
            "weight": 0.25,
            "critical": False,
            "questions": [
                "Is the pricing model theoretically grounded?",
                "Are factor choices justified?",
                "Are risk premia properly specified?"
            ]
        },
        # Add more criteria...
    ],
    "methodology": [
        {
            "name": "estimation_technique",
            "description": "Appropriateness of econometric methods",
            "weight": 0.35,
            "critical": True,  # Fatal-flaw
            "questions": [
                "Are estimation methods suitable for financial data?",
                "Is endogeneity addressed?",
                "Are standard errors robust to heteroskedasticity/autocorrelation?"
            ]
        },
        # Add more criteria...
    ],
    # Add criteria for other sections...
}
```

**Guidelines for criteria design**:
- **critical=True**: For fundamental methodological issues
- **weight**: Sum to 1.0 within each section
- **questions**: 3-5 specific evaluation questions

### 3. Merge into ALL_CRITERIA

**File**: `app_system/section_eval/criteria/base.py` (around line 300+)

```python
_ALL_CRITERIA = {
    "universal": _UNIVERSAL,
    "empirical": {**_UNIVERSAL, **_EMPIRICAL},
    "theoretical": {**_UNIVERSAL, **_THEORETICAL},
    "policy": {**_UNIVERSAL, **_POLICY},
    "finance": {**_UNIVERSAL, **_FINANCE},  # ADD THIS
    "macro": {**_UNIVERSAL, **_MACRO},
    "systematic_review": {**_UNIVERSAL, **_SYSREV},
}
```

**Note**: Universal criteria are shared across all paper types.

### 4. Add Section Importance Weights

**File**: `app_system/section_eval/scoring.py` (around line 20)

```python
SECTION_IMPORTANCE = {
    "empirical": {
        "abstract": 0.05,
        "introduction": 0.10,
        "methodology": 0.30,  # Most important
        "results": 0.25,
        "discussion": 0.15,
        "conclusion": 0.05,
        "literature_review": 0.10,
    },
    "finance": {  # ADD THIS
        "abstract": 0.05,
        "introduction": 0.08,
        "literature_review": 0.10,
        "data": 0.20,           # Finance-specific emphasis
        "methodology": 0.25,     # Most critical
        "results": 0.20,
        "discussion": 0.10,
        "conclusion": 0.02,
    },
    # Other paper types...
}
```

**Guidelines**:
- Weights should sum to 1.0
- Emphasize methodologically critical sections (0.20-0.30)
- De-emphasize descriptive sections (0.05-0.10)

### 5. Create Paper Type Context Prompt

**Location**: `app_system/prompts/section_evaluator/paper_type_contexts/finance/v1.0.txt`

```bash
# Create directory
mkdir -p app_system/prompts/section_evaluator/paper_type_contexts/finance

# Create prompt file
nano app_system/prompts/section_evaluator/paper_type_contexts/finance/v1.0.txt
```

**Template**:
```
# Finance Paper Evaluation Guidance

## Domain-Specific Considerations

### Data Quality
- Financial data often has survivorship bias, look-ahead bias
- Market microstructure issues (bid-ask bounce, non-trading)
- Sample selection must be carefully justified

### Methodology
- Asset pricing models require careful factor specification
- Returns data exhibits time-varying volatility (GARCH effects)
- High-frequency data has unique challenges (market microstructure noise)
- Standard errors must account for cross-sectional and time-series dependence

### Common Issues to Flag
- **Data Snooping**: Overfitting to known patterns
- **Transaction Costs**: Ignoring realistic trading costs
- **Risk Adjustment**: Inadequate control for known factors
- **Statistical Significance**: Multiple testing without correction

### Critical Sections
- **Data**: Source, cleaning, filters, biases
- **Methodology**: Factor models, estimation, robustness
- **Results**: Economic significance, out-of-sample performance

## Evaluation Standards

### High Quality (4-5)
- Clean data from reputable sources
- Theoretically-grounded methodology
- Robust to specifications
- Addresses known biases

### Acceptable (3)
- Standard data sources
- Conventional methods applied correctly
- Some robustness checks

### Needs Revision (2)
- Data quality concerns
- Methodological gaps
- Limited robustness

### Inadequate (1)
- Serious data issues
- Flawed methodology
- Results not credible
```

### 6. Update config.yaml

**File**: `app_system/prompts/section_evaluator/config.yaml`

```yaml
paper_type_contexts:
  empirical:
    version: "v1.0"
    file: "paper_type_contexts/empirical/{version}.txt"
  
  finance:  # ADD THIS
    version: "v1.0"
    file: "paper_type_contexts/finance/{version}.txt"
```

### 7. Add Section Aliases (if needed)

**File**: `app_system/section_eval/criteria/base.py` (around line 350)

```python
_SECTION_ALIASES = {
    # Existing aliases...
    
    # Finance-specific
    "empirical strategy": "methodology",
    "portfolio construction": "methodology",
    "trading strategy": "methodology",
    "factor models": "methodology",
    "market data": "data",
}

_KEYWORD_MAP = {
    # Existing keywords...
    
    # Finance-specific fuzzy matching
    "asset pricing": "methodology",
    "risk factors": "methodology",
    "portfolio": "methodology",
    "returns": "results",
}
```

**Purpose**: Maps section headers in papers to canonical section types.

### 8. Test the Integration

**A. Test criteria loading**:
```bash
cd app_system
python -c "
from section_eval.criteria.base import PAPER_TYPES, _ALL_CRITERIA
print('finance' in PAPER_TYPES)
print('finance' in _ALL_CRITERIA)
print(_ALL_CRITERIA['finance'].keys())
"
```

**B. Test scoring**:
```bash
cd app_system
python -c "
from section_eval.scoring import SECTION_IMPORTANCE
print('finance' in SECTION_IMPORTANCE)
print(sum(SECTION_IMPORTANCE['finance'].values()))  # Should be 1.0
"
```

**C. Test in UI**:
```bash
cd app_system
streamlit run app.py

# Go to Section Evaluator tab
# Select "Finance Paper" from dropdown
# Upload sample finance paper
# Verify evaluation runs correctly
```

**D. Run tests**:
```bash
cd app_system
python -m pytest tests/test_section_evaluator_prompts.py
```

### 9. Create Test Paper Set

**Recommended**: 3-5 sample finance papers covering:
- Asset pricing study
- Portfolio analysis
- Market microstructure paper
- Risk management paper
- Corporate finance paper

Test each to ensure:
- Section detection works
- Criteria apply appropriately
- Scores are reasonable
- Fatal-flaw logic triggers correctly

### 10. Document the Change

**Update** `app_system/docs/changelog.md`:

```markdown
## [2026-04-27] - Added Finance Paper Type

### Added
- New "Finance Paper" type to section evaluator
- Finance-specific evaluation criteria emphasizing:
  - Data quality (survivorship bias, market microstructure)
  - Asset pricing methodology (factor models, risk adjustment)
  - Transaction costs and practical implementation
- Section importance weights favoring data (0.20) and methodology (0.25)
- Paper type context prompt with finance domain guidance

### Critical Criteria
- data_quality (fatal-flaw if ≤ 1.5)
- estimation_technique (fatal-flaw if ≤ 1.5)

### Files Modified
- `section_eval/criteria/base.py`: Added _FINANCE criteria, section aliases
- `section_eval/scoring.py`: Added finance section weights
- `prompts/section_evaluator/config.yaml`: Added finance configuration
- `prompts/section_evaluator/paper_type_contexts/finance/v1.0.txt`: New prompt

### Testing
Validated on 5 sample papers covering asset pricing, portfolio analysis, 
and market microstructure. Section detection accurate, criteria appropriate.
```

## Verification Checklist

Before committing:

- ✅ Paper type added to PAPER_TYPES and PAPER_TYPE_LABELS
- ✅ Section defaults defined in SECTION_DEFAULTS
- ✅ Criteria dictionary created (_FINANCE or similar)
- ✅ Criteria merged into _ALL_CRITERIA
- ✅ Section importance weights added (sum to 1.0)
- ✅ Paper type context prompt created
- ✅ config.yaml updated
- ✅ Section aliases added (if needed)
- ✅ Tests pass
- ✅ Tested on real papers (3-5 samples)
- ✅ Fatal-flaw logic verified
- ✅ Changelog updated

## Common Issues

**Criteria not loading**:
- Check _ALL_CRITERIA merge includes new type
- Verify key names match between _FINANCE and SECTION_DEFAULTS

**Section importance weights error**:
- Ensure weights sum to 1.0
- Check all sections in SECTION_DEFAULTS have weights

**Fatal-flaw not triggering**:
- Verify critical=True on appropriate criteria
- Check FATAL_FLAW_SCORE_THRESHOLD constant

**Section detection failing**:
- Add domain-specific aliases to _SECTION_ALIASES
- Update _KEYWORD_MAP for fuzzy matching

**Prompt not loading**:
- Check file path matches config.yaml pattern
- Verify {version} placeholder in config

## Advanced: Subsection Handling

For paper types with complex section hierarchies:

```python
SECTION_DEFAULTS = {
    "finance": [
        "abstract",
        "introduction",
        {
            "parent": "methodology",
            "subsections": ["data_description", "estimation", "robustness"]
        },
        "results",
        "conclusion"
    ]
}
```

See `section_eval/hierarchy.py` for grouping logic.

## Related Skills
- `/version-prompt` - Update paper type context prompt
- `/test-changes` - Run tests after adding paper type
