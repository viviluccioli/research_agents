# Section Evaluator Prompts - Version Control System

This directory contains all prompts for the section evaluator, organized for easy version control and iteration.

## 📁 Directory Structure

```
prompts/section_evaluator/
├── config.yaml                            # Main configuration file
├── prompt_loader.py                       # Prompt loading module
├── README.md                              # This file
├── paper_type_contexts/                   # Paper type-specific context (6 types)
│   ├── empirical_v1.0.txt
│   ├── theoretical_v1.0.txt
│   ├── policy_v1.0.txt
│   ├── finance_v1.0.txt
│   ├── macro_v1.0.txt
│   └── systematic_review_v1.0.txt
├── section_type_guidance/                 # Section-specific guidance (22 types)
│   ├── abstract_v1.0.txt
│   ├── introduction_v1.0.txt
│   ├── literature_review_v1.0.txt
│   ├── data_v1.0.txt
│   ├── methodology_v1.0.txt
│   ├── model_setup_v1.0.txt
│   ├── proofs_v1.0.txt
│   ├── extensions_v1.0.txt
│   ├── results_v1.0.txt
│   ├── discussion_v1.0.txt
│   ├── robustness_checks_v1.0.txt
│   ├── identification_strategy_v1.0.txt
│   ├── calibration_v1.0.txt
│   ├── simulations_v1.0.txt
│   ├── stylized_facts_v1.0.txt
│   ├── policy_context_v1.0.txt
│   ├── recommendations_v1.0.txt
│   ├── search_methodology_v1.0.txt
│   ├── inclusion_criteria_v1.0.txt
│   ├── synthesis_v1.0.txt
│   ├── conclusion_v1.0.txt
│   └── background_v1.0.txt
└── master_prompts/                        # Master evaluation templates
    ├── scoring_philosophy_v1.0.txt        # Top journal standards
    ├── sophistication_assessment_v1.0.txt # Quality checks
    ├── task_instructions_v1.0.txt         # Evaluation instructions
    └── quote_validation_v1.0.txt          # Quote verification
```

## 🎯 Design Philosophy

### Three-Layer System
1. **Paper Type Context**: Sets expectations for paper type (empirical, theoretical, policy, etc.)
2. **Section Type Guidance**: Provides standards for each section (introduction, methodology, results, etc.)
3. **Master Prompts**: Defines scoring philosophy, task instructions, and evaluation framework

### Why External Prompts?
- **Version Control**: Track evaluation criteria changes over time
- **A/B Testing**: Test different scoring rubrics
- **Discipline-Specific**: Easy to create field-specific variants
- **Transparency**: Clear documentation of evaluation standards
- **Iteration**: Refine criteria based on user feedback

## 🚀 How to Use

### Switching to a New Prompt Version

1. **Create a new version file**:
   ```bash
   cd prompts/section_evaluator/section_type_guidance/
   cp methodology_v1.0.txt methodology_v1.1.txt
   # Edit v1.1.txt with your changes
   ```

2. **Update config.yaml**:
   ```yaml
   section_type_guidance:
     methodology:
       version: "v1.1"  # Changed from v1.0
       file: "section_type_guidance/methodology_{version}.txt"
   ```

3. **Restart the app** or call `_prompt_loader.reload_prompts()` in code

### Adding a New Section Type

1. **Create the prompt file**:
   ```bash
   echo "The NEW SECTION should:..." > section_type_guidance/new_section_v1.0.txt
   ```

2. **Add to config.yaml**:
   ```yaml
   section_type_guidance:
     new_section:
       version: "v1.0"
       file: "section_type_guidance/new_section_{version}.txt"
   ```

3. **Update criteria registry** in `section_eval/criteria/base.py` to define evaluation criteria for this section

## ⚙️ Configuration File (`config.yaml`)

```yaml
# Paper Type Contexts
paper_type_contexts:
  empirical:
    version: "v1.0"
    file: "paper_type_contexts/empirical_{version}.txt"

# Section Type Guidance
section_type_guidance:
  methodology:
    version: "v1.0"
    file: "section_type_guidance/methodology_{version}.txt"

# Master Prompts
master_prompts:
  scoring_philosophy:
    version: "v1.0"
    file: "master_prompts/scoring_philosophy_{version}.txt"
```

## 📝 Prompt Structure Guidelines

### Paper Type Context Prompts
Should define:
- **Key characteristics** of this paper type
- **Expected components** (e.g., identification strategy for empirical)
- **Evaluation emphasis** (what matters most for this type)

### Section Type Guidance Prompts
Should specify:
- **Purpose** of the section
- **Required elements** (what must be included)
- **Quality levels** with concrete examples (5=excellent, 3=adequate, 1=poor)
- **Common weaknesses** to watch for

### Master Prompt Templates
- **scoring_philosophy**: Define 1-5 scale with concrete examples
- **sophistication_assessment**: Checklist for discriminating evaluation
- **task_instructions**: Step-by-step evaluation process
- **quote_validation**: How to verify textual evidence

## 🔄 Version Control Best Practices

### Versioning Convention
- **Format**: `<name>_v<major>.<minor>.txt`
- **Major version** (e.g., v1.0 → v2.0): Significant changes to evaluation approach
- **Minor version** (e.g., v1.0 → v1.1): Refinements, clarifications, bug fixes

### Committing Changes
```bash
git add section_type_guidance/methodology_v1.1.txt
git add config.yaml
git commit -m "feat(section-eval): add v1.1 methodology prompt with enhanced identification guidance"
```

### Performance Tracking
Create a log to track prompt performance:

```markdown
# Section Evaluator Prompt Performance Log

## Methodology v1.1 (2026-03-27)
- **Change**: Added emphasis on falsification tests
- **Performance**: Improved detection of weak identification by 25%
- **Decision**: Keep as default

## Results v1.0 → v1.1 (2026-03-20)
- **Change**: Stronger emphasis on economic magnitude
- **Performance**: Reduced score compression (more 2s and 4s, fewer 3s)
- **Decision**: Adopted as default
```

## 🧪 Testing Prompt Changes

### Run Test Suite
```bash
python3 test_section_evaluator_prompts.py
```

### Expected Output
```
✓ Paper Type Context Prompts: 6/6 loaded
✓ Section Type Guidance Prompts: 22/22 loaded
✓ Master Prompt Templates: 4/4 loaded
✓ Cache working correctly
✓ Reload working correctly
✓ Backward compatibility maintained
```

### A/B Testing Framework
1. Create v1.1 version with changes
2. Evaluate same papers with v1.0 and v1.1
3. Compare score distributions and feedback quality
4. Switch config.yaml to winner

## 📊 Monitoring Prompt Performance

### Recommended Metrics
1. **Score Distribution**: Are you using the full 1-5 range?
2. **Inter-Rater Reliability**: Do different runs produce consistent scores?
3. **User Satisfaction**: Is feedback actionable and specific?
4. **False Positive Rate**: Penalizing papers inappropriately?
5. **False Negative Rate**: Missing real issues?

### Example Analysis
```python
# Track score distributions across prompt versions
results = {
    'methodology_v1.0': {'1': 5, '2': 15, '3': 50, '4': 25, '5': 5},  # Compressed
    'methodology_v1.1': {'1': 10, '2': 20, '3': 40, '4': 20, '5': 10}  # Better spread
}
```

## 🔧 Troubleshooting

### "Prompt file not found" Error
- Check that the file exists at the path specified in config.yaml
- Verify version number matches filename
- Ensure you're running from the correct directory (`app_system/`)

### "Unknown section type" Error
- Verify section type name in config.yaml matches code
- Check that section type is in `SECTION_TYPE_PROMPTS` dict
- Section types use underscores (e.g., `literature_review`, not `literature-review`)

### Prompts Not Updating
- Restart the Streamlit app (prompts load on startup)
- Or call `_prompt_loader.reload_prompts()` programmatically
- Verify config.yaml points to correct version

### Scores Too Compressed (Mostly 3s)
- Check `scoring_philosophy_v1.0.txt` - may need stronger language
- Consider adding `sophistication_assessment` to relevant paper types
- Review section guidance for concrete 1/2/4/5 examples

## 💡 Best Practices

### When to Create New Versions
- ✅ Significant changes to evaluation criteria
- ✅ Adding/removing quality anchors
- ✅ Changing scoring philosophy
- ✅ Field-specific customization
- ❌ Minor typo fixes (just edit current version)
- ❌ Reformatting without content changes

### Keep Old Versions
- Don't delete old version files
- They serve as documentation of criteria evolution
- Useful for understanding past evaluation decisions
- Can revert if new version underperforms

### Document Rationale
Add comments at top of new version files:
```
# v1.1 - Added emphasis on falsification tests
# Rationale: v1.0 was too lenient on identification strategy
# Expected impact: More 2s and 3s for weak identification
# Date: 2026-03-27
```

## 🤝 Contributing

When proposing prompt changes:
1. ✅ Create new version file (don't modify existing)
2. ✅ Document rationale for changes
3. ✅ Test on representative papers
4. ✅ Include performance comparison
5. ✅ Update this README if adding new categories

## 📚 Additional Resources

- Main code: `../../section_eval/prompts/templates.py`
- Prompt loader: `./prompt_loader.py`
- Configuration: `./config.yaml`
- Criteria registry: `../../section_eval/criteria/base.py`
- Test suite: `../../test_section_evaluator_prompts.py`

---

**Last Updated**: 2026-03-27
**Current Active Versions**: All v1.0 (baseline)
**Maintainer**: Research Agents Team
