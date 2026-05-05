# Skill: Version a Prompt

This skill guides creating a new version of an existing prompt.

## When to Use
- Improving prompt clarity or effectiveness
- Fixing issues found in testing
- Adding new instructions or criteria
- A/B testing different approaches

## Prerequisites
- Identify which prompt to version
- Understand current prompt structure
- Have test cases ready for validation

## Steps

### 1. Identify Target Prompt

**Find the prompt file**:
```bash
# For personas
ls app_system/prompts/multi_agent_debate/personas/*/

# For debate rounds
ls app_system/prompts/multi_agent_debate/debate_rounds/*/

# For section evaluator
ls app_system/prompts/section_evaluator/paper_type_contexts/*/
```

**Check current version**:
```bash
cat app_system/prompts/multi_agent_debate/config.yaml | grep -A 2 "theorist:"
```

Output:
```yaml
theorist:
  version: "v1.0"
  file: "personas/theorist/{version}.txt"
```

### 2. Copy Existing Version

**Create new version file**:

```bash
# For subdirectory-structured prompts
cp app_system/prompts/multi_agent_debate/personas/theorist/v1.0.txt \
   app_system/prompts/multi_agent_debate/personas/theorist/v1.1.txt

# For flat file prompts
cp app_system/prompts/section_evaluator/section_type_guidance/abstract_v1.0.txt \
   app_system/prompts/section_evaluator/section_type_guidance/abstract_v1.1.txt
```

**Version numbering guide**:
- **Minor (v1.0 → v1.1)**: Wording, clarifications, non-breaking
- **Major (v1.x → v2.0)**: Structural changes, new sections, breaking changes

### 3. Edit the New Version

```bash
nano app_system/prompts/multi_agent_debate/personas/theorist/v1.1.txt
```

**Common improvements**:
- Clarify ambiguous instructions
- Add concrete examples
- Refine evaluation criteria
- Improve output format specification
- Add domain-specific guidance

**Example change**:
```diff
## EVALUATION CRITERIA
-1. **Mathematical Rigor**: Are proofs sound?
+1. **Mathematical Rigor**: Are proofs sound and complete?
+   - Check for gaps in logical steps
+   - Verify all assumptions are stated
+   - Confirm notation is consistent

-2. **Originality**: Is the contribution novel?
+2. **Theoretical Originality**: What is the novel contribution?
+   - Compare to existing literature
+   - Identify incremental vs. fundamental advances
+   - Assess impact on field
```

### 4. Add Version Metadata (Optional)

**At top of file** (not parsed, for documentation):

```
<!--
Version: v1.1
Date: 2026-04-27
Author: Research Team
Changes:
  - Clarified mathematical rigor criteria
  - Added specific checks for theoretical originality
  - Improved output format examples
Rationale: Testing showed ambiguity in "rigor" interpretation
-->

You are the **Theorist** in a multi-agent referee panel...
```

### 5. Test the New Version (Before Deploying)

**A. Update config.yaml temporarily**:
```yaml
personas:
  theorist:
    version: "v1.1"  # Point to new version
    file: "personas/theorist/{version}.txt"
```

**B. Test with sample papers**:
```bash
cd app_system
streamlit run app.py

# Test on:
# 1. Paper that worked well with v1.0 (regression test)
# 2. Paper that had issues with v1.0 (validation)
# 3. New paper (general quality check)
```

**C. Compare outputs**:
```python
# Run A/B comparison
results_v1_0 = run_with_version("v1.0", paper)
results_v1_1 = run_with_version("v1.1", paper)

# Compare:
# - Output quality
# - Consistency
# - Clarity of findings
# - Appropriateness of verdicts
```

### 6. Deploy New Version

**Update config.yaml**:
```bash
nano app_system/prompts/multi_agent_debate/config.yaml
```

```yaml
personas:
  theorist:
    version: "v1.1"  # Updated from v1.0
    file: "personas/theorist/{version}.txt"
```

**Restart app** (Streamlit auto-reloads):
```bash
# If running, just save the file - Streamlit reloads
# If not running:
cd app_system
streamlit run app.py
```

### 7. Document the Change

**Update changelog** (`app_system/docs/changelog.md`):

```markdown
## [2026-04-27] - Theorist Prompt v1.1

### Changed
- Clarified mathematical rigor evaluation criteria
- Added specific checks for proof completeness
- Improved theoretical originality assessment with comparisons

### Rationale
Testing showed ambiguity in interpreting "rigor" - some reports focused 
only on correctness, missing completeness and clarity aspects.

### Impact
Expected to improve consistency of theoretical evaluations and reduce 
vague "needs more rigor" feedback.

### Files Modified
- `prompts/multi_agent_debate/personas/theorist/v1.1.txt`: New version
- `prompts/multi_agent_debate/config.yaml`: Version bump
```

**Update CLAUDE.md if major change**:
Only if the change affects documented behavior or workflows.

### 8. Monitor Performance

**Track metrics over next N runs**:
- Output quality (manual review)
- Consistency across papers
- User feedback
- Token usage (if prompt length changed significantly)

**If issues arise**:
- Document the issue
- Revert to v1.0 immediately (update config.yaml)
- Fix in v1.2 or v2.0
- Re-test thoroughly

## Rollback Procedure

**Immediate rollback**:
```bash
# Update config.yaml
nano app_system/prompts/multi_agent_debate/config.yaml
```

```yaml
personas:
  theorist:
    version: "v1.0"  # Reverted from v1.1
    file: "personas/theorist/{version}.txt"
```

Streamlit auto-reloads. No restart needed.

**Investigate**:
- Review outputs that caused rollback
- Identify specific prompt issues
- Compare with previous version
- Plan fixes for next version

## A/B Testing Strategy

**For major changes**:

1. **Create experimental version** (v2.0-beta):
```bash
cp v1.0.txt v2.0-beta.txt
# Make significant changes
```

2. **Run parallel tests**:
```python
# Test same papers with both versions
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
for paper in papers:
    result_v1 = evaluate_with_version("v1.0", paper)
    result_v2 = evaluate_with_version("v2.0-beta", paper)
    compare_and_log(result_v1, result_v2)
```

3. **Analyze results**:
- Quality improvement?
- Consistency maintained?
- Unexpected behaviors?
- Token cost impact?

4. **Decide**:
- ✅ Deploy as v2.0 if clearly better
- 🔄 Refine and test again
- ❌ Abandon if worse or unclear benefit

## Version Management Tips

**Preserve old versions**:
- Never delete old version files
- Keep for rollback capability
- Reference in future improvements

**Version history**:
```
theorist/
├── v1.0.txt  # Original
├── v1.1.txt  # Clarified rigor criteria
├── v1.2.txt  # Added examples
└── v2.0.txt  # Major restructure
```

**Branch for major changes**:
```bash
git checkout -b prompt-theorist-v2
# Make changes
# Test thoroughly
git commit -m "feat: theorist prompt v2.0 - restructured evaluation"
# PR for review before merging
```

## Common Prompt Issues & Fixes

**Issue**: Vague or inconsistent outputs
**Fix**: Add concrete examples and clearer criteria

**Issue**: Ignoring certain aspects
**Fix**: Make those criteria more prominent, add to evaluation checklist

**Issue**: Too long/verbose outputs
**Fix**: Add output length guidance, specify priority focus areas

**Issue**: Missing required sections
**Fix**: Make output format more explicit, add template

**Issue**: Wrong tone or style
**Fix**: Add tone guidance, examples of good vs. bad phrasing

## Testing Checklist

Before deploying new version:

- ✅ Tested on 3+ papers (varied types)
- ✅ Compared with previous version
- ✅ Output format matches expectations
- ✅ No performance regression
- ✅ Token usage reasonable
- ✅ Changelog updated
- ✅ config.yaml updated
- ✅ Rollback plan ready

## Related Skills
- `/add-persona` - Add new persona (includes prompt creation)
- `/test-changes` - Run test suite after prompt updates
