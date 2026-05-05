# Prompts & Versioning Context

This rule activates when working on prompt files, versioning, or configuration.

## File Scope
- `app_system/prompts/`
- `app_system/prompts/*/config.yaml`
- `*.txt` prompt files

## Directory Structure

### Standard Pattern
```
prompts/{system}/
├── config.yaml                    # Version control
├── {category}/
│   ├── {item}/
│   │   ├── v1.0.txt              # Versioned prompt
│   │   ├── v1.1.txt              # Newer version
│   │   └── v2.0.txt              # Major revision
```

### Flat File Exception
Some categories use flat files instead of subdirectories:
```
prompts/section_evaluator/
├── section_type_guidance/
│   ├── abstract_v1.0.txt         # Flat (not abstract/v1.0.txt)
│   ├── introduction_v1.0.txt
│   └── methodology_v1.0.txt
```

**Use flat files when**:
- 20+ items in category
- Rarely version independently
- Tightly coupled (e.g., section guidance)

**Use subdirectories when**:
- 5-10 items in category
- Version independently
- Conceptually distinct (e.g., personas, paper types)

## Config.yaml Structure

```yaml
# Category name
personas:
  # Item name
  theorist:
    version: "v1.0"                           # Active version
    file: "personas/theorist/{version}.txt"   # Path template

# Nested categories
additional_context:
  paper_type_contexts:
    empirical:
      version: "v1.0"
      file: "additional_context/paper_type_contexts/empirical/{version}.txt"
```

**Key Rules**:
1. `{version}` placeholder is replaced by `PromptLoader`
2. Path is relative to `prompts/{system}/`
3. Version string format: `v{MAJOR}.{MINOR}`

## Versioning Workflow

### Creating a New Version

**1. Copy existing version**:
```bash
cp prompts/multi_agent_debate/personas/theorist/v1.0.txt \
   prompts/multi_agent_debate/personas/theorist/v1.1.txt
```

**2. Edit the new version**:
```bash
nano prompts/multi_agent_debate/personas/theorist/v1.1.txt
```

**3. Update config.yaml**:
```yaml
personas:
  theorist:
    version: "v1.1"  # Changed from v1.0
    file: "personas/theorist/{version}.txt"
```

**4. Test the change**:
```python
# In Streamlit app, reload prompts
from referee.engine import reload_prompts
reload_prompts()
```

### Version Numbering

**Minor version (v1.0 → v1.1)**:
- Wording improvements
- Clarifications
- Non-breaking changes

**Major version (v1.x → v2.0)**:
- Structural changes
- New instructions
- Breaking changes

## Prompt Components

### Persona Prompts (MAD System)

**Required sections**:
1. **Role definition**: "You are a [role] reviewing this paper..."
2. **Expertise**: Specific focus areas
3. **Evaluation criteria**: What to look for
4. **Output format**: Structured report template
5. **Error severity guide**: `{_ERROR_SEVERITY_GUIDE}` placeholder

**Example structure**:
```
You are the [ROLE] in a multi-agent referee panel...

EXPERTISE:
- Area 1
- Area 2

EVALUATION CRITERIA:
1. Criterion 1
2. Criterion 2

{_ERROR_SEVERITY_GUIDE}

OUTPUT FORMAT:
## Section 1
...
```

### Debate Round Prompts

**Context variables**:
- `{paper_text}` — Full paper text
- `{peer_reports}` — Round 1 reports (Round 2A)
- `{qa_transcript}` — Q&A from 2A+2B (Round 2C)
- `{debate_transcript}` — Full discussion (Round 3)

**Round-specific focus**:
- **Round 0**: Selection prompt (choose 3 personas + weights)
- **Round 1**: Independent analysis
- **Round 2A**: Generate questions for peers
- **Round 2B**: Answer received questions
- **Round 2C**: Amend initial report based on discussion
- **Round 3**: Editor synthesizes consensus

### Paper Type Contexts

**Purpose**: Inject domain-specific guidance into persona prompts.

**Loaded at runtime**: Appended to persona system prompt based on paper type.

**Example** (`empirical/v1.0.txt`):
```
EMPIRICAL PAPER GUIDANCE:
- Focus on identification strategy
- Evaluate instrument validity
- Check robustness tests
...
```

## Prompt Loader System

### Multi-Agent Debate
```python
from prompts.multi_agent_debate import PromptLoader

loader = PromptLoader()
theorist_prompt = loader.get_persona_prompt("theorist")
round_2a_prompt = loader.get_debate_prompt("round_2a_cross_exam")
```

### Section Evaluator
```python
from section_eval.prompts import PromptLoader

loader = PromptLoader()
empirical_context = loader.get_paper_type_context("empirical")
abstract_guidance = loader.get_section_guidance("abstract")
```

### Fallback Handling
```python
# Prompt loader returns None if file not found
prompt = loader.get_persona_prompt("theorist") or """
Fallback hardcoded prompt...
"""
```

## Testing Prompts

### Manual Testing
```bash
cd app_system
streamlit run app.py

# Upload test paper
# Select paper type / personas
# Review output quality
```

### Automated Testing
```bash
cd app_system
python -m pytest tests/test_prompt_loader.py
python -m pytest tests/test_referee_quick.py  # End-to-end
```

### A/B Testing
```python
# Run with v1.0
results_v1 = run_evaluation(paper, prompt_version="v1.0")

# Run with v1.1
results_v1_1 = run_evaluation(paper, prompt_version="v1.1")

# Compare
compare_results(results_v1, results_v1_1)
```

## Common Patterns

### Variable Injection
```python
prompt = template.format(
    paper_text=paper_text,
    peer_reports="\n\n".join(reports.values()),
    custom_context=custom_context or ""
)
```

### Error Severity Guide
```python
_ERROR_SEVERITY_GUIDE = """
CRITICAL ERRORS: Fundamental flaws that invalidate findings...
MAJOR ISSUES: Significant problems requiring revision...
MODERATE ISSUES: Concerns that should be addressed...
MINOR ISSUES: Small improvements that would strengthen the paper...
"""

# Inject into persona prompts
prompt = persona_prompt.format(_ERROR_SEVERITY_GUIDE=_ERROR_SEVERITY_GUIDE)
```

## Prompt Best Practices

✅ **Do**:
- Use clear section headers (##, ###)
- Provide concrete examples
- Specify output format explicitly
- Use consistent terminology
- Version control all changes
- Test on multiple papers

❌ **Don't**:
- Make prompts too long (>8K tokens)
- Use ambiguous language
- Forget to inject required variables
- Skip config.yaml updates
- Break backwards compatibility without major version bump

## Validation Checklist

**Before committing prompt changes**:

1. ✅ Syntax check (no unterminated strings, valid placeholders)
2. ✅ Config.yaml updated with new version
3. ✅ Tested on 2-3 representative papers
4. ✅ Output format matches expected schema
5. ✅ No performance regression (cost, quality)
6. ✅ Documentation updated if major change
7. ✅ Old version preserved (don't delete)

## Rollback Procedure

**If new version causes issues**:

1. **Immediate rollback** (config.yaml):
   ```yaml
   theorist:
     version: "v1.0"  # Revert to previous
   ```

2. **Restart app** (Streamlit auto-reloads)

3. **Investigate issue**:
   - Check logs for errors
   - Review output quality
   - Compare with previous version

4. **Fix and re-deploy**:
   - Fix issues in prompt file
   - Increment version (v1.1 → v1.2)
   - Test thoroughly before updating config

## Metadata in Prompts

**Optional frontmatter** (not parsed, for documentation):
```
<!--
Version: v1.1
Date: 2026-04-27
Author: Research Team
Changes: Clarified identification strategy guidance
-->

You are the Econometrician...
```

## Related Documentation

- `CLAUDE.md` — Overall prompt organization rules
- `app_system/docs/changelog.md` — Major prompt changes log
- `prompts/*/config.yaml` — Active version configuration
