# Multi-Agent Debate Prompts - Version Control System

This directory contains all system prompts for the multi-agent debate system, organized for easy version control and iteration.

## 📁 Directory Structure

```
prompts/multi_agent_debate/
├── config.yaml                     # Main configuration file
├── prompt_loader.py                # Prompt loading module
├── README.md                       # This file
├── personas/                       # Persona system prompts
│   ├── theorist/
│   │   ├── v1.0.txt               # Version 1.0 of Theorist prompt
│   │   └── v1.1.txt               # Version 1.1 (example)
│   ├── empiricist/
│   │   └── v1.0.txt
│   ├── historian/
│   │   └── v1.0.txt
│   ├── visionary/
│   │   └── v1.0.txt
│   └── policymaker/
│       └── v1.0.txt
└── debate_rounds/                  # Debate round prompts
    ├── round_0_selection/
    │   └── v1.0.txt               # Persona selection prompt
    ├── round_2a_cross_exam/
    │   └── v1.0.txt               # Cross-examination prompt
    ├── round_2b_direct_exam/
    │   └── v1.0.txt               # Direct examination prompt
    ├── round_2c_final_amendment/
    │   └── v1.0.txt               # Final amendment prompt
    └── round_3_editor/
        └── v1.0.txt               # Editor report prompt
```

## 🎯 Design Philosophy

### Why External Prompts?
1. **Version Control**: Track prompt changes over time with git
2. **A/B Testing**: Easy to switch between prompt versions
3. **Collaboration**: Multiple people can work on prompts without code conflicts
4. **Documentation**: Prompt evolution is visible in git history
5. **Experimentation**: Test new prompt versions without touching code

### Versioning Convention
- **Format**: `v<major>.<minor>.txt`
- **Major version** (e.g., v1.0 → v2.0): Significant changes to prompt structure or approach
- **Minor version** (e.g., v1.0 → v1.1): Small tweaks, clarifications, or bug fixes

## 🚀 How to Use

### Switching to a New Prompt Version

1. **Create a new version file**:
   ```bash
   cd prompts/multi_agent_debate/personas/theorist/
   cp v1.0.txt v1.1.txt
   # Edit v1.1.txt with your changes
   ```

2. **Update config.yaml**:
   ```yaml
   personas:
     theorist:
       version: "v1.1"  # Changed from v1.0
       file: "personas/theorist/{version}.txt"
   ```

3. **Restart the app** or call `_prompt_loader.reload_prompts()` in code

### Creating a New Persona

1. **Create directory and prompt file**:
   ```bash
   mkdir -p prompts/multi_agent_debate/personas/new_persona/
   echo "Your prompt here" > prompts/multi_agent_debate/personas/new_persona/v1.0.txt
   ```

2. **Add to config.yaml**:
   ```yaml
   personas:
     new_persona:
       version: "v1.0"
       file: "personas/new_persona/{version}.txt"
   ```

3. **Update multi_agent_debate.py** to include the new persona in the selection pool

## ⚙️ Configuration File (`config.yaml`)

The config file specifies which version of each prompt to use:

```yaml
# Persona System Prompts
personas:
  theorist:
    version: "v1.0"                    # Which version to use
    file: "personas/theorist/{version}.txt"  # File path template

# Debate Round Prompts
debate_rounds:
  round_0_selection:
    version: "v1.0"
    file: "debate_rounds/round_0_selection/{version}.txt"

# Model Configuration (optional)
model:
  temperature: 1.0
  max_tokens:
    round_0_1_2a_2b: 4096
    round_2c: 6144
    round_3_editor: 8192
```

### Key Fields:
- **`version`**: Which version file to load (e.g., "v1.0", "v2.1")
- **`file`**: Path template with `{version}` placeholder
- **`model`**: Optional model configuration settings

## 📝 Prompt Structure Guidelines

### Persona Prompts Should Include:
1. **ROLE** - Who the persona is and their domain expertise
2. **OBJECTIVE** - What they focus on when evaluating
3. **ERROR SEVERITY GUIDE** - Classification system ([FATAL], [MAJOR], [MINOR])
4. **OUTPUT FORMAT** - Required structure for responses

### Debate Round Prompts Should Include:
1. **CONTEXT** - What information the persona has access to
2. **OBJECTIVE** - What they need to accomplish in this round
3. **OUTPUT FORMAT** - Expected response structure
4. **CRITICAL NOTES** - Important constraints or requirements

## 🔄 Version Control Best Practices

### Committing Prompt Changes
```bash
git add prompts/multi_agent_debate/personas/theorist/v1.1.txt
git add prompts/multi_agent_debate/config.yaml
git commit -m "feat(prompts): add v1.1 Theorist prompt with improved math focus"
```

### Commit Message Convention:
- `feat(prompts): <description>` - New prompt version
- `fix(prompts): <description>` - Fix bugs in existing prompt
- `docs(prompts): <description>` - Documentation updates
- `refactor(prompts): <description>` - Restructure without changing behavior

### Tracking Performance
Create a log file to track which prompt versions perform best:

```markdown
# Prompt Performance Log

## Theorist v1.1 (2026-03-27)
- **Change**: Added emphasis on assumption verification
- **Performance**: 15% more FATAL flaw detection
- **Decision**: Keep as default

## Empiricist v1.0 → v1.1 (2026-03-20)
- **Change**: Clarified identification strategy criteria
- **Performance**: Reduced false positives by 20%
- **Decision**: Rolled back due to lower recall
```

## 🧪 Testing Prompt Changes

### A/B Testing Framework
1. Run same papers through v1.0 and v1.1
2. Compare verdicts and quality of findings
3. Document differences in performance log
4. Switch config.yaml to winner

### Regression Testing
When changing prompts, test on a set of known papers to ensure:
- Critical flaws are still detected
- Verdict consistency is maintained
- Output format compliance

## 🔧 Troubleshooting

### "Prompt file not found" Error
- Check that the file exists at the path specified in config.yaml
- Verify the version number matches the filename
- Ensure you're running from the correct directory

### "Unknown persona" Error
- Verify the persona name in config.yaml matches the code
- Check that persona is in the SYSTEM_PROMPTS loading dict

### Prompts Not Updating
- Restart the Streamlit app (changes only load on startup)
- Or call `_prompt_loader.reload_prompts()` programmatically
- Check that config.yaml points to the correct version

## 📊 Monitoring Prompt Performance

### Recommended Metrics
1. **Verdict Distribution**: PASS/REVISE/FAIL ratios
2. **Fatal Flaw Detection Rate**: % of known critical issues found
3. **Output Format Compliance**: % of responses matching expected structure
4. **Verdict Changes Round 1 → Round 2C**: Measure debate impact

### Example Analysis
```python
# Track prompt version performance
results = {
    'theorist_v1.0': {'pass': 20, 'revise': 50, 'fail': 30},
    'theorist_v1.1': {'pass': 15, 'revise': 45, 'fail': 40}  # More strict
}
```

## 🤝 Contributing

When proposing prompt changes:
1. Create a new version file (don't modify existing versions)
2. Document the rationale for changes
3. Test on representative papers
4. Update this README if adding new sections
5. Submit PR with performance comparison

## 📚 Additional Resources

- Main codebase: `../../multi_agent_debate.py`
- Prompt loader: `./prompt_loader.py`
- Configuration: `./config.yaml`
- Section evaluator prompts: `../section_eval/` (future work)

---

**Last Updated**: 2026-03-27
**Current Active Versions**: All v1.0 (baseline)
**Maintainer**: Research Agents Team
