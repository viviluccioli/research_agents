# Prompt Management System - Quick Start Guide

## 🎯 What Changed?

All system prompts for the multi-agent debate system have been moved to **external versioned files** for better version control and experimentation.

### Before (Hardcoded):
```python
# multi_agent_debate.py
SYSTEM_PROMPTS = {
    "Theorist": """
    ### ROLE
    You are a rigorous Economic Theorist...
    """
}
```

### After (External Files):
```
prompts/multi_agent_debate/
├── config.yaml          # Configuration
├── personas/
│   └── theorist/
│       └── v1.0.txt    # Versioned prompt file
└── debate_rounds/
    └── round_0_selection/
        └── v1.0.txt    # Versioned prompt file
```

## 🚀 Quick Start

### Viewing Current Prompts
```bash
cd prompts/multi_agent_debate/
cat personas/theorist/v1.0.txt
cat debate_rounds/round_0_selection/v1.0.txt
```

### Switching Prompt Versions
1. **Edit** `config.yaml`:
   ```yaml
   personas:
     theorist:
       version: "v1.1"  # Change from v1.0
   ```

2. **Restart** the Streamlit app

### Creating a New Prompt Version
```bash
# Copy existing version
cp personas/theorist/v1.0.txt personas/theorist/v1.1.txt

# Edit the new version
nano personas/theorist/v1.1.txt

# Update config to use new version
nano config.yaml
```

## 📁 File Organization

```
app_system/
├── multi_agent_debate.py                    # Main orchestration (now loads external prompts)
├── test_prompt_loader.py                    # Test script
└── prompts/
    └── multi_agent_debate/
        ├── README.md                        # Detailed documentation
        ├── config.yaml                      # Active version configuration
        ├── prompt_loader.py                 # Loading logic
        ├── personas/                        # Persona system prompts
        │   ├── theorist/
        │   │   ├── v1.0.txt
        │   │   └── v1.1.txt (future)
        │   ├── empiricist/
        │   ├── historian/
        │   ├── visionary/
        │   └── policymaker/
        └── debate_rounds/                   # Debate round prompts
            ├── round_0_selection/
            ├── round_2a_cross_exam/
            ├── round_2b_direct_exam/
            ├── round_2c_final_amendment/
            └── round_3_editor/
```

## ✅ Benefits

### Version Control
- Track prompt changes with git
- See history of prompt evolution
- Revert to previous versions easily

### Collaboration
- Multiple people can edit prompts without code conflicts
- Clear separation between prompt engineering and code

### Experimentation
- A/B test different prompt versions
- Switch between versions instantly
- Document performance of each version

### Safety
- Old versions are preserved
- No risk of losing working prompts
- Easy rollback if new version underperforms

## 🧪 Testing

```bash
# Run test suite
python3 test_prompt_loader.py

# Show sample prompt
python3 test_prompt_loader.py --show-sample
```

## 📊 Best Practices

### Versioning Convention
- `v1.0` → `v1.1`: Minor tweaks
- `v1.0` → `v2.0`: Major restructuring

### Git Commits
```bash
# Good commit message
git commit -m "feat(prompts): add v1.1 Theorist prompt with improved math focus"

# Include performance notes
git commit -m "feat(prompts): switch to Empiricist v1.2 - reduces false positives by 15%"
```

### Documentation
- Keep performance log in `prompts/multi_agent_debate/README.md`
- Document why you created each new version
- Note which versions work best for different paper types

## 🔄 Migration

### What Was Moved?
All hardcoded prompts from `multi_agent_debate.py`:
- ✅ `SELECTION_PROMPT` → `round_0_selection/v1.0.txt`
- ✅ `SYSTEM_PROMPTS["Theorist"]` → `personas/theorist/v1.0.txt`
- ✅ `SYSTEM_PROMPTS["Empiricist"]` → `personas/empiricist/v1.0.txt`
- ✅ `SYSTEM_PROMPTS["Historian"]` → `personas/historian/v1.0.txt`
- ✅ `SYSTEM_PROMPTS["Visionary"]` → `personas/visionary/v1.0.txt`
- ✅ `SYSTEM_PROMPTS["Policymaker"]` → `personas/policymaker/v1.0.txt`
- ✅ `DEBATE_PROMPTS["Round_2A_Cross_Examination"]` → `round_2a_cross_exam/v1.0.txt`
- ✅ `DEBATE_PROMPTS["Round_2B_Direct_Examination"]` → `round_2b_direct_exam/v1.0.txt`
- ✅ `DEBATE_PROMPTS["Round_2C_Final_Amendment"]` → `round_2c_final_amendment/v1.0.txt`
- ✅ `DEBATE_PROMPTS["Round_3_Editor"]` → `round_3_editor/v1.0.txt`

### Backward Compatibility
The code still works exactly the same way. The variables `SELECTION_PROMPT`, `SYSTEM_PROMPTS`, and `DEBATE_PROMPTS` still exist - they're just loaded from files now instead of being hardcoded.

## 📚 Additional Resources

- **Detailed docs**: `prompts/multi_agent_debate/README.md`
- **Code**: `prompts/multi_agent_debate/prompt_loader.py`
- **Config**: `prompts/multi_agent_debate/config.yaml`

## 🆘 Troubleshooting

### App won't start
- Check that all prompts files exist
- Run `python3 test_prompt_loader.py` to diagnose
- Verify config.yaml syntax is valid

### Prompts not updating
- Restart the Streamlit app (prompts load on startup)
- Check that config.yaml points to correct version

### "File not found" errors
- Ensure you're running from `app_system/` directory
- Check file paths in config.yaml match actual files

---

**Questions?** See the detailed README: `prompts/multi_agent_debate/README.md`
