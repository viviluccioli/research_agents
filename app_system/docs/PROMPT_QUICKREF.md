# Prompt Management - Quick Reference Card

## 📂 File Locations

```
prompts/
├── multi_agent_debate/
│   ├── config.yaml                          ← Version configuration
│   ├── personas/{name}_v{X.Y}.txt          ← Persona prompts
│   └── debate_rounds/{round}_v{X.Y}.txt    ← Round prompts
│
└── section_evaluator/
    ├── config.yaml                          ← Version configuration
    ├── paper_type_contexts/{type}_v{X.Y}.txt        ← Paper type contexts
    ├── section_type_guidance/{section}_v{X.Y}.txt   ← Section guidance
    └── master_prompts/{template}_v{X.Y}.txt         ← Master templates
```

## ⚡ Common Tasks

### View Current Prompts
```bash
# Multi-agent debate persona
cat prompts/multi_agent_debate/personas/theorist/v1.0.txt

# Section evaluator guidance
cat prompts/section_evaluator/section_type_guidance/methodology_v1.0.txt
```

### Create New Version
```bash
# 1. Copy existing version
cp theorist/v1.0.txt theorist/v1.1.txt

# 2. Edit new version
nano theorist/v1.1.txt

# 3. Update config
nano prompts/multi_agent_debate/config.yaml
# Change: version: "v1.1"
```

### Switch Versions
```yaml
# Edit config.yaml
personas:
  theorist:
    version: "v1.1"  # Was v1.0
```

### Test Changes
```bash
# Test multi-agent debate
python3 test_prompt_loader.py

# Test section evaluator
python3 test_section_evaluator_prompts.py

# Test everything
python3 test_prompt_loader.py && python3 test_section_evaluator_prompts.py
```

## 🎯 Quick Examples

### Example 1: Improve Methodology Guidance
```bash
# Step 1: Create new version
cp section_type_guidance/methodology_v1.0.txt section_type_guidance/methodology_v1.1.txt

# Step 2: Edit (add better examples)
nano section_type_guidance/methodology_v1.1.txt

# Step 3: Update config
sed -i 's/methodology.*version: "v1.0"/methodology:\n    version: "v1.1"/' prompts/section_evaluator/config.yaml

# Step 4: Test
python3 test_section_evaluator_prompts.py

# Step 5: Restart app to apply changes
```

### Example 2: A/B Test Scoring Philosophy
```bash
# Create experimental version
cp master_prompts/scoring_philosophy_v1.0.txt master_prompts/scoring_philosophy_v1.1-experimental.txt

# Edit experimental version
nano master_prompts/scoring_philosophy_v1.1-experimental.txt

# Test with v1.0 (baseline)
# ... run evaluations, record results ...

# Switch to v1.1-experimental
# Edit config: version: "v1.1-experimental"
# ... run same evaluations, compare results ...

# Keep better version
```

## 📋 Checklists

### Before Creating New Version
- [ ] Identify specific issue with current version
- [ ] Document what you want to change
- [ ] Consider impact on existing evaluations

### When Creating New Version
- [ ] Copy from most recent version
- [ ] Make targeted changes (not wholesale rewrite)
- [ ] Add comment at top explaining changes
- [ ] Update config.yaml
- [ ] Run test suite
- [ ] Test on sample papers

### After Creating New Version
- [ ] Document performance (better/worse/same)
- [ ] Commit to git with descriptive message
- [ ] Update README performance log
- [ ] Keep old version (don't delete)

## 🔍 Troubleshooting

| Problem | Solution |
|---------|----------|
| "File not found" error | Check version in config.yaml matches filename |
| Prompts not updating | Restart app (prompts load at startup) |
| Import errors | Verify you're in `app_system/` directory |
| Tests failing | Run `python3 test_*_prompts.py` to diagnose |
| Syntax errors | Check YAML indentation in config.yaml |

## 📊 Version Naming

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Typo fix | None (edit in place) | v1.0 → v1.0 |
| Clarification | Minor | v1.0 → v1.1 |
| New criteria | Minor | v1.1 → v1.2 |
| Complete rewrite | Major | v1.2 → v2.0 |
| Experimental | Add suffix | v1.0 → v1.1-exp |

## 🎓 Best Practices

### DO ✅
- Keep old versions for rollback
- Test on sample papers before deploying
- Document why you made changes
- Use descriptive version comments
- Commit to git regularly

### DON'T ❌
- Delete old versions
- Skip testing after changes
- Make undocumented changes
- Edit current version for major changes
- Forget to update config.yaml

## 📚 Documentation

| File | Purpose |
|------|---------|
| `PROMPT_MANAGEMENT.md` | Quick start guide |
| `PROMPT_SYSTEM_COMPLETE.md` | Complete system overview |
| `prompts/*/README.md` | Detailed system-specific docs |
| `RESTRUCTURING_SUMMARY.md` | Initial restructuring details |

## 🚀 Git Commands

```bash
# View prompt changes
git log prompts/multi_agent_debate/personas/theorist/

# Compare versions
git diff theorist/v1.0.txt theorist/v1.1.txt

# Revert to old version
git checkout <commit-hash> prompts/section_evaluator/config.yaml

# Commit new version
git add prompts/
git commit -m "feat(prompts): add methodology v1.1 with improved identification guidance"
```

## 🔢 Statistics

| System | Prompts | Config | Loader | Docs | Test |
|--------|---------|--------|--------|------|------|
| Multi-Agent Debate | 10 | ✓ | ✓ | ✓ | ✓ |
| Section Evaluator | 32 | ✓ | ✓ | ✓ | ✓ |
| **Total** | **42** | **2** | **2** | **5** | **2** |

---

**Quick Help**: Run `python3 test_prompt_loader.py --help` or `python3 test_section_evaluator_prompts.py --help`

**Full Docs**: See `PROMPT_SYSTEM_COMPLETE.md` for comprehensive guide
