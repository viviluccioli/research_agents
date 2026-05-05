# .claude/ Structure Overview

## Quick Stats
- **Total Lines**: 3,479 lines of documentation
- **Rules**: 5 context-specific guides
- **Skills**: 5 workflow guides
- **Coverage**: All major subsystems and workflows

## Visual Structure

```
.claude/
│
├── 📖 README.md (456 lines)
│   └── Overview, usage guide, examples
│
├── 📁 rules/ (1,868 lines)
│   │
│   ├── 🎯 referee-system.md (294 lines)
│   │   ├── Multi-agent debate architecture
│   │   ├── Adding personas (4-step process)
│   │   ├── Utilities (cache, dedup, quote validation)
│   │   └── Experiment 4 & memo system
│   │
│   ├── 📊 section-evaluator.md (238 lines)
│   │   ├── 5-stage pipeline
│   │   ├── Criteria system & fatal-flaws
│   │   ├── Section detection algorithm
│   │   └── Paper type management
│   │
│   ├── 🧪 experiments.md (349 lines)
│   │   ├── Batch referee reports
│   │   ├── Persona selection experiments
│   │   ├── Ground truth evaluation
│   │   └── Referee classifier
│   │
│   ├── 📝 prompts.md (413 lines)
│   │   ├── Directory structure (subdirs vs flat)
│   │   ├── config.yaml management
│   │   ├── Version numbering & rollback
│   │   └── A/B testing strategy
│   │
│   └── 🔧 utilities.md (574 lines)
│       ├── LLM call patterns (3 types)
│       ├── Caching system (SHA256-based)
│       ├── Deduplication (similarity metrics)
│       ├── Quote validation (fuzzy matching)
│       └── PDF extraction (PyMuPDF)
│
└── 📁 skills/ (1,155 lines)
    │
    ├── ➕ add-persona.md (296 lines)
    │   └── 7-step process: prompt → config → engine → UI → test → document
    │
    ├── ➕ add-paper-type.md (391 lines)
    │   └── 10-step process: metadata → criteria → weights → prompts → test
    │
    ├── 🔄 version-prompt.md (269 lines)
    │   └── 8-step process: copy → edit → test → deploy → monitor
    │
    ├── ✅ test-changes.md (364 lines)
    │   └── Test matrix: referee/section-eval/utilities × quick/full
    │
    └── 🚀 run-experiment.md (291 lines)
        └── Batch experiments: setup → test → run → analyze
```

## Rule Activation Map

```
File Pattern                           → Activated Rule(s)
─────────────────────────────────────────────────────────────────
app_system/referee/                    → referee-system.md
app_system/referee/_utils/             → referee-system.md + utilities.md
app_system/section_eval/               → section-evaluator.md
app_system/section_eval/criteria/      → section-evaluator.md
app_system/prompts/                    → prompts.md
app_system/prompts/*/config.yaml       → prompts.md
app_system/utils.py                    → utilities.md
app_system/config.py                   → utilities.md
experiment/                            → experiments.md
persona_exp/                           → experiments.md
mad_experiments/                       → experiments.md
```

## Workflow Coverage Matrix

```
Task                          Rule          Skill              Time
──────────────────────────────────────────────────────────────────────
Add new persona              referee       /add-persona        30 min
Add paper type               section-eval  /add-paper-type     60 min
Version prompt               prompts       /version-prompt     15-30 min
Test changes                 (any)         /test-changes       5-10 min
Run batch experiment         experiments   /run-experiment     varies
Add utility function         utilities     —                   varies
Modify caching logic         utilities     /test-changes       varies
Update criteria weights      section-eval  —                   15 min
Create experiment variant    experiments   —                   varies
```

## Documentation Density by Topic

```
Topic                    Lines    Percentage
─────────────────────────────────────────────
Utilities                574      16.5%
Prompts                  413      11.9%
Add Paper Type Skill     391      11.2%
Test Changes Skill       364      10.5%
Experiments              349      10.0%
Add Persona Skill        296      8.5%
Section Evaluator        238      6.8%
Referee System           294      8.5%
Run Experiment Skill     291      8.4%
Version Prompt Skill     269      7.7%
─────────────────────────────────────────────
Total                    3,479    100%
```

## Key Patterns Documented

### Architecture Patterns
- ✅ Multi-agent debate (5-round structure)
- ✅ Section evaluation pipeline (5 stages)
- ✅ Prompt versioning (subdirs + config.yaml)
- ✅ LLM call patterns (stateless, stateful, direct)
- ✅ Caching strategy (per-round granularity)

### Development Workflows
- ✅ Adding personas (referee + exp_4 + memo)
- ✅ Adding paper types (criteria + weights + prompts)
- ✅ Versioning prompts (minor vs major)
- ✅ Testing (subsystem-specific + integration)
- ✅ Running experiments (batch + persona selection)

### Common Pitfalls
- ✅ 20+ documented anti-patterns
- ✅ Troubleshooting guides
- ✅ Rollback procedures
- ✅ Validation checklists

## Integration with Project Documentation

```
Documentation Layer          Purpose                    Audience
────────────────────────────────────────────────────────────────────
CLAUDE.md (root)            Project-wide guidance      Claude Code
.claude/rules/              Context-specific detail    Claude Code
.claude/skills/             Step-by-step workflows     Claude Code
app_system/README.md        User-facing guide          End users
app_system/docs/*.md        Technical deep dives       Developers
```

## Usage Patterns

### For Claude Code AI
1. **Automatic**: Rules load based on file context
2. **On-demand**: Skills invoked by name (e.g., `/add-persona`)
3. **Contextual**: Multiple rules can be active simultaneously

### For Human Developers
1. **Quick reference**: Browse `.claude/` for workflows
2. **Onboarding**: Read README → Rules → Skills
3. **Troubleshooting**: Search for error patterns in rules

## Metrics & Coverage

### Subsystem Coverage
- **Referee System**: ✅ Comprehensive (294 lines rule + 296 lines skill)
- **Section Evaluator**: ✅ Comprehensive (238 lines rule + 391 lines skill)
- **Experiments**: ✅ Comprehensive (349 lines rule + 291 lines skill)
- **Prompts**: ✅ Comprehensive (413 lines rule + 269 lines skill)
- **Utilities**: ✅ Comprehensive (574 lines rule + 364 lines skill)

### Workflow Coverage
- **Adding features**: ✅ (add-persona, add-paper-type)
- **Versioning**: ✅ (version-prompt)
- **Testing**: ✅ (test-changes)
- **Research**: ✅ (run-experiment)
- **Debugging**: ✅ (troubleshooting sections)
- **Deployment**: ⚠️  (partial, in individual rules)

### Documentation Quality
- **Clarity**: ✅ Step-by-step guides with examples
- **Completeness**: ✅ Prerequisites, steps, validation, troubleshooting
- **Maintainability**: ✅ Versioned with codebase
- **Discoverability**: ✅ README with clear structure

## Future Expansion Opportunities

### Potential New Rules
- **ui-development.md** — Streamlit UI patterns
- **testing-strategies.md** — Deep dive on test patterns
- **deployment.md** — Production deployment guide

### Potential New Skills
- **/debug-persona** — Debug persona output issues
- **/optimize-cost** — Reduce API costs
- **/migrate-experiment** — Move experiment → production
- **/benchmark-changes** — A/B test system changes

## Maintenance Guidelines

### When to Update
- **After major features**: Document new patterns
- **After debugging**: Add troubleshooting tips
- **After onboarding**: Clarify confusing sections
- **Quarterly**: Review for outdated information

### Quality Checklist
- ✅ Clear headings and structure
- ✅ Code examples that work
- ✅ Concrete file paths (not abstract)
- ✅ Validation checklists
- ✅ Troubleshooting sections
- ✅ Related links

## Success Metrics

This `.claude/` scaffolding is successful if:
1. ✅ Context switches feel natural (right info at right time)
2. ✅ Common tasks have clear workflows (no guessing)
3. ✅ Errors are preventable (checklists catch issues)
4. ✅ Onboarding is faster (new contributors self-sufficient)
5. ✅ Knowledge is preserved (tribal knowledge documented)
