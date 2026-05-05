# .claude/ Directory

This directory contains **rules** and **skills** to help Claude Code provide context-aware assistance for this research agents project.

## Structure

```
.claude/
├── README.md           # This file
├── rules/              # Context-specific guidance
│   ├── referee-system.md      # Multi-agent debate (MAD) system
│   ├── section-evaluator.md   # Section evaluation system
│   ├── experiments.md          # Batch experiments & research
│   ├── prompts.md              # Prompt versioning & management
│   └── utilities.md            # Shared utilities & infrastructure
└── skills/             # Recurring workflow guides
    ├── add-persona.md          # Add new persona to MAD system
    ├── add-paper-type.md       # Add new paper type to evaluator
    ├── version-prompt.md       # Create new prompt version
    ├── test-changes.md         # Run appropriate tests
    └── run-experiment.md       # Run batch experiments
```

## How It Works

### Rules (Automatic Context)

**Rules** provide specialized guidance when working on specific subsystems. Claude Code automatically loads relevant rules based on the files you're working with.

**Example**: When editing `app_system/referee/engine.py`, the `referee-system.md` rule activates, providing context about:
- MAD architecture (5-round debate structure)
- Adding personas (4-step process)
- Prompt loading and versioning
- Testing requirements
- Common pitfalls

**When rules activate**:
- **referee-system.md** → Working on `app_system/referee/`, `prompts/multi_agent_debate/`
- **section-evaluator.md** → Working on `app_system/section_eval/`, `prompts/section_evaluator/`
- **experiments.md** → Working on `experiment/`, `persona_exp/`, `mad_experiments/`
- **prompts.md** → Working on any `prompts/` files or `config.yaml`
- **utilities.md** → Working on `utils.py`, `config.py`, `_utils/`

### Skills (On-Demand Workflows)

**Skills** are step-by-step guides for recurring development workflows. Invoke them explicitly when needed.

**Example**: To add a new persona, invoke `/add-persona` skill:
1. Create prompt file (`personas/{name}/v1.0.txt`)
2. Update `config.yaml`
3. Update `engine.py` SYSTEM_PROMPTS
4. Update SELECTION_PROMPT
5. Add UI styling
6. Test integration
7. Document change

**How to invoke**: In Claude Code, type the skill name (e.g., `/add-persona`) or ask Claude to use it.

## Available Rules

### 1. referee-system.md
**Activates when**: Working on MAD system (referee reports)

**Key topics**:
- 5-round debate architecture
- Adding/modifying personas
- Quote validation, deduplication, caching
- Experiment 4 (10-persona system)
- Memo evaluation system

**Common tasks**:
- Adding new persona → See skill `/add-persona`
- Versioning persona prompt → See skill `/version-prompt`
- Testing changes → See skill `/test-changes`

### 2. section-evaluator.md
**Activates when**: Working on section evaluation system

**Key topics**:
- 5-stage pipeline (extraction → detection → hierarchy → evaluation → scoring)
- Paper types and evaluation criteria
- Fatal-flaw logic
- Section detection algorithm
- Prompt organization

**Common tasks**:
- Adding paper type → See skill `/add-paper-type`
- Updating criteria weights
- Versioning prompts → See skill `/version-prompt`

### 3. experiments.md
**Activates when**: Working on batch experiments or research evaluation

**Key topics**:
- Batch referee reports
- Persona selection consistency experiments
- Ground truth evaluation
- Cost estimation

**Common tasks**:
- Running batch experiment → See skill `/run-experiment`
- Analyzing results
- A/B testing prompts

### 4. prompts.md
**Activates when**: Working on prompt files or versioning

**Key topics**:
- Directory structure (subdirs vs. flat files)
- config.yaml management
- Version numbering (major vs. minor)
- PromptLoader system
- A/B testing

**Common tasks**:
- Creating new version → See skill `/version-prompt`
- Rollback procedure
- Testing prompts

### 5. utilities.md
**Activates when**: Working on shared utilities

**Key topics**:
- LLM call patterns (`single_query`, `safe_query`, `ConversationManager`)
- Caching system architecture
- Deduplication algorithm
- Quote validation
- PDF extraction (PyMuPDF)

**Common tasks**:
- Modifying cache logic
- Adding utility function
- Testing utilities → See skill `/test-changes`

## Available Skills

### /add-persona
**Add new persona to MAD system**

**When**: Creating a new reviewer type (e.g., Statistician, ML_Expert)

**Steps**: 7 steps from prompt creation to testing

**Time**: ~30 minutes

---

### /add-paper-type
**Add new paper type to section evaluator**

**When**: Supporting a new research domain (e.g., finance, macro)

**Steps**: 10 steps from criteria definition to testing

**Time**: ~1 hour

---

### /version-prompt
**Create new version of existing prompt**

**When**: Improving prompt clarity, fixing issues, A/B testing

**Steps**: 8 steps from copying to deployment

**Time**: ~15-30 minutes

---

### /test-changes
**Run appropriate tests for your changes**

**When**: After any code or prompt changes, before committing

**Quick reference**:
- Referee changes → `pytest tests/test_referee_quick.py`
- Section eval changes → `pytest tests/test_section_evaluator_prompts.py`
- Utility changes → `pytest tests/test_{utility}.py`
- Before commit → `pytest tests/`

---

### /run-experiment
**Run batch experiments with proper setup**

**When**: Evaluating on multiple papers, comparing to ground truth

**Types**:
- Batch referee reports (with ground truth)
- Persona selection consistency

**Steps**: 7 steps from preparation to analysis

**Time**: Variable (2-5 min per paper)

## Usage Examples

### Example 1: Adding a New Persona

**Context**: You want to add a "Statistician" persona to the referee system.

**Workflow**:
1. Invoke `/add-persona` skill
2. Follow the 7-step guide
3. `referee-system.md` rule provides additional context
4. Test with `/test-changes` skill

**Result**: New persona integrated, tested, and documented.

---

### Example 2: Improving a Prompt

**Context**: Theorist persona outputs are too vague.

**Workflow**:
1. Invoke `/version-prompt` skill
2. Create v1.1 with clearer criteria
3. Test with sample papers
4. Deploy or rollback based on results
5. `prompts.md` rule provides versioning context

**Result**: Improved prompt deployed, old version preserved for rollback.

---

### Example 3: Running Batch Evaluation

**Context**: Evaluate system on 20 papers with ground truth.

**Workflow**:
1. Invoke `/run-experiment` skill
2. Prepare papers and ground truth CSV
3. Test on 1-2 papers first
4. Run full batch
5. Analyze results with provided code
6. `experiments.md` rule provides additional context

**Result**: Batch complete, accuracy computed, findings documented.

## Benefits

### 1. Context-Aware Assistance
Claude Code provides relevant guidance based on what you're working on.

### 2. Consistency
Standard workflows ensure changes follow project conventions.

### 3. Knowledge Preservation
Captures recurring patterns and tribal knowledge.

### 4. Faster Onboarding
New contributors can follow structured guides.

### 5. Reduced Errors
Checklists and validation steps catch common mistakes.

## Maintenance

### When to Update Rules

**Add new section** when:
- New subsystem added
- Major architectural change
- New utility or pattern introduced

**Update existing section** when:
- API changes
- Common pitfall discovered
- Best practice evolved

### When to Create New Skill

**Create skill** when:
- Workflow done 3+ times
- Process has 5+ steps
- High error rate without guide
- Complex multi-file changes

### Version Control

Rules and skills are versioned with the codebase:
```bash
git add .claude/
git commit -m "docs: update referee-system rule for caching"
```

## Related Documentation

- **CLAUDE.md** (root) — High-level project guidance
- **app_system/README.md** — User-facing documentation
- **app_system/docs/changelog.md** — Change history
- **app_system/docs/*.md** — Technical deep dives

## Questions?

If you're unsure which rule or skill applies:
1. Ask Claude Code: "What's the best approach for [task]?"
2. Check this README for overview
3. Browse rules and skills directly

Claude Code will suggest relevant rules and skills based on your question.
