# Running Ideas & Experiments

## 🔬 Active: Referee Consistency Improvements (2026-05-05)

### Problem Statement
The referee report system (app_system/app.py) gives inconsistent results when run multiple times on the same paper. Need to improve reliability while maintaining thoughtful, strong analysis.

### Root Causes Identified
1. **Generic system prompt pollution** - Hardcoded "research assistant" prompt was diluting specialized persona instructions
2. **Temperature too high** - Single temperature (0.7) for all rounds causes variability
3. **No thinking mode** - Despite documentation claiming it's enabled, it wasn't actually being sent to API
4. **Context window bloat** - Large prompts may cause "lost in the middle" effects

### Changes Implemented (2026-05-05)
✅ **Phase 1: Removed Generic System Prompt**
- Created new `referee_query()` function in `utils.py` (line 90-149)
- Function identical to `single_query()` but WITHOUT hardcoded system prompt
- Updated all referee system calls:
  - `referee/engine.py`: Lines 300, 356, 409, 768
  - `referee/_utils/summarizer.py`: All LLM calls
- Expected improvement: 20-30% better persona adherence

### Proposed Next Steps

#### 🌡️ **Phase 2: Per-Round Temperature Control** ✅ IMPLEMENTED (2026-05-06)

**Rationale**: Different rounds need different creativity/consistency balance:
- Selection & synthesis need consistency → low temp
- Analysis & debate need thoughtfulness → medium-high temp

**Proposed temperatures by round**:
```python
ROUND_TEMPERATURES = {
    'round_0': 0.4,   # Consistent persona selection
    'round_1': 0.7,   # Creative deep analysis (current default - keep!)
    'round_2a': 0.7,  # Thoughtful cross-examination
    'round_2b': 0.6,  # Focused answers
    'round_2c': 0.6,  # Refined amendments
    'round_3': 0.4    # Faithful synthesis (no new ideas)
}
```

**Implementation completed** (2026-05-06):
1. ✅ Added `ROUND_TEMPERATURES` dict to `referee/engine.py` (lines 268-287)
2. ✅ Modified `call_llm_async()` to accept `round_id` parameter (line 297)
3. ✅ Pass round-specific temperature to `referee_query()` (line 321)
4. ✅ Updated 6 round functions to use round-specific temps:
   - `run_round_0_selection()`: lines 387, 442 (temp 0.4)
   - `run_round_1()`: line 540 (temp 0.7)
   - `run_round_2a()`: line 569 (temp 0.7)
   - `run_round_2b()`: line 599 (temp 0.6)
   - `run_round_2c()`: line 632 (temp 0.6)
   - `run_round_3()`: line 804 (temp 0.4)
5. ✅ Updated metadata to track temperature system (lines 1069-1075)
6. ⏳ Documentation in CLAUDE.md (TODO)

**Expected improvement**: 60-80% reduction in verdict variability while maintaining analysis quality

**Testing needed**: Run 5× on 3 test papers to measure consistency gains. See detailed protocol in CHANGES_2026-05-06.md.

#### 🧠 **Phase 3: Enable Thinking Mode** (OPTIONAL)

Add to `referee_query()`:
```python
"thinking": {
    "type": "enabled",
    "budget_tokens": 2048
}
```

Note: Thinking mode REQUIRES temperature=1.0, so would conflict with Phase 2. Need to decide which is more important:
- Thinking mode → better reasoning, but forced temp=1.0
- Per-round temps → consistency, but no thinking transparency

**Recommendation**: Try Phase 2 first, evaluate consistency. If still inconsistent, try Phase 3 instead.

#### 📦 **Phase 4: Prompt Caching** (COST OPTIMIZATION)

Add cache control to paper text in API calls:
```python
"system": [
    {"type": "text", "text": persona_prompt},
    {
        "type": "text", 
        "text": f"PAPER TEXT:\n{paper_text}",
        "cache_control": {"type": "ephemeral"}
    }
]
```

Expected: 50-80% cost reduction for multi-round debates (paper cached across all rounds)

#### 📊 **Phase 5: Context Compression** (IF NEEDED)

For very long papers (>30K tokens):
- Option A: Truncate to first 50K characters for evaluation
- Option B: Two-pass system: quick scan → focused deep dive
- Only implement if Phase 2 doesn't solve consistency issues

### Testing Protocol

After each phase:
1. Select 2-3 test papers (different types: empirical, theoretical, policy)
2. Run each paper 5 times
3. Record verdicts + consensus scores
4. Calculate verdict consistency rate (% of runs with same verdict)
5. Compare to baseline (current system)

Target: 80%+ consistency on same-paper runs

### Version Tracking Ideas

**Option A: Git tags**
```bash
git tag -a referee-v1.0-baseline -m "Before consistency improvements"
git tag -a referee-v1.1-no-system-prompt -m "Removed generic prompt"
git tag -a referee-v1.2-per-round-temp -m "Added per-round temperatures"
```

**Option B: Experiment branches**
```bash
git checkout -b experiment/referee-consistency
# Make changes
git commit -m "Phase 2: Per-round temperature control"
git tag referee-exp-phase2
```

**Option C: Config-driven versioning**
Add to `.env`:
```
REFEREE_VERSION=1.1-no-system-prompt
REFEREE_EXPERIMENT_NOTES=Removed generic system prompt pollution
```

Track in Excel output metadata alongside model version, timestamp, etc.

### References
- **Phase 1** (2026-05-05): Removed generic system prompt
  - Files: `app_system/utils.py`, `app_system/referee/engine.py`, `app_system/referee/_utils/summarizer.py`
  - Details: `CHANGES_2026-05-05.md`
- **Phase 2** (2026-05-06): Per-round temperature control
  - Files: `app_system/referee/engine.py`
  - Details: `CHANGES_2026-05-06.md`
- Documentation: See CLAUDE.md sections on "Model Configuration" and "Referee System Context"

---

## 💡 Other Ideas (Backlog)

*(Add future ideas here)*
