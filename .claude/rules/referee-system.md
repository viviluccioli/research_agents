# Referee System Context (Multi-Agent Debate)

This rule activates when working on the referee report system (MAD architecture).

## File Scope
- `app_system/referee/`
- `app_system/prompts/multi_agent_debate/`
- `app_system/app.py` (Tab 1)
- `experiment/batch_referee_reports.py`

## Key Architecture

**5-Round Debate Structure**:
1. Round 0: LLM selects 3 of 5 personas + assigns weights
2. Round 1: All personas write independent reports
3. Round 2A: Cross-examination (questions for peers)
4. Round 2B: Direct examination (answers)
5. Round 2C: Final amendments based on discussion
6. Round 3: Editor synthesizes weighted consensus

**Parallel Execution**: Rounds 1, 2A, 2B, 2C use `asyncio.gather()` to run personas in parallel.

**Context Isolation**: Each round receives only appropriate context (not full debate history).

## Key Files

- **`engine.py`**: Core debate orchestration, 5 base personas
- **`engine_exp_4.py`**: 10-persona variant (experimental)
- **`workflow.py`**: Production UI with summarization
- **`memo_engine.py`**: Memo evaluation variant (5 memo-specific analysts)
- **`_utils/`**: Internal utilities (cache, dedup, quote validation, PDF extraction)

## Adding a New Persona

**CRITICAL**: Must update 4 locations:

1. **System Prompt** (`engine.py:SYSTEM_PROMPTS`):
   ```python
   SYSTEM_PROMPTS["NewPersona"] = load_persona_prompt("new_persona") or """
   fallback hardcoded prompt...
   {_ERROR_SEVERITY_GUIDE}  # MUST include
   """
   ```

2. **Selection Prompt** (`engine.py:SELECTION_PROMPT`):
   - Add to persona list with description
   - Ensure still selects exactly 3

3. **UI Styling** (`workflow.py`):
   - Add CSS class (`.persona-newpersona`)
   - Add icon to persona card display
   - Update manual selection interface

4. **Prompt File** (`prompts/multi_agent_debate/personas/new_persona/v1.0.txt`):
   - Create subdirectory + versioned file
   - Update `config.yaml` if using PromptLoader

## Prompt Versioning

All prompts use subdirectory structure:
```
prompts/multi_agent_debate/
├── personas/theorist/v1.0.txt
├── debate_rounds/round_2a_cross_exam/v1.0.txt
└── additional_context/error_severity/v1.0.txt
```

Version changes require updating `config.yaml`.

## Testing

Run these after changes:
```bash
cd app_system
python -m pytest tests/test_referee_quick.py  # Fast smoke test
python -m pytest tests/test_consensus_calculation.py
python -m pytest tests/test_exp4_personas.py  # If working on exp_4
```

## Utilities Context

**Caching** (`_utils/cache.py`):
- Per-round granularity: `{cache_key}/round_{0,1,2a,2b,2c,3}.json`
- Cache key from SHA256(paper_text + personas + weights + model)
- UI checkbox enables/disables

**Deduplication** (`_utils/deduplicator.py`):
- Runs after Round 1 and 2C
- Uses quote overlap + semantic similarity + keyword matching
- Enable via `.env`: `ENABLE_DEDUPLICATION=true`

**Quote Validation** (`_utils/quote_validator.py`):
- Validates after Round 1 and 2C
- Fuzzy matching: 95% for math, 85% for prose
- Disable via `.env`: `DISABLE_QUOTE_VALIDATION=true`

**PDF Extraction** (`_utils/pdf_extractor_v2.py`):
- PyMuPDF-based with figure/table extraction
- Falls back to pdfplumber if unavailable
- Returns `ExtractedContent` with text + figures

## Model Configuration

**CRITICAL**: ALL referee calls use **Claude 4.5 Sonnet** via `MODEL_PRIMARY`.

- `single_query()`: temperature=1, thinking enabled (2048 tokens)
- Retries 3× with 5s delay on failure
- Token counting via `tiktoken` (cl100k_base)

## Common Pitfalls

❌ **Don't** modify `_archived/` — it's deprecated code
❌ **Don't** bypass `ConversationManager` — referee uses stateless `single_query()`
❌ **Don't** forget `_ERROR_SEVERITY_GUIDE` in persona prompts
❌ **Don't** hardcode prompts — use `load_persona_prompt()` or YAML config
✅ **Do** test with both auto and manual persona selection
✅ **Do** verify quote validation results in Excel output
✅ **Do** check cache invalidation when changing prompts

## Related Systems

- **Experiment 4**: 10-persona system in `engine_exp_4.py` + `app_exp_4.py`
- **Memo System**: Policy memo evaluation in `memo_engine.py` + `app-memo.py`
- **Batch Processing**: `experiment/batch_referee_reports.py` for ground truth evaluation
