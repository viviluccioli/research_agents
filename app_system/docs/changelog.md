# Changelog — app_system

## 03/20

### Feature: CSV/tabular download for benchmarking

#### Added
- **`_build_csv_report()`** in `section_eval/main.py`: Generates a flat CSV with one row per criterion across all evaluated sections. Columns include section name, raw/adjusted scores, importance multiplier, criterion name, criterion score, weight, weighted contribution, justification, both quotes + validity flags, fatal-flaw columns, paper type, overall score, publication readiness, eval timestamp, and model version. Enables tracking scores over time and comparing across papers.
- Download button added to results UI (3-column layout alongside existing MD and PDF buttons).

---

### Feature: Model version + timestamp logging per evaluation

#### Changed
- **`SectionEvaluator.evaluate_section()`** in `section_eval/evaluator.py`: Records `eval_timestamp` (UTC ISO-8601) and `model_version` (from `utils.model_selection`) immediately before each LLM call. Both fields are stored in the returned result dict and persisted in the session-state cache.
- **`_fallback_result()`**: Now accepts and propagates `eval_timestamp` and `model_version` so fallback results are also traceable.

---

### Feature: Fatal-flaw floor on section scoring

#### Added
- **`FATAL_FLAW_SCORE_THRESHOLD = 1.5`** and **`FATAL_FLAW_SCORE_CAP = 2.5`** constants in `section_eval/criteria/base.py`.
- **`critical: True`** flag on the following criteria (criteria where a score ≤ 1.5 invalidates the section regardless of other scores):
  - `data.appropriateness`, `methodology.identification`, `results.alignment` (empirical)
  - `model_setup.assumptions`, `proofs.correctness`, `proofs.rigor` (theoretical)
  - `identification_strategy.instrument_validity`, `identification_strategy.exclusion` (finance)
- **`is_fatal_criterion`** field now propagated through `_sanitize_criteria_evals()` and `_fallback_result()` in `evaluator.py`.

#### Changed
- **`compute_section_score()`** in `section_eval/scoring.py`: If any critical criterion scores ≤ `FATAL_FLAW_SCORE_THRESHOLD`, `raw_score` is capped at `FATAL_FLAW_SCORE_CAP`. Returns two new fields: `fatal_flaw_triggered` (bool) and `fatal_flaw_criteria` (list of criterion names).
- **`_render_results()`** in `section_eval/main.py`: Shows a red error banner when `fatal_flaw_triggered=True`, naming the offending criteria. Section expander label gains a `⚠️ FATAL FLAW` suffix. Critical criteria are labeled with a red `CRITICAL` badge in the breakdown.

---

### Feature: Error severity labels in referee persona prompts

#### Changed
- **`SYSTEM_PROMPTS`** in `multi_agent_debate.py`: Added a shared `_ERROR_SEVERITY_GUIDE` block (FATAL / MAJOR / MINOR taxonomy with concrete examples per category) injected into every persona's system prompt. Each persona's output format now includes a **Severity-Labeled Findings** section requiring `[FATAL]`, `[MAJOR]`, or `[MINOR]` labels on every identified flaw. Verdict must be explicitly consistent with severity labels (any FATAL → FAIL; 2+ MAJOR → REVISE; only MINOR → PASS). Persona-specific examples of what constitutes each severity level added for Theorist, Empiricist, Historian, Visionary, and Policymaker.
