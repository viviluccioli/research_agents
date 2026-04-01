# Changelog — app_system

## 04/01 (Continued)

### Refactor: Documentation consolidation
**Removed**
- Deleted 8 redundant/outdated documentation files:
  - `docs/API_CONFIGURATION.md` (redundant with CLAUDE.md)
  - `docs/architecture.md` (1476 lines, overlapped with CLAUDE.md)
  - `docs/REORGANIZATION.md` (outdated historical document)
  - `docs/RESTRUCTURING_SUMMARY.md` (outdated historical document)
  - `docs/PROMPT_MANAGEMENT.md` (documents unused versioning system)
  - `docs/PROMPT_QUICKREF.md` (documents unused versioning system)
  - `docs/PROMPT_SYSTEM_COMPLETE.md` (documents unused versioning system)
  - `referee/REFEREE_PACKAGE_STRUCTURE.md` (redundant with CLAUDE.md)
- Deleted 2 misleading README files:
  - `prompts/multi_agent_debate/README.md` (documents unused external prompt system)
  - `prompts/section_evaluator/README.md` (documents unused external prompt system)

**Changed**
- Updated CLAUDE.md to de-emphasize documentation creation
- Relaxed changelog update requirement to major changes only
- Updated file organization guidelines to discourage new .md files

**Result**
- Reduced from 15+ markdown files to 7 essential docs
- Docs directory reduced from 132K to 71K
- Removed misleading documentation about unused features
- Clearer guidance to avoid over-documentation

## 04/01

### Feature: Enhanced PDF extraction with table support
**Added**
- Table extraction from PDFs using `pdfplumber.extract_tables()`
- Tables are formatted as markdown and included in text sent to LLM
- Each table labeled with sequential numbering and page location
- Extraction diagnostics displayed in UI:
  - Total pages processed
  - Number of tables extracted
  - Total character count
- Warning message if no tables detected (suggests using LaTeX source instead)
- Helper method `_format_table_as_markdown()` for clean table formatting
- Page break markers in extracted text for better structure

**Changed**
- `referee/workflow.py:extract_text_from_pdf()` now returns text with embedded tables
- PDF extraction now provides real-time feedback to users

**Benefits**
- Empiricist persona can now see regression tables and statistical results
- All personas have access to tabular data (summary statistics, robustness checks, etc.)
- Users can assess extraction quality before running full debate
- Significantly improves evaluation reliability for empirical papers

### Feature: Cost tracking and token usage analytics
**Added**
- `calculate_token_usage_and_cost()` function in `referee/engine.py`
- Comprehensive token counting for all debate rounds and summarization calls
- Cost estimation based on Claude 3.7 Sonnet pricing ($3/1M input, $15/1M output)
- Token usage metadata added to debate results:
  - Paper token count
  - Input/output tokens split by debate vs. summarization
  - Total token count
  - LLM call counts (debate vs. summarization)
  - Detailed cost breakdown (input, output, total)
  - Pricing information for transparency
- Prominent cost display at top of results: "💰 Estimated Cost: $X.XX (27 LLM calls, X tokens)"
- Detailed cost breakdown in metadata section with two-column layout:
  - Token breakdown (paper, input, output, by category)
  - Cost breakdown (pricing, costs, LLM calls)
- New "Cost Tracking" sheet in Excel export with full token and cost metrics

**Changed**
- `referee/engine.py:execute_debate_pipeline()` now calculates and includes token/cost data in metadata
- `referee/workflow.py:display_debate_results()` displays cost prominently
- `referee/workflow.py:create_excel_export()` includes cost tracking sheet
- Added `count_tokens` import to `referee/engine.py`

**Benefits**
- Users can estimate costs before running large batches of papers
- Full transparency on token usage and LLM API costs
- Excel export enables cost tracking across experiments
- Helps with budget planning and cost optimization
- Token counts useful for debugging and optimization

### Tool: PDF extraction testing utility
**Added**
- `tests/test_pdf_extraction.py` - standalone tool for testing PDF extraction quality
- Command-line interface: `python test_pdf_extraction.py <pdf_path>`
- Features:
  - Extracts text and tables from any PDF
  - Shows per-page statistics (characters, table count)
  - Displays extraction summary (pages, tables, total chars, estimated tokens)
  - Lists all detected tables with row/column counts
  - Optional verbose mode (`-v`) shows sample table contents
  - Saves full extracted text to file for manual inspection
  - Quality assessment with color-coded warnings:
    - 🔴 NO TABLES: Critical for empirical papers
    - 🟡 FEW TABLES / SHORT TEXT: Potential issues
    - 🟢 LOOKS REASONABLE: Good extraction
- Executable script with comprehensive help text

**Benefits**
- Users can test PDF quality before running expensive debate pipeline
- Helps diagnose extraction issues (missing tables, garbled text)
- Shows exactly what the LLM receives
- Enables quality control for paper corpus
- Useful for comparing PDFs vs. LaTeX source extraction

### Documentation
**Added**
- `docs/PDF_PROCESSING_AND_COSTS.md` - comprehensive analysis document covering:
  - Current PDF extraction implementation and limitations
  - Reliability assessment for different paper types
  - Complete LLM call breakdown (27 calls per run)
  - Token usage estimation methodology
  - Cost calculation details with pricing
  - Testing recommendations and scripts
  - Immediate, medium-term, and long-term improvement suggestions

## 03/30

### Refactor: Referee package restructure for clarity
**Changed**
- Restructured `referee/` package to clearly separate production code from utilities and archived code
- Renamed files for clarity:
  - `summarized.py` → `workflow.py` (main production UI)
  - `debate.py` → `engine.py` (orchestration engine)
  - `core.py` → `_archived/full_output_ui.py` (archived alternate UI)
  - `summarizer.py` → `_utils/summarizer.py` (internal utility)
- Created `_utils/` subdirectory for internal helpers (underscore indicates internal)
- Created `_archived/` subdirectory for alternate implementations (underscore indicates not main)
- Updated all imports throughout codebase
- Renamed main class: `RefereeReportCheckerSummarized` → `RefereeWorkflow`

**Added**
- Backward compatibility aliases in `referee/__init__.py`
- `docs/REFEREE_PACKAGE_STRUCTURE.md` documenting new structure
- Clear docstrings indicating production vs. archived status

**Benefits**
- Only 2 files at top level → immediately clear what's production code
- Underscore prefix convention signals "internal" or "archived"
- Colleagues can immediately identify main code path
- Professional structure following Python best practices
- No breaking changes (old imports still work via aliases)

### Feature: Experiment tracking and versioning system
**Added**
- Prompt version tracking in debate metadata — automatically loads and records all prompt versions from `prompts/multi_agent_debate/config.yaml`
- Excel export functionality for experiment comparison with three sheets:
  - Configuration sheet: model settings, temperature, thinking mode, prompt versions
  - Results sheet: personas, weights, Round 1 verdicts, Round 2C verdicts, verdict changes
  - Consensus sheet: weighted score, final decision, justification
- New download button: "Download Excel (Experiment Tracking)"
- Enhanced metadata to include prompt versions for all personas and debate rounds

**Benefits**
- Enables systematic experimentation with different prompt versions and model configurations
- Easy comparison across multiple runs on the same paper
- Full traceability of all configuration parameters
- Structured data export for analysis and benchmarking

### UI: Persona display reorganization
**Changed**
- Moved persona descriptions into "How This System Works" expander
- Redesigned persona cards to display horizontally (5 side-by-side)
- Added beautiful gradient backgrounds to persona cards
- More compact, professional appearance
- Removed experimental/summarized language from UI text

**Fixed**
- Changed UI title from "Multi-Agent Referee Evaluation (Summarized)" to "Multi-Agent Referee Evaluation"
- Changed info message from "EXPERIMENTAL" to "NOTE"

### Refactor: Complete directory reorganization
**Changed**
- Reorganized entire `app_system/` directory to follow Python best practices and improve maintainability
- Created `referee/` package consolidating all referee-related modules:
  - `referee/core.py` (formerly `referee.py`) — Full output referee report
  - `referee/summarized.py` (formerly `referee_summarized.py`) — Summarized referee report
  - `referee/debate.py` (formerly `multi_agent_debate.py`) — Multi-agent debate orchestration
  - `referee/summarizer.py` (formerly `debate_summarizer.py`) — LLM summarization utilities
- Created `tests/` directory for all test files (moved 5 test files)
- Moved alternate app entry points to `demos/`:
  - `demos/app_full_output.py` (formerly `full_output_app.py`)
  - `demos/app_summarized_only.py` (formerly `app_summary.py`)
- Updated all import statements throughout codebase to use new package structure
- Updated `CLAUDE.md` with comprehensive file organization rules and guidelines
- Created `docs/REORGANIZATION.md` documenting the complete restructuring

**Added**
- `referee/__init__.py` — Package initialization exposing main classes and functions
- `tests/__init__.py` — Test package initialization
- File organization rules in `CLAUDE.md` specifying where different file types should be placed

**Benefits**
- Clean root directory (only 4 essential files: app.py, utils.py, README.md, run_app.sh)
- Standard Python package structure with proper `__init__.py` files
- Clear separation of concerns (tests, demos, docs, packages)
- Better code discoverability and maintainability
- Unified imports: `from referee import RefereeReportChecker, execute_debate_pipeline`

**Breaking Changes**
- Import paths changed for referee functionality (old direct imports from `multi_agent_debate` no longer work)
- All referee classes now imported from unified `referee` package
- Demo apps moved to `demos/` subdirectory

---

## 03/26

### Feature: Official letter format for editor decision

#### Changed
- **Round_3_Editor prompt** in `multi_agent_debate.py`: Restructured output format to match formal referee letter style. New structure includes:
  - **Weight Calculation**: Explicit mathematical breakdown showing each persona's verdict, weight, and weighted contribution
  - **Debate Synthesis**: 2-3 sentences summarizing panel alignment
  - **Final Decision**: The computed decision (unchanged)
  - **Official Referee Report**: Formal letter to authors in prose paragraph form (replaces bullet-pointed **Strengths** and **Critical Issues** sections)
- **Letter format requirement**: Editor now writes the referee report as a formal academic letter with complete paragraphs, not bullet points. The report discusses panel assessment in narrative form, covering strengths and concerns naturally within the letter structure.

#### Rationale
- Matches standard academic journal editorial letter format
- Provides more cohesive, readable feedback to authors
- Maintains all technical detail (weight calculation, debate synthesis) while presenting final report in professional letter style
- Inspired by madexp2.py reference implementation

---

### Fix: LLM output truncation causing incomplete responses

#### Changed
- **Max tokens parameter** in `utils.py`: Added `max_tokens` parameter to `single_query()` function (default 4096, configurable per call). Allows different rounds to use appropriate output limits based on complexity.
- **Round 2C max tokens** in `multi_agent_debate.py`: Increased from 4096 to 6144 tokens. Round 2C personas receive full debate transcript (10,000+ input tokens) and need space to write complete final verdicts. Removed paper_text from Round 2C input (already included in transcript) to reduce input token usage.
- **Round 3 editor max tokens** in `multi_agent_debate.py`: Increased from 4096 to 8192 tokens. Editor receives all Round 2C reports and needs to write comprehensive referee report with evidence.
- **Metadata tracking** in `multi_agent_debate.py`: Changed from single `max_tokens` to separate `max_tokens_round_0_1_2a_2b` (4096), `max_tokens_round_2c` (6144), `max_tokens_round_3_editor` (8192) to show per-round configuration.

#### Fixed
- **Round 2C verdict hallucination**: Fixed issue where personas would write "upgrading my verdict from REVISE to FAIL" but then output would be cut off at 4096 tokens before writing "**Final Verdict:** FAIL", causing verdict extraction to fail and show stale "REVISE" verdict. Now with 6144 tokens, complete responses are captured.
- **Editor report truncation**: Fixed issue where editor report would cut off mid-sentence after "Critical Issues Requiring Revision" heading due to 4096 token limit. Now with 8192 tokens, full report displays correctly.
- **HTML rendering in Round 2C and 2A**: Fixed issue where severity labels (`<span class="severity-fatal">⚠️ FATAL</span>`) were displaying as raw HTML tags instead of rendering. Added `unsafe_allow_html=True` to all `st.markdown()` calls that display formatted text with severity labels (Round 2C sections, Round 2A insights, Editor report).

#### Removed
- **UI truncation limits** in `referee.py`: Removed ALL display limits:
  - Character limits: 200 chars for Round 2C, 250 chars for Round 2A, 300 chars for editor → **No character limits**
  - Section limits: Round 2C limited to 3 sections, Editor limited to 8 sections → **No section limits**
  - Now displays complete, untruncated content for all rounds. Fixes issue where important content was hidden behind "...see full report for details" messages or silently cut off after N sections.

---

### Feature: Deterministic weighted consensus calculation

#### Changed
- **Round 3 decision-making** in `multi_agent_debate.py`: Completely restructured to compute weighted consensus BEFORE calling the LLM. Added `compute_weighted_consensus()` function that deterministically extracts verdicts from Round 2C reports, calculates weighted score (PASS=1.0, REVISE=0.5, FAIL/REJECT=0.0), and applies decision thresholds (>0.75=ACCEPT, 0.40-0.75=REJECT & RESUBMIT, <0.40=REJECT). The editor LLM now receives the computed score and decision as fixed inputs and writes justification only.
- **Editor prompt** in `multi_agent_debate.py`: Rewritten `Round_3_Editor` prompt to receive `{computed_score}`, `{computed_decision}`, and `{calculation_details}` as fixed inputs. Editor's task changed from "calculate consensus" to "write report justifying the already-computed decision." Removes LLM discretion from mathematical decision-making.
- **Results structure** in `multi_agent_debate.py`: `run_round_3()` now returns `{'consensus': {...}, 'editor_report': '...'}` instead of just the editor text. `execute_debate_pipeline()` stores both `results['consensus']` (with verdicts, weighted_score, decision, calculation_details) and `results['final_decision']` (editor report text).

#### Added
- **Consensus display** in `referee.py`: Added visual "🔢 Deterministic Weighted Consensus" section showing individual verdicts with weights and contributions, consensus score metric, decision thresholds, and computed decision in highlighted box. Displayed before editor's report to emphasize that decision is mathematical, not LLM-generated.
- **Consensus in downloads** in `referee.py`: Full report download now includes "ROUND 3: WEIGHTED CONSENSUS CALCULATION" section with breakdown of individual verdicts, contributions, thresholds, and computed decision before the editor's report.

#### Fixed
- **Critical bug: Editor decision hallucination** in `referee.py`: Fixed bug where editor would correctly calculate consensus score (e.g., 0.175) and correctly compute decision (REJECT), but then final decision extraction would fail and display "ACCEPT" or "UNKNOWN" due to LLM formatting issues. Now decision is ALWAYS taken from `results['consensus']['decision']` (computed deterministically), never from parsing editor's text. Added note in UI: "(Decision computed deterministically from weighted consensus, not extracted from text)".
- **Decision reliability**: Final decision is now 100% deterministic and reproducible based on persona verdicts and weights. Eliminates cases where LLM might deviate from mathematical consensus or use "editorial discretion" to override the formula.
- **Verdict extraction failures**: Added warning in `compute_weighted_consensus()` when verdict cannot be extracted from Round 2C report. Treats UNKNOWN verdicts as FAIL (0.0) conservatively and displays "⚠️ VERDICT NOT FOUND" in calculation details.

---

### Fix: HTML rendering and verdict extraction bugs

#### Fixed
- **HTML table rendering** in `referee.py`: Replaced unreliable HTML `<table>` with Pandas DataFrame using `st.dataframe()`. Fixes issue where raw HTML tags (`<tr>`, `<td>`) were displaying as text instead of rendering properly. DataFrame approach is more robust and handles Streamlit's rendering pipeline better.
- **Double colon display** in `referee.py`: Fixed bug where section headers showed double colons (e.g., "Insights Absorbed::") by stripping trailing colons from parsed headers before adding display colon. Applied to both Round 2C and Round 3 (Editor) section displays.
- **Verdict extraction robustness** in `referee.py`: Enhanced `extract_verdict()` regex patterns to handle multiple colon formats (`::` vs `:`), markdown variations, and added fallback pattern to find standalone verdict words anywhere in text. Fixes "Unable to extract verdict" warnings when LLM uses slightly different formatting.

---

### Feature: Execution metadata tracking and display

#### Added
- **Metadata tracking** in `multi_agent_debate.py`: `execute_debate_pipeline()` now tracks execution metadata including model version, start/end timestamps, total runtime, temperature, thinking mode status, max tokens, and retry configuration. Stored in `results['metadata']` dictionary.
- **Metadata display** in `referee.py`: Added "⚙️ Execution Metadata" section at bottom of results showing two-column layout with Model Configuration (model version, temperature, max tokens, thinking mode, retries) and Execution Time (start time, end time, formatted runtime). Enables version control and reproducibility tracking.
- **Max token limit** in `utils.py`: Added explicit `max_tokens: 4096` parameter to `single_query()` to prevent excessively long responses that could cause performance issues.

#### Changed
- **Thinking mode disabled** in `utils.py`: Removed extended thinking mode from both `single_query()` and `ConversationManager.conv_query()`. Thinking was causing unpredictable response quality and requiring temperature=1. Now uses standard generation.
- **Temperature reduced** in `utils.py`: Changed temperature from 1.0 to 0.7 in both LLM call functions for more consistent, focused outputs. Temperature=1 was only required when thinking mode was enabled.

---

### Fix: Round 2C blank outputs and HTML rendering errors

#### Changed
- **Round_2C_Final_Amendment prompt** in `multi_agent_debate.py`: Completely restructured prompt with much more detailed guidance. Added explicit context about what the agent has seen, clear rules for what to include, and mandatory 4-section output format (Insights Absorbed, Changes to Original Assessment, Final Verdict, Final Rationale). Added critical instruction: "You MUST provide all four sections above. Do not skip any section." Prevents LLM from returning blank or poorly formatted responses.
- **Verdict color coding** in `referee.py`: Fixed `generate_summary_table()` to use `format_verdict()` helper function instead of raw CSS classes. Adds proper handling for UNKNOWN verdicts (displays in gray) to prevent CSS class errors when Round 2C outputs are missing.
- **Round 2C display** in `referee.py`: Added blank output detection with user-facing warning: "⚠️ {role} did not provide a Round 2C response. This may indicate an LLM error." Also warns if verdict extraction fails. Prevents silent failures and helps diagnose LLM issues.

#### Added
- **CSS class for unknown verdicts** in `referee.py`: Added `.verdict-unknown` style (gray background, dark gray text) to handle cases where verdict extraction fails. Prevents raw HTML from displaying when CSS classes don't exist.
- **Visual architecture displays** in `app.py`: Added `_render_section_evaluator_architecture()` and `_render_referee_report_architecture()` functions that display beautiful numbered step boxes with gradient backgrounds, color-coded paper type comparison cards, and clear visual hierarchy. Replaces old ASCII-art diagrams. Architecture now shown at top of each tab.
- **Paper type selection UI** in `app.py`: Added prominent button-based paper type selector with color-coded display boxes (blue for Empirical, green for Theoretical, red for Policy). Paper type now selected globally at top of page and passed to Section Evaluator workflow.

#### Removed
- **Old architecture displays** in `section_eval/main.py` and `referee.py`: Removed duplicate ASCII-art architecture diagrams. Single beautiful source of truth now in main `app.py`.

#### Fixed
- **HTML table rendering** in `referee.py`: Fixed raw HTML display issue by properly handling UNKNOWN verdicts and using existing `format_verdict()` helper. Table now renders correctly even when Round 2C outputs are blank or malformed.

---

## 03/25

### UI: Major referee report display improvements

#### Changed
- **Severity labels** in `referee.py`: Replaced bracketed text `[MAJOR]`/`[MINOR]`/`[FATAL]` with styled badges featuring emoji icons (🔴 MAJOR in dark red, 🟠 MINOR in orange, ⚠️ FATAL in white-on-red) via new `format_severity_labels()` function. Applied throughout all round displays.
- **Verdict formatting** in `referee.py`: All verdicts (PASS/REVISE/REJECT/FAIL) now display with color-coded backgrounds (✅ PASS in light green, ⚠️ REVISE in light yellow, ❌ REJECT in light red) using `format_verdict()` function. Consistent across all rounds and personas.
- **Final editor decision** in `referee.py`: Editor's final verdict (Round 3) now displayed with larger font (28px), thicker border, and more prominent styling via `format_final_verdict()`. Fixes inconsistent presentation bug where third persona's verdict appeared in different font size.
- **Round 2C display** in `referee.py`: Final verdicts for all three personas now displayed in consistent format using columns, ensuring uniform font size and styling across all personas.
- **UI text density**: Reduced verbose output in UI display. Round 1 and Round 2C now show concise summaries with "see full report for details" notes for content exceeding character limits. Full detailed text preserved in downloadable reports.

#### Added
- **System prompt transparency** in `referee.py`: Added collapsible expanders for each round displaying the exact system prompts and debate prompts used to guide that round. Appears at start of each round section for transparency.
- **Summary table download** in `referee.py`: Added `generate_summary_table()` function that creates color-coded HTML table showing persona weights, Round 1 verdicts, Final (Round 2C) verdicts, whether opinions changed, and final editorial decision. Table displayed at end of results and available as separate markdown download.
- **Concede/Defend emphasis** in `referee.py`: Round 2B responses now parse and highlight CONCEDE vs DEFEND positions with colored boxes (⚠️ yellow for concessions, 🛡️ green for defenses). Each response to a peer now includes explicit position statement.
- **Helper functions** in `referee.py`: New utility functions `format_severity_labels()`, `format_verdict()`, `format_final_verdict()`, `extract_verdict()`, `summarize_text()`, `format_round1_output()`, `format_round2c_output()`, and `generate_summary_table()` for consistent formatting across all displays.

#### Changed (Prompts)
- **`SYSTEM_PROMPTS`** in `multi_agent_debate.py`: Updated all five persona prompts (Theorist, Empiricist, Historian, Visionary, Policymaker) to enforce standardized output structure. Added explicit instruction: "CRITICAL: Place source evidence IMMEDIATELY under each finding, not in a separate section." Ensures consistent format where each severity-labeled finding is followed directly by its supporting quote.
- **`DEBATE_PROMPTS["Round_2A_Cross_Examination"]`** in `multi_agent_debate.py`: Enforced mandatory three-section structure (Cross-Domain Insights, Constructive Pushback, Clarification Requests) with explicit instructions for each section. Added critical instruction: "Do NOT re-list your findings with severity labels here. That was Round 1. This round is about synthesis and dialogue." Prevents personas from incorrectly including Round 1 content in Round 2A.
- **`DEBATE_PROMPTS["Round_2B_Direct_Examination"]`** in `multi_agent_debate.py`: Restructured output format to require explicit **Position**: [CONCEDE or DEFEND] statement immediately after each peer response, eliminating ambiguity about whether persona is conceding or defending to specific arguments.

#### Fixed
- **Round 2A display consistency** in `referee.py`: Added severity label and verdict formatting to Round 2A display (was only applied in Round 1). Created `format_round2a_output()` helper function that filters out incorrectly placed Round 1 content (findings, verdicts), applies severity styling, and enforces standardized three-section structure. Displays warning if non-standard format detected. Added character limits (250 chars per section) for UI conciseness while preserving full text in downloadable report.

---

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
