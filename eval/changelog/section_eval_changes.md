Cumulative changelog of **eval/section_eval_llm_vivi.py** → **eval/section_eval/** (package)

---

## 03/03

### Fix: criteria "(Not evaluated)" for long sections due to LLM history pruning

#### Changed
- **`section_eval/utils.py`** — **`safe_query()`**: Rewrote to bypass `ConversationManager.conv_query()` entirely. Now makes a direct stateless API call using `model_selection` (strong model) at temperature 0.3. This prevents long evaluation prompts from being silently truncated by the 8000-token conversation history budget. Falls back to `single_query` on error.
- **`prompts/templates.py`**: Raised section text cap from 12,000 to 20,000 characters so long sections (e.g. Introduction) are not truncated before being sent to the LLM.

---

### Feature: "figures in appendix" checkbox to avoid penalizing external tables/figures

#### Added
- **`main.py`**: Checkbox "Tables and figures are in the appendix or not embedded in the uploaded text" appears after paper type selection. Stored in session state, passed through to both auto-detect and freeform evaluate calls.
- **`evaluator.py`**: `evaluate_section()` accepts `figures_external: bool`. Cache key includes the flag. When `True`, calls `_adjust_criteria_for_external_figures()` before building the prompt.
- **`evaluator.py`** — **`_adjust_criteria_for_external_figures()`**: Drops `presentation` criterion and redistributes its weight proportionally to remaining criteria. No-ops if section has no `presentation` criterion.
- **`prompts/templates.py`**: `build_evaluation_prompt()` accepts `figures_external: bool`. When `True`, injects a note instructing the evaluator not to penalize for absent inline figures/tables.

#### Future
- Allow uploading appendix PDF or individual chart images for inclusion in evaluation context.

---

### Fix: PDF words glued together (missing spaces) degrading LLM evaluation quality

#### Changed
- **`text_extraction.py`**: Rewrote `extract_text_from_pdf()` to detect per-page space deficiency and fall back to `page.extract_words()` (bounding-box reconstruction) when `extract_text()` produces concatenated words. Added `_page_has_missing_spaces()` (space/non-space ratio heuristic < 0.05) and `_extract_page_text()` (per-page dispatch). `_fix_missing_spaces()` retained as secondary pass for residual camelCase gluing.

---

### Fix: missing `import re` in criteria/base.py causing NameError on load

#### Fixed
- **`criteria/base.py`**: Added missing `import re` — `_PREFIX_RE = re.compile(...)` was defined but `re` was never imported, causing a `NameError` at module load time. The section-type alias matching fix from the previous entry was silently failing because the module could not initialize.

---

### Fix: section text extraction and render bugs in main.py

#### Fixed

- **`_apply_merges`**: Condition `action == "keep" or action not in ("remove",) and not action.startswith("merge")` was incorrect due to Python operator precedence — sections with `"merge → X"` actions were being included as independent sections. Replaced with explicit `action != "remove" and not action.startswith("merge → ")`.
- **Preview cache not cleared on rescan**: `se_v3_preview_{manuscript}` was never invalidated, so stale section previews were shown after rescanning the same file. Fixed by clearing preview and hierarchy caches inside the "Scan for Sections" button handler.
- **`group_subsections` called on every Streamlit re-render**: The hierarchy LLM call ran on every page interaction. Now cached under `se_v3_hierarchy_{manuscript}` after first computation, cleared on rescan.
- **Section text preview was a 300-char caption snippet**: Replaced with the full preview UI from the old script — `text_area` with word/char count, subsection text appended to parent preview, "Show full text" / "Show less" toggle for sections over 3000 chars, and a warning when no text could be extracted. Section label now shows type annotation (e.g. `Introduction (introduction)`).

---

### Refactor: modular section_eval/ package with paper-type-aware evaluation

Replaced the single `section_eval_llm_vivi.py` file (~1300 lines) with a modular `eval/section_eval/` subdirectory. All original functionality is preserved; new features (paper type selection, weighted criteria, quote extraction) are added.

#### Added

- **`eval/section_eval/`**: New package directory replacing the monolithic file.
- **`text_extraction.py`**: Houses `extract_text_from_pdf()`, `strip_latex()`, `decode_file()`, `_fix_missing_spaces()` — migrated verbatim from the old file.
- **`section_detection.py`**: Houses heuristic candidate scoring, LLM classification, `detect_sections()`, `extract_sections_from_text()`, `search_missing_section()`, and fallback logic — migrated and refactored as module-level functions.
- **`hierarchy.py`**: Houses `detect_numbering_style()`, `group_subsections()`, `group_subsections_llm()`, `extract_section_identifier()` — migrated from the old file.
- **`utils.py`**: Houses `parse_json_from_text()`, `hash_text()`, `safe_query()`, `extract_short_phrases()` — shared helpers extracted from the old file.
- **`criteria/base.py`**: New criteria registry. Maps `(paper_type, section_type)` → list of weighted criteria dicts. Covers 6 paper types (empirical, theoretical, policy, finance, macro, systematic_review) × standard + paper-type-specific sections. Weights sum to 1.0 within each section. Includes `_canonical_section_type()` alias mapper.
- **`criteria/__init__.py`**: Exports `get_criteria`, `PAPER_TYPES`, `PAPER_TYPE_LABELS`, `SECTION_DEFAULTS`.
- **`prompts/templates.py`**: Houses `PAPER_TYPE_CONTEXTS` (6 paper types), `SECTION_TYPE_PROMPTS` (14 section types), `build_evaluation_prompt()` (unified master prompt), and `QUOTE_VALIDATION_PROMPT`. Paper-type context and section guidance are injected into every evaluation prompt.
- **`prompts/__init__.py`**: Exports prompt template objects and `build_evaluation_prompt`.
- **`scoring.py`**: `compute_section_score()` computes weighted average of criteria scores plus a section-importance multiplier per paper type. `compute_overall_score()` aggregates section scores into an overall score and maps to publication readiness string.
- **`evaluator.py`**: `SectionEvaluator` class. Single-pass LLM evaluation using `build_evaluation_prompt`. Validates quotes against source text via `_validate_quote()` (fuzzy/partial match). Falls back gracefully when LLM output cannot be parsed. Results cached by content hash.
- **`main.py`**: `SectionEvaluatorApp` class — Streamlit UI with paper type selection dropdown, both auto-detect (upload) and freeform (paste sections) flows, optional paper context summarization, progress bar, and rich per-criterion output with quote display and validity markers.
- **`__init__.py`**: Package root exports `SectionEvaluatorApp`, `SectionEvaluator`, `PAPER_TYPES`, `PAPER_TYPE_LABELS`, `SECTION_DEFAULTS`.

#### Changed

- **`eval/app.py`**: Updated import from `section_eval_llm_vivi.SectionEvaluator` → `section_eval.SectionEvaluatorApp`. WORKFLOWS dict updated accordingly.

#### Removed

- **`eval/section_eval_llm_vivi.py`**: Superseded by the new package. File retained on disk until verified working in production but no longer imported by `app.py`.

#### Feature: paper-type-aware criteria
- 6 paper types with type-specific section lists shown in UI.
- Each (paper_type, section) combination has 3–5 named criteria with explicit weights.
- Generic 4-criterion fallback (`clarity`, `depth`, `relevance`, `technical_quality`) for unrecognized sections.

#### Feature: weighted scoring
- `compute_section_score()` computes weighted criterion average + section importance multiplier.
- `compute_overall_score()` aggregates sections weighted by importance and maps to readiness label.

#### Feature: mandatory quote extraction and validation
- Each criterion evaluation includes 2 required quotes from the section text.
- `_validate_quote()` checks each quote against the source text (partial match, whitespace-normalized).
- UI displays quote text with validity checkmark (✓) or approximate marker (~).

#### Feature: freeform section input
- New "Paste sections manually" flow: user selects paper type, sees standard sections for that type, and pastes text into per-section text areas with per-section evaluate checkboxes.
- Custom sections can be added beyond defaults.

---

## 02/23

### Fix: detecting correct text in each section

#### Changed

- **`extract_sections_from_text`**: Rewrote section text extraction to slice document text using stored `line_idx` positions from `detect_sections()` rather than re-scanning with fuzzy regex pattern matching. Eliminates premature boundary cutoffs caused by keyword matches in body text.

#### Added

- **`_find_references_start()`**: New helper method that detects the start of References/Bibliography/Appendix sections to cap section extraction and prevent downstream content from bleeding into evaluated sections.

#### Removed

- **`_build_pattern()`**: Fuzzy keyword matching logic used to re-detect section boundaries at extraction time.

#### Fixed

- Sections returning incomplete text due to mid-paragraph keyword matches incorrectly triggering boundary detection.

### Feature: evaluation checkboxes for section selection

#### Changed

- **`render_ui`**: Replaced 2-column layout (section name + action) with 3-column layout (checkbox + section name + action). Users can now select which sections to send through LLM evaluation independently of keep/remove/merge actions.

#### Added

- **`eval_selected`** dict in `render_ui`: Tracks per-section checkbox state. Only checked sections are passed to `evaluate_section()`.
- Column headers row ("Evaluate", "Section", "Action") for UI clarity.
- Post-merge filter that removes unchecked sections before evaluation, with a warning if none are selected.

#### Changed (model assignments)

- **`utils.py`**: Swapped `model_selection` and `model_selection3` so that `model_selection` (used by `conv_query` for section detection) now maps to `MODEL_STRONG` and `model_selection3` maps to `MODEL_FAST`. Previously these were reversed.

### Fix: missing spaces in extracted PDF text (v3 — regex post-processing)

#### Changed

- **`extract_text_from_pdf`**: Reverted to `page.extract_text()` (stable line structure) with regex post-processing via `_fix_missing_spaces()`. The character-level approach (`page.chars`) produced inconsistent line counts that broke `line_idx`-based section slicing.

#### Added

- **`_fix_missing_spaces()`**: Regex-based post-processor that inserts missing spaces in common gluing patterns: lowercase→uppercase ("sectionPresents"), punctuation→uppercase ("2009).The"), lowercase→open-paren ("in(2009)"), close-paren→lowercase ("(2009)and"), digit→lowercase ("32participants").

#### Removed

- **`_estimate_space_width()`**: Character-level gap analysis removed — caused section text cutoff by altering line structure relative to stored `line_idx` values.

### Feature: text paste input and LaTeX source support

#### Added

- **`_strip_latex()`**: 13-pass regex-based LaTeX stripper that converts raw `.tex` source to clean plain text. Handles preamble/postamble removal, comment stripping, discard environments (figure, table, tikzpicture, etc.), citation unwrapping, section header preservation, formatting command unwrapping, and whitespace normalization. No external dependencies.
- **File-type routing** in `render_ui`: Detects `.tex` files (routes through `_strip_latex()`), `.txt` files (decoded directly), and PDFs (existing `extract_text_from_pdf` path). Applied in both the scan handler and the fallback extraction path.

#### Changed (app.py)

- **`st.file_uploader`**: Added `.tex` to accepted file types.
- **Paste text area**: New text input area below the file uploader with a "Plain text" / "LaTeX source" radio toggle. Pasted text is stored in `st.session_state.file_data` as `"Pasted Text.txt"` or `"Pasted Text.tex"` depending on the selected format.
