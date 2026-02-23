Cumulative changelog of **eval/section_eval_llm_vivi.py**

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
