# CLAUDE.md — Project Instructions for Claude Code

## Project overview

Research Agents is a Streamlit-based tool for evaluating academic economics papers. It provides two workflows:

1. **Section Evaluator** (`eval/section_eval_llm_vivi.py`) — Hybrid heuristic + LLM pipeline that detects paper sections, extracts their text, and runs multi-pass evaluation (qualitative assessment, structured strengths/weaknesses/improvements, scoring).
2. **Referee Report Checker** (`eval/referee.py`) — Analyzes referee reports and suggests revisions.

The app entry point is `eval/app.py`. Shared LLM infrastructure (API calls, conversation management, token counting) lives in `eval/utils.py`.

## Key architecture

- **LLM provider**: Configurable via `.env` (`LLM_PROVIDER` = openai | anthropic | gemini). `eval/utils.py` handles all API calls through a unified `requests`-based interface using OpenAI-compatible chat completions format.
- **Model tiers**: Three tiers (`MODEL_FAST`, `MODEL_GENERAL`, `MODEL_STRONG`) with sensible defaults per provider.
- **Section detection**: Two-stage pipeline — (1) heuristic candidate scoring of all lines, (2) LLM classification of candidates. See `eval/.scratchpad.md` for detailed technical writeup.
- **Input formats**: Accepts PDF (via `pdfplumber`), plain text, and LaTeX source (`.tex`). LaTeX is stripped to plain text via `_strip_latex()` (regex, no external deps). Users can also paste text directly in the UI.
- **Section extraction**: Slices document text between detected header `line_idx` positions. Caps at References/Bibliography/Appendix boundary.
- **Evaluation**: Three LLM passes per section — qualitative assessment, structured extraction (JSON), scoring (JSON). Results are cached by content hash in `st.session_state`.

## File map

| File | Purpose |
|------|---------|
| `eval/app.py` | Streamlit entry point, tab routing, file upload + text paste input |
| `eval/section_eval_llm_vivi.py` | **Primary active file** — section detection, extraction, evaluation |
| `eval/utils.py` | LLM API config, `single_query()`, `ConversationManager`, token counting |
| `eval/referee.py` | Referee report workflow |
| `eval/section_eval_new.py` | Earlier version of section evaluator (reference only) |
| `eval/section_eval.py` | Original section evaluator (reference only) |
| `eval/routing.py` | Routing agent (not currently used in Streamlit app) |
| `eval/madexp.py` | Separate Gemini/Colab experiment script (not part of Streamlit app) |
| `eval/.scratchpad.md` | Technical writeup of the section detection framework |
| `eval/changelog/section_eval_changes.md` | Cumulative changelog for `section_eval_llm_vivi.py` |

## Changelog rule

**Every time you modify `eval/section_eval_llm_vivi.py`, you MUST also update `eval/changelog/section_eval_changes.md`.**

Follow this format:

```markdown
## MM/DD

### <Category>: <short description>

#### Changed
- **`method_name`**: What changed and why.

#### Added
- **`method_name()`**: What it does.

#### Removed
- **`method_name()`**: What it was and why it was removed.

#### Fixed
- Description of the bug fix.
```

- Categories: `Fix`, `Feature`, `Refactor`, `Performance`, `UI`
- Keep entries concise — one line per item
- Group related changes under a single `###` heading
- If changes also touch `eval/utils.py` or `eval/app.py`, note them under a separate `#### Changed (filename)` sub-heading within the same entry
- Append new entries under the current date header, or create a new date header if the date has changed

## Development notes

- Run with: `streamlit run eval/app.py`
- Python environment: uses venv or conda (`dataviz5200`)
- Dependencies: `requirements.txt` (requests, pandas, tiktoken, pdfplumber, streamlit, python-dotenv)
- Never commit `.env` — it contains API keys
- PDF text extraction uses `pdfplumber`; some PDFs produce text without spaces (a known extraction artifact that does not significantly affect LLM evaluation quality)
