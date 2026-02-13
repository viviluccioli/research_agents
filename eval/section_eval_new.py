import streamlit as st
import json
import tempfile
import os
import re
import hashlib
from typing import Dict, List, Any, Optional
import pdfplumber
from utils import cm, single_query

class SectionEvaluator:
    """
    Robust SectionEvaluator:
    - Two-pass evaluation: qualitative -> structured (strengths, weaknesses, improvements) -> scoring
    - Strict score schema and filtering (no ad-hoc extraction)
    - LLM fallback and robust JSON parsing
    - Caching in st.session_state
    """

    # Allowed score keys (closed schema)
    ALLOWED_SCORE_KEYS = ("clarity", "depth", "relevance", "technical_accuracy") # too many: remove clarity? 
    CACHE_PREFIX = "se_cache_v2"

    DEFAULT_SECTIONS = [
        "Abstract", "Introduction", "Literature Review", "Theory/Model",
        "Methodology", "Results", "Discussion", "Conclusion", "Appendix"
    ]

    def __init__(self, llm=cm, cache_prefix: str = CACHE_PREFIX):
        self.llm = llm
        self.cache_prefix = cache_prefix
        st.session_state.setdefault(self.cache_prefix, {})

    # --------------------
    # Low-level helpers
    # --------------------
    def _safe_query(self, prompt: str, max_chars: Optional[int] = None) -> str:
        """
        Use conversation manager if available, else single_query.
        Truncate prompt if max_chars given.
        """
        try:
            p = prompt if max_chars is None else prompt[:max_chars]
            if hasattr(self.llm, "conv_query"):
                return self.llm.conv_query(p)
            else:
                return single_query(p)
        except Exception:
            try:
                return single_query(prompt if max_chars is None else prompt[:max_chars])
            except Exception as exc:
                return f"LLM error: {exc}"

    @staticmethod
    def _parse_json_from_text(text: str) -> Optional[Any]:
        """
        Robustly try to extract a JSON object/array from model output.
        Returns Python object if parsing succeeded, otherwise None.
        """
        if not text:
            return None

        candidates = []

        # Try balanced-brace extraction for {...}
        start = None
        depth = 0
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}' and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i+1])
                    start = None

        # Try bracket extraction for [...]
        start = None
        depth = 0
        for i, ch in enumerate(text):
            if ch == '[':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ']' and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i+1])
                    start = None

        # Try candidates
        for c in candidates:
            try:
                return json.loads(c)
            except Exception:
                continue

        # As a last attempt, try to load the entire text
        try:
            return json.loads(text)
        except Exception:
            return None

    @staticmethod
    def _hash_text(s: str) -> str:
        h = hashlib.sha256()
        h.update(s.encode("utf-8"))
        return h.hexdigest()

    # --------------------
    # PDF extraction
    # --------------------
    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text from PDF bytes using pdfplumber. Returns concatenated text.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        text_parts = []
        try:
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    try:
                        txt = page.extract_text() or ""
                        if txt:
                            text_parts.append(txt + "\n\n")
                    except Exception:
                        # skip problematic pages
                        continue
        except Exception as e:
            st.warning(f"PDF read error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return "".join(text_parts).strip()

    # --------------------
    # Section extraction
    # --------------------
    def extract_sections_from_text(self, text: str, desired_sections: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Try deterministic header-based segmentation first. If result is sparse,
        ask the LLM to segment the text into the requested sections and return JSON.
        Returns a dict of section_name -> section_text.
        """
        desired_sections = desired_sections # or self.DEFAULT_SECTIONS # CHANGE THIS: change default sections 

        # Attempt lightweight deterministic segmentation by detecting header-like lines
        lines = text.splitlines()
        found: Dict[str, List[str]] = {}
        current: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            # short line likely a header; check if it matches any desired section (case-insensitive, fuzzy)
            if 0 < len(stripped) <= 120:
                for sec in desired_sections:
                    # match either exact words or beginnings like "1. Introduction"
                    if re.search(r'\b' + re.escape(sec.split('/')[0].split()[0]) + r'\b', stripped, flags=re.IGNORECASE):
                        # start new section
                        current = sec
                        found.setdefault(current, [])
                        break
            if current:
                found[current].append(line)

        # Convert to text and prune empties
        found_text = {k: "\n".join(v).strip() for k, v in found.items() if v and "\n".join(v).strip()}

        # If segmentation looks reasonable (we found at least a few sections), return them
        if found_text and len(found_text) >= max(2, len(desired_sections) // 3):
            return found_text

        # Otherwise ask LLM to segment into JSON
        prompt = f"""
You are an assistant that extracts main sections from an academic economics paper.
Given the paper text below, split it into the following section names if present (use exactly these names as keys):
{desired_sections}

Return a single JSON object where keys are section names and values are the section text.
If a section is not present, omit it. Keep values strictly to the section text only.

PAPER TEXT:
{text[:120000]}
"""
        resp = self._safe_query(prompt, max_chars=120000)
        parsed = self._parse_json_from_text(resp)
        if isinstance(parsed, dict) and parsed:
            # ensure text values and trim excessively large outputs
            cleaned = {k: (v.strip()[:200000] if isinstance(v, str) else "") for k, v in parsed.items() if isinstance(v, str) and v.strip()}
            if cleaned:
                return cleaned

        # fallback: return whole document as "Full Text"
        return {"Full Text": text}

    # --------------------
    # Two-pass evaluation + strict scoring
    # --------------------
    def evaluate_section(self, section_name: str, section_text: str) -> Dict[str, Any]:
        """
        Two-pass evaluation:
          1) Qualitative free-form assessment (2-6 sentences)
          2) Structured JSON extraction (strengths, weaknesses, improvements)
          3) Scoring with strict allowed keys only (1-5) - derived via JSON
        Returns a dict with keys: qualitative, strengths, weaknesses, improvements, scores, raw
        """
        # caching by content hash
        cache_key = self._hash_text(section_name + "|" + section_text[:200000])
        cache = st.session_state[self.cache_prefix]
        if cache_key in cache:
            return cache[cache_key]

        # PASS 1: qualitative
        qual_prompt = f"""
You are a senior reviewer for top economics journals. Read the following section titled "{section_name}" and provide a concise qualitative assessment (2-6 sentences) that covers:
- The section's purpose,
- 1?2 main strengths,
- 1?2 main shortcomings or missing elements,
- One sentence that gives the most impactful fix.

SECTION TEXT:
{section_text[:12000]}
"""
        qualitative = self._safe_query(qual_prompt, max_chars=12000)

        # PASS 2: structured extraction (strict JSON)
        struct_prompt = f"""
Based ONLY on the qualitative assessment below, output EXACTLY one JSON object with keys:
- "strengths": array of up to 3 short phrases (strings)
- "weaknesses": array of up to 3 short phrases (strings)
- "improvements": array of up to 4 short actionable phrases (strings)

Do not include any other keys, text, or commentary. Return valid JSON only.

QUALITATIVE:
{qualitative}
"""
        structured_raw = self._safe_query(struct_prompt, max_chars=4000)
        structured = self._parse_json_from_text(structured_raw)

        strengths, weaknesses, improvements = [], [], []
        if isinstance(structured, dict):
            strengths = structured.get("strengths") or []
            weaknesses = structured.get("weaknesses") or []
            improvements = structured.get("improvements") or []

        # Fallback heuristics if JSON extraction failed: extract short clauses from qualitative text
        if not strengths:
            strengths = self._extract_short_phrases(qualitative, keywords=("strength", "strong", "well"))
        if not weaknesses:
            weaknesses = self._extract_short_phrases(qualitative, keywords=("weakness", "problem", "missing", "lack"))
        if not improvements:
            improvements = self._extract_short_phrases(qualitative, keywords=("suggest", "fix", "improv", "recommend"))

        # sanitize lists
        strengths = [str(s).strip() for s in strengths][:3]
        weaknesses = [str(s).strip() for s in weaknesses][:3]
        improvements = [str(s).strip() for s in improvements][:4]

        # PASS 3: scoring - ask LLM to output ONLY the allowed score keys in JSON
        score_prompt = f"""
Based only on the qualitative assessment and the structured lists below, assign integer scores 1-5 for the following keys:
{list(self.ALLOWED_SCORE_KEYS)}

Return EXACTLY one JSON object with these keys and integer values between 1 and 5.
If unsure, prefer conservative/neutral scoring (3).

QUALITATIVE:
{qualitative}

STRUCTURED:
{json.dumps({'strengths': strengths, 'weaknesses': weaknesses, 'improvements': improvements})}
"""
        score_raw = self._safe_query(score_prompt, max_chars=3000)
        score_obj = self._parse_json_from_text(score_raw)

        scores: Dict[str, int] = {}
        if isinstance(score_obj, dict):
            for k in self.ALLOWED_SCORE_KEYS:
                if k in score_obj:
                    try:
                        val = int(score_obj[k])
                        scores[k] = max(1, min(5, val))
                    except Exception:
                        # ignore invalid
                        pass

        # If scoring failed (any key missing), fill defaults (neutral = 3)
        for k in self.ALLOWED_SCORE_KEYS:
            scores.setdefault(k, 3)

        # compute overall as rounded average (one decimal)
        avg = sum(scores[k] for k in self.ALLOWED_SCORE_KEYS) / len(self.ALLOWED_SCORE_KEYS)
        scores["overall"] = round(avg, 1)

        # Final defensive assertion - ensure no extra keys leaked
        assert set(scores.keys()) == set(self.ALLOWED_SCORE_KEYS).union({"overall"}), \
            f"Score schema violated: {scores.keys()}"

        result = {
            "qualitative": qualitative.strip(),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvements": improvements,
            "scores": scores,
            "raw": {
                "structured_raw": structured_raw,
                "score_raw": score_raw
            }
        }

        # cache and return
        st.session_state[self.cache_prefix][cache_key] = result
        return result

    def _extract_short_phrases(self, text: str, keywords: tuple = ("strength",)) -> List[str]:
        """
        Simple heuristic to pull short candidate phrases when structured JSON is not available.
        Looks for candidate clauses containing keywords and returns short phrases.
        """
        results = []
        if not text:
            return results
        # split into sentences and pick ones containing keywords
        sents = re.split(r'(?<=[\.\?\!])\s+', text)
        for s in sents:
            s_low = s.lower()
            if any(k in s_low for k in keywords):
                # extract a short clause (first 140 chars)
                candidate = s.strip()
                if len(candidate) > 140:
                    candidate = candidate[:140].rsplit(' ', 1)[0] + "..."
                results.append(candidate)
            if len(results) >= 3:
                break
        return results

    # --------------------
    # Aggregate overall assessment
    # --------------------
    def generate_overall_assessment(self, sections: Dict[str, str], evaluations: Dict[str, Any]) -> str:
        """
        Build a concise overall assessment based on per-section scores.
        """
        lines = []
        for name, ev in evaluations.items():
            sc = ev.get("scores", {})
            lines.append(f"{name}: overall={sc.get('overall')}, clarity={sc.get('clarity')}, depth={sc.get('depth')}")

        prompt = f"""
You are an experienced editor for top economics journals. Based on the per-section score summary below,
produce the following EXACT format:

## Key Strengths
1.
2.
3.

## Key Weaknesses
1.
2.
3.

## Priority Improvements
1.
2.
3.

## Publication Readiness
[One of: Not ready, Needs major revisions, Needs minor revisions, Ready] - one sentence justification.

Per-section summary:
{chr(10).join(lines)}
"""
        return self._safe_query(prompt, max_chars=4000)

    # --------------------
    # Streamlit UI
    # --------------------
    def render_ui(self, files: Optional[Dict[str, bytes]] = None):
        st.subheader("Manuscript Section Evaluation (v2)")
        st.write("Two-pass evaluation (qualitative ? structured) and strict scoring schema. See concise recommendations first; expand for full detail.")

        if not files:
            st.info("Upload a PDF using the main uploader to begin.")
            return

        file_keys = list(files.keys())
        manuscript = st.selectbox("Select manuscript", options=file_keys, key="se_v2_select")
        if manuscript is None:
            st.info("Select a file to proceed.")
            return

        # choose sections (checkboxes)
        st.write("Select sections to evaluate (defaults on).")
        chosen_sections = []
        for s in self.DEFAULT_SECTIONS:
            if st.checkbox(s, value=True, key=f"se_v2_chk_{s}"):
                chosen_sections.append(s)

        if st.button("Evaluate Manuscript (Sections)", key=f"se_v2_run_{manuscript}"):
            file_bytes = files[manuscript]
            with st.spinner("Extracting text..."):
                paper_text = self.extract_text_from_pdf(file_bytes)
                seg_hash = self._hash_text(paper_text + "|" + ",".join(chosen_sections))
            seg_cache_key = f"seg_{seg_hash}"
            cache = st.session_state[self.cache_prefix]
            if seg_cache_key in cache:
                sections = cache[seg_cache_key]
            else:
                with st.spinner("Segmenting sections (LLM fallback)..."):
                    sections = self.extract_sections_from_text(paper_text, chosen_sections)
                cache[seg_cache_key] = sections

            # Evaluate each section
            evaluations = {}
            total = len(sections)
            prog = st.progress(0)
            status = st.empty()
            for i, (sec_name, sec_text) in enumerate(sections.items(), start=1):
                status.text(f"Evaluating {sec_name} ({i}/{total})")
                evaluations[sec_name] = self.evaluate_section(sec_name, sec_text)
                prog.progress(i / max(1, total))
            status.text("Generating overall assessment...")
            overall = self.generate_overall_assessment(sections, evaluations)
            status.text("Done.")

            # store results for UI and download
            st.session_state.setdefault("se_v2_last", {})
            st.session_state["se_v2_last"]["manuscript"] = manuscript
            st.session_state["se_v2_last"]["sections"] = sections
            st.session_state["se_v2_last"]["evaluations"] = evaluations
            st.session_state["se_v2_last"]["overall"] = overall
            st.success("Evaluation complete.")

        # Display last results if present
        if "se_v2_last" in st.session_state:
            last = st.session_state["se_v2_last"]
            st.subheader("Overall Assessment")
            st.markdown(last.get("overall", ""))

            st.subheader("Section Summaries")
            evals = last.get("evaluations", {})
            for sec_name, ev in evals.items():
                sc = ev.get("scores", {})
                overall_score = sc.get("overall", "N/A")
                # top-line summary
                st.markdown(f"### {sec_name} ? {overall_score}/5")
                # one-line qualitative first sentence
                qual = ev.get("qualitative", "")
                if qual:
                    first_line = qual.splitlines()[0]
                    st.write(first_line)
                else:
                    st.write("(No concise qualitative assessment available.)")

                # Quick lists
                st.markdown("**Top issues (quick view):**")
                if ev.get("weaknesses"):
                    for w in ev["weaknesses"]:
                        st.write(f"- {w}")
                else:
                    st.write("- (no specific weaknesses found)")

                st.markdown("**Top quick fixes:**")
                if ev.get("improvements"):
                    for imp in ev["improvements"]:
                        st.write(f"- {imp}")
                else:
                    st.write("- (no specific improvements suggested)")

                # Expand for details and strict score display
                with st.expander("Full evaluation & scores"):
                    st.markdown("**Qualitative assessment**")
                    st.write(ev.get("qualitative", ""))

                    st.markdown("**Structured lists**")
                    st.write("**Strengths**")
                    for s in ev.get("strengths", []):
                        st.write(f"- {s}")
                    st.write("**Weaknesses**")
                    for s in ev.get("weaknesses", []):
                        st.write(f"- {s}")
                    st.write("**Improvements**")
                    for s in ev.get("improvements", []):
                        st.write(f"- {s}")

                    st.markdown("**Scores (strict schema)**")
                    # Show only allowed keys and overall in fixed order
                    for k in list(self.ALLOWED_SCORE_KEYS) + ["overall"]:
                        if k in ev.get("scores", {}):
                            label = k.replace("_", " ").capitalize()
                            st.write(f"- {label}: {ev['scores'][k]}/5")

                    st.markdown("**Raw LLM outputs (for debugging)**")
                    st.code(json.dumps(ev.get("raw", {}), indent=2))

            # Download report button
            if st.button("Prepare download: Full report (markdown)", key="se_v2_dl"):
                report_md = self._build_markdown_report(
                    st.session_state["se_v2_last"]["manuscript"],
                    st.session_state["se_v2_last"]["sections"],
                    st.session_state["se_v2_last"]["evaluations"],
                    st.session_state["se_v2_last"]["overall"]
                )
                st.download_button("Download Report", data=report_md, file_name="manuscript_evaluation.md", mime="text/markdown")

    # --------------------
    # Reporting
    # --------------------
    def _build_markdown_report(self, manuscript_name: str, sections: Dict[str, str], evaluations: Dict[str, Any], overall_text: str) -> str:
        md = f"# Manuscript Evaluation Report\n\n**File:** {manuscript_name}\n\n## Overall Assessment\n\n{overall_text}\n\n## Sections\n\n"
        for name, txt in sections.items():
            ev = evaluations.get(name, {})
            sc = ev.get("scores", {})
            md += f"### {name} ? Overall: {sc.get('overall', 'N/A')}/5\n\n"
            md += f"**Qualitative assessment**\n\n{ev.get('qualitative', '')}\n\n"
            if ev.get("strengths"):
                md += "**Strengths**\n"
                for s in ev["strengths"]:
                    md += f"- {s}\n"
                md += "\n"
            if ev.get("weaknesses"):
                md += "**Weaknesses**\n"
                for s in ev["weaknesses"]:
                    md += f"- {s}\n"
                md += "\n"
            if ev.get("improvements"):
                md += "**Improvements**\n"
                for s in ev["improvements"]:
                    md += f"- {s}\n"
                md += "\n"
            md += "\n---\n\n"
        return md
