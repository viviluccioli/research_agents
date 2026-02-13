# hybrid heuristic + LLM pipeline designed to generalize across diverse formatting styles

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
    Robust SectionEvaluator: - Two-pass evaluation: qualitative -> structured (strengths, weaknesses, improvements) -> scoring - Strict score schema and filtering (no ad-hoc extraction) - LLM fallback and robust JSON parsing - Caching in st.session_state
    """

    # Allowed score keys (closed schema)
    ALLOWED_SCORE_KEYS = ("clarity", "depth", "relevance", "technical_accuracy")
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
    # Section detection (hybrid: heuristic + LLM)
    # --------------------
    def _heuristic_candidate_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Score each line in the paper as a potential section header using
        structural heuristics. Returns a list of candidate dicts sorted by
        position, each with keys: text, line_idx, score.
        """
        lines = text.splitlines()
        candidates = []

        # Patterns to SKIP (captions, footnotes, equations, etc.)
        # Allow optional space/digit after keyword (PDFs often strip spaces: "Figure1:")
        skip_re = re.compile(
            r"^\s*(?:figure|fig\.|table|source|note|notes|algorithm|listing|exhibit|scheme|panel)[\s\d.:)]",
            re.IGNORECASE,
        )
        # Math/equation artifacts: lines with +, =, ×, ≤, etc. or variable-like tokens
        math_re = re.compile(
            r"[+=%×≤≥∑∏∫≈∈∀∃←→]"  # math operators/symbols
            r"|^\s*\d+\s*:"           # numbered pseudocode lines like "10: endwhile"
            r"|\b(?:endwhile|endif|endfor|return)\b"  # pseudocode keywords
        )
        # Table/metadata identifiers (common parameter names and metadata fields)
        table_metadata_re = re.compile(
            r"^\s*(?:"
            r"parameter|value|dataset|data|model|accuracy|balanced|"
            r"issn|doi|jel|classification|keywords|abstract|"
            r"batch\s*size|embedding|dimensions|initial|max|min|"
            r"number\s*of|considered|performance|measures|runs|"
            r"experimental|teacher|student|oracle|schemes|random|"
            r"curacy\s*curacy|"  # PDF parsing artifact like "curacy curacy"
            r"al\s*batch|gnad|comments"
            r")\b",
            re.IGNORECASE,
        )
        # Chart/figure axis labels and repeated words (PDF artifacts from figures)
        chart_artifact_re = re.compile(
            r"^\s*(?:"
            r"accuracy\s+accuracy|"  # Repeated axis labels
            r"balanced\s+ac-|"       # Hyphenated labels split across lines
            r"curacy\s+curacy|"      # More PDF artifacts
            r"[\w\-]+\s+[\w\-]+\s*$"  # Two words where both are identical or very short
            r")",
            re.IGNORECASE,
        )
        # Citations: patterns like "& Lu, 2025)", "et al., 2024)", "(Author, 2023)"
        citation_re = re.compile(
            r"(?:&\s+\w+,?\s+\d{4}\)|"     # & Lu, 2025)
            r"et\s+al\.?,?\s+\d{4}\)|"     # et al., 2024)
            r"\(\w+,?\s+\d{4}\)|"          # (Author, 2023)
            r"\d{4}\)\s*\.|"               # 2024).
            r"^\s*\([^\)]{0,30}\)\s*$)",   # short parenthetical
            re.IGNORECASE,
        )
        # Mathematical variables with subscripts/superscripts (Unicode and spacing patterns)
        # Matches patterns like: "𝑗𝑡 𝑗 𝑡", "log(𝐴𝐼_𝑝𝑎𝑡𝑒𝑛𝑡_𝑟𝑎𝑡𝑒 )", "𝐻𝐻𝐼 , or"
        math_var_re = re.compile(
            r"(?:[\U0001D400-\U0001D7FF]|"  # Mathematical alphanumeric symbols Unicode block
            r"_\w+_|"                        # Underscored variables like _patent_
            r"\b(?:log|exp|sin|cos|ln)\(|"   # Function notation
            r"^\s*\d+\.\s+[\w_\(\)]+\s*[,\)]|"  # Numbered list items with variables: "1. log(...)"
            r"^\s*[\w𝑗𝑡𝑠𝑟]+\s+[\w𝑗𝑡𝑠𝑟]+\s+[\w𝑗𝑡𝑠𝑟]+\s*$)"  # Spaced subscript notation
        )
        # Roman numeral prefix pattern
        roman_re = re.compile(
            r"^\s*(?:I{1,3}|IV|VI{0,3}|IX|X{0,3})[\.\):\s]",
            re.IGNORECASE,
        )
        # Numbered prefix pattern (e.g. "1.", "2.1", "A.")
        numbered_re = re.compile(r"^\s*(?:\d+[\.\):]|\d+\.\d+[\.\):]?|[A-D][\.\)])\s")

        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            word_count = len(line.split())
            words = line.split()

            # Must be short enough to be a header
            if len(line) > 80 or word_count > 12:
                continue
            # Must have at least TWO characters and at least 2 alpha chars
            # (filters out single-letter PDF artifacts like "T", "S")
            if len(line) < 3 or sum(c.isalpha() for c in line) < 2:
                continue

            # Skip lines consisting only of repeated single-character words (e.g., "t t", "s s s")
            # These are typically mathematical variables or PDF parsing artifacts
            if word_count >= 2 and all(len(w) == 1 for w in words):
                continue

            # Skip lines where all words are single characters or just numbers/punctuation
            # Real headers need at least one word with 2+ letters
            if not any(sum(c.isalpha() for c in word) >= 2 for word in words):
                continue

            # Skip captions / footnotes
            if skip_re.match(line):
                continue
            # Skip math/equation fragments and pseudocode
            if math_re.search(line):
                continue
            # Skip mathematical variables with subscripts/superscripts
            if math_var_re.search(line):
                continue
            # Skip citations
            if citation_re.search(line):
                continue
            # Skip table metadata and parameter names
            if table_metadata_re.match(line):
                continue
            # Skip chart/figure artifacts
            if chart_artifact_re.match(line):
                continue
            # Skip lines with long no-space words (PDF glued text like "Payoffswerepaidout.")
            # A real header word shouldn't exceed ~18 chars without spaces
            if any(len(word) > 18 for word in line.split()):
                continue

            # Additional filter: skip lines that look like table rows
            # Characteristics: short phrase with common table keywords
            line_lower = line.lower()
            table_keywords = ["parameter", "value", "dataset", "batch", "size", "accuracy",
                            "model", "embeddings", "dimensions", "runs", "measure"]
            if word_count <= 4 and any(kw in line_lower for kw in table_keywords):
                # Could be a table row, not a section header
                # Check if it lacks typical header formatting
                if not (line.isupper() or numbered_re.match(line) or roman_re.match(line)):
                    continue

            # Enhanced variable detection: check for Unicode mathematical symbols
            # and spacing patterns that suggest subscripted variables
            if any(ord(c) >= 0x1D400 and ord(c) <= 0x1D7FF for c in line):
                # Contains mathematical alphanumeric symbols (includes italic/bold variants)
                continue

            # Check for suspicious spacing patterns (variables with spaces between chars)
            # E.g., "𝑗 𝑡" or "a b c" where words are mostly single chars
            if word_count >= 2 and word_count <= 6:
                single_char_words = sum(1 for w in words if len(w) <= 2)
                if single_char_words >= word_count * 0.7:  # 70%+ are 1-2 chars
                    # Likely subscripted variables parsed with spaces
                    continue

            score = 0

            # --- Formatting signals ---
            if line.isupper() and word_count >= 2:
                score += 3  # ALL CAPS is a strong header signal (require 2+ words)
            elif line.istitle() and word_count >= 2:
                score += 1

            # --- Numbering signals ---
            if numbered_re.match(line):
                score += 2
            if roman_re.match(line):
                score += 2

            # --- Length signals (only reward if there's actual content) ---
            if 2 <= word_count <= 5:
                score += 1
            if 2 <= word_count <= 3:
                score += 1

            # --- Context: followed by a longer paragraph ---
            next_nonempty = ""
            for j in range(idx + 1, min(idx + 4, len(lines))):
                if lines[j].strip():
                    next_nonempty = lines[j].strip()
                    break
            if next_nonempty and len(next_nonempty) > 2 * len(line):
                score += 1

            # --- Context: preceded by a blank line ---
            if idx > 0 and not lines[idx - 1].strip():
                score += 1

            if score >= 3:
                candidates.append({
                    "text": line,
                    "line_idx": idx,
                    "score": score,
                })

        return candidates

    @staticmethod
    def _normalize_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure all section dicts have required fields with sensible defaults.
        Required fields: text, type, line_idx
        """
        normalized = []
        for i, sec in enumerate(sections):
            normalized.append({
                "text": sec.get("text", "Unknown Section"),
                "type": sec.get("type", "other"),
                "line_idx": sec.get("line_idx", i)
            })
        return normalized

    def detect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Detect section headers in a paper using heuristics + LLM confirmation.
        Returns a list of dicts: [{"text": "3. Empirical Strategy", "type": "methodology"}, ...]
        """
        candidates = self._heuristic_candidate_headers(text)

        if not candidates:
            # Nothing found heuristically — ask LLM to detect from raw text
            detected = self._llm_detect_sections_raw(text)
            return self._normalize_sections(detected)

        # Build numbered candidate list for the LLM
        candidate_list = "\n".join(
            f'{i+1}. "{c["text"]}"'
            for i, c in enumerate(candidates[:40])  # cap at 40 to limit prompt size
        )

        prompt = f"""You are analyzing an economics research paper to identify its section headers.

Below are candidate lines extracted from the paper based on formatting cues.
Determine which ones are genuine **main section headers** (not sub-headers, figure captions, author names, table titles, mathematical notation, or variables).

IMPORTANT: Reject candidates that are:
- Single characters or repeated single characters (e.g., "t", "t t", "s s s")
- Mathematical variables, subscripts, or notation (e.g., "𝑗𝑡 𝑗 𝑡", "log(𝐴𝐼_𝑝𝑎𝑡𝑒𝑛𝑡_𝑟𝑎𝑡𝑒)", "𝐻𝐻𝐼 , or", "𝐺𝑖𝑛𝑖𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡")
- Numbered list items containing only variables (e.g., "1. log(...)", "2. 𝐻𝐻𝐼 , or")
- Equation fragments or function notation
- Citation fragments (e.g., "& Lu, 2025)", "et al., 2024)")
- Figure/table labels or captions
- Chart/graph axis labels (e.g., "Accuracy Accuracy", "Balanced Accuracy")
- Table headers or metadata fields (e.g., "Parameter Value", "Dataset", "Batch Size")
- Metadata identifiers (e.g., "ISSN", "JEL Classification", "DOI")
- Author names, affiliations, or dates
- Hyphenated word fragments from line breaks (e.g., "curacy curacy", "GNAD Balanced Ac-")
- Isolated phrases from text bodies (e.g., "the results.", "banking regulation.", "financial institutions.")

For each genuine header, classify its type from this list:
abstract, introduction, literature_review, theory, methodology, data, results, discussion, robustness, conclusion, references, appendix, other

Candidates:
{candidate_list}

First ~600 characters of the paper for context:
{text[:600]}

Return ONLY a JSON array. Each element should have:

- "index": the candidate number (1-based)
- "is_header": true or false
- "type": one of the types above (only if is_header is true)

Example: [{{"index": 1, "is_header": true, "type": "introduction"}}, {{"index": 2, "is_header": false}}]
Return valid JSON only, no other text."""

        resp = self._safe_query(prompt, max_chars=8000)
        parsed = self._parse_json_from_text(resp)

        detected = []
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if item.get("is_header") is True:
                    idx_1based = item.get("index")
                    if isinstance(idx_1based, int) and 1 <= idx_1based <= len(candidates):
                        cand = candidates[idx_1based - 1]
                        header_text = cand["text"]

                        # Final defensive filter: reject obvious non-headers
                        # Skip if all words are single characters (e.g., "t t", "s s")
                        words = header_text.split()
                        if words and all(len(w) == 1 for w in words):
                            continue

                        # Skip if text is less than 3 chars or no multi-letter words
                        if len(header_text) < 3 or not any(sum(c.isalpha() for c in w) >= 2 for w in words):
                            continue

                        # Skip table/metadata patterns
                        header_lower = header_text.lower()
                        table_keywords = ["parameter", "value", "dataset", "accuracy", "batch", "size",
                                        "issn", "jel", "classification", "balanced", "embedding",
                                        "curacy", "gnad", "al batch"]
                        if any(kw in header_lower for kw in table_keywords) and len(words) <= 4:
                            # Likely table content, not a section header
                            continue

                        # Skip repeated words (chart artifacts like "Accuracy Accuracy")
                        if len(words) == 2 and words[0].lower() == words[1].lower():
                            continue

                        # Skip citations
                        if re.search(r"(?:&\s+\w+,?\s+\d{4}\)|et\s+al\.?,?\s+\d{4}\)|\d{4}\))", header_text):
                            continue

                        # Skip mathematical variables with Unicode symbols
                        if any(ord(c) >= 0x1D400 and ord(c) <= 0x1D7FF for c in header_text):
                            continue

                        # Skip if mostly single-char words (likely spaced variables)
                        if len(words) >= 2 and sum(1 for w in words if len(w) <= 2) >= len(words) * 0.7:
                            continue

                        # Skip if contains function notation or underscored variables
                        if re.search(r"(?:log|exp|ln|sin|cos)\(|_\w+_", header_text):
                            continue

                        # Skip sentence fragments (ends with period but not a proper header)
                        if header_text.endswith('.') and not re.match(r'^\s*\d+[\.\):]', header_text):
                            # Ends with period but doesn't start with number - likely a sentence fragment
                            if len(words) <= 4:  # Short fragments like "the results."
                                continue

                        detected.append({
                            "text": header_text,
                            "type": item.get("type", "other"),
                            "line_idx": cand["line_idx"],
                        })

        # If LLM classification returned too few, fall back to top heuristic candidates
        if len(detected) < 2:
            detected = self._fallback_from_heuristics(candidates)

        # Normalize to ensure all required fields are present
        detected = self._normalize_sections(detected)

        # Sort by document order
        detected.sort(key=lambda d: d["line_idx"])

        # Filter out everything after conclusion
        detected = self._filter_post_conclusion(detected)

        return detected

    def _llm_detect_sections_raw(self, text: str) -> List[Dict[str, str]]:
        """Fallback: ask LLM to identify sections directly from paper text."""
        prompt = f"""You are analyzing an economics research paper. Identify ALL main section headers

present in the paper text below. Return a JSON array where each element has:

- "text": the exact section header as it appears in the paper
- "type": one of: abstract, introduction, literature_review, theory, methodology, data, results, discussion, robustness, conclusion, references, appendix, other

Paper text (first 8000 chars):
{text[:8000]}

Return valid JSON only, no other text."""

        resp = self._safe_query(prompt, max_chars=10000)
        parsed = self._parse_json_from_text(resp)

        if isinstance(parsed, list) and len(parsed) >= 2:
            return [
                {"text": str(item.get("text", "")), "type": str(item.get("type", "other")), "line_idx": 0}
                for item in parsed
                if isinstance(item, dict) and item.get("text")
            ]

        # Ultimate fallback: return DEFAULT_SECTIONS as suggestions
        return [{"text": s, "type": "suggestion", "line_idx": i} for i, s in enumerate(self.DEFAULT_SECTIONS)]

    def _fallback_from_heuristics(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Use top-scoring heuristic candidates when LLM classification fails."""
        sorted_cands = sorted(candidates, key=lambda c: c["line_idx"])
        result = []
        for c in sorted_cands:
            header_text = c["text"]
            words = header_text.split()

            # Apply final defensive filter
            # Skip if all words are single characters
            if words and all(len(w) == 1 for w in words):
                continue

            # Skip if no multi-letter words
            if not any(sum(ch.isalpha() for ch in w) >= 2 for w in words):
                continue

            result.append({"text": header_text, "type": "other", "line_idx": c.get("line_idx", 0)})

        if not result:
            return [{"text": s, "type": "suggestion", "line_idx": i} for i, s in enumerate(self.DEFAULT_SECTIONS)]
        return result

    # --------------------
    # Section extraction
    # --------------------
    def extract_sections_from_text(self, text: str, desired_sections: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Try deterministic header-based segmentation first. If result is sparse,
        ask the LLM to segment the text into the requested sections and return JSON.
        Returns a dict of section_name -> section_text.

        desired_sections can be either:
          - Simple names: ["Introduction", "Methodology"]
          - Exact header text from detect_sections(): ["3. Empirical Strategy", "IV. Data"]
        """
        desired_sections = desired_sections or self.DEFAULT_SECTIONS

        # Build matching patterns: for each desired section, try to match either
        # the exact header text or the core words (stripping numbering prefixes).
        def _build_pattern(sec: str) -> re.Pattern:
            # Strip leading numbering like "3.", "IV.", "A)" etc.
            core = re.sub(r"^\s*(?:\d+[\.\):]?\s*|[IVXivx]+[\.\):]?\s*|[A-D][\.\)]\s*)", "", sec).strip()
            if not core:
                core = sec.strip()
            # Escape for regex and match the core text as a whole phrase
            escaped = re.escape(core)
            # Also match the first significant word for fuzzy matching
            first_word = core.split('/')[0].split()[0] if core.split() else core
            first_escaped = re.escape(first_word)
            # Match either the full phrase or at least the first keyword
            return re.compile(
                rf"(?:{escaped}|\b{first_escaped}\b)",
                re.IGNORECASE,
            )

        patterns = [(sec, _build_pattern(sec)) for sec in desired_sections]

        # Attempt lightweight deterministic segmentation by detecting header-like lines
        lines = text.splitlines()
        found: Dict[str, List[str]] = {}
        current: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            # short line likely a header; check if it matches any desired section
            if 0 < len(stripped) <= 120:
                for sec, pat in patterns:
                    if pat.search(stripped):
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
    # Section post-processing and hierarchical grouping
    # --------------------

    @staticmethod
    def _detect_numbering_style(detected: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze detected sections to determine the document's numbering style.
        Returns a dict with detected patterns and confidence level.

        Returns: {
            "primary_style": "arabic" | "roman_upper" | "roman_lower" | "letter_upper" | "letter_lower" | "mixed" | "none",
            "subsection_style": "numeric" | "letter_upper" | "letter_lower" | "roman_lower" | "none",
            "confidence": float (0-1),
            "pattern_counts": dict of pattern frequencies
        }
        """
        patterns = {
            "arabic": 0,           # 1, 2, 3...
            "roman_upper": 0,      # I, II, III...
            "roman_lower": 0,      # i, ii, iii...
            "letter_upper": 0,     # A, B, C...
            "letter_lower": 0,     # a, b, c...
            "numeric_dot": 0,      # 1.1, 2.1, 3.2...
        }

        # Count repeated single letters (indicates subsection style)
        letter_counts = {}

        for sec in detected:
            text = sec.get("text", "").strip()

            # Check for arabic numerals
            if re.match(r'^\s*\d+[\.\):\s]', text):
                patterns["arabic"] += 1

            # Check for numeric subsections (e.g., 4.1, 5.2)
            if re.match(r'^\s*\d+\.\d+', text):
                patterns["numeric_dot"] += 1

            # Check for roman numerals (upper)
            if re.match(r'^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3})[\.\):\s]', text):
                patterns["roman_upper"] += 1

            # Check for roman numerals (lower)
            if re.match(r'^\s*(?:i{1,3}|iv|v|vi{0,3}|ix|x|xi{0,3})[\.\):\s]', text):
                patterns["roman_lower"] += 1

            # Check for letter-based sections (A., B., C.)
            letter_match = re.match(r'^\s*([A-Z])[\.\):\s]', text)
            if letter_match:
                letter = letter_match.group(1)
                patterns["letter_upper"] += 1
                letter_counts[letter] = letter_counts.get(letter, 0) + 1

            letter_match_lower = re.match(r'^\s*([a-z])[\.\):\s]', text)
            if letter_match_lower:
                letter = letter_match_lower.group(1)
                patterns["letter_lower"] += 1
                letter_counts[letter.upper()] = letter_counts.get(letter.upper(), 0) + 1

        # Determine primary style
        total_sections = len(detected)
        if total_sections == 0:
            return {
                "primary_style": "none",
                "subsection_style": "none",
                "confidence": 0.0,
                "pattern_counts": patterns
            }

        # Check for repeated letters (indicates subsection level)
        repeated_letters = [letter for letter, count in letter_counts.items() if count > 1]
        has_repeated_letters = len(repeated_letters) > 0

        # Calculate confidence and determine primary style
        max_pattern = max(patterns.items(), key=lambda x: x[1])
        primary_style = max_pattern[0]
        confidence = max_pattern[1] / total_sections if total_sections > 0 else 0

        # Determine subsection style
        subsection_style = "none"
        if patterns["numeric_dot"] > 0:
            subsection_style = "numeric"
        elif has_repeated_letters:
            if patterns["letter_upper"] > patterns["letter_lower"]:
                subsection_style = "letter_upper"
            else:
                subsection_style = "letter_lower"
        elif patterns["roman_lower"] > 0 and patterns["roman_upper"] > patterns["roman_lower"]:
            subsection_style = "roman_lower"

        # Adjust primary style if mostly subsections
        if primary_style == "numeric_dot" and patterns["arabic"] > patterns["numeric_dot"] / 2:
            primary_style = "arabic"

        return {
            "primary_style": primary_style,
            "subsection_style": subsection_style,
            "confidence": confidence,
            "pattern_counts": patterns,
            "repeated_letters": repeated_letters
        }

    @staticmethod
    def _extract_section_identifier(header_text: str, style_info: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Extract section identifier based on detected numbering style.
        Returns dict with 'identifier', 'level', and 'type' (primary/subsection).
        """
        text = header_text.strip()
        primary = style_info.get("primary_style", "arabic")
        subsection = style_info.get("subsection_style", "none")

        # Try numeric patterns (most common)
        # Multi-level numeric (1.1, 2.3.1)
        match = re.match(r'^\s*(\d+(?:\.\d+)+)', text)
        if match:
            return {
                "identifier": match.group(1),
                "level": match.group(1).count('.') + 1,
                "type": "subsection"
            }

        # Single-level numeric (1, 2, 3)
        match = re.match(r'^\s*(\d+)[\.\):\s]', text)
        if match:
            return {
                "identifier": match.group(1),
                "level": 1,
                "type": "primary"
            }

        # Roman numerals (upper)
        match = re.match(r'^\s*((?:I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}))[\.\):\s]', text)
        if match and primary == "roman_upper":
            return {
                "identifier": match.group(1),
                "level": 1,
                "type": "primary"
            }

        # Roman numerals (lower) - typically subsections
        match = re.match(r'^\s*((?:i{1,3}|iv|v|vi{0,3}|ix|x|xi{0,3}))[\.\):\s]', text)
        if match:
            return {
                "identifier": match.group(1),
                "level": 2,
                "type": "subsection"
            }

        # Letter-based (A, B, C or a, b, c)
        match = re.match(r'^\s*([A-Za-z])[\.\):\s]', text)
        if match:
            letter = match.group(1)
            is_repeated = letter.upper() in style_info.get("repeated_letters", [])

            return {
                "identifier": letter,
                "level": 2 if is_repeated else 1,
                "type": "subsection" if is_repeated else "primary"
            }

        return None

    def _group_subsections_enhanced(self, detected: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Enhanced hierarchical grouping using detected numbering style.

        Returns: {
            "hierarchy": {parent_header: [child_header1, child_header2, ...]},
            "style_info": detected style information,
            "top_level": [list of top-level section texts]
        }
        """
        # Step 1: Detect numbering style
        style_info = self._detect_numbering_style(detected)

        # Step 2: Extract identifiers for all sections
        section_map = {}  # identifier -> full header text
        identifier_to_info = {}  # identifier -> {text, level, type}

        for sec in detected:
            text = sec.get("text", "")
            id_info = self._extract_section_identifier(text, style_info)

            if id_info:
                identifier = id_info["identifier"]
                section_map[identifier] = text
                identifier_to_info[identifier] = {
                    "text": text,
                    "level": id_info["level"],
                    "type": id_info["type"]
                }

        # Step 3: Build parent-child relationships
        hierarchy = {}
        top_level = []

        # Low confidence - fallback to LLM
        if style_info["confidence"] < 0.3:
            return self._group_subsections_llm(detected, style_info)

        # Group by parent-child relationships
        for identifier, info in identifier_to_info.items():
            text = info["text"]
            level = info["level"]

            if level == 1 or info["type"] == "primary":
                # Top-level section
                top_level.append(text)
                hierarchy[text] = []
            else:
                # Find parent
                parent_text = self._find_parent_section(identifier, info, identifier_to_info, style_info)
                if parent_text:
                    if parent_text not in hierarchy:
                        hierarchy[parent_text] = []
                    hierarchy[parent_text].append(text)
                else:
                    # Can't find parent, treat as top-level
                    top_level.append(text)

        return {
            "hierarchy": hierarchy,
            "style_info": style_info,
            "top_level": top_level
        }

    @staticmethod
    def _find_parent_section(child_id: str, child_info: Dict, all_sections: Dict, style_info: Dict) -> Optional[str]:
        """
        Find the parent section for a given child identifier.
        """
        # Numeric subsections (4.1 -> 4)
        if '.' in child_id:
            parent_id = child_id.rsplit('.', 1)[0]
            if parent_id in all_sections:
                return all_sections[parent_id]["text"]

        # Letter-based subsections under numeric sections
        # Look for the most recent numeric section before this letter
        if child_id.isalpha():
            child_text = child_info["text"]
            # Find preceding numeric section
            for sec_id, sec_info in all_sections.items():
                if sec_info["level"] == 1 and sec_id.isdigit():
                    # This could be the parent
                    # Check document order (would need line_idx)
                    return sec_info["text"]

        return None

    def _group_subsections_llm(self, detected: List[Dict[str, str]], style_info: Dict) -> Dict[str, Any]:
        """
        Fallback: Use LLM to determine hierarchy when static methods have low confidence.
        """
        section_list = "\n".join([f'{i+1}. "{sec.get("text", "")}"' for i, sec in enumerate(detected[:50])])

        prompt = f"""Analyze these section headers from an academic paper and determine their hierarchical relationships.

Style analysis suggests: {style_info}

Sections:
{section_list}

For each section, determine:
1. Is it a top-level (main) section or a subsection?
2. If it's a subsection, which section number is its parent?

Return a JSON object with:
- "top_level": array of section numbers (1-based) that are main sections
- "subsections": object mapping parent section number to array of child section numbers

Example: {{"top_level": [1, 4, 7], "subsections": {{"4": [5, 6], "7": [8, 9]}}}}

Return only valid JSON."""

        resp = self._safe_query(prompt, max_chars=6000)
        parsed = self._parse_json_from_text(resp)

        hierarchy = {}
        top_level = []

        if isinstance(parsed, dict):
            top_level_indices = parsed.get("top_level", [])
            subsections_map = parsed.get("subsections", {})

            # Build hierarchy from LLM response
            for i in top_level_indices:
                if 1 <= i <= len(detected):
                    top_text = detected[i-1].get("text", "")
                    top_level.append(top_text)
                    hierarchy[top_text] = []

            for parent_idx_str, child_indices in subsections_map.items():
                try:
                    parent_idx = int(parent_idx_str)
                    if 1 <= parent_idx <= len(detected):
                        parent_text = detected[parent_idx-1].get("text", "")
                        if parent_text not in hierarchy:
                            hierarchy[parent_text] = []

                        for child_idx in child_indices:
                            if 1 <= child_idx <= len(detected):
                                child_text = detected[child_idx-1].get("text", "")
                                hierarchy[parent_text].append(child_text)
                except (ValueError, TypeError):
                    continue

        return {
            "hierarchy": hierarchy,
            "style_info": style_info,
            "top_level": top_level if top_level else [sec.get("text", "") for sec in detected]
        }

    def _filter_post_conclusion(self, detected: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Remove all sections detected after 'Conclusion' section.
        Assumes conclusion is a terminal section and anything after is spurious (appendix tables, etc.).
        """
        conclusion_idx = None
        for i, sec in enumerate(detected):
            sec_text_lower = sec.get("text", "").lower()
            sec_type = sec.get("type", "").lower()

            # Look for conclusion by type or text content
            if sec_type == "conclusion" or "conclusion" in sec_text_lower:
                conclusion_idx = i
                break

        if conclusion_idx is not None:
            # Keep everything up to and including conclusion
            return detected[:conclusion_idx + 1]

        return detected

    def _group_subsections(self, detected: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """
        Wrapper for backward compatibility. Calls enhanced grouping and returns just the hierarchy dict.
        """
        result = self._group_subsections_enhanced(detected)
        return result.get("hierarchy", {})

    def _search_missing_section(self, text: str, section_hint: str) -> Optional[Dict[str, str]]:
        """
        Search for a specific section that may have been missed in initial detection.
        section_hint: user-provided hint like "discussion" or "8. Discussion"
        Returns a dict with 'text' and 'type' if found, else None.
        """
        # Clean up the hint
        hint_lower = section_hint.strip().lower()

        # Try to find in the text using flexible matching
        lines = text.splitlines()

        # Build pattern: look for the hint as a standalone header-like line
        for idx, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Skip if line is too long to be a header
            if len(line_stripped) > 120:
                continue

            # Check if this line contains the hint keyword
            if hint_lower in line_lower:
                # Additional checks to ensure it's likely a header
                word_count = len(line_stripped.split())
                if 1 <= word_count <= 12:
                    # Check if preceded by blank line (common for headers)
                    preceded_by_blank = (idx > 0 and not lines[idx - 1].strip())

                    # Check if it looks like a section header
                    looks_like_header = (
                        line_stripped[0].isupper() or  # Starts with capital
                        re.match(r'^\d+[\.\):]', line_stripped) or  # Starts with number
                        line_stripped.isupper()  # All caps
                    )

                    if looks_like_header or preceded_by_blank:
                        # Determine type
                        section_type = "other"
                        if "discussion" in line_lower:
                            section_type = "discussion"
                        elif "introduction" in line_lower:
                            section_type = "introduction"
                        elif "conclusion" in line_lower:
                            section_type = "conclusion"
                        elif "method" in line_lower:
                            section_type = "methodology"
                        elif "result" in line_lower:
                            section_type = "results"

                        return {
                            "text": line_stripped,
                            "type": section_type,
                            "line_idx": idx
                        }

        return None

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
        st.write("Upload a paper, scan for sections, then choose which ones to evaluate.")

        if not files:
            st.info("Upload a PDF using the main uploader to begin.")
            return

        file_keys = list(files.keys())
        manuscript = st.selectbox("Select manuscript", options=file_keys, key="se_v2_select")
        if manuscript is None:
            st.info("Select a file to proceed.")
            return

        # ---- Phase 1: Scan for sections ----
        scan_state_key = f"se_v2_detected_{manuscript}"

        if st.button("Scan for Sections", key=f"se_v2_scan_{manuscript}"):
            file_bytes = files[manuscript]
            with st.spinner("Extracting text from PDF..."):
                paper_text = self.extract_text_from_pdf(file_bytes)
            with st.spinner("Detecting section headers..."):
                detected = self.detect_sections(paper_text)
            # Store detected sections and extracted text in session state
            st.session_state[scan_state_key] = detected
            st.session_state[f"se_v2_text_{manuscript}"] = paper_text
            st.success(f"Found {len(detected)} section(s).")

        # ---- Phase 1.5: Check for missing sections ----
        detected = st.session_state.get(scan_state_key)

        # Normalize detected sections to ensure all have required fields
        if detected:
            detected = self._normalize_sections(detected)
            st.session_state[scan_state_key] = detected

        if detected:
            st.write("---")
            st.write("**Did we miss any sections?** Type a section name or hint below and we'll search for it:")

            col1, col2 = st.columns([3, 1])
            with col1:
                missing_section_hint = st.text_input(
                    "Section hint (e.g., 'Discussion' or '8. Discussion')",
                    key=f"se_v2_missing_{manuscript}",
                    placeholder="e.g., Discussion"
                )
            with col2:
                st.write("")  # spacing
                st.write("")  # spacing
                search_missing = st.button("Search", key=f"se_v2_search_missing_{manuscript}")

            if search_missing and missing_section_hint:
                paper_text = st.session_state.get(f"se_v2_text_{manuscript}", "")
                if paper_text:
                    with st.spinner(f"Searching for '{missing_section_hint}'..."):
                        found_section = self._search_missing_section(paper_text, missing_section_hint)

                    if found_section:
                        # Insert the found section in the appropriate position based on line_idx
                        inserted = False
                        for i, sec in enumerate(detected):
                            if sec.get("line_idx", 999999) > found_section.get("line_idx", 0):
                                detected.insert(i, found_section)
                                inserted = True
                                break
                        if not inserted:
                            detected.append(found_section)

                        st.session_state[scan_state_key] = detected
                        st.success(f"Added: {found_section['text']}")
                        st.rerun()
                    else:
                        st.warning(f"Could not find a section matching '{missing_section_hint}'. Try a different hint.")
                else:
                    st.error("Paper text not found. Please rescan.")

        # ---- Phase 2: Show detected sections with hierarchical grouping ----
        detected = st.session_state.get(scan_state_key)

        if detected:
            # Build hierarchy for display using enhanced grouping
            hierarchy_result = self._group_subsections_enhanced(detected)
            hierarchy = hierarchy_result.get("hierarchy", {})
            style_info = hierarchy_result.get("style_info", {})
            top_level_from_analysis = hierarchy_result.get("top_level", [])

            # Separate top-level sections from subsections
            # Subsections are those that have a parent in the hierarchy
            subsection_texts = set()
            for parent, children in hierarchy.items():
                subsection_texts.update(children)

            # Filter to show only top-level sections by default
            # Use the analysis result if available, otherwise filter manually
            if top_level_from_analysis:
                top_level_sections = [sec for sec in detected if sec["text"] in top_level_from_analysis]
            else:
                top_level_sections = [sec for sec in detected if sec["text"] not in subsection_texts]

            is_suggestion = all(d.get("type") == "suggestion" for d in detected)
            if is_suggestion:
                st.info("Could not auto-detect sections. Showing default suggestions -- feel free to edit.")
            else:
                st.write("---")

                # Show detected numbering style
                primary_style = style_info.get("primary_style", "unknown")
                subsection_style = style_info.get("subsection_style", "none")
                confidence = style_info.get("confidence", 0)

                style_display = {
                    "arabic": "Arabic numerals (1, 2, 3...)",
                    "roman_upper": "Roman numerals (I, II, III...)",
                    "roman_lower": "Roman numerals (i, ii, iii...)",
                    "letter_upper": "Letters (A, B, C...)",
                    "letter_lower": "Letters (a, b, c...)",
                    "numeric_dot": "Decimal numbering (1.1, 1.2...)",
                    "mixed": "Mixed styles",
                    "none": "No clear pattern"
                }

                st.write(f"**Detected numbering style:** {style_display.get(primary_style, primary_style)}")
                if subsection_style != "none":
                    st.write(f"**Subsection style:** {style_display.get(subsection_style, subsection_style)}")
                if confidence < 0.5:
                    st.warning(f"⚠️ Low confidence ({confidence:.0%}) in style detection. Using LLM-assisted hierarchy analysis.")

                st.write("**Detected sections** — subsections will be automatically grouped under their parent sections during evaluation:")

                # Show hierarchy info in expander
                if hierarchy:
                    with st.expander("📋 View complete section hierarchy (including subsections)"):
                        st.write("Subsections will be automatically merged into their parent sections when you evaluate.")
                        st.write("")
                        for parent, children in hierarchy.items():
                            st.write(f"**{parent}**")
                            for child in children:
                                st.write(f"  └─ {child}")
                        st.write("")
                        st.info(f"Showing {len(top_level_sections)} top-level sections below. {len(subsection_texts)} subsections are hidden but will be included during evaluation.")

                st.write("For each top-level section, choose to **keep**, **remove**, or **merge into** another section:")

            # Use top-level sections for the UI display
            display_sections = top_level_sections if not is_suggestion else detected
            section_names = [sec["text"] for sec in display_sections]

            # For each detected section, show a selectbox: Keep / Remove / Merge into <other>
            actions = {}  # sec_text -> "keep" | "remove" | target_sec_text

            # Auto-keep subsections (they'll be merged during evaluation)
            for sec_text in subsection_texts:
                actions[sec_text] = "keep"

            for i, sec in enumerate(display_sections):
                sec_text = sec["text"]
                sec_type = sec.get("type", "")
                display_label = f"{sec_text}  ({sec_type})" if sec_type and sec_type not in ("other", "suggestion") else sec_text

                # Build options: Keep, Remove, Merge into each other section
                merge_targets = [f"Merge into: {s}" for s in section_names if s != sec_text]
                options = ["Keep", "Remove"] + merge_targets

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"**{display_label}**")
                with col2:
                    choice = st.selectbox(
                        "Action",
                        options=options,
                        index=0,
                        key=f"se_v2_action_{manuscript}_{i}",
                        label_visibility="collapsed",
                    )

                if choice == "Keep":
                    actions[sec_text] = "keep"
                elif choice == "Remove":
                    actions[sec_text] = "remove"
                elif choice.startswith("Merge into: "):
                    target = choice[len("Merge into: "):]
                    actions[sec_text] = target

            # ---- Phase 3: Evaluate selected sections ----
            if st.button("Evaluate Selected Sections", key=f"se_v2_run_{manuscript}"):
                # Build final section list: resolve merges
                # kept sections are the primary ones; merged sections get their text appended
                kept = [s for s, a in actions.items() if a == "keep"]
                merges = {s: a for s, a in actions.items() if a not in ("keep", "remove")}

                if not kept:
                    st.warning("Please keep at least one section to evaluate.")
                    return

                paper_text = st.session_state.get(f"se_v2_text_{manuscript}", "")
                if not paper_text:
                    file_bytes = files[manuscript]
                    with st.spinner("Extracting text..."):
                        paper_text = self.extract_text_from_pdf(file_bytes)

                # Automatically group subsections under parent sections
                # E.g., if "4. Experimental design" is kept, also include "4.1", "4.2", etc.
                hierarchy = self._group_subsections(detected)
                auto_merged = {}  # track which subsections get auto-merged

                for parent_header in kept:
                    if parent_header in hierarchy:
                        # This section has subsections - merge them in
                        for child_header in hierarchy[parent_header]:
                            # Only auto-merge if the child was kept (not manually removed)
                            if child_header in actions and actions[child_header] == "keep":
                                auto_merged[child_header] = parent_header

                # Update merges dict with auto-merged subsections
                for child, parent in auto_merged.items():
                    if child in kept:
                        kept.remove(child)
                    merges[child] = parent

                # Show info about auto-grouping
                if auto_merged:
                    st.info(f"Auto-grouped {len(auto_merged)} subsection(s) under their parent sections: " +
                            ", ".join([f"{child} → {parent}" for child, parent in list(auto_merged.items())[:3]]) +
                            ("..." if len(auto_merged) > 3 else ""))

                # Extract text for ALL sections (kept + merge sources) so we can combine
                all_needed = list(set(kept + list(merges.keys())))
                seg_hash = self._hash_text(paper_text + "|" + ",".join(sorted(all_needed)))
                seg_cache_key = f"seg_{seg_hash}"
                cache = st.session_state[self.cache_prefix]
                if seg_cache_key in cache:
                    raw_sections = cache[seg_cache_key]
                else:
                    with st.spinner("Extracting section text..."):
                        raw_sections = self.extract_sections_from_text(paper_text, all_needed)
                    cache[seg_cache_key] = raw_sections

                # Apply merges: append merged section text to target section
                sections = {}
                for sec_name in kept:
                    sections[sec_name] = raw_sections.get(sec_name, "")
                for src, target in merges.items():
                    if target in sections:
                        src_text = raw_sections.get(src, "")
                        if src_text:
                            sections[target] = sections[target] + "\n\n" + src_text

                if not sections:
                    st.warning("No section text could be extracted.")
                    return

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
        else:
            st.info("Click **Scan for Sections** to detect the paper's structure before evaluating.")

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
