"""
Section detection: heuristic candidate scoring + LLM classification.
"""

import re
from typing import Any, Dict, List, Optional

from .utils import parse_json_from_text, safe_query


DEFAULT_SECTIONS = [
    "Abstract", "Introduction", "Literature Review", "Theory/Model",
    "Methodology", "Results", "Discussion", "Conclusion", "Appendix"
]


def _heuristic_candidate_headers(text: str) -> List[Dict[str, Any]]:
    """
    Score each line as a potential section header using structural heuristics.
    Returns candidates sorted by position with keys: text, line_idx, score.
    """
    lines = text.splitlines()
    candidates = []

    skip_re = re.compile(
        r"^\s*(?:figure|fig\.|table|source|note|notes|algorithm|listing|exhibit|scheme|panel)[\s\d.:)]",
        re.IGNORECASE,
    )
    math_re = re.compile(
        r"[+=%×≤≥∑∏∫≈∈∀∃←→]"
        r"|^\s*\d+\s*:"
        r"|\b(?:endwhile|endif|endfor|return)\b"
    )
    table_metadata_re = re.compile(
        r"^\s*(?:"
        r"parameter|value|dataset|data|model|accuracy|balanced|"
        r"issn|doi|jel|classification|keywords|abstract|"
        r"batch\s*size|embedding|dimensions|initial|max|min|"
        r"number\s*of|considered|performance|measures|runs|"
        r"experimental|teacher|student|oracle|schemes|random|"
        r"curacy\s*curacy|"
        r"al\s*batch|gnad|comments"
        r")\b",
        re.IGNORECASE,
    )
    chart_artifact_re = re.compile(
        r"^\s*(?:"
        r"accuracy\s+accuracy|"
        r"balanced\s+ac-|"
        r"curacy\s+curacy|"
        r"[\w\-]+\s+[\w\-]+\s*$"
        r")",
        re.IGNORECASE,
    )
    citation_re = re.compile(
        r"(?:&\s+\w+,?\s+\d{4}\)|"
        r"et\s+al\.?,?\s+\d{4}\)|"
        r"\(\w+,?\s+\d{4}\)|"
        r"\d{4}\)\s*\.|"
        r"^\s*\([^\)]{0,30}\)\s*$)",
        re.IGNORECASE,
    )
    math_var_re = re.compile(
        r"(?:[\U0001D400-\U0001D7FF]|"
        r"_\w+_|"
        r"\b(?:log|exp|sin|cos|ln)\(|"
        r"^\s*\d+\.\s+[\w_\(\)]+\s*[,\)]|"
        r"^\s*[\w𝑗𝑡𝑠𝑟]+\s+[\w𝑗𝑡𝑠𝑟]+\s+[\w𝑗𝑡𝑠𝑟]+\s*$)"
    )
    roman_re = re.compile(r"^\s*(?:I{1,3}|IV|VI{0,3}|IX|X{0,3})[\.\):\s]", re.IGNORECASE)
    numbered_re = re.compile(r"^\s*(?:\d+[\.\):]|\d+\.\d+[\.\):]?|[A-D][\.\)])\s")

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        word_count = len(line.split())
        words = line.split()

        if len(line) > 80 or word_count > 12:
            continue
        if len(line) < 3 or sum(c.isalpha() for c in line) < 2:
            continue
        if word_count >= 2 and all(len(w) == 1 for w in words):
            continue
        if not any(sum(c.isalpha() for c in word) >= 2 for word in words):
            continue
        if skip_re.match(line):
            continue
        if math_re.search(line):
            continue
        if math_var_re.search(line):
            continue
        if citation_re.search(line):
            continue
        if table_metadata_re.match(line):
            continue
        if chart_artifact_re.match(line):
            continue
        if any(len(word) > 18 for word in line.split()):
            continue

        line_lower = line.lower()
        table_keywords = ["parameter", "value", "dataset", "batch", "size", "accuracy",
                          "model", "embeddings", "dimensions", "runs", "measure"]
        if word_count <= 4 and any(kw in line_lower for kw in table_keywords):
            if not (line.isupper() or numbered_re.match(line) or roman_re.match(line)):
                continue

        if any(ord(c) >= 0x1D400 and ord(c) <= 0x1D7FF for c in line):
            continue

        if word_count >= 2 and word_count <= 6:
            single_char_words = sum(1 for w in words if len(w) <= 2)
            if single_char_words >= word_count * 0.7:
                continue

        score = 0

        if line.isupper() and word_count >= 2:
            score += 3
        elif line.istitle() and word_count >= 2:
            score += 1

        if numbered_re.match(line):
            score += 2
        if roman_re.match(line):
            score += 2

        if 2 <= word_count <= 5:
            score += 1
        if 2 <= word_count <= 3:
            score += 1

        next_nonempty = ""
        for j in range(idx + 1, min(idx + 4, len(lines))):
            if lines[j].strip():
                next_nonempty = lines[j].strip()
                break
        if next_nonempty and len(next_nonempty) > 2 * len(line):
            score += 1

        if idx > 0 and not lines[idx - 1].strip():
            score += 1

        if score >= 3:
            candidates.append({"text": line, "line_idx": idx, "score": score})

    return candidates


def _normalize_sections(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for i, sec in enumerate(sections):
        normalized.append({
            "text": sec.get("text", "Unknown Section"),
            "type": sec.get("type", "other"),
            "line_idx": sec.get("line_idx", i),
        })
    return normalized


def _find_header_line_idx(lines: List[str], header_text: str) -> int:
    header_stripped = header_text.strip().lower()
    for idx, line in enumerate(lines):
        if line.strip().lower() == header_stripped:
            return idx
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if 0 < len(stripped) <= 120 and header_stripped in stripped.lower():
            return idx
    return 0


def _fallback_from_heuristics(candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sorted_cands = sorted(candidates, key=lambda c: c["line_idx"])
    result = []
    for c in sorted_cands:
        header_text = c["text"]
        words = header_text.split()
        if words and all(len(w) == 1 for w in words):
            continue
        if not any(sum(ch.isalpha() for ch in w) >= 2 for w in words):
            continue
        result.append({"text": header_text, "type": "other", "line_idx": c.get("line_idx", 0)})
    if not result:
        return [{"text": s, "type": "suggestion", "line_idx": i} for i, s in enumerate(DEFAULT_SECTIONS)]
    return result


def _filter_post_conclusion(detected: List[Dict[str, str]]) -> List[Dict[str, str]]:
    conclusion_idx = None
    for i, sec in enumerate(detected):
        sec_text_lower = sec.get("text", "").lower()
        sec_type = sec.get("type", "").lower()
        if sec_type == "conclusion" or "conclusion" in sec_text_lower:
            conclusion_idx = i
            break
    if conclusion_idx is not None:
        return detected[:conclusion_idx + 1]
    return detected


def _llm_detect_sections_raw(text: str, llm) -> List[Dict[str, str]]:
    prompt = f"""You are analyzing an economics research paper. Identify ALL main section headers

present in the paper text below. Return a JSON array where each element has:

- "text": the exact section header as it appears in the paper
- "type": one of: abstract, introduction, literature_review, theory, methodology, data, results, discussion, robustness, conclusion, references, appendix, other

Paper text (first 8000 chars):
{text[:8000]}

Return valid JSON only, no other text."""

    resp = safe_query(llm, prompt, max_chars=10000)
    parsed = parse_json_from_text(resp)

    if isinstance(parsed, list) and len(parsed) >= 2:
        results = []
        lines = text.splitlines()
        for item in parsed:
            if not isinstance(item, dict) or not item.get("text"):
                continue
            header_text = str(item["text"])
            line_idx = _find_header_line_idx(lines, header_text)
            results.append({
                "text": header_text,
                "type": str(item.get("type", "other")),
                "line_idx": line_idx,
            })
        if results:
            return results

    return [{"text": s, "type": "suggestion", "line_idx": i} for i, s in enumerate(DEFAULT_SECTIONS)]


def detect_sections(text: str, llm) -> List[Dict[str, str]]:
    """
    Detect section headers using heuristics + LLM confirmation.
    Returns list of dicts: [{"text": "...", "type": "...", "line_idx": N}, ...]
    """
    candidates = _heuristic_candidate_headers(text)

    if not candidates:
        detected = _llm_detect_sections_raw(text, llm)
        return _normalize_sections(detected)

    candidate_list = "\n".join(
        f'{i+1}. "{c["text"]}"'
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are analyzing an economics research paper to identify its section headers.

Below are candidate lines extracted from the paper based on formatting cues.
Determine which ones are genuine **main section headers** (not sub-headers, figure captions, author names, table titles, mathematical notation, or variables).

IMPORTANT: Reject candidates that are:
- Single characters or repeated single characters (e.g., "t", "t t", "s s s")
- Mathematical variables, subscripts, or notation
- Numbered list items containing only variables
- Equation fragments or function notation
- Citation fragments (e.g., "& Lu, 2025)", "et al., 2024)")
- Figure/table labels or captions
- Chart/graph axis labels
- Table headers or metadata fields
- Metadata identifiers (e.g., "ISSN", "JEL Classification", "DOI")
- Author names, affiliations, or dates
- Hyphenated word fragments from line breaks
- Isolated phrases from text bodies (e.g., "the results.", "banking regulation.")

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

    resp = safe_query(llm, prompt, max_chars=8000)
    parsed = parse_json_from_text(resp)

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

                    words = header_text.split()
                    if words and all(len(w) == 1 for w in words):
                        continue
                    if len(header_text) < 3 or not any(sum(c.isalpha() for c in w) >= 2 for w in words):
                        continue

                    header_lower = header_text.lower()
                    table_keywords = ["parameter", "value", "dataset", "accuracy", "batch", "size",
                                      "issn", "jel", "classification", "balanced", "embedding",
                                      "curacy", "gnad", "al batch"]
                    if any(kw in header_lower for kw in table_keywords) and len(words) <= 4:
                        continue

                    if len(words) == 2 and words[0].lower() == words[1].lower():
                        continue
                    if re.search(r"(?:&\s+\w+,?\s+\d{4}\)|et\s+al\.?,?\s+\d{4}\)|\d{4}\))", header_text):
                        continue
                    if any(ord(c) >= 0x1D400 and ord(c) <= 0x1D7FF for c in header_text):
                        continue
                    if len(words) >= 2 and sum(1 for w in words if len(w) <= 2) >= len(words) * 0.7:
                        continue
                    if re.search(r"(?:log|exp|ln|sin|cos)\(|_\w+_", header_text):
                        continue
                    if header_text.endswith('.') and not re.match(r'^\s*\d+[\.\):]', header_text):
                        if len(words) <= 4:
                            continue

                    detected.append({
                        "text": header_text,
                        "type": item.get("type", "other"),
                        "line_idx": cand["line_idx"],
                    })

    if len(detected) < 2:
        detected = _fallback_from_heuristics(candidates)

    detected = _normalize_sections(detected)
    detected.sort(key=lambda d: d["line_idx"])
    detected = _filter_post_conclusion(detected)
    return detected


def search_missing_section(text: str, section_hint: str) -> Optional[Dict[str, str]]:
    """
    Search for a specific section that may have been missed in initial detection.
    Returns a dict with 'text', 'type', 'line_idx' if found, else None.
    """
    hint_lower = section_hint.strip().lower()
    lines = text.splitlines()

    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()

        if len(line_stripped) > 120:
            continue

        if hint_lower in line_lower:
            word_count = len(line_stripped.split())
            if 1 <= word_count <= 12:
                preceded_by_blank = (idx > 0 and not lines[idx - 1].strip())
                looks_like_header = (
                    line_stripped[0].isupper() or
                    re.match(r'^\d+[\.\):]', line_stripped) or
                    line_stripped.isupper()
                )
                if looks_like_header or preceded_by_blank:
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
                    return {"text": line_stripped, "type": section_type, "line_idx": idx}

    return None


def find_references_start(lines: List[str]) -> Optional[int]:
    ref_re = re.compile(
        r"^\s*(?:\d+[\.\):]?\s*)?(?:references|bibliography|works\s+cited|appendix|appendices)\s*$",
        re.IGNORECASE,
    )
    for idx, line in enumerate(lines):
        if ref_re.match(line.strip()):
            return idx
    return None


def extract_sections_from_text(
    text: str,
    all_detected: List[Dict[str, Any]],
    desired_sections: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Extract section text by slicing the document between detected header positions.
    Returns dict of section_name -> section_text.
    """
    desired_sections = desired_sections or DEFAULT_SECTIONS
    lines = text.splitlines()

    header_positions = []
    for sec in all_detected:
        line_idx = sec.get("line_idx")
        header_text = sec.get("text", "")
        if line_idx is not None and isinstance(line_idx, int):
            header_positions.append((line_idx, header_text))

    refs_start = find_references_start(lines)

    if header_positions:
        header_positions.sort(key=lambda x: x[0])
        all_sections = {}
        for i, (start_idx, header_text) in enumerate(header_positions):
            end_idx = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(lines)
            if refs_start is not None and start_idx < refs_start and end_idx > refs_start:
                end_idx = refs_start
            section_lines = lines[start_idx:end_idx]
            section_text = "\n".join(section_lines).strip()
            all_sections[header_text] = section_text

        desired_set = set(desired_sections)
        result = {k: v for k, v in all_sections.items() if k in desired_set and v}
        if result:
            return result

    # Fallback: locate desired headers directly
    header_positions = []
    for sec_name in desired_sections:
        for idx, line in enumerate(lines):
            if line.strip() == sec_name or line.strip() == sec_name.strip():
                header_positions.append((idx, sec_name))
                break

    if header_positions:
        header_positions.sort(key=lambda x: x[0])
        result = {}
        for i, (start_idx, header_text) in enumerate(header_positions):
            end_idx = header_positions[i + 1][0] if i + 1 < len(header_positions) else len(lines)
            if refs_start is not None and start_idx < refs_start and end_idx > refs_start:
                end_idx = refs_start
            section_text = "\n".join(lines[start_idx:end_idx]).strip()
            if section_text:
                result[header_text] = section_text
        if result:
            return result

    return {"Full Text": text}
