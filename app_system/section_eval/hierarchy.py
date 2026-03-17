"""
Section hierarchy detection: numbering style analysis and subsection grouping.
"""

import re
from typing import Any, Dict, List, Optional

from .utils import parse_json_from_text, safe_query


def detect_numbering_style(detected: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Analyze detected sections to determine the document's numbering style.
    Returns: {
        "primary_style": str,
        "subsection_style": str,
        "confidence": float,
        "pattern_counts": dict,
        "repeated_letters": list
    }
    """
    patterns = {
        "arabic": 0,
        "roman_upper": 0,
        "roman_lower": 0,
        "letter_upper": 0,
        "letter_lower": 0,
        "numeric_dot": 0,
    }
    letter_counts: Dict[str, int] = {}

    for sec in detected:
        text = sec.get("text", "").strip()

        if re.match(r'^\s*\d+[\.\):\s]', text):
            patterns["arabic"] += 1
        if re.match(r'^\s*\d+\.\d+', text):
            patterns["numeric_dot"] += 1
        if re.match(r'^\s*(?:I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3})[\.\):\s]', text):
            patterns["roman_upper"] += 1
        if re.match(r'^\s*(?:i{1,3}|iv|v|vi{0,3}|ix|x|xi{0,3})[\.\):\s]', text):
            patterns["roman_lower"] += 1

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

    total_sections = len(detected)
    if total_sections == 0:
        return {
            "primary_style": "none",
            "subsection_style": "none",
            "confidence": 0.0,
            "pattern_counts": patterns,
            "repeated_letters": [],
        }

    repeated_letters = [letter for letter, count in letter_counts.items() if count > 1]
    has_repeated_letters = len(repeated_letters) > 0

    if has_repeated_letters:
        patterns_without_letters = {k: v for k, v in patterns.items()
                                     if k not in ["letter_upper", "letter_lower"]}
        if patterns_without_letters:
            max_pattern = max(patterns_without_letters.items(), key=lambda x: x[1])
            primary_style = max_pattern[0]
            primary_count = max_pattern[1]
            total_primary = sum(patterns_without_letters.values())
            confidence = primary_count / total_primary if total_primary > 0 else 0.5
        else:
            primary_style = "letter_upper" if patterns["letter_upper"] > patterns["letter_lower"] else "letter_lower"
            confidence = 0.5
        subsection_style = "letter_upper" if patterns["letter_upper"] > patterns["letter_lower"] else "letter_lower"
    else:
        max_pattern = max(patterns.items(), key=lambda x: x[1])
        primary_style = max_pattern[0]
        confidence = max_pattern[1] / total_sections if total_sections > 0 else 0
        subsection_style = "none"
        if patterns["numeric_dot"] > 0:
            subsection_style = "numeric"
        elif patterns["roman_lower"] > 0 and patterns["roman_upper"] > patterns["roman_lower"]:
            subsection_style = "roman_lower"

    if primary_style == "numeric_dot" and patterns["arabic"] > patterns["numeric_dot"] / 2:
        primary_style = "arabic"

    return {
        "primary_style": primary_style,
        "subsection_style": subsection_style,
        "confidence": confidence,
        "pattern_counts": patterns,
        "repeated_letters": repeated_letters,
    }


def extract_section_identifier(header_text: str, style_info: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Extract section identifier based on detected numbering style.
    Returns dict with 'identifier', 'level', 'type' (primary/subsection).
    """
    text = header_text.strip()
    primary = style_info.get("primary_style", "arabic")
    repeated_letters = style_info.get("repeated_letters", [])

    # Multi-level numeric (1.1, 2.3.1)
    match = re.match(r'^\s*(\d+(?:\.\d+)+)', text)
    if match:
        return {
            "identifier": match.group(1),
            "level": match.group(1).count('.') + 1,
            "type": "subsection",
        }

    # Letter-based sections
    letter_match = re.match(r'^\s*([A-Za-z])[\.\):\s]', text)
    if letter_match:
        letter = letter_match.group(1)
        is_repeated = letter.upper() in repeated_letters
        if is_repeated:
            return {"identifier": letter.upper(), "level": 2, "type": "subsection"}
        else:
            return {"identifier": letter.upper(), "level": 1, "type": "primary"}

    # Roman numerals (upper)
    roman_upper_match = re.match(
        r'^\s*((?:I{1,3}|IV|V|VI{0,3}|IX|X|XI{0,3}|XII{0,3}|XIII|XIV|XV))[\.\):\s]', text
    )
    if roman_upper_match:
        if repeated_letters or primary == "roman_upper":
            return {"identifier": roman_upper_match.group(1), "level": 1, "type": "primary"}

    # Roman numerals (lower)
    roman_lower_match = re.match(r'^\s*((?:i{1,3}|iv|v|vi{0,3}|ix|x|xi{0,3}))[\.\):\s]', text)
    if roman_lower_match:
        return {"identifier": roman_lower_match.group(1), "level": 2, "type": "subsection"}

    # Single-level numeric
    numeric_match = re.match(r'^\s*(\d+)[\.\):\s]', text)
    if numeric_match:
        if repeated_letters or primary == "arabic":
            return {"identifier": numeric_match.group(1), "level": 1, "type": "primary"}

    return None


def _find_parent_section(
    child_id: str,
    child_info: Dict,
    all_sections: Dict,
    style_info: Dict,
    detected: List[Dict[str, str]]
) -> Optional[str]:
    # Numeric subsections (4.1 -> 4)
    if '.' in child_id:
        parent_id = child_id.rsplit('.', 1)[0]
        if parent_id in all_sections:
            return all_sections[parent_id]["text"]

    # Letter-based: find most recent primary section in document order
    if child_id.isalpha() or child_info.get("type") == "subsection":
        child_text = child_info["text"]
        child_idx = None
        for idx, sec in enumerate(detected):
            if sec.get("text") == child_text:
                child_idx = idx
                break
        if child_idx is None:
            return None
        for idx in range(child_idx - 1, -1, -1):
            sec = detected[idx]
            sec_text = sec.get("text", "")
            id_info = extract_section_identifier(sec_text, style_info)
            if id_info and id_info.get("type") == "primary":
                return sec_text

    return None


def group_subsections_llm(
    detected: List[Dict[str, str]],
    style_info: Dict,
    llm
) -> Dict[str, Any]:
    """Fallback: use LLM to determine hierarchy when static methods have low confidence."""
    section_list = "\n".join([f'{i+1}. "{sec.get("text", "")}"' for i, sec in enumerate(detected)])

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

    resp = safe_query(llm, prompt, max_chars=6000)
    parsed = parse_json_from_text(resp)

    hierarchy: Dict[str, List[str]] = {}
    top_level: List[str] = []

    if isinstance(parsed, dict):
        top_level_indices = parsed.get("top_level", [])
        subsections_map = parsed.get("subsections", {})

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
        "top_level": top_level if top_level else [sec.get("text", "") for sec in detected],
    }


def group_subsections(detected: List[Dict[str, str]], llm) -> Dict[str, Any]:
    """
    Hierarchical grouping using detected numbering style.
    Returns: {"hierarchy": {...}, "style_info": {...}, "top_level": [...]}
    """
    style_info = detect_numbering_style(detected)

    if style_info["confidence"] < 0.3:
        return group_subsections_llm(detected, style_info, llm)

    section_map: Dict[str, str] = {}
    identifier_to_info: Dict[str, Dict] = {}

    for sec in detected:
        text = sec.get("text", "")
        id_info = extract_section_identifier(text, style_info)
        if id_info:
            identifier = id_info["identifier"]
            section_map[identifier] = text
            identifier_to_info[identifier] = {
                "text": text,
                "level": id_info["level"],
                "type": id_info["type"],
            }

    hierarchy: Dict[str, List[str]] = {}
    top_level: List[str] = []

    for identifier, info in identifier_to_info.items():
        text = info["text"]
        level = info["level"]
        if level == 1 or info["type"] == "primary":
            top_level.append(text)
            hierarchy[text] = []
        else:
            parent_text = _find_parent_section(identifier, info, identifier_to_info, style_info, detected)
            if parent_text:
                if parent_text not in hierarchy:
                    hierarchy[parent_text] = []
                hierarchy[parent_text].append(text)
            else:
                top_level.append(text)

    return {"hierarchy": hierarchy, "style_info": style_info, "top_level": top_level}
