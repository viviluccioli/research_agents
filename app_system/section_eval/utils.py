"""
Shared utilities: JSON parsing, content hashing, caching helpers.
"""

import json
import hashlib
import re
from typing import Any, Optional


def parse_json_from_text(text: str) -> Optional[Any]:
    """
    Robustly extract a JSON object/array from model output.
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

    for c in candidates:
        try:
            return json.loads(c)
        except Exception:
            continue

    try:
        return json.loads(text)
    except Exception:
        return None


def hash_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def safe_query(llm, prompt: str, max_chars: Optional[int] = None) -> str:
    """
    Call LLM for evaluation using a stateless single request with the strong model.
    Uses single_query directly (bypasses ConversationManager history) so long
    evaluation prompts are not affected by history pruning or token budget limits.
    max_chars kept for compatibility but prompt is no longer truncated here —
    section text is already capped inside build_evaluation_prompt.
    """
    import requests as _requests
    from utils import url_chat_completions, API_KEY, model_selection  # parent eval/ utils.py

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model_selection,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }
    try:
        resp = _requests.post(url_chat_completions, headers=headers, json=data, timeout=120)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        # On API error fall back to single_query
        from utils import single_query
        return single_query(prompt)
    except Exception as exc:
        try:
            from utils import single_query
            return single_query(prompt)
        except Exception as exc2:
            return f"LLM error: {exc2}"


def extract_short_phrases(text: str, keywords: tuple = ("strength",)) -> list:
    """
    Simple heuristic to pull short candidate phrases when structured JSON is unavailable.
    """
    results = []
    if not text:
        return results
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    for s in sents:
        s_low = s.lower()
        if any(k in s_low for k in keywords):
            candidate = s.strip()
            if len(candidate) > 140:
                candidate = candidate[:140].rsplit(' ', 1)[0] + "..."
            results.append(candidate)
        if len(results) >= 3:
            break
    return results
