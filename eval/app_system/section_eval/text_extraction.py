"""
Text extraction utilities for PDF, LaTeX, and plain text inputs.
"""

import re
import tempfile
import os
from typing import Optional


def _page_has_missing_spaces(txt: str) -> bool:
    """
    Heuristic: if the ratio of spaces to non-space characters is below 0.05
    (roughly 1 space per 20 chars), the page likely has concatenated words.
    Normal prose has ~1 space per 5 chars (ratio ~0.20).
    """
    if not txt:
        return False
    non_space = len(txt.replace(" ", ""))
    if non_space == 0:
        return False
    return (txt.count(" ") / non_space) < 0.05


def _extract_page_text(page) -> str:
    """
    Extract text from a single pdfplumber page.
    Uses extract_text() first; if the result looks space-deficient, falls back
    to extract_words() which reconstructs word boundaries from bounding boxes.
    """
    try:
        txt = page.extract_text() or ""
    except Exception:
        txt = ""

    if _page_has_missing_spaces(txt):
        try:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            if words:
                # Reconstruct lines: group words by vertical position, join with spaces
                lines: dict = {}
                for w in words:
                    # Round top to nearest 2pt to group words on the same line
                    line_key = round(w["top"] / 2) * 2
                    lines.setdefault(line_key, []).append(w)
                rebuilt = []
                for key in sorted(lines):
                    line_words = sorted(lines[key], key=lambda w: w["x0"])
                    rebuilt.append(" ".join(w["text"] for w in line_words))
                txt = "\n".join(rebuilt)
        except Exception:
            pass  # stick with original txt

    return txt


def extract_text_from_pdf(file_bytes: bytes, warn_fn=None) -> str:
    """
    Extract text from PDF bytes using pdfplumber. Returns concatenated text.
    warn_fn: optional callable(message) for warnings (e.g. st.warning).
    """
    import pdfplumber

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    text_parts = []
    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                txt = _extract_page_text(page)
                if txt:
                    text_parts.append(txt + "\n\n")
    except Exception as e:
        if warn_fn:
            warn_fn(f"PDF read error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    raw = "".join(text_parts).strip()
    return _fix_missing_spaces(raw)


def _fix_missing_spaces(text: str) -> str:
    """
    Regex post-processor that inserts missing spaces in common PDF gluing patterns.
    """
    # lowercase→uppercase boundary: "sectionPresents" → "section Presents"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # punctuation→uppercase: "2009).The" → "2009). The"
    text = re.sub(r'([.!?)])([A-Z])', r'\1 \2', text)
    # lowercase→open-paren: "in(2009)" → "in (2009)"
    text = re.sub(r'([a-z])\(', r'\1 (', text)
    # close-paren→lowercase: "(2009)and" → "(2009) and"
    text = re.sub(r'\)([a-z])', r') \1', text)
    # digit→lowercase: "32participants" → "32 participants"
    text = re.sub(r'(\d)([a-z])', r'\1 \2', text)
    return text


def strip_latex(text: str) -> str:
    """
    Strip LaTeX markup from raw source, returning readable plain text.
    Preserves math expressions for LLM consumption.
    Pure regex — no external dependencies.
    """
    # Pass 0: Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Pass 1: Remove preamble and postamble
    m = re.search(r'\\begin\{document\}', text)
    if m:
        text = text[m.end():]
    m = re.search(r'\\end\{document\}', text)
    if m:
        text = text[:m.start()]

    # Pass 2: Remove comments (% not preceded by \)
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Pass 3: Remove environments to discard entirely
    for env in ['figure', r'figure\*', 'table', r'table\*', 'tikzpicture',
                 'tabular', r'tabular\*', 'longtable', 'sidewaystable',
                 'sidewaysfigure', 'lstlisting', 'verbatim', 'thebibliography']:
        text = re.sub(
            r'\\begin\{' + env + r'\}.*?\\end\{' + env + r'\}',
            '', text, flags=re.DOTALL
        )

    # Pass 4: Remove \input, \include, \bibliography, \bibliographystyle, \usepackage
    text = re.sub(r'\\(?:input|include|bibliography|bibliographystyle|usepackage)(?:\[[^\]]*\])?\{[^}]*\}', '', text)

    # Pass 5: Remove \newcommand, \renewcommand, \def definitions
    text = re.sub(r'\\(?:new|renew|provide)command\*?\{[^}]*\}(?:\[\d+\])?(?:\[[^\]]*\])?\{[^}]*\}', '', text)
    text = re.sub(r'\\def\\[a-zA-Z]+[^{]*\{[^}]*\}', '', text)

    # Pass 6: Remove \label, \ref, \eqref, \pageref, etc.
    text = re.sub(r'\\(?:label|ref|eqref|pageref|nameref|autoref|cref|Cref)\{[^}]*\}', '', text)

    # Pass 7: Handle \cite variants — keep citation key as readable text
    text = re.sub(r'\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:citeauthor|citeyear|citealp)\*?(?:\[[^\]]*\])*\{([^}]*)\}', r'\1', text)

    # Pass 8: Section headers — keep title on its own line
    text = re.sub(
        r'\\(?:section|subsection|subsubsection|paragraph|subparagraph|chapter)\*?\{([^}]*)\}',
        r'\n\n\1\n', text
    )

    # Pass 9: Unwrap formatting commands (iterative for nesting)
    _unwrap_re = re.compile(
        r'\\(?:textbf|textit|emph|underline|textsc|textrm|textsf|texttt|'
        r'mbox|hbox|footnote|title|author|thanks|centering|text)\{([^}]*)\}'
    )
    text = re.sub(r'\\href\{[^}]*\}\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\url\{([^}]*)\}', r'\1', text)
    prev = None
    while prev != text:
        prev = text
        text = _unwrap_re.sub(r'\1', text)

    # Pass 10: Strip environment wrappers (keep content inside)
    for env in ['abstract', 'quote', 'quotation', 'center', 'flushleft',
                 'flushright', 'itemize', 'enumerate', 'description',
                 'minipage', 'spacing', 'singlespace', 'doublespace',
                 'equation', r'equation\*', 'align', r'align\*', 'gather',
                 r'gather\*', 'multline', r'multline\*', 'displaymath', 'math']:
        text = re.sub(r'\\begin\{' + env + r'\}(?:\[[^\]]*\])?', '', text)
        text = re.sub(r'\\end\{' + env + r'\}', '', text)

    # Pass 11: \item -> bullet point
    text = re.sub(r'\\item\s*(?:\[[^\]]*\])?\s*', '\n- ', text)

    # Pass 12: Clean up spacing, line breaks, special characters
    text = re.sub(r'\\\\(?:\[[^\]]*\])?', '\n', text)
    text = re.sub(r'\\(?:newline|linebreak)\b', '\n', text)
    text = re.sub(r'\\(?:par|newpage|clearpage)\b', '\n\n', text)
    text = re.sub(r'\\(?:noindent|indent|bigskip|medskip|smallskip|vfill|hfill|hspace|vspace)\*?(?:\{[^}]*\})?', '', text)
    text = re.sub(r'\\(?:maketitle|tableofcontents|listoffigures|listoftables|appendix)\b', '', text)
    text = re.sub(r'\\(?:centering|raggedright|raggedleft|normalsize|small|large|Large|LARGE|huge|Huge|tiny|footnotesize|scriptsize)\b', '', text)
    text = re.sub(r'\\[,;:!]', ' ', text)
    text = text.replace('~', ' ')
    for esc, ch in [('\\&', '&'), ('\\%', '%'), ('\\$', '$'), ('\\#', '#'), ('\\_', '_'), ('\\{', '{'), ('\\}', '}')]:
        text = text.replace(esc, ch)
    text = text.replace('---', '\u2014')
    text = text.replace('--', '\u2013')
    text = re.sub(r"``", '"', text)
    text = re.sub(r"''", '"', text)

    # Pass 13: Collapse whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?m)^[ \t]+', '', text)
    text = re.sub(r'(?m)[ \t]+$', '', text)
    return text.strip()


def decode_file(filename: str, file_bytes: bytes, warn_fn=None) -> str:
    """
    Route file bytes to the appropriate extractor based on file extension.
    Returns plain text.
    """
    if filename.endswith('.tex'):
        return strip_latex(file_bytes.decode('utf-8', errors='replace'))
    elif filename.endswith('.txt'):
        return file_bytes.decode('utf-8', errors='replace')
    else:
        return extract_text_from_pdf(file_bytes, warn_fn=warn_fn)
