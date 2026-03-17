# Section Evaluator Architecture Documentation

## DETAILED ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI (main.py)                                  │
│                     SectionEvaluatorApp - Main Entry Point                       │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ├──► User Input Flow
                     │    ├─ Paper Type Selection (empirical/theoretical/policy)
                     │    ├─ File Upload (PDF/TXT/DOCX/TEX)
                     │    └─ OR Manual Text Paste (Freeform Mode)
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        TEXT EXTRACTION LAYER                                     │
│                         (text_extraction.py)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  decode_file()                                                                   │
│    ├─ PDF → pdfplumber extraction                                              │
│    ├─ LaTeX → strip commands, extract content                                  │
│    ├─ DOCX → python-docx extraction                                            │
│    └─ TXT → direct read                                                        │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      SECTION DETECTION LAYER                                     │
│                       (section_detection.py)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Heuristic Candidate Scoring                                           │
│    ├─ Line-by-line structural analysis                                         │
│    ├─ Scoring based on: capitalization, word count, position                   │
│    ├─ Filter out: figures, tables, citations, math expressions                 │
│    └─ Generate scored candidates                                               │
│                                                                                  │
│  Phase 2: LLM Classification (if heuristics insufficient)                       │
│    ├─ Send candidates to LLM for classification                               │
│    ├─ LLM returns: {section_name, type, confidence}                           │
│    └─ Filter by confidence threshold                                           │
│                                                                                  │
│  Phase 3: Hierarchy Detection                                                   │
│    ├─ Identify parent-child relationships (e.g., 2.1 under 2)                 │
│    ├─ Group subsections under parent sections                                  │
│    └─ Return: flat list + hierarchy map                                        │
│                                                                                  │
│  Output: List of detected sections with metadata                                │
│    └─ {text: "Introduction", type: "introduction", line_idx: 42}              │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    USER SECTION SELECTION & ACTIONS                              │
│                           (main.py UI)                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  For each detected section, user can:                                           │
│    ├─ ☑ Select for evaluation                                                  │
│    ├─ ✕ Remove (skip)                                                          │
│    ├─ ⤴ Merge into another section                                             │
│    └─ 👁 Preview extracted text                                                 │
│                                                                                  │
│  Optional: Add paper context (abstract/summary) for LLM                         │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       CRITERIA LOOKUP LAYER                                      │
│                      (criteria/base.py)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  get_criteria(paper_type, section_name)                                         │
│    │                                                                             │
│    ├─ Normalize section name → canonical type                                  │
│    │   ├─ Strip numbering (3., II., A.)                                        │
│    │   ├─ Check exact aliases ("Methods" → "methodology")                      │
│    │   ├─ Check keyword matches ("Empirical Strategy" → "methodology")         │
│    │   └─ Fallback to "other"                                                  │
│    │                                                                             │
│    ├─ Look up paper_type + canonical_type in criteria registry                 │
│    │   ├─ UNIVERSAL criteria (all papers): abstract, intro, lit review, etc.   │
│    │   ├─ EMPIRICAL criteria: data, methodology, results                       │
│    │   ├─ THEORETICAL criteria: model_setup, proofs, extensions                │
│    │   ├─ POLICY criteria: policy_context, recommendations, background         │
│    │   └─ FALLBACK criteria if no match                                        │
│    │                                                                             │
│    └─ Return: [{name, weight, description}, ...]                               │
│         where weights sum to 1.0                                                │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      EVALUATION ORCHESTRATION                                    │
│                         (evaluator.py)                                           │
│                   SectionEvaluator.evaluate_section()                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Input: section_name, section_text, paper_type, paper_context                   │
│                                                                                  │
│  Step 1: Check Cache                                                             │
│    └─ Hash: paper_type|section_name|section_text[0:50000]                      │
│    └─ If cached → return cached result                                          │
│                                                                                  │
│  Step 2: Get Criteria                                                            │
│    └─ criteria = get_criteria(paper_type, section_name)                        │
│                                                                                  │
│  Step 3: Build LLM Prompt (prompts/templates.py)                                │
│    ├─ Include: paper_type context, section expectations                        │
│    ├─ Include: all criteria with descriptions                                  │
│    ├─ Request: qualitative assessment, per-criterion scores,                   │
│    │           supporting quotes, improvement suggestions                        │
│    └─ Format: JSON schema for structured output                                 │
│                                                                                  │
│  Step 4: LLM Query                                                               │
│    ├─ Single unified pass (not multi-pass)                                     │
│    ├─ Uses safe_query() wrapper                                                │
│    └─ Response: qualitative text + JSON with scores/quotes/improvements        │
│                                                                                  │
│  Step 5: Parse & Validate                                                        │
│    ├─ Extract qualitative assessment                                           │
│    ├─ Parse JSON for:                                                          │
│    │   ├─ criteria_evaluations: [{criterion, score, justification,            │
│    │   │                           quote_1, quote_2, supports_assessment}]     │
│    │   └─ improvements: [{priority, suggestion, rationale}]                    │
│    ├─ Validate quotes against section_text                                     │
│    │   └─ Mark each quote as valid/invalid                                     │
│    └─ Sanitize scores (clamp 1-5)                                              │
│                                                                                  │
│  Step 6: Compute Weighted Score (scoring.py)                                    │
│    ├─ For each criterion: score × weight                                       │
│    ├─ Raw score = Σ(criterion_score × criterion_weight)                        │
│    ├─ Importance multiplier = SECTION_IMPORTANCE[paper_type][section_type]     │
│    ├─ Adjusted score = raw_score × importance_multiplier                       │
│    └─ Store breakdown: {raw_score, adjusted_score, criteria_breakdown,         │
│                         weight_breakdown, importance_multiplier}                │
│                                                                                  │
│  Step 7: Cache & Return                                                          │
│    └─ Return: {qualitative_assessment, criteria_evaluations,                   │
│                improvements, section_score, paper_type, section_name}           │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OVERALL SCORING LAYER                                    │
│                            (scoring.py)                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│  compute_overall_score(section_scores, paper_type)                              │
│    │                                                                             │
│    ├─ Collect all adjusted_scores from evaluated sections                      │
│    ├─ Overall = mean(adjusted_scores)                                          │
│    ├─ Map to publication readiness:                                             │
│    │   ├─ < 2.5: "Not Ready"                                                   │
│    │   ├─ 2.5-3.4: "Needs Major Revisions"                                     │
│    │   ├─ 3.5-4.2: "Needs Minor Revisions"                                     │
│    │   └─ > 4.2: "Ready"                                                       │
│    └─ Return: {overall_score, publication_readiness}                           │
└────────────────────┬────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RESULTS RENDERING                                        │
│                      (main.py - _render_results)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Display Overview:                                                               │
│    ├─ Overall Score (1-5)                                                       │
│    ├─ Publication Readiness                                                     │
│    └─ Number of Sections Evaluated                                              │
│                                                                                  │
│  Generate Overall Assessment (optional):                                         │
│    └─ LLM synthesizes cross-section insights                                    │
│                                                                                  │
│  Per-Section Expandable Cards:                                                   │
│    ├─ Section name + raw score + adjusted score                                │
│    ├─ Qualitative assessment                                                   │
│    ├─ Criteria breakdown:                                                       │
│    │   ├─ Each criterion: name, score, weight                                  │
│    │   ├─ Justification text                                                   │
│    │   └─ Supporting quotes with validation marks (✓ valid / ~ invalid)        │
│    ├─ Priority improvements (sorted by priority)                                │
│    └─ Score breakdown visualization                                             │
│                                                                                  │
│  Download Options:                                                               │
│    ├─ 📄 Markdown Report (_build_markdown_report)                              │
│    └─ 📕 PDF Report (generate_pdf_report)                                       │
│         ├─ Professional formatting with headers/footers                         │
│         ├─ Page numbers                                                         │
│         └─ Complete evaluation details                                          │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                            SUPPORTING MODULES
═══════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│  utils.py                                                                        │
│    ├─ parse_json_from_text(): Extract JSON from LLM responses                  │
│    ├─ hash_text(): SHA256 hash for caching                                     │
│    ├─ safe_query(): Wrapper for LLM API calls with error handling              │
│    └─ extract_short_phrases(): Fallback heuristic for criteria extraction      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  hierarchy.py                                                                    │
│    └─ group_subsections(): Identify parent-child section relationships          │
│        ├─ Detect numbering patterns (1.1 under 1, etc.)                        │
│        ├─ Build hierarchy map: {parent: [children]}                            │
│        └─ Merge subsection text into parent for evaluation                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│  prompts/templates.py                                                            │
│    └─ build_evaluation_prompt(): Construct LLM evaluation prompt                │
│        ├─ Section-specific guidance                                            │
│        ├─ Criteria list with descriptions                                      │
│        ├─ JSON schema for structured output                                    │
│        └─ Quote extraction instructions                                         │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
                              DATA FLOW SUMMARY
═══════════════════════════════════════════════════════════════════════════════════

USER INPUT
   │
   ├─> Paper Type + Uploaded File
   │
   ▼
TEXT EXTRACTION (decode_file)
   │
   ├─> Raw paper text
   │
   ▼
SECTION DETECTION (detect_sections)
   │
   ├─> List of section candidates + hierarchy
   │
   ▼
USER SELECTION & MERGING
   │
   ├─> Selected sections with text
   │
   ▼
FOR EACH SECTION:
   ├─> CRITERIA LOOKUP (get_criteria)
   │     │
   │     ├─> [{name, weight, description}, ...]
   │     │
   │     ▼
   ├─> EVALUATION (evaluate_section)
   │     │
   │     ├─> Build prompt with criteria
   │     ├─> LLM query (single pass)
   │     ├─> Parse & validate response
   │     ├─> Compute weighted score
   │     │
   │     ▼
   │   {qualitative, criteria_evaluations, improvements, section_score}
   │
   ▼
OVERALL SCORING (compute_overall_score)
   │
   ├─> {overall_score, publication_readiness}
   │
   ▼
RESULTS RENDERING + PDF/MD EXPORT
```

---

## CONCISE ARCHITECTURE DIAGRAM

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                             │
│  Upload PDF/Paste Text → Select Paper Type → Preview Sections    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Text Extract  │  (pdfplumber, LaTeX parser)
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │Section Detect │  (Heuristics + LLM)
                 └───────┬───────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                      EVALUATION CORE                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For Each Section:                                              │
│                                                                 │
│  1. Get Criteria ──→ [Paper Type + Section] → Criteria List   │
│                                                                 │
│  2. Build Prompt ──→ Criteria + Section Text + Context        │
│                                                                 │
│  3. LLM Query ─────→ Qualitative + Scores + Quotes            │
│                                                                 │
│  4. Validate ──────→ Quote verification, JSON parsing          │
│                                                                 │
│  5. Score ─────────→ Weighted criteria score × importance      │
│                                                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │Overall Score  │  (Average adjusted scores)
                 └───────┬───────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                       OUTPUT                                    │
│  • Visual Results (Streamlit UI)                               │
│  • Markdown Report Download                                    │
│  • PDF Report Download                                         │
└────────────────────────────────────────────────────────────────┘
```

---

## KEY COMPONENTS

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **main.py** | Streamlit UI orchestration | render_ui, _run_evaluation, _render_results |
| **text_extraction.py** | File parsing | decode_file (PDF/LaTeX/DOCX/TXT) |
| **section_detection.py** | Find section headers | detect_sections, extract_sections_from_text |
| **criteria/base.py** | Criteria registry | get_criteria, PAPER_TYPES, SECTION_DEFAULTS |
| **evaluator.py** | Core evaluation logic | evaluate_section, generate_overall_assessment |
| **scoring.py** | Score computation | compute_section_score, compute_overall_score |
| **prompts/templates.py** | LLM prompt builder | build_evaluation_prompt |
| **hierarchy.py** | Section relationships | group_subsections |
| **utils.py** | Helper utilities | parse_json, hash_text, safe_query |

---

## CACHING STRATEGY

```
Cache Key = SHA256(paper_type + section_name + section_text[0:50000])
              │
              ├─> Stored in: st.session_state["se_v3_eval_cache"]
              └─> Avoids re-evaluation of unchanged sections
```

---

## SCORING FORMULA

```
For each criterion i:
  criterion_score[i] = LLM score (1-5)
  criterion_weight[i] = from criteria registry (sums to 1.0)

RAW SECTION SCORE = Σ(criterion_score[i] × criterion_weight[i])

IMPORTANCE MULTIPLIER = SECTION_IMPORTANCE[paper_type][section_type]
  Examples:
    - Empirical: methodology=1.3, results=1.2, abstract=0.7
    - Theoretical: proofs=1.4, model_setup=1.3
    - Policy: recommendations=1.3, policy_context=1.2

ADJUSTED SECTION SCORE = RAW SECTION SCORE × IMPORTANCE MULTIPLIER

OVERALL SCORE = mean(all adjusted_section_scores)

PUBLICATION READINESS:
  < 2.5      → "Not Ready"
  2.5 - 3.4  → "Needs Major Revisions"
  3.5 - 4.2  → "Needs Minor Revisions"
  > 4.2      → "Ready"
```

---

## LLM INTERACTION PATTERN

```
SINGLE-PASS EVALUATION per section:

INPUT:
  - Paper type context
  - Section text (max ~50K chars)
  - Criteria list with descriptions
  - Optional paper context (abstract/summary)

OUTPUT (JSON):
  {
    "qualitative_assessment": "2-6 sentence overview...",
    "criteria_evaluations": [
      {
        "criterion": "clarity",
        "score": 4,
        "weight": 0.25,
        "justification": "...",
        "quote_1": {"text": "...", "supports_assessment": true},
        "quote_2": {"text": "...", "supports_assessment": false}
      },
      ...
    ],
    "improvements": [
      {
        "priority": 1,
        "suggestion": "...",
        "rationale": "..."
      },
      ...
    ]
  }

POST-PROCESSING:
  - Validate quotes against source text
  - Sanitize scores (clamp 1-5)
  - Compute weighted section score
```

