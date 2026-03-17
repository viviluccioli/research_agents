# Section Evaluator - Quick Reference Guide

## 🎯 What It Does
Evaluates economics research papers section-by-section using paper-type-specific criteria, providing scores, feedback, and actionable improvements.

---

## 📊 Simple Flow

```
Upload Paper → Detect Sections → Select & Evaluate → Get Scores & Feedback → Download Report
     │                │                  │                    │                      │
   (PDF)        (Auto-detect)      (Choose which)      (Per-section +         (PDF/Markdown)
                                                         Overall)
```

---

## 📝 Paper Types

| Type | Focus | Key Sections |
|------|-------|-------------|
| **Empirical** | Data analysis, econometrics | Data, Methodology, Results |
| **Theoretical** | Mathematical models | Model Setup, Proofs, Extensions |
| **Policy** | Policy analysis | Background, Policy Context, Recommendations |

---

## 🔍 What Gets Evaluated

### Every Section Gets:
1. **Qualitative Assessment** - Overall narrative feedback (2-6 sentences)
2. **Criteria Scores** - 4-6 specific criteria rated 1-5 with justifications
3. **Supporting Quotes** - 2 quotes per criterion extracted from the text
4. **Improvement Suggestions** - 3-5 prioritized actionable recommendations
5. **Weighted Score** - Combined score accounting for criterion importance

### Example Section Evaluation Card:

```
┌─────────────────────────────────────────────────────────┐
│ Introduction — Score: 4.2/5.0 (adjusted: 4.2)           │
├─────────────────────────────────────────────────────────┤
│ Qualitative: The introduction effectively establishes   │
│ the research territory and identifies a clear gap...    │
│                                                          │
│ Criteria Breakdown:                                      │
│   ✓ Territory: 4/5 (15% weight)                        │
│   ✓ Niche: 5/5 (20% weight)                            │
│   ✓ Contribution: 4/5 (20% weight)                     │
│   ⚠ Thesis: 3/5 (20% weight)                           │
│   ✓ Roadmap: 4/5 (10% weight)                          │
│   ✓ Scope: 5/5 (15% weight)                            │
│                                                          │
│ Priority Improvements:                                   │
│   1. Sharpen the research question statement           │
│   2. Add preview of methodology at end                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🎚️ Scoring System

### Individual Scores (1-5 scale)
- **5** = Excellent, exemplary
- **4** = Good, above average
- **3** = Adequate, meets standards
- **2** = Significant weaknesses
- **1** = Major problems

### Section Score Calculation
```
Raw Score = Σ(criterion_score × criterion_weight)
                    ↓
Adjusted Score = Raw Score × Section Importance Multiplier
                    ↓
Overall Score = Average of all Adjusted Scores
```

### Publication Readiness
| Score | Status | Meaning |
|-------|--------|---------|
| < 2.5 | 🔴 Not Ready | Major revisions needed |
| 2.5-3.4 | 🟡 Needs Major Revisions | Significant work required |
| 3.5-4.2 | 🟢 Needs Minor Revisions | Good quality, polish needed |
| > 4.2 | ⭐ Ready | High quality |

---

## 📋 Quick Criteria Lookup

### Universal (All Papers)
- **Abstract**: Completeness, Clarity, Accuracy, Conciseness
- **Introduction**: Territory, Niche, Contribution, Thesis, Roadmap, Scope
- **Literature Review**: Coverage, Organization, Synthesis, Gap, Recency
- **Conclusion**: Consistency, Contribution, Limitations, Future Research, Implications
- **Discussion**: Interpretation, Mechanisms, Limitations, Implications

### Empirical Papers
- **Data**: Source, Appropriateness, Limitations, Sample, Variables
- **Methodology** ⭐: Specification, Identification, Assumptions, Robustness Plan, Replicability
- **Results**: Alignment, Statistical, Economic Magnitude, Anomalies, Presentation

### Theoretical Papers
- **Model Setup**: Assumptions, Notation, Motivation, Tractability, Relation to Lit
- **Proofs** ⭐: Correctness, Logical Flow, Completeness, Intuition
- **Extensions**: Meaningful Variation, Comparative Statics, Robustness, Relevance

### Policy Papers
- **Background**: Context, Data Landscape, Framing
- **Policy Context**: Institutional, Historical, Stakeholders, Current Debate
- **Recommendations** ⭐: Evidence Basis, Feasibility, Tradeoffs, Specificity

*(⭐ = Highest importance multiplier for that paper type)*

---

## 🎯 Section Importance Weights

### What Matters Most Per Paper Type:

**Empirical:**
```
Methodology      █████████████ 1.3  ← Most critical
Data, Results    ████████████  1.2
Robustness       ███████████   1.1
Introduction     ██████████    1.0
Discussion       █████████     0.9
Conclusion       ████████      0.8
Abstract         ███████       0.7
```

**Theoretical:**
```
Proofs           ██████████████ 1.4  ← Most critical
Model Setup      █████████████  1.3
Extensions       ███████████    1.1
Introduction     ██████████     1.0
Discussion       █████████      0.9
Conclusion       █████████      0.9
Abstract         ███████        0.7
```

**Policy:**
```
Recommendations  █████████████ 1.3  ← Most critical
Policy Context   ████████████  1.2
Introduction     ███████████   1.1
Background       ██████████    1.0
Conclusion       ███████████   1.1
Discussion       ██████████    1.0
Abstract         ███████       0.7
```

---

## 🚀 Usage Workflow

### 1️⃣ Setup
```
Select Paper Type: [Empirical] [Theoretical] [Policy]
Upload File: [Choose PDF/TXT/DOCX/LaTeX]
```

### 2️⃣ Section Detection (Automatic)
- System detects section headers
- Shows hierarchy (subsections grouped under parents)
- Preview extracted text before evaluation

### 3️⃣ Section Selection
For each detected section:
- ☑ **Keep** - Evaluate this section
- ⤴ **Merge** - Combine with another section
- ✕ **Remove** - Skip this section

### 4️⃣ Evaluation
- Click "Evaluate Selected Sections"
- Progress bar shows real-time status
- Results appear as expandable cards

### 5️⃣ Review Results
- Overall score + publication readiness
- Optional: Generate overall assessment (LLM synthesis)
- Expand individual sections for details

### 6️⃣ Export
- 📄 **Markdown** - Plain text format
- 📕 **PDF** - Professional formatted report

---

## 🔧 Technical Features

### Smart Section Detection
- **Heuristic scoring** - Analyzes capitalization, position, word count
- **LLM fallback** - Uses AI when heuristics uncertain
- **Hierarchy aware** - Recognizes numbered subsections (2.1, 2.2 under 2)

### Quote Validation
- Extracts supporting quotes from text
- Validates quotes actually appear in section
- Marks valid (✓) vs. invalid (~) quotes

### Caching
- Results cached by content hash
- Re-evaluating unchanged sections is instant
- Cache persists during session

### Error Handling
- Graceful fallback on JSON parsing failures
- Heuristic extraction if structured output fails
- Conservative default scores when uncertain

---

## 💡 Best Practices

### For Best Results:
1. **Choose correct paper type** - Determines which criteria apply
2. **Review detected sections** - Fix any misdetections before evaluating
3. **Merge related subsections** - e.g., "2.1 Data" + "2.2 Variables" → "Data"
4. **Provide paper context** - Optional abstract/summary helps LLM understand paper
5. **Evaluate key sections first** - Start with methodology/model/recommendations

### Common Issues:
- **Low scores on abstracts** - Often too terse or missing elements
- **Introduction missing "niche"** - Research gap not clearly stated
- **Methodology lacks identification** - Causal strategy unclear
- **Results ignore economic magnitude** - Focus on p-values only
- **Proofs missing intuition** - Pure math without economic interpretation

---

## 📁 File Structure

```
section_eval/
├── main.py                 # Streamlit UI + orchestration
├── evaluator.py           # Core evaluation logic
├── criteria/
│   └── base.py           # Criteria registry (all rules here)
├── section_detection.py  # Find section headers
├── text_extraction.py    # Parse PDF/LaTeX/etc
├── scoring.py            # Score computation + weights
├── prompts/
│   └── templates.py      # LLM prompt builder
├── hierarchy.py          # Parent-child section grouping
└── utils.py              # JSON parsing, caching, etc
```

---

## 🎨 UI Components

### Main Tabs
- **Section Evaluator** - Full evaluation workflow
- **Referee Report** - Different workflow (not covered here)

### Evaluation Steps
1. Paper type selector (radio buttons)
2. File uploader (shared across app)
3. Auto-detect or freeform input toggle
4. Section checkboxes + action dropdowns
5. Context input (optional)
6. Evaluate button
7. Results display (expandable cards)
8. Download buttons

### Result Display
- Overall metrics (score, readiness, count)
- Progress bar visualization
- Optional overall assessment
- Per-section expandable cards with:
  - Qualitative text
  - Criteria table with quotes
  - Improvements list
  - Score breakdown

---

## 🔗 Integration Points

### LLM Integration
- Uses `utils.cm` (ConversationManager) from parent eval/utils.py
- MartinAI Federal Reserve API
- Model: Claude Sonnet 4.5

### Session State
- `st.session_state["se_v3_eval_cache"]` - Evaluation cache
- `st.session_state["se_v3_results_{manuscript}"]` - Current results
- `st.session_state["se_v3_overall_text"]` - Overall assessment

### File Handling
- Files passed as `Dict[str, bytes]` from app.py
- Multiple file uploads supported
- Pasted text creates virtual files

---

## 📊 Example Output

```
════════════════════════════════════════════════════
OVERALL SUMMARY
════════════════════════════════════════════════════
Overall Score: 3.8 / 5.0
Publication Readiness: Needs Minor Revisions
Sections Evaluated: 6
Paper Type: Empirical

════════════════════════════════════════════════════
SECTION SCORES
════════════════════════════════════════════════════
Introduction        4.1 / 5.0  (adjusted: 4.1)
Literature Review   3.5 / 5.0  (adjusted: 3.2)
Data                4.2 / 5.0  (adjusted: 5.0) ⭐
Methodology         4.5 / 5.0  (adjusted: 5.0) ⭐
Results             3.8 / 5.0  (adjusted: 4.6)
Conclusion          3.2 / 5.0  (adjusted: 2.6)

════════════════════════════════════════════════════
TOP IMPROVEMENTS
════════════════════════════════════════════════════
1. Strengthen causal identification discussion
2. Add economic magnitude interpretation to Table 3
3. Expand discussion of limitations in conclusion
4. Include more recent literature (post-2022)
5. Clarify variable construction for key measures
```

