# Section Evaluator Documentation Index

## 📚 Documentation Files

| File | Size | Purpose | Read This If... |
|------|------|---------|-----------------|
| **QUICK_REFERENCE.md** | 12K | Quick overview | You want a 5-minute understanding of the system |
| **ARCHITECTURE.md** | 31K | System architecture | You need to understand how components work together |
| **CRITERIA_REFERENCE.md** | 13K | Complete criteria guide | You want detailed criteria explanations and examples |
| **CRITERIA_MATRIX.md** | 17K | Criteria by paper type | You need exact criteria for specific section/paper combos |

---

## 🚀 Quick Start

**New to the system?** Read in this order:
1. **QUICK_REFERENCE.md** - Get the basics (10 min read)
2. **ARCHITECTURE.md** (Concise section) - Understand the flow (5 min)
3. **CRITERIA_MATRIX.md** - See what's evaluated (reference)

**Need specific information?**
- **"How does evaluation work?"** → ARCHITECTURE.md (Detailed diagram)
- **"What criteria are used for X section?"** → CRITERIA_MATRIX.md
- **"Why did my paper get this score?"** → CRITERIA_REFERENCE.md (Scoring formulas)
- **"How do I use the UI?"** → QUICK_REFERENCE.md (Usage workflow)

---

## 📖 Document Summaries

### QUICK_REFERENCE.md
**Best for:** Users, quick lookups, understanding output

**Contains:**
- 1-page system overview
- Simple flow diagrams
- Scoring system explained
- Quick criteria tables
- Usage workflow
- Common issues & best practices
- Example output

**Key Sections:**
- "What It Does" (30 seconds)
- "Simple Flow" (visual)
- "Quick Criteria Lookup" (tables)
- "Section Importance Weights" (bars)
- "Usage Workflow" (6 steps)

---

### ARCHITECTURE.md
**Best for:** Developers, system understanding, technical details

**Contains:**
- Detailed layer-by-layer architecture diagram
- Concise architecture diagram (simplified)
- Data flow visualization
- Module responsibilities
- Caching strategy
- Scoring formulas
- LLM interaction patterns

**Key Sections:**
- "DETAILED ARCHITECTURE DIAGRAM" (comprehensive)
- "CONCISE ARCHITECTURE DIAGRAM" (simple)
- "DATA FLOW SUMMARY"
- "SCORING FORMULA"
- "LLM INTERACTION PATTERN"

---

### CRITERIA_REFERENCE.md
**Best for:** Understanding evaluation logic, score interpretation

**Contains:**
- Universal criteria (all papers)
- Empirical paper criteria
- Theoretical paper criteria
- Policy paper criteria
- Section importance multipliers
- Publication readiness scale
- Score calculation examples
- Design philosophy

**Key Sections:**
- "UNIVERSAL CRITERIA" (Abstract, Intro, Lit Review, etc.)
- "EMPIRICAL PAPER CRITERIA" (Data, Methodology, Results)
- "THEORETICAL PAPER CRITERIA" (Model, Proofs, Extensions)
- "POLICY PAPER CRITERIA" (Background, Context, Recommendations)
- "SCORE CALCULATION EXAMPLE" (worked example)

---

### CRITERIA_MATRIX.md
**Best for:** Looking up specific criteria, complete reference

**Contains:**
- Complete Paper Type × Section Type matrix
- Every criterion with weight & multiplier
- Empirical: 10 sections, 42 total criteria
- Theoretical: 8 sections, 32 total criteria
- Policy: 9 sections, 38 total criteria
- Summary heatmap
- Fallback criteria

**Key Sections:**
- "EMPIRICAL PAPERS" (section-by-section tables)
- "THEORETICAL PAPERS" (section-by-section tables)
- "POLICY PAPERS" (section-by-section tables)
- "SUMMARY HEATMAP" (quick importance comparison)
- "FALLBACK CRITERIA" (when no match found)

---

## 🔍 Finding Information

### By Question Type

**"How does X work?"**
- Text extraction → ARCHITECTURE.md (Text Extraction Layer)
- Section detection → ARCHITECTURE.md (Section Detection Layer)
- Evaluation → ARCHITECTURE.md (Evaluation Orchestration)
- Scoring → ARCHITECTURE.md (Scoring Formula) or CRITERIA_REFERENCE.md
- PDF generation → ARCHITECTURE.md (Results Rendering)

**"What criteria apply to X?"**
- Specific section/paper combo → CRITERIA_MATRIX.md (lookup table)
- All criteria for paper type → CRITERIA_REFERENCE.md
- Quick lookup → QUICK_REFERENCE.md (Quick Criteria Lookup)

**"Why did I get this score?"**
- Scoring formula → ARCHITECTURE.md or CRITERIA_REFERENCE.md
- Importance multipliers → CRITERIA_MATRIX.md (heatmap)
- Publication readiness → CRITERIA_REFERENCE.md (scale)

**"How do I use it?"**
- Step-by-step → QUICK_REFERENCE.md (Usage Workflow)
- Best practices → QUICK_REFERENCE.md (Best Practices)
- Common issues → QUICK_REFERENCE.md (Common Issues)

---

## 📊 Visual Guides

### Architecture Diagrams
- **Detailed** → ARCHITECTURE.md (DETAILED ARCHITECTURE DIAGRAM)
- **Concise** → ARCHITECTURE.md (CONCISE ARCHITECTURE DIAGRAM)
- **Data Flow** → ARCHITECTURE.md (DATA FLOW SUMMARY)
- **Simple** → QUICK_REFERENCE.md (Simple Flow)

### Criteria Tables
- **By Paper Type** → CRITERIA_MATRIX.md (complete tables)
- **Quick Reference** → QUICK_REFERENCE.md (abbreviated tables)
- **With Examples** → CRITERIA_REFERENCE.md (detailed explanations)

### Importance Weights
- **Heatmap** → CRITERIA_MATRIX.md (SUMMARY HEATMAP)
- **Visual Bars** → QUICK_REFERENCE.md (Section Importance Weights)
- **Lists** → CRITERIA_REFERENCE.md (multiplier lists)

---

## 🎯 Paper Type Coverage

### Supported Types
- ✅ **Empirical** - Econometric analysis, field experiments
- ✅ **Theoretical** - Mathematical models, game theory
- ✅ **Policy** - Policy analysis, recommendations

### Not Currently Covered
- ❌ Finance (can be added from base.py)
- ❌ Macro (can be added from base.py)
- ❌ Systematic Review (can be added from base.py)

---

## 🔑 Key Concepts

### Section Importance Multiplier
Adjusts section scores based on their importance to the paper type.
- Example: Methodology for empirical papers = 1.3×
- See: CRITERIA_REFERENCE.md or CRITERIA_MATRIX.md

### Criterion Weight
How much each criterion contributes to the section score.
- Weights sum to 1.0 within a section
- See: Any CRITERIA_* file

### Publication Readiness
Overall assessment derived from the overall score.
- < 2.5: Not Ready
- 2.5-3.4: Needs Major Revisions
- 3.5-4.2: Needs Minor Revisions
- > 4.2: Ready

### Quote Validation
System validates that quotes actually appear in the section text.
- ✓ = Valid quote found
- ~ = Quote not found or mismatched

---

## 🛠️ For Developers

### Code Organization
```
section_eval/
├── main.py              # UI orchestration
├── evaluator.py         # Core evaluation logic
├── criteria/base.py     # ALL CRITERIA DEFINED HERE ⭐
├── section_detection.py # Find section headers
├── text_extraction.py   # Parse files
├── scoring.py           # Score computation
├── prompts/templates.py # LLM prompt builder
├── hierarchy.py         # Section grouping
└── utils.py            # Helpers
```

### Modifying Criteria
1. Open `criteria/base.py`
2. Find the relevant section (e.g., `_EMPIRICAL`, `_THEORETICAL`, `_POLICY`)
3. Modify criterion dicts: `{"name": "...", "weight": 0.xx, "description": "..."}`
4. Ensure weights sum to 1.0
5. Update documentation files if needed

### Adding Paper Types
1. Uncomment in `PAPER_TYPES` list in `criteria/base.py`
2. Update `_ALL_CRITERIA` dict to include new type
3. Update `SECTION_IMPORTANCE` in `scoring.py`
4. Update documentation

---

## 📝 Examples

### Example 1: Finding Methodology Criteria for Empirical Papers
**File:** CRITERIA_MATRIX.md
**Section:** "EMPIRICAL PAPERS" → "Section: Methodology"
**Result:** 5 criteria with weights and descriptions

### Example 2: Understanding Why Proofs Are Important
**File:** CRITERIA_REFERENCE.md
**Section:** "THEORETICAL PAPER CRITERIA" → "Proofs / Derivations"
**Result:** 1.4× multiplier explanation + common issues

### Example 3: Learning the Evaluation Workflow
**File:** QUICK_REFERENCE.md
**Section:** "Usage Workflow"
**Result:** 6-step process with descriptions

### Example 4: Seeing All Paper Type Importance Weights
**File:** CRITERIA_MATRIX.md
**Section:** "SUMMARY HEATMAP"
**Result:** Complete comparison table with stars

---

## 🔄 Document Updates

When updating documentation:
1. Update the relevant detail files first (CRITERIA_*, ARCHITECTURE)
2. Update QUICK_REFERENCE.md with any major changes
3. Update this index if new sections added
4. Keep examples current

---

## 📧 Support

For questions about:
- **Usage** → Check QUICK_REFERENCE.md first
- **Scores** → Check CRITERIA_REFERENCE.md
- **Technical** → Check ARCHITECTURE.md
- **Specific Criteria** → Check CRITERIA_MATRIX.md

---

## 📄 File Locations

All documentation files are in:
```
/ofs/home/m1vcl00/FS-CASL/research_agents-main/eval/
```

Supporting code is in:
```
/ofs/home/m1vcl00/FS-CASL/research_agents-main/eval/section_eval/
```

