# Evaluation Agent Framework
## AI-Assisted Academic Review for Economics Research

**Federal Reserve System Research Evaluation Framework**
Version 3.0 | March 2026

---

## Overview

We built a dual-system framework for evaluating economics research papers using AI:

1. **Multi-Agent Debate (MAD)**: Simulates peer review through debates between specialized AI personas → produces holistic accept/reject recommendations
2. **Section Evaluator**: Provides detailed, criteria-based scoring of individual sections → generates actionable improvement feedback

Both systems are **paper-type aware** (empirical/theoretical/policy), require **textual evidence** for all claims, and use **proportional error weighting** to avoid over-penalizing minor issues.

---

## The Problem

### Why Current Peer Review Struggles

- **Bottleneck**: 3-6 month review cycles, increasing submissions
- **Inconsistency**: Different reviewers apply different standards
- **Expertise gaps**: No single reviewer masters theory + empirics + policy
- **Subjectivity**: Implicit biases influence decisions

### Why Economics Papers Are Hard to Evaluate

Economics spans incompatible paradigms:
- **Empirical**: Needs causal identification, robustness checks, data quality assessment
- **Theoretical**: Needs proof verification, economic insight evaluation, mathematical rigor
- **Policy**: Needs institutional context, feasibility analysis, welfare implications

Standard AI review fails because it uses one-size-fits-all criteria and hallucinates criticisms without evidence.

---

## System 1: Multi-Agent Debate

**Purpose**: Simulate peer review panel deliberation

**How it works**:

```
Paper Upload
     ↓
[R0] AI selects 3 relevant personas from pool of 5
     → Assigns expertise weights (e.g., Empiricist 40%, Historian 30%, Policymaker 30%)
     ↓
[R1] Each persona independently evaluates → verdict: PASS/REVISE/FAIL
     ↓
[R2A] Personas read each other's reports → pose challenging questions
     ↓
[R2B] Personas answer peer questions → concede flaws or defend positions
     ↓
[R2C] Personas update verdicts after full debate
     ↓
[R3] Editor calculates weighted consensus → final decision

     Score = Σ(verdict × weight)
     PASS=1.0, REVISE=0.5, FAIL=0.0

     >0.75 → ACCEPT
     0.40-0.75 → REJECT & RESUBMIT
     <0.40 → REJECT
```

**Available personas**:

| Persona | Focus | Key Questions |
|---------|-------|---------------|
| **Theorist** | Math, proofs, models | Are derivations correct? Are assumptions justified? |
| **Empiricist** | Data, identification | Is causality credible? Are robustness checks sufficient? |
| **Historian** | Literature, gaps | Is the claimed gap real? Are citations appropriate? |
| **Visionary** | Innovation, impact | Is this genuinely novel or incremental? |
| **Policymaker** | Real-world application | Can central banks use this? What are welfare implications? |

**Output**: Referee report with weighted decision, synthesized feedback, full debate transcript

---

## System 2: Section Evaluator

**Purpose**: Granular feedback on specific manuscript sections

**How it works**:

```
Paper Upload → Extract text (PDF/LaTeX) → Detect sections → Identify paper type
     ↓
For each section:
  1. Load paper-type + section-specific criteria
     (e.g., empirical methodology: identification=30%, robustness=20%, ...)
  2. Evaluate section against criteria → score 1-5 per criterion
  3. Extract 2 verbatim quotes per criterion as evidence
  4. Validate quotes exist in source text
  5. Weight scores by criterion importance
     ↓
Aggregate scores across sections:
  - Apply section importance multipliers (methodology=1.3×, abstract=0.7×)
  - Check minimum score gate (no section below threshold)
  - Generate publication readiness assessment
```

**Example criteria weights** (empirical methodology):
- Identification strategy: **30%** (most critical)
- Robustness depth: 20%
- Specification: 20%
- Assumptions: 20%
- Replicability: 10%

**Output**: Per-section scores, criterion breakdowns, prioritized improvements, evidence quotes

---

## Key Innovations

### 1. Endogenous Persona Selection
**Problem**: A DSGE policy paper needs different reviewers than an empirical labor study.

**Solution**: AI reads the paper first, then selects the 3 most relevant personas from the pool and assigns expertise weights.

**Example**: Monetary policy paper → Theorist (35%), Policymaker (40%), Historian (25%)
vs. Empirical labor paper → Empiricist (50%), Historian (30%), Policymaker (20%)

### 2. Structured Debate Protocol
**Problem**: Single-pass AI evaluations miss the iterative refinement of real peer review.

**Solution**: Multi-round cross-examination forces personas to:
- Challenge each other's assumptions with specific questions
- Defend claims with textual evidence
- Acknowledge valid counterarguments
- Update beliefs based on debate

**Why it matters**: Catches errors like "Historian claims this fills a gap, but Empiricist finds the gap was already filled by prior work."

### 3. Proportional Error Weighting
**Problem**: AI treats all errors equally (typo = fatal identification flaw).

**Solution**: Evaluate errors by: (% of paper affected) × (severity to core claim)

**Example**:
> ❌ Without: "FAIL: Paper contains statistical errors"
>
> ✅ With: "REVISE: One robustness specification (Appendix Table A4, ~5% of results) has incorrect standard errors. Main findings (Tables 1-5) use correct specification. Fix A4 and verify other appendix tables. Moderate revision, not rejection."

### 4. Paper-Type-Aware Criteria
**Problem**: Empirical and theoretical papers need different standards.

**Solution**: Dynamic criteria loading based on paper + section type.

**Empirical methodology** emphasizes:
- Identification strategy (30%)
- Robustness depth (20%)
- Assumptions defense (20%)

**Theoretical model** emphasizes:
- Derivation rigor (25%)
- Economic depth (25%)
- Sophistication (20%)

### 5. Evidence-Backed Assessment
**Problem**: LLMs hallucinate criticisms not in the paper.

**Solution**: For every criterion score, extract 2 verbatim quotes. Algorithmically validate quotes exist in source. Flag invalid quotes with ⚠️.

**Example**:
```
Criterion: Identification Strategy (Score: 3/5)

Quote 1 (✓ Valid):
"We exploit state-level banking deregulation as an instrument..." (p.12)

Quote 2 (⚠️ Invalid):
"Potential confounders include state-level shocks..."
[This phrase not found in paper]
```

### 6. Hierarchical Importance Weighting
**Problem**: Weak abstract is less concerning than flawed methodology.

**Solution**: Two-level scoring:

**Level 1**: Criterion weights within section (sum to 100%)
**Level 2**: Section importance multipliers across paper

**Empirical paper multipliers**:
- Methodology: **1.3×** (critical)
- Data: 1.2×
- Results: 1.2×
- Introduction: 1.0× (baseline)
- Abstract: 0.7×

**Publication readiness** requires both:
- Overall score > threshold (e.g., 4.5/5)
- Minimum section score (e.g., all sections > 3.5/5)

This prevents stellar intro from masking fatal methodology.

---

## When to Use Which System

### Multi-Agent Debate
✅ **Use for**:
- Complete manuscripts ready for submission
- High-stakes decisions (tenure review, journal submission)
- Papers crossing multiple subfields
- Holistic accept/reject recommendation needed

⏱️ **Runtime**: 3-5 minutes (13-16 LLM calls)

📄 **Output**: Overall verdict, referee report, debate transcript

### Section Evaluator
✅ **Use for**:
- Works in progress (iterative feedback)
- Targeted revision (improving specific sections)
- Training junior researchers
- Quick feedback during writing

⏱️ **Runtime**: 1-2 minutes for typical paper

📄 **Output**: Section scores, criterion breakdowns, improvement priorities

### Complementary Workflow
1. **Early drafting**: Section Evaluator for iterative feedback
2. **Pre-submission**: Multi-Agent Debate for holistic check
3. **Post-review**: Section Evaluator to target referee concerns

---

## Technical Architecture

```
┌─────────────────────────────────────────┐
│        Web Interface (Streamlit)        │
│   [Referee Tab]  [Section Eval Tab]    │
└─────────────────────────────────────────┘
              ↓                ↓
    ┌─────────────┐    ┌──────────────┐
    │ MAD Engine  │    │  Section     │
    │ • Personas  │    │  Evaluator   │
    │ • Debate    │    │  • Criteria  │
    │ • Voting    │    │  • Scoring   │
    └─────────────┘    └──────────────┘
              ↓                ↓
         ┌──────────────────────────┐
         │   LLM Infrastructure     │
         │   • Claude 3.7 (MAD)     │
         │   • Claude 4.5 (Section) │
         │   • Extended thinking    │
         │   • Caching              │
         └──────────────────────────┘
```

**Models**:
- Claude Sonnet 3.7: MAD (balanced reasoning + creativity)
- Claude Sonnet 4.5: Section Eval (high consistency)
- Extended thinking: 2048 token budget for internal reasoning

**Document processing**:
- PDF: pdfplumber with OCR fixes
- LaTeX: Pure regex stripping (preserves math)
- Auto section detection: Heuristics + LLM confirmation

---

## Validation & Quality Control

### Quote Validation
- Algorithmic verification that quotes exist in source
- Tolerance for whitespace/OCR variations
- Target: >95% validity rate

### Consistency Testing
- Same paper → stable persona selections
- Score variance <10% across runs
- Qualitative assessments maintain themes

### Expert Calibration
- Compare AI vs. human expert assessments
- Agreement on accept/revise/reject decisions
- Target: 80%+ expert agreement that feedback is useful

---

## Current Limitations

### What the System Cannot Do
1. **Verify proofs**: Cannot rigorously check mathematical derivations (flags suspicious steps for human review)
2. **Run replication code**: Evaluates reported methodology, not actual results
3. **Access unpublished work**: Training cutoff means very recent papers may be missed
4. **Resolve subjective trade-offs**: Surfaces "novelty vs. rigor" tensions but leaves decision to humans

### Ethical Guardrails
- **Transparency**: Users know feedback is AI-generated; full reasoning visible
- **Human authority**: AI recommends, doesn't decide
- **Bias mitigation**: Blind evaluation (no author names), multi-persona debate, regular audits
- **Privacy**: Federal Reserve internal API, no model training on uploads
- **Accountability**: Feedback loops for users to flag bad advice

---

## Use Cases

### 1. Pre-Submission Review
Economist uploads draft → Section Evaluator flags weak methodology → revise → MAD confirms readiness → submit

### 2. Editorial Screening
Journal receives 200 submissions → MAD triages borderline papers → informs reviewer assignment

### 3. PhD Training
Students upload term papers → receive detailed Section Evaluator feedback → instructor reviews before final grade

### 4. Internal Policy Papers
Fed economist drafts analysis → Section Evaluator checks policy relevance → MAD assesses before senior review

### 5. Replication Studies
Research team audits published paper → Section Evaluator highlights robustness gaps → guides replication efforts

---

## Future Enhancements

**Near-term** (6-12 months):
- Citation graph integration (validate "gap" claims against literature)
- Figure/table quality assessment (multimodal LLMs)
- Interactive dialogue (authors ask follow-up questions)

**Long-term** (1-3 years):
- Replication integration (link to code verification)
- Continuous evaluation (real-time feedback as you write)
- Cross-disciplinary expansion (political science, natural sciences)

---

## Summary

This framework advances AI-assisted review through:

✅ **Multi-agent deliberation** (not single-pass scoring)
✅ **Paper-type awareness** (not one-size-fits-all)
✅ **Evidence requirements** (not hallucinated criticisms)
✅ **Proportional weighting** (not absolute judgments)
✅ **Human augmentation** (not replacement)

**Philosophy**: AI handles systematic evaluation of structure, methodology, and evidence. Humans make final judgments on novelty, significance, and fit.

**Status**: Production-ready, active use in Federal Reserve System

---

**Contact**: research-agents@federalreserve.gov
**Document Version**: 3.0 | March 12, 2026
