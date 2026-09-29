# Comparison: PaperReviewer vs CoEconomist (MAD System)

**Date**: May 6-7, 2026  
**Setup**: Three IFDP papers evaluated by both systems:
- **CoEconomist (MAD)**: 10-persona multi-agent debate system with 5 rounds, numeric scoring (max 30)
- **PaperReviewer**: 4-agent adversarial system (Core Analysis + Specialized + Author Defense + Orchestrator), severity-based recommendations

---

## Overview: All Papers

| Paper | Ground Truth | CoEconomist Verdict | CoEconomist Score | PaperReviewer Verdict | PaperReviewer Score | Match? |
|-------|--------------|---------------------|-------------------|----------------------|---------------------|--------|
| ifdp-2020-2 | Tier 4 | REVISE | 16/30 (53%) | **FAILED** (Error) | N/A | ❌ Error |
| ifdp-2020-4 | Tier 1 | FAIL | 12/30 (40%) | REVISE | 2.88/5.0 (58%) | ❌ Disagree |
| ifdp-2020-9 | Tier 4 | FAIL | 9/30 (30%) | **ACCEPT** | 2.45/5.0 (49%) | ❌ Disagree |

**Key**: 
- Tier 1 = Highest quality, Tier 4 = Lowest quality
- CoEconomist Score = Sum of 3 persona final scores (max 30)
- PaperReviewer Score = Average across 8 criteria (max 5.0)

**Success Rate**:
- CoEconomist: 3/3 papers processed successfully (100%)
- PaperReviewer: 2/3 papers processed successfully (67%)

**Agreement Rate**: 0/2 completed papers agree on verdict (0%)

---

## Paper-by-Paper Analysis

### **ifdp-2020-2**: "When is Bad News Good News? U.S. Monetary Policy, Macroeconomic News, and Financial Conditions in Emerging Markets"

**Ground Truth**: Tier 4 (Low quality)

| System | Verdict | Score | Notes |
|--------|---------|-------|-------|
| **CoEconomist** | REVISE | 16/30 (53%) | Unanimous agreement in Round 2C. Mid-range REVISE verdict. |
| **PaperReviewer** | **ERROR** | N/A | JSON parsing error in specialized cyber agent. Core analysis completed (12 criticisms), but specialized analysis failed. |

**Failure Mode**: PaperReviewer detected both "cyber" and "ai" technologies. Core economics agent completed successfully, but the cyber security specialist agent returned malformed JSON causing a `TypeError: float() argument must be a string or a real number, not 'NoneType'` error.

---

### **ifdp-2020-4**: "The Elusive Gains from Nationally-Oriented Monetary Policy"

**Ground Truth**: Tier 1 (Highest quality)

| System | Verdict | Score | Rationale |
|--------|---------|-------|-----------|
| **CoEconomist** | FAIL | 12/30 (40%) | Partial agreement (not unanimous). Personas identified significant methodological concerns. Low score reflects harsh evaluation. |
| **PaperReviewer** | **REVISE** | 2.88/5.0 (58%) | **1 unresolved MAJOR issue** triggered REVISE despite higher score. 70% confidence, estimated 1-2 months revision. |

**Key Difference**: 
- **CoEconomist** evaluated harshly (40% score) and recommended FAIL
- **PaperReviewer** gave higher score (58%) but still recommended REVISE due to **severity-based logic**

**Major Issue Identified by PaperReviewer**:
> "Open-loop Nash equilibrium is overly restrictive and may not capture realistic policy interactions"

**CoEconomist Concerns** (from May 6 run):
- Low consensus score (0.125) indicates personas found fundamental problems
- Partial agreement suggests divergent views on paper quality

**Verdict Paradox**: Ground truth says Tier 1 (highest quality), but both systems recommended rejection/revision. This suggests either:
1. The paper has genuine methodological issues both systems detected
2. Ground truth "Tier 1" may not mean "ready for publication" but rather "high-tier working paper"
3. Both systems may be overly harsh on methodological rigor

---

### **ifdp-2020-9**: "Sovereign Risk Matters: The Effects of Endogenous Default Risk on the Time-Varying Volatility of Interest Rate Spreads"

**Ground Truth**: Tier 4 (Low quality)

| System | Verdict | Score | Rationale |
|--------|---------|-------|-----------|
| **CoEconomist** | FAIL | 9/30 (30%) | Unanimous FAIL. Lowest score of the batch. Consensus score 0.0 indicates all personas rejected. |
| **PaperReviewer** | **ACCEPT** | 2.45/5.0 (49%) | **No MAJOR issues**, only 2 MODERATE issues. 50% confidence (lowest possible for ACCEPT). |

**Massive Disagreement**: 
- **CoEconomist**: Harshest verdict, unanimous rejection, lowest score
- **PaperReviewer**: **Recommended acceptance** despite lower score than ifdp-2020-4

**Why PaperReviewer Accepted Despite Low Score (2.45/5.0)**:

This is the **key insight** into PaperReviewer's recommendation logic. The decision isn't primarily based on the numeric score - it's heavily influenced by **the severity classification of unresolved issues**.

**ifdp-2020-9** (ACCEPT at 2.45/5.0):
- **Unresolved Issues**: 3 total
- **Severity**: 2 **MODERATE** issues + 1 minor
- **No MAJOR issues**
- **Justification**: "Excellent paper with solid contributions and methodology. Successfully defended 0% of criticisms. No significant unresolved issues. Recommend acceptance."

**ifdp-2020-4** (REVISE at 2.88/5.0):
- **Unresolved Issues**: 5 total
- **Severity**: **1 MAJOR issue** + 3 MODERATE issues
- **Major Issue**: "Open-loop Nash equilibrium is overly restrictive and may not capture realistic policy interactions"

**PaperReviewer's Severity-Based Logic**:

According to PaperReviewer's documentation (CLAUDE.md, fairness improvements from May 2026):

1. **CRITICAL issues** → Immediate REJECT (unless partially resolved)
2. **MAJOR unresolved issues** → Typically triggers REVISE AND RESUBMIT
3. **Only MODERATE/MINOR issues** → Can still get ACCEPT even with lower scores

The system implements these fairness mechanisms:
- **Severity downgrade**: Partially resolved issues get downgraded (MAJOR→MODERATE)
- **Impact downgrade**: Critical issues that are partially addressed don't auto-reject
- **Score-based thresholds**: Better papers get more tolerance for minor issues
- **MAJOR issues override scores**: Even high-scoring papers with MAJOR unresolved issues get REVISE

**The Logic**: Having even one unresolved MAJOR issue is worse than having a lower score with only MODERATE issues.

**Academic Rationale**: This makes sense from an academic review perspective - a paper can have lower overall polish (lower score) but if it doesn't have **fundamental methodological flaws** (MAJOR issues), it can still be acceptable. Whereas a higher-scoring paper with a fundamental flaw needs substantial revision regardless of its other strengths.

**Why CoEconomist Failed It**:
- CoEconomist uses **numeric scoring + consensus voting** without severity classification
- All three personas gave low scores (likely 2-3/10 each, totaling 9/30)
- Unanimous FAIL suggests personas identified fundamental issues that couldn't be overcome
- No author defense mechanism to challenge criticisms

---

## Architectural Differences

### **CoEconomist (MAD System)**

**Structure**:
- **Round 0**: LLM selects 3 of 10 personas + assigns weights
- **Round 1**: Independent analysis by all 3 personas
- **Rounds 2A-2C**: Multi-turn debate (cross-examination, answers, amendments)
- **Round 3**: Editor synthesizes weighted consensus

**Scoring**: 
- Each persona gives 1-10 confidence score per round
- Total score = sum of 3 final scores (max 30)
- Verdict based on weighted consensus: PASS=1.0, REVISE=0.5, FAIL=0.0

**Key Features**:
- Parallel execution of personas (async)
- Per-round temperature control (0.4-0.7)
- Quote validation (optional)
- Caching (optional)
- No author defense mechanism

---

### **PaperReviewer**

**Structure**:
- **Core Analysis Agent**: Domain-specific (Economics, Finance, CS, etc.) evaluates 8 criteria
- **Specialized Agent**: Conditional on tech keywords (AI/ML, Cyber, Quantum)
- **Author Agent**: Adversarial defense using ONLY paper text (no hallucinations)
- **Orchestrator**: Coordinates multi-turn debates (up to 6 turns)

**Scoring**:
- Each criterion scored 1-5 (Significance, Novelty, Methodology, Results, Clarity, Reproducibility, Related Work, Limitations)
- Average score computed across criteria
- **Severity classification**: MINOR, MODERATE, MAJOR, CRITICAL
- **Verdict logic**: Severity-based thresholds override numeric scores

**Key Features**:
- Adversarial architecture with author defense
- Multi-layer quote validation (fuzzy matching, 75% threshold)
- Fairness mechanisms (severity downgrade for partial resolutions)
- Professional referee report generation
- **Known issue**: Occasional JSON parsing errors in specialized agents

---

## Key Findings

### 1. **Fundamentally Different Decision Logic**

**CoEconomist**: 
- Relies on **weighted consensus** of categorical verdicts
- Numeric scores inform but don't override consensus
- Lower scores correlate with FAIL verdicts (9/30 → FAIL, 16/30 → REVISE)

**PaperReviewer**:
- Uses **severity-based thresholds** that override scores
- **One MAJOR issue can trigger REVISE** even with 58% score
- **No MAJOR issues can allow ACCEPT** even with 49% score

### 2. **Agreement Paradox**

For the two papers both systems completed:
- **ifdp-2020-4**: CoEconomist=FAIL (40%), PaperReviewer=REVISE (58%)
- **ifdp-2020-9**: CoEconomist=FAIL (30%), PaperReviewer=ACCEPT (49%)

**Both disagreed**, but in opposite directions:
- CoEconomist was **harsher** on both papers
- PaperReviewer's severity-based logic led to **more lenient** outcomes for papers without major flaws

### 3. **Ground Truth Mismatch**

Neither system aligned with ground truth tiers:
- **ifdp-2020-4** (Tier 1/highest): Both systems rejected/required revision
- **ifdp-2020-9** (Tier 4/lowest): CoEconomist failed, PaperReviewer accepted

This suggests:
1. "Tier" may not directly map to publication readiness
2. Both systems may evaluate differently than human reviewers who assigned tiers
3. Tier 1 papers may still have methodological issues worth flagging

### 4. **Reliability**

**CoEconomist**: 100% completion rate (3/3 papers)
**PaperReviewer**: 67% completion rate (2/3 papers)
- Failed on paper with multiple technology tags (cyber + ai)
- Known issue: JSON parsing errors in specialized agents

### 5. **Severity vs. Consensus**

The most striking difference:

**Scenario A** (ifdp-2020-9):
- **Lower score** (2.45/5.0 = 49%)
- **No major issues** → ACCEPT
- PaperReviewer: "No significant unresolved issues"

**Scenario B** (ifdp-2020-4):
- **Higher score** (2.88/5.0 = 58%)
- **1 major issue** → REVISE
- PaperReviewer: "1 major unresolved issue(s) requiring substantial work"

This demonstrates that **PaperReviewer prioritizes issue severity over overall score**, while **CoEconomist treats scores as the primary signal**.

---

## Recommendations

### For CoEconomist Improvements

1. **Add Severity Classification**: Implement MINOR/MODERATE/MAJOR/CRITICAL labels for identified issues
2. **Consider Severity-Based Overrides**: Single critical flaw should trigger specific recommendations
3. **Author Defense Mechanism**: Allow papers to "respond" to criticisms (could reduce false negatives)

### For PaperReviewer Improvements

1. **Fix JSON Parsing**: Specialized agents need more robust output parsing
2. **Calibrate Severity Thresholds**: 49% score + ACCEPT seems too lenient
3. **Add Consensus Voting**: Multiple agents should vote on severity classifications, not just one

### For Benchmarking

1. **Expand Test Set**: 2 completed papers insufficient to draw strong conclusions
2. **Investigate Ground Truth**: Understand what "Tier 1" actually means for interpretation
3. **Human Baseline**: Compare both systems to actual human referee reports
4. **Severity Annotations**: Add human-annotated severity labels to ground truth

---

## Conclusion

**CoEconomist** and **PaperReviewer** implement fundamentally different philosophies:

- **CoEconomist**: Democratic consensus with numeric confidence → More conservative (both papers failed)
- **PaperReviewer**: Severity-based quality gates with author defense → More nuanced (ACCEPT vs REVISE based on flaw types)

**Neither system agreed with ground truth**, suggesting:
1. Both systems may be harsher than human reviewers (Tier 1 paper rejected by both)
2. Or ground truth tiers don't map directly to publication recommendations
3. Or papers have genuine flaws both systems detected but humans overlooked

**Key Insight**: PaperReviewer's severity-based logic reveals that **how you score matters more than what you score** - a paper with 49% overall quality but no fundamental flaws can be publishable, while a 58% paper with one major methodological issue is not.

The 0% agreement rate on completed papers highlights the need for **hybrid approaches** that combine:
- Consensus voting (CoEconomist strength)
- Severity classification (PaperReviewer strength)  
- Author defense (PaperReviewer strength)
- Multiple perspectives (CoEconomist strength)

---

**Generated**: May 7, 2026  
**Systems Tested**:
- CoEconomist (MAD): v2.0 with numeric scoring (May 6, 2026 results)
- PaperReviewer: v1.0 with fairness improvements (May 2026)
