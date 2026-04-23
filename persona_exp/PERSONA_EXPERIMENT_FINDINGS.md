# Persona Selection Experiment Findings

**Experiment Date:** April 20, 2026
**Configuration:** Claude 4.5 Sonnet, Temperature 1.0, 10 personas available, 3 selected per run

## Available Personas (10 Total)

The following personas were available for selection in the experiment:

1. **Theorist** - Focuses on mathematical logic, proofs, correct derivations, and models with mathematical insight
2. **Econometrician** - Focuses on causal inference, endogeneity, identification strategies, and robustness of results
3. **ML_Expert** - Focuses on model architecture, hyperparameter tuning, train/test validity, interpretability
4. **Data_Scientist** - Focuses on data pipeline, cleaning, feature engineering, EDA, data leakage, preprocessing biases
5. **CS_Expert** - Focuses on algorithm creation, computational complexity, memory efficiency, hardware constraints
6. **Historian** - Focuses on literature lineage, context, gap analysis, and claims of novelty
7. **Visionary** - Focuses on paradigm-shifting potential, novelty, creativity, and intellectual risk
8. **Policymaker** - Focuses on policy applicability, welfare implications, and actionable insights for real-world use
9. **Ethicist** - Focuses on moral/social values, privacy, consent, fairness, accountability, philosophical implications
10. **Perspective** - Focuses on distributional consequences, dataset biases, fairness, impact on marginalized groups

**Note:** An "Empiricist" persona also exists in the codebase but was not included in this experiment.

## Summary

This experiment tested the consistency of persona selection across 10 runs for 4 different papers. The goal was to determine whether the LLM makes stable persona choices for a given paper.

---

## Paper 1: Patents (AI Patent Patterns in Financial Institutions)

**Tokens:** 20,390
**Total Runs:** 10
**Unique Combinations:** 1 (100% consistency)

### Most Common Combination (100%)
- Econometrician, Data_Scientist, Policymaker

### Persona Selection Frequency
- **Econometrician:** 10/10 runs (100%)
- **Data_Scientist:** 10/10 runs (100%)
- **Policymaker:** 10/10 runs (100%)

### Average Weights
- Econometrician: 0.38
- Data_Scientist: 0.33
- Policymaker: 0.29

### First Persona Frequency
- Econometrician: 7/10 runs (70%)
- Data_Scientist: 3/10 runs (30%)

**Finding:** Perfect consistency - all 10 runs selected the same three personas, though the ordering and exact weights varied slightly.

---

## Paper 2: KD (Machine Learning Paper)

**Tokens:** 13,993
**Total Runs:** 10
**Unique Combinations:** 2 (70% consistency for most common)

### Most Common Combination (70%)
- CS_Expert, Data_Scientist, ML_Expert

### Persona Selection Frequency
- **ML_Expert:** 10/10 runs (100%)
- **Data_Scientist:** 10/10 runs (100%)
- **CS_Expert:** 7/10 runs (70%)
- **Econometrician:** 3/10 runs (30%)

### Average Weights
- ML_Expert: 0.50
- Data_Scientist: 0.29
- CS_Expert: 0.22
- Econometrician: 0.20

### First Persona Frequency
- ML_Expert: 10/10 runs (100%)

**Finding:** High consistency - ML_Expert and Data_Scientist selected in all runs. CS_Expert vs Econometrician as third persona showed some variation.

---

## Paper 3: Taming Animal Spirits (Policy/Economic Theory)

**Tokens:** 35,571
**Total Runs:** 10
**Unique Combinations:** 3 (50% consistency for most common)

### Most Common Combination (50%)
- Data_Scientist, Econometrician, Policymaker

### Persona Selection Frequency
- **Econometrician:** 10/10 runs (100%)
- **Policymaker:** 10/10 runs (100%)
- **Data_Scientist:** 5/10 runs (50%)
- **ML_Expert:** 3/10 runs (30%)
- **Ethicist:** 2/10 runs (20%)

### Average Weights
- Econometrician: 0.43
- Data_Scientist: 0.31
- Policymaker: 0.30
- ML_Expert: 0.25
- Ethicist: 0.20

### First Persona Frequency
- Econometrician: 10/10 runs (100%)

**Finding:** Moderate consistency - Econometrician and Policymaker selected in all runs. Third persona varied between Data_Scientist (most common), ML_Expert, and Ethicist.

---

## Paper 4: Beigebook (Economic Report Analysis)

**Tokens:** 13,315
**Total Runs:** 10
**Unique Combinations:** 2 (90% consistency for most common)

### Most Common Combination (90%)
- Data_Scientist, Econometrician, Policymaker

### Persona Selection Frequency
- **Econometrician:** 10/10 runs (100%)
- **Data_Scientist:** 10/10 runs (100%)
- **Policymaker:** 9/10 runs (90%)
- **ML_Expert:** 1/10 runs (10%)

### Average Weights
- Econometrician: 0.50
- Data_Scientist: 0.30
- Policymaker: 0.20
- ML_Expert: 0.20

### First Persona Frequency
- Econometrician: 10/10 runs (100%)

**Finding:** Very high consistency - same three personas in 9/10 runs. One run replaced Policymaker with ML_Expert.

---

## Cross-Paper Analysis

### Most Frequently Selected Personas (Overall)
1. **Econometrician:** Selected in 100% of runs across all papers
2. **Data_Scientist:** Selected in 88% of all runs (35/40 total)
3. **Policymaker:** Selected in 73% of all runs (29/40 total)
4. **ML_Expert:** Selected in 35% of all runs (14/40 total)
5. **CS_Expert:** Selected in 18% of runs (7/40 total)
6. **Ethicist:** Selected in 5% of runs (2/40 total)

### Personas Never Selected (0% Selection Rate)
The following 4 personas were **never selected** across all 40 runs:
- **Theorist** - Mathematical logic and proofs
- **Historian** - Literature lineage and gap analysis
- **Visionary** - Paradigm-shifting potential and novelty
- **Perspective** - Distributional consequences and DEI concerns

### Rarely Selected Personas (<10% Selection Rate)
- **Ethicist** - Selected only 2/40 times (5%), both times for "Taming Animal Spirits" paper

### Consistency Patterns
- **Perfect consistency (100%):** 1 paper (Patents)
- **Very high consistency (≥90%):** 1 paper (Beigebook)
- **High consistency (≥70%):** 1 paper (KD)
- **Moderate consistency (50%):** 1 paper (Taming Animal Spirits)

### Key Observations
1. **Econometrician dominance:** Selected in all 40 runs and was the first-ranked persona in 37/40 runs
2. **Core trio:** Econometrician + Data_Scientist + Policymaker was the most common combination across papers (appears in 3/4 papers)
3. **Technical papers:** ML/CS-focused papers (KD) selected technical personas (ML_Expert, CS_Expert) over policy personas
4. **Unused personas:** 4 personas never selected, suggesting they may not be relevant for typical Fed research papers
5. **Weight stability:** When personas are selected, their weights remain relatively stable across runs

### Implications
- The persona selection system shows good-to-excellent consistency for most papers
- Paper type/content strongly influences persona selection (empirical economics → Econometrician/Data_Scientist/Policymaker; ML → ML_Expert/Data_Scientist/CS_Expert)
- Some personas may be redundant or rarely applicable to typical Fed research
- The selection appears deterministic enough to be reliable but flexible enough to adapt to paper content

---

## Additional Insights

### Why Were Certain Personas Never Selected?

**Theorist (0%):** Despite being a core academic persona, Theorist was never selected. This suggests:
- The papers tested were primarily empirical/applied rather than theoretical
- The "Econometrician" persona may be covering theoretical aspects sufficiently
- Fed research may emphasize empirical validation over pure mathematical theory

**Historian (0%):** Never selected across all papers, indicating:
- Literature review and gap analysis may not be viewed as critical for technical evaluation
- The selection prompt may prioritize technical/methodological review over contextual positioning
- Papers may have adequately covered their literature positioning, making this less critical

**Visionary (0%):** Complete absence suggests:
- The papers were incremental/applied rather than paradigm-shifting
- Fed research context prioritizes methodological rigor over intellectual novelty
- The selection system may favor technical precision over innovation assessment

**Perspective (0%):** Never selected, which reveals:
- DEI and distributional concerns may not be central to the papers tested
- Technical papers (ML, patents, economics) may not trigger distributional fairness concerns in the selection logic
- The persona may be too specialized for general Fed research papers

**Ethicist (5% - only 2 selections):** Rarely selected, appearing only for "Taming Animal Spirits":
- Ethics considerations are not viewed as critical for most economics/ML papers
- May only be relevant for papers with explicit social/welfare implications
- The two selections suggest the paper had normative content that triggered ethics review

### Dominant Persona: Econometrician

**Key Statistics:**
- Selected in 100% of runs (40/40)
- First-ranked persona in 92.5% of runs (37/40)
- Average weight: 0.43 (highest among all personas)

**Why is Econometrician so dominant?**
- **Fed context:** All papers were Fed research, which typically emphasizes empirical rigor and causal inference
- **Broad applicability:** Even ML/CS papers (KD) included econometric evaluation in 3/10 runs
- **Core competency:** Identification, causality, and statistical validity are fundamental to Fed research standards
- **Overlap with other personas:** May subsume some theoretical and empirical aspects that would otherwise go to Theorist/Historian

### Paper Type Determines Persona Mix

**Empirical Economics Papers (Patents, Beigebook, Taming Animal Spirits):**
- Core trio: Econometrician + Data_Scientist + Policymaker
- Average consistency: 80%
- Reasoning: Fed economics research requires causal inference (Econometrician), data quality (Data_Scientist), and policy relevance (Policymaker)

**Technical ML Papers (KD):**
- Core trio: ML_Expert + Data_Scientist + (CS_Expert OR Econometrician)
- Average consistency: 70%
- Reasoning: ML papers need model evaluation (ML_Expert), data pipeline review (Data_Scientist), and either algorithmic efficiency (CS_Expert) or causal interpretation (Econometrician)

### Consistency vs Paper Complexity

Inverse relationship observed between paper complexity and selection consistency:
- **Simple empirical paper (Patents, 20K tokens):** 100% consistency
- **Simple report (Beigebook, 13K tokens):** 90% consistency
- **Simple ML paper (KD, 14K tokens):** 70% consistency
- **Complex policy paper (Taming Animal Spirits, 36K tokens):** 50% consistency

**Hypothesis:** Longer, more complex papers may present multiple valid evaluation angles, reducing determinism in persona selection.

### Weight Distribution Patterns

**Econometrician consistently receives highest weight:**
- Range: 0.38 - 0.50 across papers
- Interpretation: Viewed as most critical reviewer regardless of paper type

**Data_Scientist receives stable mid-range weight:**
- Range: 0.29 - 0.33 across papers
- Interpretation: Data quality is universally important but not the primary concern

**Third persona receives lowest weight:**
- Range: 0.20 - 0.33
- Interpretation: Secondary considerations (policy, CS, ML) are important but subordinate

**Weight hierarchy reflects Fed priorities:** Methodological rigor > Data quality > Domain-specific concerns

### Selection Stability Within Paper Types

**High stability personas (selected when relevant):**
- Econometrician: Always selected, always highly weighted
- Data_Scientist: Selected in 88% of runs, stable weight
- Policymaker: Selected in 73% of runs when paper has policy implications

**Unstable personas (compete for third slot):**
- ML_Expert vs CS_Expert: Compete in ML papers (7 vs 3 in KD paper)
- ML_Expert vs Data_Scientist vs Ethicist: Compete in policy papers (3 vs 5 vs 2 in Taming Animal Spirits)

**Interpretation:** First two persona slots are highly deterministic; third slot is context-dependent and less stable.

### Recommendations for System Improvement

**Consider removing or merging underutilized personas:**
1. **Remove:** Theorist, Historian, Visionary, Perspective (0% usage)
2. **Keep as optional:** Ethicist (5% usage but domain-specific)
3. **Core personas:** Econometrician, Data_Scientist, Policymaker, ML_Expert, CS_Expert

**Alternative: Two-tier selection system:**
- **Tier 1 (always available):** Econometrician, Data_Scientist, Policymaker
- **Tier 2 (domain-specific):** ML_Expert, CS_Expert, Ethicist, Theorist, Historian

**Increase personas selected from 3 to 4:**
- Current system may force unnecessary trade-offs (e.g., choosing between Policymaker and ML_Expert)
- Four personas would reduce selection variance and cover more evaluation dimensions

**Test with non-Fed papers:**
- Current experiment only tested Fed research, which may bias toward Econometrician/Policymaker
- Test with pure theory papers, CS papers, or social science papers to validate persona utility

### Methodological Observations

**Temperature 1.0 provides reasonable consistency:**
- Even at high temperature, 3/4 papers showed ≥70% consistency
- Suggests the selection prompt and paper content strongly constrain the decision space

**Selection appears primarily content-driven:**
- Persona choices align with paper content (ML → ML_Expert, policy → Policymaker)
- Randomness primarily affects marginal decisions (third persona choice)

**Thinking budget (2048 tokens) may help consistency:**
- Extended reasoning may help the model converge on optimal personas
- Would be valuable to test with/without thinking budget to measure impact on consistency
