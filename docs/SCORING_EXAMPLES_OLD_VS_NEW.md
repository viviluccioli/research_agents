# Scoring Examples: Old System vs New System

This document illustrates how the same paper sections would be scored under the old vs new evaluation system.

---

## Example 1: Model/Proofs Section (Theoretical Paper)

### Paper Excerpt A (High Quality - Rithika's approach)

> "To derive the price-dividend ratio, we must first establish convergence conditions for the infinite sum. The price at time 0 can be expressed as P₀ = E₀[∑ᵗ₌₁^∞ Mₜ Dₜ], where Mₜ is the stochastic discount factor. For this sum to converge, we require that the expected discounted dividend growth rate be less than unity.
>
> Formally, convergence requires: βᵞ E[e^(-γb + h)] < 1, which simplifies to: -γb + h + (γ²b²)/2 < -ln(β).
>
> This condition has clear economic interpretation: the disaster risk premium (captured by the γb term) must dominate the AI growth premium (h term) after accounting for risk aversion and precautionary savings (γ²b²/2 term). When b is small or h is large, the model predicts explosive valuations, reflecting insufficient hedging value relative to growth exposure.
>
> We can decompose the risk premium into two components: E[R - Rf] = Consumption Risk Premium + Disaster Risk Premium. The consumption risk premium equals (γ²σ²)/2 in the no-disaster state, while the disaster risk premium equals p·γb/(1-p). Calibrating to historical consumption volatility (σ=0.02) and reasonable risk aversion (γ=2), the disaster component dominates when p > 0.001, suggesting that even low-probability AI risks substantially affect valuations.
>
> Parameter b=0.6 implies a 45% consumption drop in disaster states, consistent with labor share estimates if AI fully displaces human labor while ownership remains concentrated (Acemoglu and Restrepo, 2020)."

### OLD SYSTEM SCORING

**Criteria:**
- `correctness` (35%): **Score 4** - "Derivation is mathematically correct"
- `logical_flow` (25%): **Score 4** - "Steps follow logically"
- `completeness` (20%): **Score 3** - "Derivation is complete"
- `intuition` (20%): **Score 4** - "Economic interpretation provided"

**Overall: 3.80**

**Problems with old scoring:**
- Doesn't reward the explicit convergence condition derivation
- Doesn't credit the risk premium decomposition
- Doesn't recognize parameter calibration to empirical literature
- Treats "provides intuition" as binary (yes/no) rather than evaluating depth

---

### NEW SYSTEM SCORING

**Criteria:**
- `correctness` (20%): **Score 5** - "Derivation is mathematically correct with no errors"
- `rigor` (25%): **Score 5** - "Convergence condition explicitly derived with clear parameter restrictions and economic interpretation of when it binds"
- `economic_depth` (25%): **Score 5** - "Provides multi-layered interpretation: (1) convergence condition economic meaning, (2) risk premium decomposition into components, (3) calibration of parameters to empirical estimates with citation (Acemoglu and Restrepo, 2020), (4) welfare interpretation of b parameter"
- `sophistication` (15%): **Score 5** - "Analysis goes beyond routine derivation: decomposes risk premium, calibrates parameters, connects to labor economics literature"
- `clarity` (15%): **Score 5** - "Derivation is clear and well-explained"

**Overall: 5.00**

**Justification for high score:**
- Explicitly derives convergence condition with economic interpretation ✓
- Decomposes risk premium into consumption risk vs disaster risk ✓
- Calibrates parameters to empirical estimates with citations ✓
- Provides welfare interpretation (b as labor displacement) ✓
- Multiple layers of economic reasoning ✓

---

## Example 2: Model/Proofs Section (Basic Quality - Andrew's approach)

### Paper Excerpt B (Low Quality)

> "The price-dividend ratio can be derived using standard asset pricing theory. We have P₀ = E₀[∑ᵗ₌₁^∞ Mₜ Dₜ]. After some algebra, this gives us:
>
> P₀/D₀ = (1-p) + p·e^h / [1 - β·((1-p) + p·e^(-γb+h))]
>
> This formula shows that the price-dividend ratio increases with h (the AI growth parameter) and decreases with b (the disaster magnitude). This makes economic sense: higher AI growth makes stocks more valuable, while larger disasters make them less valuable. The risk premium is E[R] - Rf, which we can compute from the model."

### OLD SYSTEM SCORING

**Criteria:**
- `correctness` (35%): **Score 3** - "Derivation appears correct but skips steps ('after some algebra')"
- `logical_flow` (25%): **Score 3** - "Logical but compressed"
- `completeness` (20%): **Score 3** - "Main result presented but derivation incomplete"
- `intuition` (20%): **Score 3** - "Basic intuition provided"

**Overall: 3.00**

**Problems with old scoring:**
- Gives same "adequate" score despite lacking depth
- Doesn't penalize absence of convergence conditions
- Doesn't differentiate shallow intuition from deep interpretation
- Doesn't note missing parameter justification

---

### NEW SYSTEM SCORING

**Criteria:**
- `correctness` (20%): **Score 3** - "Formula appears correct but derivation skips steps ('after some algebra')"
- `rigor` (25%): **Score 2** - "No convergence condition derived or discussed. No mention of when the infinite sum converges, what parameter restrictions are required, or when the formula is valid. The phrase 'after some algebra' hides critical steps."
- `economic_depth` (25%): **Score 2** - "Intuition is superficial and tautological: 'higher growth makes stocks more valuable' is obvious. No decomposition of risk premium, no discussion of mechanisms, no welfare interpretation, no calibration of parameters to data."
- `sophistication` (15%): **Score 2** - "Routine application of standard asset pricing formula without advancement. No novel insights, no connections to broader literature, parameters chosen without justification."
- `clarity` (15%): **Score 4** - "Writing is clear though abbreviated"

**Overall: 2.50**

**Justification for low score:**
- No convergence condition derived ✗
- No decomposition of results ✗
- Intuition is tautological ("high growth → high value") ✗
- No parameter calibration or empirical grounding ✗
- "After some algebra" hides important steps ✗

---

## Example 3: Methodology Section (Empirical Paper)

### Paper Excerpt A (High Quality)

> "Our identification strategy exploits the staggered rollout of AI regulation across EU countries. The key threat to identification is that countries adopting regulations earlier may differ systematically in unobservable ways. We address this through three approaches:
>
> First, we demonstrate parallel pre-trends: AI stock valuations in early-adopter vs late-adopter countries moved in parallel for 24 months pre-treatment (see Figure 2). We formally test this using the event study specification in equation (3), finding no significant pre-treatment coefficients (joint F-test p=0.42).
>
> Second, we include extensive controls for country-level economic conditions that might affect both regulation timing and valuations: GDP growth, tech sector employment share, and prior banking crises. Our results are robust to these controls (Table 4, columns 2-3).
>
> Third, we conduct placebo tests assigning pseudo-treatment dates 12 and 24 months before actual regulation. These yield null effects (Table 5), supporting the parallel trends assumption.
>
> The exclusion restriction requires that regulation timing affects valuations only through actual regulatory changes, not through correlated factors. We argue this is plausible because: (1) regulation dates were set by EU-wide negotiations largely determined by bureaucratic timing rather than country fundamentals (see Appendix A for institutional detail), (2) controlling for economic conditions eliminates the most obvious confounds, (3) placebo tests show no effects at arbitrary dates.
>
> We acknowledge remaining concerns: if countries anticipated regulation and adjusted behavior before official dates, our estimates understate true effects. We partially address this using Google Trends data on 'AI regulation' searches..."

### OLD SYSTEM SCORING

**Criteria:**
- `specification` (25%): **Score 4** - "Model clearly specified"
- `identification` (30%): **Score 4** - "Identification strategy stated and justified"
- `assumptions` (20%): **Score 4** - "Assumptions stated"
- `robustness_plan` (15%): **Score 4** - "Robustness checks planned"
- `replicability` (10%): **Score 4** - "Sufficient detail provided"

**Overall: 4.00**

---

### NEW SYSTEM SCORING

**Criteria:**
- `specification` (20%): **Score 5** - "Model clearly specified with event study design"
- `identification` (30%): **Score 5** - "Compelling identification argument: (1) parallel pre-trends formally tested, (2) extensive controls for confounds, (3) multiple placebo tests, (4) exclusion restriction defended with institutional detail about EU bureaucratic timing. Three-pronged approach addresses key threats rigorously."
- `assumptions` (20%): **Score 5** - "Assumptions not just stated but defended: parallel trends tested formally (F-test), exclusion restriction argued with institutional detail, remaining concerns (anticipation effects) acknowledged with partial solution (Google Trends)"
- `robustness_depth` (20%): **Score 5** - "Comprehensive pre-specified robustness: event study with multiple leads/lags, controls added systematically, placebo tests at multiple arbitrary dates. Clear falsification strategy."
- `replicability` (10%): **Score 5** - "Highly detailed: references specific equations, tables, figures, appendix sections"

**Overall: 5.00**

---

### Paper Excerpt B (Low Quality)

> "We use a difference-in-differences design to estimate the effect of AI regulation on valuations. Our identification strategy relies on the staggered adoption of regulations across countries.
>
> The main assumption is parallel trends, which we believe is reasonable given that all countries are developed economies with similar tech sectors. We control for GDP growth to address potential confounds.
>
> Our baseline specification is: Valuation_it = α + β·Regulation_it + γ·GDP_it + δ_i + θ_t + ε_it
>
> We will also check robustness using alternative specifications."

### OLD SYSTEM SCORING

**Criteria:**
- `specification` (25%): **Score 3** - "Model specified"
- `identification` (30%): **Score 3** - "Identification strategy stated"
- `assumptions` (20%): **Score 3** - "Parallel trends assumption mentioned"
- `robustness_plan` (15%): **Score 3** - "Mentions robustness checks"
- `replicability` (10%): **Score 4** - "Equation provided"

**Overall: 3.05**

---

### NEW SYSTEM SCORING

**Criteria:**
- `specification` (20%): **Score 4** - "Model clearly specified with equation"
- `identification` (30%): **Score 2** - "Identification strategy merely asserted. Parallel trends 'believed to be reasonable' without evidence. No pre-trend tests mentioned. No discussion of exclusion restriction. No falsification tests planned. Claims similarity justifies parallel trends but provides no evidence."
- `assumptions` (20%): **Score 2** - "Assumptions stated but not defended. Parallel trends assumption crucial but no testing or evidence provided. Simply asserts 'we believe' this is reasonable."
- `robustness_depth` (20%): **Score 2** - "Robustness plan is vague ('alternative specifications') without specificity. What specifications? What threats do they address? Single control variable (GDP) is minimal. No discussion of sample variation, measurement alternatives, or systematic robustness strategy."
- `replicability` (10%): **Score 4** - "Equation provided with clear notation"

**Overall: 2.60**

**Justification for low score:**
- Parallel trends assumption asserted without evidence ✗
- No pre-trend testing planned ✗
- No placebo/falsification tests ✗
- Exclusion restriction not discussed ✗
- Robustness plan is vague and unspecific ✗
- Identification relies on "we believe" rather than evidence ✗

---

## Summary: Score Differentiation

| Paper Quality | OLD System | NEW System | Spread |
|--------------|-----------|-----------|--------|
| **Example 1A (High)** | 3.80 | 5.00 | +1.20 |
| **Example 1B (Low)** | 3.00 | 2.50 | -0.50 |
| **Example 3A (High)** | 4.00 | 5.00 | +1.00 |
| **Example 3B (Low)** | 3.05 | 2.60 | -0.45 |

**Key Improvement:** New system creates 2.0-2.5 point spread between high and low quality (vs 0.75-1.0 in old system).

---

## What Changed?

### Old System Problems:
1. ✓ Has convergence condition → Score 4
2. ✓ Has intuition → Score 3-4
3. ✓ Mentions parallel trends → Score 3
4. **Result: Everything gets 3-4**

### New System Advantages:
1. **Depth matters**: One-sentence intuition scores 2-3, multi-layered interpretation scores 4-5
2. **Rigor matters**: "After some algebra" scores 2, explicit derivation scores 5
3. **Evidence matters**: "We believe" scores 2, formal tests score 5
4. **Sophistication matters**: Routine application scores 2-3, novel insights score 4-5
5. **Comprehensiveness matters**: Minimal checks score 2, extensive pre-specified checks score 5

**Bottom line:** New system differentiates on quality of execution, not just presence of elements.
