# Memo Evaluation System - Usage Guidance

## Running Threat Assessment / Risk Briefing Memos

The memo evaluation system has been updated (April 2026) to better handle forward-looking threat assessments, especially for emerging technology risks like AI-enabled cybersecurity threats.

### Key Improvements

1. **New Memo Type**: "Threat Assessment / Risk Briefing"
   - Specifically designed for cybersecurity, AI risks, emerging tech threats
   - Recognizes that limited historical precedent is EXPECTED
   - Focuses on threat characterization quality rather than implementation costs

2. **Footnote Citation Support**
   - System now checks for numbered references [1], [2], etc.
   - Claims won't be labeled "unbacked" if footnote references exist
   - Analysts instructed to note "See footnote [X]" in their evaluations

3. **Appropriate Standards for Forward-Looking Analysis**
   - Scenario analysis and expert judgment recognized as valid evidence
   - Quantitative models + qualitative scenarios are complementary, not contradictory
   - Risk characterization logic evaluated, not historical actuarial precision

### How to Use for AI Cybersecurity Threat Memos

When running a memo like "AI and Cybersecurity Risks to Financial Stability":

1. **Select the correct memo type**: Choose **"Threat Assessment / Risk Briefing"** from the dropdown
   - This will guide the system to select Financial Stability Analyst (for loss scenarios) and Data Analyst (for scenario reasoning)
   - Policy Analyst will assess threat characterization logic

2. **Ensure footnotes are included**: Make sure the full text extraction includes footnote/endnote sections
   - PDF extraction via pdfplumber should capture these automatically
   - For manual text paste, include the footnotes section at the end

3. **Optional custom context**: Use the custom evaluation context field to emphasize:
   ```
   Focus on threat characterization quality and scenario reasoning for emerging AI-enabled
   cyber threats. Limited historical precedent is expected. Evaluate whether loss scenarios
   are credible and risk characterization is sound.
   ```

### What Changed in Evaluation Criteria

#### For Threat Assessment Memos, Analysts Now Understand:

**Data Analyst:**
- ✅ Scenario analysis is valid evidence when historical data is limited
- ✅ Quantified estimates (base-case models) + qualitative scenarios (tail risks) serve different purposes
- ✅ Evaluates QUALITY of scenario reasoning, not just existence of historical precedent

**Financial Stability Analyst:**
- ✅ Threat briefings should quantify POTENTIAL LOSSES, not intervention costs
- ✅ Cost-benefit analysis of interventions is NOT required for risk identification memos
- ✅ Scenario ranges (e.g., $100M base case to $50B worst case) are appropriate risk management
- ✅ For emerging threats: assesses risk characterization LOGIC, not actuarial precision

**Policy Analyst:**
- ✅ Threat assessments raise awareness and inform preparedness—complete implementation plans not required
- ✅ Precautionary emphasis for novel threats is appropriate if uncertainty is acknowledged
- ✅ Evaluates analytical soundness, not completeness of implementation details

### Example Memo Characteristics That Should Now PASS

A well-written threat assessment memo about AI-enabled cybersecurity threats should:

✅ **Characterize the threat landscape** with clear scenario analysis
✅ **Quantify potential losses** using reasonable modeling assumptions (even with wide uncertainty ranges)
✅ **Cite sources** via footnotes for claims about Claude Code leak, Mythos release, attack vectors
✅ **Acknowledge uncertainty** explicitly where historical precedent is limited
✅ **Present scenario ranges** from base-case quantified estimates to worst-case qualitative scenarios
✅ **Suggest response directions** proportionate to identified threats (even without full implementation plans)

### What Would Still Trigger REJECT

Fatal flaws that would still fail a threat assessment:

❌ **Fabricated evidence** or claims about events that didn't occur
❌ **Fundamental mischaracterization** of the threat (e.g., claiming AI can't be used for cyber attacks)
❌ **Internal logical contradictions** where different sections make mutually exclusive claims
❌ **Missing critical evidence** with NO citations (not even footnotes) for key factual claims
❌ **Demonstrable mathematical errors** that void the core risk characterization

### Expected Analyst Selection for AI Cyber Threat Memos

For "AI and Cybersecurity Risks to Financial Stability" with memo type "Threat Assessment":

**Typical Selection:**
- Financial Stability Analyst: 0.45 (evaluates systemic risk and loss scenarios)
- Data Analyst: 0.35 (evaluates scenario reasoning and evidence quality)
- Policy Analyst: 0.20 (evaluates threat characterization logic)

**Alternative Selection:**
- Data Analyst: 0.40 (if evidence quality is the primary concern)
- Financial Stability Analyst: 0.35
- Policy Analyst: 0.25

## Troubleshooting

### If memo still gets rejected for "unbacked claims":

1. **Check footnote extraction**: Verify that the PDF/text extraction captured the footnotes section
2. **Verify footnote numbering**: Ensure claims have visible `[1]`, `[2]` references in the text
3. **Add custom context**: Emphasize "Check footnote citations before labeling claims as unbacked"

### If memo gets rejected for "lack of historical precedent":

1. **Verify memo type selection**: Must select "Threat Assessment / Risk Briefing"
2. **Check threat framing**: Explicitly acknowledge in memo text that threat is emerging/novel
3. **Add custom context**: "Evaluate scenario reasoning quality for emerging threats with limited historical data"

### If memo gets rejected for "cost-benefit analysis missing":

1. **Verify memo type**: Should be "Threat Assessment", not "Policy Recommendation"
2. **Check if interventions are specific**: If memo proposes detailed interventions (not just general response directions), it may need cost estimates
3. **Clarify memo purpose**: Make clear in intro whether memo is identifying threats vs. proposing specific policies

## Testing the Changes

To verify the updated prompts work correctly, try running your "AI and Cybersecurity Risks to Financial Stability" memo again with:

```
Memo Type: Threat Assessment / Risk Briefing
Custom Context: "Focus on threat characterization quality and scenario reasoning for emerging
AI-enabled cyber threats. Evaluate whether loss scenarios are credible and appropriately bounded."
```

The evaluation should now:
- ✅ Recognize footnote citations as valid evidence
- ✅ Accept forward-looking scenario analysis as appropriate methodology
- ✅ Evaluate loss characterization rather than demanding intervention cost-benefit analysis
- ✅ Not penalize for "lack of historical precedent" on AI-enabled attack vectors
- ✅ Accept scenario ranges from quantified base-case to qualitative worst-case as sound risk management

## Files Modified

- `app_system/referee/memo_prompts.py` - Updated all analyst system prompts
- `app_system/app-memo.py` - Added "Threat Assessment / Risk Briefing" memo type option
