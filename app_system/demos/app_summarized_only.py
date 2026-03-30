# app_summarized_only.py - Standalone Demo of Main Referee Workflow
"""
Standalone demo showcasing the main production referee workflow.

This is identical to what's used in the main app.py. It demonstrates
the multi-agent debate system with LLM-powered summarization for
cleaner UI display.
"""

import streamlit as st
import json
import tempfile
import os
import asyncio
import datetime
import re
from typing import Dict, List, Any
import pdfplumber
import sys
from pathlib import Path

# Add parent directory to path so we can import from app_system
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import cm, single_query
from referee.engine import execute_debate_pipeline, SELECTION_PROMPT, SYSTEM_PROMPTS, DEBATE_PROMPTS
from referee._utils.summarizer import summarize_all_rounds

# Import helper functions from archived full output UI
from referee._archived.full_output_ui import (
    CUSTOM_CSS,
    format_severity_labels,
    format_verdict,
    format_final_verdict,
    extract_verdict,
    summarize_text,
    format_round1_output,
    format_round2a_output,
    format_round2c_output,
    generate_summary_table
)


class RefereeReportCheckerSummarized:
    """
    Multi-agent debate paper evaluation with LLM summarization for cleaner UI.

    This workflow adds an additional LLM pass to compress debate outputs while preserving
    full details in expanders. The system uses 5 rounds of debate between specialized personas.
    """

    def __init__(self, llm=cm):
        """Initialize with conversation manager from utils."""
        self.llm = llm

    def render_ui(self, files=None):
        """Render the Streamlit UI for the multi-agent referee report."""
        # Apply custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        st.title("🤖 Multi-Agent Referee Evaluation")

        # Add description expander at the top
        with st.expander("📖 **How This System Works**", expanded=False):
            st.markdown("""
### What This System Does

The **Multi-Agent Debate (MAD) System** evaluates research papers through structured debate between specialized AI personas.

**Note:** This workflow includes an **additional LLM summarization pass** to compress debate outputs for cleaner display. Full transcripts are available in expanders below.

### Workflow Overview

The system operates in **5 sequential rounds**:

1. **Round 0 - Persona Selection**: An LLM Chief Editor selects 3 of 5 available personas most relevant to the paper and assigns importance weights (summing to 1.0).

2. **Round 1 - Independent Evaluation**: Each selected persona independently evaluates the paper from their domain expertise, identifying flaws with severity labels ([FATAL], [MAJOR], [MINOR]) and providing an initial verdict (PASS/REVISE/FAIL).

3. **Round 2A - Cross-Examination**: Personas read each other's Round 1 evaluations and engage in cross-domain synthesis, constructive pushback, and clarification questions.

4. **Round 2B - Direct Examination**: Each persona responds to questions directed at them from Round 2A, providing evidence and taking a position (CONCEDE or DEFEND).

5. **Round 2C - Final Amendments**: After reviewing the full debate transcript, each persona submits a final amended verdict with justification.

6. **Round 3 - Editor Consensus**: A weighted consensus score is computed mathematically (PASS=1.0, REVISE=0.5, FAIL=0.0), and the Editor writes the official referee report.

### Available Personas

- **Theorist**: Mathematical logic, proofs, model soundness
- **Empiricist**: Data structures, identification strategies, statistical validity
- **Historian**: Literature lineage, contextualization, gap analysis
- **Visionary**: Novelty, paradigm-shifting potential, intellectual impact
- **Policymaker**: Real-world applicability, welfare implications, policy relevance

### Decision Thresholds

- **Consensus Score > 0.75**: ACCEPT
- **Consensus Score 0.40–0.75**: REJECT AND RESUBMIT
- **Consensus Score < 0.40**: REJECT

---

**💡 Note**: The full debate output with all persona responses is available in the expanders below and in the exported document at the bottom.
            """)

        # Add system prompts viewer
        with st.expander("🔍 **View System Prompts by Round**", expanded=False):
            st.markdown("Select tabs below to view the system prompts used to guide AI agents in each round.")

            # Create sub-tabs for each round
            prompt_tabs = st.tabs(["Round 0", "Round 1", "Round 2A", "Round 2B", "Round 2C", "Round 3"])

            with prompt_tabs[0]:
                st.markdown("### Round 0: Persona Selection Prompt")
                st.markdown("The Chief Editor uses this prompt to select 3 personas and assign weights.")
                st.code(SELECTION_PROMPT, language="text")

            with prompt_tabs[1]:
                st.markdown("### Round 1: Persona System Prompts")
                st.markdown("Each persona has a specialized system prompt defining their role and evaluation criteria.")
                persona_subtabs = st.tabs(list(SYSTEM_PROMPTS.keys()))
                for idx, persona_name in enumerate(SYSTEM_PROMPTS.keys()):
                    with persona_subtabs[idx]:
                        st.code(SYSTEM_PROMPTS[persona_name], language="text")

            with prompt_tabs[2]:
                st.markdown("### Round 2A: Cross-Examination Prompt")
                st.markdown("Guides personas to synthesize peer evaluations and ask clarification questions.")
                st.code(DEBATE_PROMPTS["Round_2A_Cross_Examination"], language="text")

            with prompt_tabs[3]:
                st.markdown("### Round 2B: Direct Examination Prompt")
                st.markdown("Instructs personas to answer questions and take a position (CONCEDE or DEFEND).")
                st.code(DEBATE_PROMPTS["Round_2B_Direct_Examination"], language="text")

            with prompt_tabs[4]:
                st.markdown("### Round 2C: Final Amendment Prompt")
                st.markdown("Asks personas to integrate all debate information and submit final verdicts.")
                st.code(DEBATE_PROMPTS["Round_2C_Final_Amendment"], language="text")

            with prompt_tabs[5]:
                st.markdown("### Round 3: Editor Consensus Prompt")
                st.markdown("The Editor receives the computed consensus and writes the official referee report.")
                st.code(DEBATE_PROMPTS["Round_3_Editor"], language="text")

        st.markdown("---")

        # Main UI content
        st.write("This tool uses a multi-agent debate system to evaluate your manuscript from multiple perspectives.")
        st.info("📊 **Note:** An additional LLM summarization pass compresses debate outputs for cleaner display. Full reports are available in expanders below.")

        # Check if files are available
        if not files:
            st.info("Please upload your manuscript using the document uploader above.")
            return

        # Agent Persona Descriptions
        st.markdown("### 👥 Available Personas (3 will be selected)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="persona-box mathematician-box">
                <h4>🔢 Theorist</h4>
                <p><strong>Focus:</strong> Mathematical logic, formal proofs, model insight</p>
                <p><strong>Criteria:</strong> Derivation correctness, assumption validity, theoretical soundness</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="persona-box mathematician-box">
                <h4>📊 Empiricist</h4>
                <p><strong>Focus:</strong> Data, econometrics, identification strategy</p>
                <p><strong>Criteria:</strong> Statistical validity, empirical soundness, data quality</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="persona-box visionary-box">
                <h4>🚀 Visionary</h4>
                <p><strong>Focus:</strong> Innovation, paradigm-shifting potential</p>
                <p><strong>Criteria:</strong> Novelty, creativity, intellectual impact</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="persona-box historian-box">
                <h4>📚 Historian</h4>
                <p><strong>Focus:</strong> Literature context, citation lineage</p>
                <p><strong>Criteria:</strong> Gap analysis, novelty claims, contextualization</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="persona-box editor-box">
                <h4>🏛️ Policymaker</h4>
                <p><strong>Focus:</strong> Real-world application, policy relevance</p>
                <p><strong>Criteria:</strong> Welfare implications, actionable insights, practical value</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("💡 The system will automatically select the 3 most relevant personas based on your paper's content and assign importance weights.")

        st.markdown("---")

        # Configuration Section
        st.markdown("### ⚙️ Configuration")

        col1, col2 = st.columns(2)

        with col1:
            model_choice = st.selectbox(
                "Model Selection",
                options=["Claude Sonnet 4.5 (Default)", "Claude 3.7 Sonnet"],
                index=0,
                help="Select which Claude model to use for evaluation",
                key="mad_model_selector"
            )

        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.1,
                help="Higher = more creative/varied, Lower = more focused/consistent. Default: 1.0",
                key="mad_temperature"
            )

        # Optional Paper Context
        with st.expander("📝 **Optional: Provide Paper Context** (Helps with persona selection)", expanded=False):
            st.markdown("""
            Optionally provide additional context about your paper to help the system select the most relevant personas and focus areas.

            **Examples:**
            - "This is primarily an empirical paper, but I'm especially concerned about the policy implications section"
            - "Focus on the theoretical contributions and mathematical rigor"
            - "I want close scrutiny of the identification strategy and robustness checks"
            """)

            paper_context = st.text_area(
                "Paper Context & Evaluation Priorities",
                placeholder="Optional: Describe your paper and what aspects you want evaluated most closely...",
                height=100,
                key="mad_paper_context"
            )

        st.markdown("---")

        # File selection
        manuscript_file = st.selectbox(
            "Select your manuscript for multi-agent evaluation",
            options=list(files.keys()),
            key="manuscript_selector"
        )

        if st.button("🚀 Run Multi-Agent Evaluation", type="primary"):
            if not manuscript_file:
                st.error("Please select a manuscript file.")
                return

            with st.spinner("Extracting text from manuscript..."):
                # Extract text from manuscript
                paper_text = self.extract_text_from_pdf(files[manuscript_file])

            # Run the multi-agent debate
            try:
                # Create progress tracking
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(stage, progress):
                    status_text.text(f"🔄 {stage}")
                    progress_bar.progress(progress)

                # Determine model selection
                if "4.5" in model_choice:
                    selected_model = "model_selection"  # Claude Sonnet 4.5
                else:
                    selected_model = "model_selection3"  # Claude 3.7 Sonnet

                # Execute debate pipeline with configuration
                with st.spinner("Running multi-agent debate..."):
                    debate_results = asyncio.run(
                        execute_debate_pipeline(
                            paper_text,
                            progress_callback=update_progress,
                            paper_context=paper_context if paper_context.strip() else None,
                            model_key=selected_model,
                            temperature=temperature
                        )
                    )

                # Run summarization pass
                with st.spinner("Compressing outputs for display (additional LLM pass)..."):
                    summaries = asyncio.run(summarize_all_rounds(debate_results))

                # Store results in session state
                st.session_state.debate_results = debate_results
                st.session_state.debate_summaries = summaries

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display results
                self.display_debate_results(debate_results, summaries)

            except Exception as e:
                st.error(f"Error during multi-agent evaluation: {e}")
                st.exception(e)


    def extract_text_from_pdf(self, file_content):
        """Extract text from a PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()

            text = ""
            try:
                with pdfplumber.open(temp_file.name) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            except Exception as e:
                st.error(f"Error extracting text from PDF: {e}")

            # Clean up
            os.unlink(temp_file.name)
            return text

    def display_debate_results(self, results: Dict, summaries: Dict):
        """Display the multi-agent debate results with summarized outputs."""
        st.success("✅ Multi-Agent Evaluation Complete!")

        # Icon mapping for all personas
        icon_map = {
            "Theorist": "🔢",
            "Empiricist": "📊",
            "Mathematician": "🔢",
            "Historian": "📚",
            "Visionary": "🚀",
            "Policymaker": "🏛️"
        }

        # Get active personas from round 0
        active_personas = results.get('round_0', {}).get('selected_personas', ["Mathematician", "Historian", "Visionary"])
        weights = results.get('round_0', {}).get('weights', {})

        # ROUND 0: Persona Selection
        if 'round_0' in results:
            st.markdown('<div class="round-header">🎯 ROUND 0: PERSONA SELECTION</div>', unsafe_allow_html=True)
            with st.expander("📋 **Selected Review Panel**", expanded=True):
                st.markdown(f"**Selected Personas:** {', '.join(active_personas)}")
                st.markdown("**Weights:**")
                for persona, weight in weights.items():
                    st.markdown(f"- **{persona}**: {weight}")
                if 'justification' in results['round_0']:
                    st.markdown(f"**Justification:** {results['round_0']['justification']}")

            # System Prompt Display
            with st.expander("🔍 **View System Prompt for Round 0**", expanded=False):
                st.code(SELECTION_PROMPT, language="text")

            st.markdown("---")

        # ROUND 1: Independent Evaluation (SUMMARIZED)
        st.markdown('<div class="round-header">⚡ ROUND 1: INDEPENDENT EVALUATION</div>', unsafe_allow_html=True)
        st.markdown("*Each persona independently evaluates the paper with severity-weighted findings.*")
        st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 1
        with st.expander("🔍 **View System Prompts for Round 1**", expanded=False):
            for persona in active_personas:
                st.markdown(f"**{persona} System Prompt:**")
                st.code(SYSTEM_PROMPTS.get(persona, "N/A"), language="text")
                st.markdown("---")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            # Display SUMMARIZED version by default
            summary_data = summaries['round_1_summaries'].get(role, {})
            summary_text = summary_data.get('summary', 'No summary available')
            verdict = summary_data.get('verdict', 'UNKNOWN')

            with st.container():
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(f"### {icon} {role.upper()}")
                st.markdown(f"**Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)
                st.markdown(summary_text, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Full report in expander
                with st.expander(f"📄 View Full {role} Report", expanded=False):
                    raw_text = results['round_1'][role]
                    formatted_text = format_severity_labels(raw_text)
                    st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2A: Cross-Examination (SUMMARIZED)
        st.markdown('<div class="round-header">🔄 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
        st.markdown("*Personas challenge each other's findings and ask clarifying questions.*")
        st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 2A
        with st.expander("🔍 **View System Prompt for Round 2A**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2A_Cross_Examination"], language="text")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            # Display SUMMARIZED version
            summary_text = summaries['round_2a_summaries'].get(role, 'No summary available')

            with st.container():
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(f"### {icon} {role.upper()} - Cross-Examination")
                st.markdown(summary_text, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Full report in expander
                with st.expander(f"📄 View Full {role} Cross-Examination", expanded=False):
                    raw_text = results['round_2a'][role]
                    formatted_text = format_severity_labels(raw_text)

                    # Parse and display structured sections
                    insights_pattern = r'\*\*Cross-Domain Insights[:\*]*\s*(.*?)(?=\*\*Constructive Pushback|\*\*Clarification Requests|\Z)'
                    insights_match = re.search(insights_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                    if insights_match:
                        st.markdown("**Cross-Domain Insights:**")
                        insights_text = insights_match.group(1).strip()
                        st.markdown(insights_text, unsafe_allow_html=True)
                        st.markdown("")

                    pushback_pattern = r'\*\*Constructive Pushback[:\*]*\s*(.*?)(?=\*\*Clarification Requests|\Z)'
                    pushback_match = re.search(pushback_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                    if pushback_match:
                        st.markdown("**Constructive Pushback:**")
                        pushback_text = pushback_match.group(1).strip()
                        st.markdown(pushback_text, unsafe_allow_html=True)
                        st.markdown("")

                    clarification_pattern = r'\*\*Clarification Requests[:\*]*\s*(.*?)(?=\Z)'
                    clarification_match = re.search(clarification_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                    if clarification_match:
                        st.markdown("**Clarification Requests:**")
                        clarification_text = clarification_match.group(1).strip()
                        st.markdown(clarification_text, unsafe_allow_html=True)

                    if not (insights_match or pushback_match or clarification_match):
                        st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2B: Direct Examination (SUMMARIZED)
        st.markdown('<div class="round-header">💬 ROUND 2B: ANSWERING QUESTIONS</div>', unsafe_allow_html=True)
        st.markdown("*Personas respond to peer questions with evidence and concessions/defenses.*")
        st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 2B
        with st.expander("🔍 **View System Prompt for Round 2B**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2B_Direct_Examination"], language="text")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            # Display SUMMARIZED version
            summary_text = summaries['round_2b_summaries'].get(role, 'No summary available')

            with st.container():
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(f"### {icon} {role.upper()} - Responses")
                st.markdown(summary_text, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Full report in expander
                with st.expander(f"📄 View Full {role} Responses", expanded=False):
                    raw_text = results['round_2b'][role]
                    peers = [p for p in active_personas if p != role]

                    for peer in peers:
                        pattern = rf'\*\*Response to {peer}[:\*]*\s*(.*?)(?=\*\*Response to |\*\*Concession or Defense|\Z)'
                        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)

                        if match:
                            response_content = match.group(1).strip()
                            st.markdown(f"**Response to {peer}:**")
                            st.markdown(response_content)
                            st.markdown("---")

                    # General concession or defense section
                    pattern = r'\*\*Concession or Defense[:\*]*\s*(.*?)(?=\Z)'
                    match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
                    if match:
                        content = match.group(1).strip()
                        st.markdown(f"**Overall Position:**")
                        st.markdown(content)

        st.markdown("---")

        # ROUND 2C: Final Amendments (SUMMARIZED)
        st.markdown('<div class="round-header">⚖️ ROUND 2C: FINAL AMENDMENTS</div>', unsafe_allow_html=True)
        st.markdown("*Personas submit final verdicts after integrating full debate.*")
        st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)

        # System Prompt Display for Round 2C
        with st.expander("🔍 **View System Prompt for Round 2C**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2C_Final_Amendment"], language="text")

        cols = st.columns(3)
        for idx, role in enumerate(active_personas):
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            # Display SUMMARIZED version
            summary_data = summaries['round_2c_summaries'].get(role, {})
            summary_text = summary_data.get('summary', 'No summary available')
            verdict = summary_data.get('verdict', 'UNKNOWN')

            with cols[idx]:
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(f"### {icon} {role.upper()}")
                st.markdown(f"**Final Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)
                st.markdown(summary_text, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Full report in expander
                with st.expander(f"📄 Full Report", expanded=False):
                    raw_text = results['round_2c'][role]
                    formatted_text = format_severity_labels(raw_text)
                    st.markdown(formatted_text, unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 3: Consensus Calculation & Editor Decision (SUMMARIZED)
        st.markdown('<div class="round-header">📜 ROUND 3: WEIGHTED CONSENSUS & EDITOR DECISION</div>', unsafe_allow_html=True)

        # System Prompt Display for Round 3
        with st.expander("🔍 **View System Prompt for Round 3**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_3_Editor"], language="text")

        # Display deterministic consensus calculation
        if 'consensus' in results:
            st.markdown("### 🔢 Deterministic Weighted Consensus")
            consensus = results['consensus']

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown("**Individual Verdicts & Weights:**")
                for persona in active_personas:
                    verdict = consensus['verdicts'].get(persona, 'UNKNOWN')
                    weight = weights.get(persona, 0)
                    verdict_value = 1.0 if verdict == 'PASS' else 0.5 if verdict == 'REVISE' else 0.0
                    contribution = weight * verdict_value
                    st.markdown(f"- **{persona}** (weight={weight}): {verdict} = {verdict_value} → contribution = {contribution:.3f}")

            with col2:
                st.markdown("**Consensus Score:**")
                st.metric("Weighted Score", f"{consensus['weighted_score']:.3f}", help="Out of 1.0")
                st.markdown("**Decision Threshold:**")
                st.code("""Score > 0.75: ACCEPT
0.40 ≤ Score ≤ 0.75: REJECT & RESUBMIT
Score < 0.40: REJECT""", language="text")

            st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border: 2px solid #388e3c; margin: 15px 0;">
                <strong style="font-size: 18px;">📊 Computed Decision: {consensus['decision']}</strong>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

        # Editor Report (SUMMARIZED)
        st.markdown('<span class="summary-badge">SUMMARIZED VIEW</span>', unsafe_allow_html=True)

        final_decision = results.get('consensus', {}).get('decision', 'UNKNOWN')
        editor_summary = summaries.get('editor_summary', 'No summary available')

        st.markdown(f"""
        <div class="persona-box editor-box">
        <h3>🏛️ Senior Editor's Report</h3>
        <div style="text-align: center; margin: 20px 0;">
        {format_final_verdict(final_decision)}
        </div>
        <p style="text-align: center; color: #666; font-size: 14px;">
        (Decision computed deterministically from weighted consensus)
        </p>
        """, unsafe_allow_html=True)

        st.markdown(editor_summary, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Full editor report in expander
        with st.expander("📄 View Full Editor Report", expanded=False):
            final_text = results['final_decision']
            sections = re.split(r'\*\*([^*]+)\*\*', final_text)
            for i in range(0, len(sections), 2):
                if i + 1 < len(sections):
                    header = sections[i + 1].strip().rstrip(':')
                    content = sections[i + 2].strip() if i + 2 < len(sections) else ""
                    st.markdown(f"**{header}:**")
                    st.markdown(content, unsafe_allow_html=True)

        st.markdown("---")

        # Summary Table
        st.subheader("📊 Evaluation Summary")

        # Create table data
        table_data = []
        r1_verdicts = {p: extract_verdict(results['round_1'].get(p, '')) for p in active_personas}
        r2c_verdicts = {p: extract_verdict(results['round_2c'].get(p, '')) for p in active_personas}

        for persona in active_personas:
            weight = weights.get(persona, 0)
            r1_v = r1_verdicts[persona]
            r2c_v = r2c_verdicts[persona]
            changed = "Yes ✅" if r1_v != r2c_v else "No ➖"
            table_data.append({
                "Persona": persona,
                "Weight": weight,
                "Round 1 Verdict": r1_v,
                "Final Verdict (R2C)": r2c_v,
                "Changed?": changed
            })

        # Display as dataframe
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        weighted_score = results.get('consensus', {}).get('weighted_score', 0)

        st.markdown(f"""
        **FINAL EDITORIAL DECISION (Weighted Score: {weighted_score:.3f}):** {format_final_verdict(final_decision)}
        """, unsafe_allow_html=True)

        # Metadata Section
        if 'metadata' in results:
            st.markdown("---")
            st.subheader("⚙️ Execution Metadata")
            metadata = results['metadata']

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Model Configuration:**")
                thinking_status = "Enabled" if metadata.get('thinking_enabled') else "Disabled"
                thinking_budget = f"{metadata.get('thinking_budget_tokens', 'N/A')} tokens" if metadata.get('thinking_enabled') else "N/A"

                if 'max_tokens_round_2c' in metadata:
                    max_tokens_display = f"""Max Output Tokens:
  - Rounds 0,1,2A,2B: {metadata.get('max_tokens_round_0_1_2a_2b', 'N/A')}
  - Round 2C: {metadata.get('max_tokens_round_2c', 'N/A')}
  - Round 3 (Editor): {metadata.get('max_tokens_round_3_editor', 'N/A')}"""
                else:
                    max_tokens_display = f"Max Output Tokens: {metadata.get('max_tokens', 'N/A')}"

                st.code(f"""Model: {metadata.get('model_version', 'N/A')}
Temperature: {metadata.get('temperature', 'N/A')}
{max_tokens_display}
Thinking Mode: {thinking_status}
Thinking Budget: {thinking_budget}
Max Retries: {metadata.get('max_retries', 'N/A')}
Retry Delay: {metadata.get('retry_delay_seconds', 'N/A')}s""", language="text")

            with col2:
                st.markdown("**Execution Time:**")
                st.code(f"""Start Time: {metadata.get('start_time', 'N/A')}
End Time: {metadata.get('end_time', 'N/A')}
Total Runtime: {metadata.get('total_runtime_formatted', 'N/A')}
({metadata.get('total_runtime_seconds', 'N/A')} seconds)""", language="text")

        # Download options
        st.markdown("---")
        st.subheader("📥 Download Reports")

        col1, col2 = st.columns(2)

        with col1:
            # Prepare full report (with summaries)
            full_report = f"""# Multi-Agent Referee Evaluation Report (Summarized Version)
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

NOTE: This report includes both summarized and full outputs from the multi-agent debate.

## ROUND 0: PERSONA SELECTION

Selected Personas: {', '.join(active_personas)}
Weights: {json.dumps(weights, indent=2)}
Justification: {results.get('round_0', {}).get('justification', 'N/A')}

---

## ROUND 1: INDEPENDENT EVALUATION

"""
            for role in active_personas:
                summary_data = summaries['round_1_summaries'].get(role, {})
                full_report += f"### {role} (Summary)\n{summary_data.get('summary', 'N/A')}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_1'][role]}\n\n"

            full_report += """---

## ROUND 2A: CROSS-EXAMINATION

"""
            for role in active_personas:
                summary_text = summaries['round_2a_summaries'].get(role, 'N/A')
                full_report += f"### {role} (Summary)\n{summary_text}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_2a'][role]}\n\n"

            full_report += """---

## ROUND 2B: ANSWERING QUESTIONS

"""
            for role in active_personas:
                summary_text = summaries['round_2b_summaries'].get(role, 'N/A')
                full_report += f"### {role} (Summary)\n{summary_text}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_2b'][role]}\n\n"

            full_report += """---

## ROUND 2C: FINAL AMENDMENTS

"""
            for role in active_personas:
                summary_data = summaries['round_2c_summaries'].get(role, {})
                full_report += f"### {role} (Summary)\n{summary_data.get('summary', 'N/A')}\n\n"
                full_report += f"### {role} (Full Report)\n{results['round_2c'][role]}\n\n"

            # Add consensus calculation
            if 'consensus' in results:
                consensus = results['consensus']
                full_report += f"""---

## ROUND 3: WEIGHTED CONSENSUS CALCULATION

### Deterministic Consensus Score: {consensus['weighted_score']:.3f}

**Individual Verdicts:**
"""
                for persona in active_personas:
                    verdict = consensus['verdicts'].get(persona, 'UNKNOWN')
                    weight = weights.get(persona, 0)
                    verdict_value = 1.0 if verdict == 'PASS' else 0.5 if verdict == 'REVISE' else 0.0
                    contribution = weight * verdict_value
                    full_report += f"- {persona} (weight={weight}): {verdict} = {verdict_value} → contribution = {contribution:.3f}\n"

                full_report += f"""
**Decision Thresholds:**
- Score > 0.75: ACCEPT
- 0.40 ≤ Score ≤ 0.75: REJECT AND RESUBMIT
- Score < 0.40: REJECT

**Computed Decision:** {consensus['decision']}

---

## EDITOR'S REPORT

### Summary
{summaries.get('editor_summary', 'N/A')}

### Full Report
{results['final_decision']}
"""

            st.download_button(
                label="📄 Download Full Evaluation Report",
                data=full_report,
                file_name=f"multi_agent_evaluation_summarized_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        with col2:
            # Prepare summary table in markdown
            summary_md = f"""# Evaluation Summary Table
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

| Persona | Weight | Round 1 Verdict | Final Verdict (R2C) | Changed? |
|---------|--------|----------------|---------------------|----------|
"""
            for persona in active_personas:
                weight = weights.get(persona, 0)
                r1_v = r1_verdicts[persona]
                r2c_v = r2c_verdicts[persona]
                changed = "Yes" if r1_v != r2c_v else "No"
                summary_md += f"| {persona} | {weight} | {r1_v} | {r2c_v} | {changed} |\n"

            summary_md += f"\n**FINAL EDITORIAL DECISION:** {final_decision}\n"

            st.download_button(
                label="📊 Download Summary Table",
                data=summary_md,
                file_name=f"evaluation_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize file uploader in session state
    if 'uploaded_files_dict' not in st.session_state:
        st.session_state.uploaded_files_dict = {}

    # File uploader at top
    st.title("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your manuscript (PDF)",
        type=['pdf'],
        accept_multiple_files=True,
        key="file_uploader_summary"
    )

    # Convert uploaded files to dict
    files = {}
    if uploaded_files:
        for file in uploaded_files:
            files[file.name] = file.read()
            file.seek(0)  # Reset file pointer

    st.markdown("---")

    # Initialize and render the app
    app = RefereeReportCheckerSummarized()
    app.render_ui(files=files)
