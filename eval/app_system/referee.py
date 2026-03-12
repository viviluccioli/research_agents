# referee_checker.py
import streamlit as st
import json
import tempfile
import os
import asyncio
import datetime
from typing import Dict, List, Any
import pdfplumber
from utils import cm, single_query
from multi_agent_debate import execute_debate_pipeline

# Custom CSS for styling
CUSTOM_CSS = """
<style>
    .persona-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .theorist-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .mathematician-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .empiricist-box {
        background-color: #e0f2f1;
        border-left: 5px solid #00796b;
    }
    .historian-box {
        background-color: #f3e5f5;
        border-left: 5px solid #7b1fa2;
    }
    .visionary-box {
        background-color: #fff3e0;
        border-left: 5px solid #f57c00;
    }
    .policymaker-box {
        background-color: #fce4ec;
        border-left: 5px solid #c2185b;
    }
    .editor-box {
        background-color: #e8f5e9;
        border-left: 5px solid #388e3c;
    }
    .round-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
    }
    .verdict-pass {
        color: #2e7d32;
        font-weight: bold;
        font-size: 18px;
    }
    .verdict-fail {
        color: #c62828;
        font-weight: bold;
        font-size: 18px;
    }
    .verdict-revise {
        color: #f57c00;
        font-weight: bold;
        font-size: 18px;
    }
    .architecture-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 20px 0;
    }
</style>
"""

class RefereeReportChecker:
    """
    Workflow for multi-agent debate paper evaluation.
    """

    def __init__(self, llm=cm):
        """Initialize with conversation manager from utils."""
        self.llm = llm

    def render_ui(self, files=None):
        """Render the Streamlit UI for the multi-agent referee report."""
        # Apply custom CSS
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        st.subheader("Multi-Agent Referee Evaluation")

        # Architecture Overview (collapsible)
        with st.expander("📐 **View Architecture & Approach**", expanded=False):
            st.markdown('<div class="architecture-box">', unsafe_allow_html=True)
            st.markdown("""
            ```
            ┌─────────────────────────────────────────────────────────────┐
            │      MULTI-AGENT REFEREE DEBATE SYSTEM (MAD)                 │
            │              With Endogenous Persona Selection               │
            └─────────────────────────────────────────────────────────────┘

            👥 FIVE AVAILABLE PERSONAS:

            🔢 THEORIST
               └─> Focus: Mathematical logic, formal proofs, model insight
               └─> Checks: Derivation correctness, assumption validity

            📊 EMPIRICIST
               └─> Focus: Data, econometrics, identification strategy
               └─> Checks: Statistical validity, empirical soundness

            📚 HISTORIAN
               └─> Focus: Literature context, citation lineage
               └─> Checks: Gap analysis, novelty claims, contextualization

            🚀 VISIONARY
               └─> Focus: Innovation, paradigm-shifting potential
               └─> Checks: Novelty, creativity, intellectual impact

            🏛️ POLICYMAKER
               └─> Focus: Real-world application, policy relevance
               └─> Checks: Welfare implications, actionable insights

            🔄 DEBATE ROUNDS:

            Round 0: ENDOGENOUS PERSONA SELECTION
               ├─> System analyzes paper content
               ├─> Selects 3 most relevant personas
               └─> Assigns importance weights (sum to 1.0)

            Round 1: INDEPENDENT EVALUATION
               ├─> Each selected agent evaluates independently
               ├─> Provides domain-specific verdict (PASS/REVISE/FAIL)
               └─> Extracts evidence with proportional error weighting

            Round 2A: CROSS-EXAMINATION
               ├─> Agents read each other's Round 1 reports
               ├─> Challenge assumptions, identify conflicts
               └─> Ask specific clarification questions

            Round 2B: ANSWERING QUESTIONS
               ├─> Agents respond to peer questions directly
               ├─> Provide textual evidence for claims
               └─> Concede flaws or defend positions

            Round 2C: FINAL AMENDMENTS
               ├─> Agents integrate full debate transcript
               ├─> Update beliefs based on peer responses
               └─> Submit final verdict with rationale

            Round 3: WEIGHTED CONSENSUS DECISION
               └─> Senior Editor calculates mathematical consensus:
                   ├─> PASS = 1.0, REVISE = 0.5, FAIL = 0.0
                   ├─> Multiply by persona weights, sum scores
                   ├─> Score > 0.75 → ACCEPT
                   ├─> 0.40 ≤ Score ≤ 0.75 → REJECT & RESUBMIT
                   ├─> Score < 0.40 → REJECT
                   └─> Synthesizes final referee report

            KEY INSIGHTS:
            • Endogenous selection: Panel composition adapts to paper type
            • Proportional error weighting: Avoids over-penalizing minor flaws
            • Structured debate: Multi-round Q&A ensures thorough examination
            • Weighted consensus: Democratic but expertise-calibrated decision
            ```
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("This tool uses a multi-agent debate system to evaluate your manuscript from multiple perspectives.")

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

                # Execute debate pipeline
                with st.spinner("Running multi-agent debate..."):
                    debate_results = asyncio.run(
                        execute_debate_pipeline(paper_text, progress_callback=update_progress)
                    )

                # Store results in session state
                st.session_state.debate_results = debate_results

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display results
                self.display_debate_results(debate_results)

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

    def display_debate_results(self, results: Dict):
        """Display the multi-agent debate results in a beautiful format."""
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
            st.markdown("---")

        # ROUND 1: Independent Evaluation
        st.markdown('<div class="round-header">⚡ ROUND 1: INDEPENDENT EVALUATION</div>', unsafe_allow_html=True)

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            with st.expander(f"{icon} **{role.upper()}** - Round 1 Assessment", expanded=False):
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(results['round_1'][role])
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2A: Cross-Examination
        st.markdown('<div class="round-header">🔄 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
        st.markdown("*Agents read each other's reports and ask clarifying questions...*")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            with st.expander(f"{icon} **{role.upper()}** - Cross-Examination", expanded=False):
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(results['round_2a'][role])
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2B: Direct Examination
        st.markdown('<div class="round-header">💬 ROUND 2B: ANSWERING QUESTIONS</div>', unsafe_allow_html=True)
        st.markdown("*Agents answer specific questions from their peers...*")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            with st.expander(f"{icon} **{role.upper()}** - Answers", expanded=False):
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(results['round_2b'][role])
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2C: Final Amendments
        st.markdown('<div class="round-header">⚖️ ROUND 2C: FINAL AMENDMENTS</div>', unsafe_allow_html=True)
        st.markdown("*Agents submit their final verdicts after integrating peer feedback...*")

        cols = st.columns(3)
        for idx, role in enumerate(active_personas):
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            with cols[idx]:
                st.markdown(f"""
                <div class="persona-box {box_class}">
                <h4>{icon} {role}</h4>
                """, unsafe_allow_html=True)
                st.markdown(results['round_2c'][role])
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 3: Editor Decision
        st.markdown('<div class="round-header">📜 ROUND 3: EDITOR DECISION</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="persona-box editor-box">
        <h3>🏛️ Senior Editor's Final Decision</h3>
        {results['final_decision']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Download option
        st.subheader("📥 Download Full Report")

        # Prepare full report
        full_report = f"""# Multi-Agent Referee Evaluation Report
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## ROUND 0: PERSONA SELECTION

Selected Personas: {', '.join(active_personas)}
Weights: {json.dumps(weights, indent=2)}
Justification: {results.get('round_0', {}).get('justification', 'N/A')}

---

## ROUND 1: INDEPENDENT EVALUATION

"""
        for role in active_personas:
            full_report += f"### {role}\n{results['round_1'][role]}\n\n"

        full_report += """---

## ROUND 2A: CROSS-EXAMINATION

"""
        for role in active_personas:
            full_report += f"### {role}\n{results['round_2a'][role]}\n\n"

        full_report += """---

## ROUND 2B: ANSWERING QUESTIONS

"""
        for role in active_personas:
            full_report += f"### {role}\n{results['round_2b'][role]}\n\n"

        full_report += """---

## ROUND 2C: FINAL AMENDMENTS

"""
        for role in active_personas:
            full_report += f"### {role}\n{results['round_2c'][role]}\n\n"

        full_report += f"""---

## ROUND 3: EDITOR DECISION

{results['final_decision']}
"""

        st.download_button(
            label="📄 Download Full Evaluation Report",
            data=full_report,
            file_name=f"multi_agent_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        ) 