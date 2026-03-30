# full_output_ui.py - Archived Full-Output Referee Report UI
"""
ARCHIVED: Alternate UI that shows full uncompressed debate outputs.

This is not the main production code. It's kept for reference and special
use cases where full verbose output is needed. The main production UI is
in referee/workflow.py.
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
from utils import cm, single_query
from referee.engine import execute_debate_pipeline, SELECTION_PROMPT, SYSTEM_PROMPTS, DEBATE_PROMPTS

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
        background-color: #c8e6c9;
        color: #1b5e20;
        font-weight: bold;
        font-size: 18px;
        padding: 8px 16px;
        border-radius: 5px;
        display: inline-block;
    }
    .verdict-reject {
        background-color: #ffcdd2;
        color: #b71c1c;
        font-weight: bold;
        font-size: 18px;
        padding: 8px 16px;
        border-radius: 5px;
        display: inline-block;
    }
    .verdict-revise {
        background-color: #fff9c4;
        color: #f57f17;
        font-weight: bold;
        font-size: 18px;
        padding: 8px 16px;
        border-radius: 5px;
        display: inline-block;
    }
    .verdict-fail {
        background-color: #ffcdd2;
        color: #b71c1c;
        font-weight: bold;
        font-size: 18px;
        padding: 8px 16px;
        border-radius: 5px;
        display: inline-block;
    }
    .verdict-unknown {
        background-color: #e0e0e0;
        color: #616161;
        font-weight: bold;
        font-size: 18px;
        padding: 8px 16px;
        border-radius: 5px;
        display: inline-block;
    }
    .final-verdict {
        background-color: #e8f5e9;
        color: #1b5e20;
        font-weight: bold;
        font-size: 28px;
        padding: 15px 25px;
        border-radius: 8px;
        display: inline-block;
        margin: 15px 0;
        border: 3px solid #388e3c;
    }
    .final-verdict-reject {
        background-color: #ffebee;
        color: #b71c1c;
        border-color: #c62828;
    }
    .final-verdict-revise {
        background-color: #fffde7;
        color: #f57f17;
        border-color: #fbc02d;
    }
    .severity-major {
        color: #b71c1c;
        font-weight: bold;
        background-color: #ffcdd2;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 0 4px;
    }
    .severity-minor {
        color: #e65100;
        font-weight: bold;
        background-color: #ffe0b2;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 0 4px;
    }
    .severity-fatal {
        color: #ffffff;
        font-weight: bold;
        background-color: #b71c1c;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        margin: 0 4px;
    }
    .architecture-box {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #ddd;
        margin: 20px 0;
    }
    .concede-box {
        background-color: #fff9c4;
        border-left: 4px solid #f57f17;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .defend-box {
        background-color: #c8e6c9;
        border-left: 4px solid #388e3c;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    .summary-table th {
        background-color: #667eea;
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: bold;
    }
    .summary-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
    }
    .summary-table tr:hover {
        background-color: #f5f5f5;
    }
</style>
"""

def format_severity_labels(text: str) -> str:
    """Replace [MAJOR], [MINOR], [FATAL] with styled versions."""
    text = re.sub(r'\[FATAL\]', '<span class="severity-fatal">⚠️ FATAL</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[MAJOR\]', '<span class="severity-major">🔴 MAJOR</span>', text, flags=re.IGNORECASE)
    text = re.sub(r'\[MINOR\]', '<span class="severity-minor">🟠 MINOR</span>', text, flags=re.IGNORECASE)
    return text

def format_verdict(verdict_text: str) -> str:
    """Format verdict with colored background."""
    verdict_text = verdict_text.strip()
    if 'PASS' in verdict_text.upper():
        return f'<span class="verdict-pass">✅ PASS</span>'
    elif 'REJECT' in verdict_text.upper() or 'FAIL' in verdict_text.upper():
        return f'<span class="verdict-reject">❌ REJECT</span>'
    elif 'REVISE' in verdict_text.upper():
        return f'<span class="verdict-revise">⚠️ REVISE</span>'
    return verdict_text

def format_final_verdict(verdict_text: str) -> str:
    """Format final editor verdict with larger, more prominent styling."""
    verdict_text = verdict_text.strip().upper()
    if 'ACCEPT' in verdict_text:
        css_class = 'final-verdict'
        icon = '✅'
        text = 'ACCEPT'
    elif 'REJECT AND RESUBMIT' in verdict_text or 'RESUBMIT' in verdict_text:
        css_class = 'final-verdict final-verdict-revise'
        icon = '⚠️'
        text = 'REJECT & RESUBMIT'
    elif 'REJECT' in verdict_text:
        css_class = 'final-verdict final-verdict-reject'
        icon = '❌'
        text = 'REJECT'
    else:
        return verdict_text
    return f'<span class="{css_class}">{icon} {text}</span>'

def extract_verdict(text: str) -> str:
    """
    Extract verdict from persona report.

    Prioritizes "Final Verdict:" to avoid matching intermediate text like
    "Verdict Change: REVISE → FAIL". Uses last occurrence as fallback.
    """
    verdict = "UNKNOWN"

    # Try "Final Verdict:" patterns first (most specific, least ambiguous)
    final_verdict_patterns = [
        r'\*\*Final Verdict\*\*\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'Final Verdict\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'\*\*Final Verdict:\*\*\s+(PASS|REVISE|REJECT|FAIL)',
    ]

    for pattern in final_verdict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # If not found, try generic "Verdict:" patterns (but these might catch wrong ones)
    generic_verdict_patterns = [
        r'\*\*Verdict\*\*\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
        r'Verdict\s*:+\s*(PASS|REVISE|REJECT|FAIL)',
    ]

    for pattern in generic_verdict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find all occurrences and take the LAST one (most likely final verdict)
    all_matches = re.findall(r'\b(PASS|REVISE|REJECT|FAIL)\b', text, re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()

    return "UNKNOWN"

def summarize_text(text: str, max_bullets: int = 5) -> str:
    """Create a concise bullet-point summary of text for UI display."""
    # Split by common section headers
    sections = re.split(r'\*\*([^*]+)\*\*', text)

    summary_parts = []
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            header = sections[i + 1].strip()
            content = sections[i + 2].strip() if i + 2 < len(sections) else ""

            # Extract first sentence or first 150 characters
            sentences = re.split(r'[.!?]\s+', content)
            first_sentence = sentences[0] if sentences else content
            if len(first_sentence) > 150:
                first_sentence = first_sentence[:150] + "..."

            summary_parts.append(f"**{header}**: {first_sentence}")

            if len(summary_parts) >= max_bullets:
                break

    return "\n\n".join(summary_parts[:max_bullets])

def format_round1_output(text: str, persona: str) -> str:
    """Format Round 1 output with standardized structure and source evidence under each finding."""
    formatted = f"### 🔍 {persona.upper()} - INDEPENDENT EVALUATION\n\n"

    # Split into sections
    sections = re.split(r'\*\*([^*]+)\*\*', text)

    current_section = None
    findings = []

    for i in range(len(sections)):
        if i % 2 == 1:  # This is a header
            current_section = sections[i].strip()
        elif i % 2 == 0 and current_section:  # This is content
            content = sections[i].strip()
            if current_section and content:
                # Check if this is the findings/severity section
                if 'finding' in current_section.lower() or 'severity' in current_section.lower():
                    # Parse individual findings with severity labels
                    finding_lines = content.split('\n')
                    for line in finding_lines:
                        if line.strip() and ('[MAJOR]' in line or '[MINOR]' in line or '[FATAL]' in line):
                            formatted += f"\n{format_severity_labels(line)}\n"
                        elif line.strip():
                            formatted += f"{line}\n"
                elif 'verdict' in current_section.lower():
                    formatted += f"\n**{current_section}**: {format_verdict(content)}\n"
                else:
                    formatted += f"\n**{current_section}**: {content}\n"

    return formatted

def format_round2a_output(text: str, persona: str) -> str:
    """Format Round 2A output with standardized structure, filtering out Round 1 content."""
    formatted = f"### 🔍 {persona.upper()} - CROSS-EXAMINATION\n\n"

    # Filter out any verdict statements that don't belong in Round 2A
    text = re.sub(r'Verdict:\s*(PASS|REVISE|REJECT|FAIL)[^\n]*\n?', '', text, flags=re.IGNORECASE)

    # Apply severity formatting (in case LLM incorrectly included them)
    text = format_severity_labels(text)

    # Extract the three required sections
    sections_found = []

    # Cross-Domain Insights
    insights_pattern = r'\*\*Cross-Domain Insights[:\*]*\s*(.*?)(?=\*\*Constructive Pushback|\*\*Clarification Requests|\Z)'
    insights_match = re.search(insights_pattern, text, re.DOTALL | re.IGNORECASE)
    if insights_match:
        formatted += "**Cross-Domain Insights:**\n"
        formatted += insights_match.group(1).strip() + "\n\n"
        sections_found.append("insights")

    # Constructive Pushback
    pushback_pattern = r'\*\*Constructive Pushback[:\*]*\s*(.*?)(?=\*\*Clarification Requests|\Z)'
    pushback_match = re.search(pushback_pattern, text, re.DOTALL | re.IGNORECASE)
    if pushback_match:
        formatted += "**Constructive Pushback:**\n"
        formatted += pushback_match.group(1).strip() + "\n\n"
        sections_found.append("pushback")

    # Clarification Requests
    clarification_pattern = r'\*\*Clarification Requests[:\*]*\s*(.*?)(?=\Z)'
    clarification_match = re.search(clarification_pattern, text, re.DOTALL | re.IGNORECASE)
    if clarification_match:
        formatted += "**Clarification Requests:**\n"
        formatted += clarification_match.group(1).strip() + "\n"
        sections_found.append("clarification")

    # If standard structure not found, display raw text (filtered)
    if not sections_found:
        formatted += text

    return formatted

def format_round2c_output(text: str, persona: str) -> str:
    """Format Round 2C output with emphasized CONCEDE/DEFEND decisions."""
    formatted = f"### ⚖️ {persona.upper()} - FINAL AMENDED VERDICT\n\n"

    # Look for concede/defend patterns
    text = re.sub(
        r'(Concession|Concede)(.*?)(?=\n\n|\Z)',
        r'<div class="concede-box">⚠️ <strong>CONCESSION</strong>\2</div>',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    text = re.sub(
        r'(Defense|Defend)(.*?)(?=\n\n|\Z)',
        r'<div class="defend-box">🛡️ <strong>DEFENSE</strong>\2</div>',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    formatted += text
    return formatted

def generate_summary_table(results: Dict, active_personas: List[str], weights: Dict) -> str:
    """Generate HTML summary table of the debate."""
    # Extract verdicts from each round
    r1_verdicts = {p: extract_verdict(results['round_1'].get(p, '')) for p in active_personas}
    r2c_verdicts = {p: extract_verdict(results['round_2c'].get(p, '')) for p in active_personas}

    # Extract final decision
    final_text = results.get('final_decision', '')
    final_decision = 'UNKNOWN'
    if 'ACCEPT' in final_text.upper() and 'RESUBMIT' not in final_text.upper():
        final_decision = 'ACCEPT'
    elif 'REJECT AND RESUBMIT' in final_text.upper() or 'RESUBMIT' in final_text.upper():
        final_decision = 'REJECT & RESUBMIT'
    elif 'REJECT' in final_text.upper():
        final_decision = 'REJECT'

    html = """
    <table class="summary-table">
        <thead>
            <tr>
                <th>Persona</th>
                <th>Weight</th>
                <th>Round 1 Verdict</th>
                <th>Final Verdict (Round 2C)</th>
                <th>Changed?</th>
            </tr>
        </thead>
        <tbody>
    """

    for persona in active_personas:
        weight = weights.get(persona, 0)
        r1_v = r1_verdicts[persona]
        r2c_v = r2c_verdicts[persona]
        changed = "✅ Yes" if r1_v != r2c_v else "➖ No"

        # Color code verdicts - handle UNKNOWN case
        if r1_v in ["PASS", "REVISE", "REJECT", "FAIL"]:
            r1_colored = format_verdict(r1_v)
        else:
            r1_colored = f'<span style="color: #999;">{r1_v}</span>'

        if r2c_v in ["PASS", "REVISE", "REJECT", "FAIL"]:
            r2c_colored = format_verdict(r2c_v)
        else:
            r2c_colored = f'<span style="color: #999;">{r2c_v}</span>'

        html += f"""
            <tr>
                <td><strong>{persona}</strong></td>
                <td>{weight}</td>
                <td>{r1_colored}</td>
                <td>{r2c_colored}</td>
                <td>{changed}</td>
            </tr>
        """

    # Add final decision row
    final_colored = format_final_verdict(final_decision)
    html += f"""
            <tr style="background-color: #f0f0f0; font-weight: bold;">
                <td colspan="4" style="text-align: right;">FINAL EDITORIAL DECISION:</td>
                <td>{final_colored}</td>
            </tr>
        </tbody>
    </table>
    """

    return html

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

            # System Prompt Display
            with st.expander("🔍 **View System Prompt for Round 0**", expanded=False):
                st.code(SELECTION_PROMPT, language="text")

            st.markdown("---")

        # ROUND 1: Independent Evaluation
        st.markdown('<div class="round-header">⚡ ROUND 1: INDEPENDENT EVALUATION</div>', unsafe_allow_html=True)
        st.markdown("*Each persona independently evaluates the paper with severity-weighted findings.*")

        # System Prompt Display for Round 1
        with st.expander("🔍 **View System Prompts for Round 1**", expanded=False):
            for persona in active_personas:
                st.markdown(f"**{persona} System Prompt:**")
                st.code(SYSTEM_PROMPTS.get(persona, "N/A"), language="text")
                st.markdown("---")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            with st.expander(f"{icon} **{role.upper()}** - Round 1 Assessment", expanded=False):
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)

                # Parse and format the output with standardized structure
                raw_text = results['round_1'][role]

                # Extract verdict first
                verdict = extract_verdict(raw_text)
                st.markdown(f"### Verdict: {format_verdict(verdict)}", unsafe_allow_html=True)

                # Format severity labels
                formatted_text = format_severity_labels(raw_text)

                # Display concise version in UI
                st.markdown(formatted_text, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2A: Cross-Examination
        st.markdown('<div class="round-header">🔄 ROUND 2A: CROSS-EXAMINATION</div>', unsafe_allow_html=True)
        st.markdown("*Personas challenge each other's findings and ask clarifying questions.*")

        # System Prompt Display for Round 2A
        with st.expander("🔍 **View System Prompt for Round 2A**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2A_Cross_Examination"], language="text")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"
            peers = [p for p in active_personas if p != role]

            with st.expander(f"{icon} **{role.upper()}** - Cross-Examination", expanded=False):
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)

                raw_text = results['round_2a'][role]

                # Apply severity label formatting
                formatted_text = format_severity_labels(raw_text)

                # Parse and display in standardized structure
                # Section 1: Cross-Domain Insights
                insights_pattern = r'\*\*Cross-Domain Insights[:\*]*\s*(.*?)(?=\*\*Constructive Pushback|\*\*Clarification Requests|\Z)'
                insights_match = re.search(insights_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                if insights_match:
                    st.markdown("**Cross-Domain Insights:**")
                    insights_text = insights_match.group(1).strip()
                    # Display full content (no truncation, allow HTML for severity labels)
                    st.markdown(insights_text, unsafe_allow_html=True)
                    st.markdown("")

                # Section 2: Constructive Pushback
                pushback_pattern = r'\*\*Constructive Pushback[:\*]*\s*(.*?)(?=\*\*Clarification Requests|\Z)'
                pushback_match = re.search(pushback_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                if pushback_match:
                    st.markdown("**Constructive Pushback:**")
                    pushback_text = pushback_match.group(1).strip()
                    # Display full content (no truncation)
                    st.markdown(pushback_text, unsafe_allow_html=True)
                    st.markdown("")

                # Section 3: Clarification Requests
                clarification_pattern = r'\*\*Clarification Requests[:\*]*\s*(.*?)(?=\Z)'
                clarification_match = re.search(clarification_pattern, formatted_text, re.DOTALL | re.IGNORECASE)

                if clarification_match:
                    st.markdown("**Clarification Requests:**")
                    clarification_text = clarification_match.group(1).strip()

                    # Parse individual questions to each peer
                    for peer in peers:
                        peer_question_pattern = rf'(?:To|to)\s+{peer}[:\s]*(.*?)(?=(?:To|to)\s+\w+:|$)'
                        peer_match = re.search(peer_question_pattern, clarification_text, re.DOTALL)
                        if peer_match:
                            question_text = peer_match.group(1).strip()
                            st.markdown(f"*To {peer}:*")
                            st.markdown(f"> {question_text}")
                            st.markdown("")

                # Warn if standard structure not found
                if not (insights_match or pushback_match or clarification_match):
                    st.warning("⚠️ Non-standard output format detected. Displaying raw content:")
                    st.markdown(formatted_text, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2B: Direct Examination
        st.markdown('<div class="round-header">💬 ROUND 2B: ANSWERING QUESTIONS</div>', unsafe_allow_html=True)
        st.markdown("*Personas respond to peer questions with evidence and concessions/defenses.*")

        # System Prompt Display for Round 2B
        with st.expander("🔍 **View System Prompt for Round 2B**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2B_Direct_Examination"], language="text")

        for role in active_personas:
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"
            peers = [p for p in active_personas if p != role]

            with st.expander(f"{icon} **{role.upper()}** - Answers", expanded=False):
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(f"### 💬 {role.upper()} - RESPONSES\n")

                raw_text = results['round_2b'][role]

                # Parse responses to each peer
                for peer in peers:
                    # Look for response to this peer
                    pattern = rf'\*\*Response to {peer}[:\*]*\s*(.*?)(?=\*\*Response to |\*\*Concession or Defense|\Z)'
                    match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)

                    if match:
                        response_content = match.group(1).strip()
                        st.markdown(f"**Response to {peer}:**")
                        st.markdown(response_content)

                        # Look for concede/defend specific to this peer
                        concede_pattern = rf'(concede|concession).*?{peer}(.*?)(?=\n\n|\Z)'
                        defend_pattern = rf'(defend|defense).*?{peer}(.*?)(?=\n\n|\Z)'

                        concede_match = re.search(concede_pattern, raw_text, re.IGNORECASE | re.DOTALL)
                        defend_match = re.search(defend_pattern, raw_text, re.IGNORECASE | re.DOTALL)

                        if concede_match:
                            st.markdown('<div class="concede-box">⚠️ <strong>CONCEDE</strong></div>', unsafe_allow_html=True)
                        elif defend_match:
                            st.markdown('<div class="defend-box">🛡️ <strong>DEFEND</strong></div>', unsafe_allow_html=True)

                        st.markdown("---")

                # General concession or defense section
                pattern = r'\*\*Concession or Defense[:\*]*\s*(.*?)(?=\Z)'
                match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
                if match:
                    content = match.group(1).strip()
                    if 'concede' in content.lower():
                        st.markdown(f'<div class="concede-box"><strong>⚠️ FINAL POSITION: CONCEDE</strong><br>{content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="defend-box"><strong>🛡️ FINAL POSITION: DEFEND</strong><br>{content}</div>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 2C: Final Amendments
        st.markdown('<div class="round-header">⚖️ ROUND 2C: FINAL AMENDMENTS</div>', unsafe_allow_html=True)
        st.markdown("*Personas submit final verdicts after integrating full debate.*")

        # System Prompt Display for Round 2C
        with st.expander("🔍 **View System Prompt for Round 2C**", expanded=False):
            st.code(DEBATE_PROMPTS["Round_2C_Final_Amendment"], language="text")

        cols = st.columns(3)
        for idx, role in enumerate(active_personas):
            icon = icon_map.get(role, "🔍")
            box_class = f"{role.lower()}-box"

            with cols[idx]:
                st.markdown(f'<div class="persona-box {box_class}">', unsafe_allow_html=True)
                st.markdown(f"### {icon} {role.upper()}")

                raw_text = results['round_2c'][role]

                # Check if output is blank or too short
                if not raw_text or len(raw_text.strip()) < 20:
                    st.warning(f"⚠️ {role} did not provide a Round 2C response. This may indicate an LLM error.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    continue

                # Extract and display verdict prominently
                verdict = extract_verdict(raw_text)
                if verdict == "UNKNOWN":
                    st.warning("⚠️ Unable to extract verdict from response")
                st.markdown(f"**Final Verdict:** {format_verdict(verdict)}", unsafe_allow_html=True)

                # Format and display the rest
                formatted_text = format_severity_labels(raw_text)

                # Display ALL sections (no limit)
                sections = re.split(r'\*\*([^*]+)\*\*', formatted_text)
                for i in range(0, len(sections), 2):
                    if i + 1 < len(sections):
                        header = sections[i + 1].strip().rstrip(':')  # Remove trailing colon
                        content = sections[i + 2].strip() if i + 2 < len(sections) else ""

                        if header.lower() not in ['verdict', 'final verdict']:
                            st.markdown(f"**{header}:**")
                            # Display full content (no truncation, allow HTML for severity labels)
                            st.markdown(content, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ROUND 3: Consensus Calculation & Editor Decision
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

        # ALWAYS use computed consensus as source of truth (don't trust LLM to format correctly)
        final_text = results['final_decision']
        final_decision = results.get('consensus', {}).get('decision', 'UNKNOWN')

        st.markdown(f"""
        <div class="persona-box editor-box">
        <h3>🏛️ Senior Editor's Report</h3>
        <div style="text-align: center; margin: 20px 0;">
        {format_final_verdict(final_decision)}
        </div>
        <p style="text-align: center; color: #666; font-size: 14px;">
        (Decision computed deterministically from weighted consensus, not extracted from text)
        </p>
        """, unsafe_allow_html=True)

        # Display complete editor report (all sections)
        sections = re.split(r'\*\*([^*]+)\*\*', final_text)
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                header = sections[i + 1].strip().rstrip(':')  # Remove trailing colon
                content = sections[i + 2].strip() if i + 2 < len(sections) else ""

                st.markdown(f"**{header}:**")
                # Display full content (no truncation, allow HTML if present)
                st.markdown(content, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

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

        # Display as dataframe (more reliable than HTML table)
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Get final decision from consensus (deterministic)
        final_decision = results.get('consensus', {}).get('decision', 'UNKNOWN')
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

                # Handle both old and new metadata formats
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

{results['final_decision']}
"""
            else:
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

        with col2:
            # Prepare summary table in markdown
            r1_verdicts = {p: extract_verdict(results['round_1'].get(p, '')) for p in active_personas}
            r2c_verdicts = {p: extract_verdict(results['round_2c'].get(p, '')) for p in active_personas}

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