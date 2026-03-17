import streamlit as st
import json
import tempfile
import os
import re
from typing import Dict, List, Any
import pdfplumber
from utils import cm, single_query
import codecs
from fpdf import FPDF
from datetime import datetime

class SectionEvaluator:
    """
    Workflow for evaluating manuscript sections, grading them,
    and providing detailed improvement suggestions with flexibility
    for economics research papers.
    """
    
    # Configurable section detection patterns - can be customized
    SECTION_PATTERNS = {
        "Abstract": r"(?:abstract|summary)",
        "Introduction": r"(?:introduction|overview|background)",
        "Literature Review": r"(?:literature|related\s+work|previous\s+research)",
        "Theory/Model": r"(?:theoretical\s+framework|model|theory|framework)",
        "Methodology": r"(?:methodology|methods|data|empirical\s+strategy|research\s+design)",
        "Results": r"(?:results|findings|analysis|empirical\s+results)",
        "Discussion": r"(?:discussion|interpretation|implications)",
        "Conclusion": r"(?:conclusion|concluding\s+remarks|final\s+remarks)",
        "Appendix": r"(?:appendix|appendices)"
    }
    
    def __init__(self, llm=cm):
        """Initialize with conversation manager from utils."""
        self.llm = llm
    
    def render_ui(self, files=None, paper_type=None):
        """Render the Streamlit UI for the section evaluator."""
        st.subheader("📝 Text Input Options")

        # Upload mode selection
        upload_mode = st.radio(
            "Choose how to input your manuscript:",
            ["Upload Full Manuscript", "Enter Text Section by Section"],
            key="upload_mode_selector",
            horizontal=True
        )

        st.markdown("---")

        manuscript_sections = {}

        # MODE 1: Upload Full Manuscript
        if upload_mode == "Upload Full Manuscript":
            st.markdown("### 📄 Full Manuscript Upload")

            # Check if files are available
            if not files:
                st.info("Please upload your manuscript using the document uploader above.")
                return

            # File selection
            manuscript_file = st.selectbox(
                "Select your manuscript",
                options=list(files.keys()),
                key="manuscript_selector_se"
            )

            # Allow customization of sections to evaluate
            st.write("**Select sections to evaluate:**")
            selected_sections = {}

            # Create columns for checkboxes
            col1, col2, col3 = st.columns(3)
            section_items = list(self.SECTION_PATTERNS.items())

            for idx, (section, pattern) in enumerate(section_items):
                col = [col1, col2, col3][idx % 3]
                with col:
                    if st.checkbox(section, value=True, key=f"check_{section}"):
                        selected_sections[section] = pattern

            if st.button("Evaluate Manuscript", type="primary", use_container_width=True):
                if not manuscript_file:
                    st.error("Please select a manuscript file.")
                    return

                with st.spinner("Extracting text from manuscript..."):
                    # Extract sections from manuscript
                    manuscript_text = self.extract_text_from_pdf(files[manuscript_file])
                    manuscript_sections = self.extract_sections_from_text(manuscript_text, selected_sections)

                if not manuscript_sections:
                    st.error("No sections could be extracted from the manuscript. Please check the file.")
                    return

                # Evaluate sections
                self._evaluate_and_display(manuscript_sections)

        # MODE 2: Enter Text Section by Section
        else:
            st.markdown("### ✍️ Enter Text Section by Section")
            st.write("Paste or type the text for each section you want to evaluate.")

            # Initialize session state for manual sections
            if "manual_sections" not in st.session_state:
                st.session_state.manual_sections = {}

            # Section selection
            st.write("**Select sections to enter:**")
            selected_manual_sections = []

            # Create columns for checkboxes
            col1, col2, col3 = st.columns(3)
            section_items = list(self.SECTION_PATTERNS.keys())

            for idx, section in enumerate(section_items):
                col = [col1, col2, col3][idx % 3]
                with col:
                    if st.checkbox(section, value=False, key=f"manual_check_{section}"):
                        selected_manual_sections.append(section)

            st.markdown("---")

            # Text input areas for selected sections
            if selected_manual_sections:
                st.markdown("### 📝 Enter Section Content")

                for section in selected_manual_sections:
                    st.markdown(f"#### {section}")
                    text_input = st.text_area(
                        f"Paste the text for **{section}**:",
                        height=200,
                        key=f"manual_text_{section}",
                        placeholder=f"Enter the {section} content here..."
                    )

                    if text_input and text_input.strip():
                        st.session_state.manual_sections[section] = text_input.strip()
                    elif section in st.session_state.manual_sections:
                        del st.session_state.manual_sections[section]

                    st.markdown("---")

                # Evaluate button
                if st.button("Evaluate Sections", type="primary", use_container_width=True):
                    # Collect all entered sections
                    manuscript_sections = {}
                    missing_sections = []

                    for section in selected_manual_sections:
                        if section in st.session_state.manual_sections and st.session_state.manual_sections[section]:
                            manuscript_sections[section] = st.session_state.manual_sections[section]
                        else:
                            missing_sections.append(section)

                    if missing_sections:
                        st.error(f"Please enter text for the following sections: {', '.join(missing_sections)}")
                        return

                    if not manuscript_sections:
                        st.error("No sections entered. Please enter text for at least one section.")
                        return

                    # Evaluate sections
                    self._evaluate_and_display(manuscript_sections)
            else:
                st.info("👆 Please select at least one section to enter text.")

    def _evaluate_and_display(self, manuscript_sections):
        """Helper method to evaluate sections and display results."""
        with st.spinner("Evaluating sections..."):
            evaluation_results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process each section
            total_sections = len(manuscript_sections)
            for i, (section_name, section_text) in enumerate(manuscript_sections.items()):
                status_text.text(f"Evaluating {section_name}...")
                evaluation = self.evaluate_section(section_name, section_text)
                evaluation_results[section_name] = evaluation
                progress_bar.progress((i + 1) / total_sections)

            # Generate overall assessment
            status_text.text("Generating overall assessment...")
            overall_assessment = self.generate_overall_assessment(
                manuscript_sections, evaluation_results
            )
            status_text.text("Evaluation complete!")

        # Display results
        self.display_results(
            manuscript_sections, evaluation_results, overall_assessment
        )
    def extract_text_from_pdf(self, file_content):
        """Extract text from a PDF file with robust encoding handling."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            
            text = ""
            try:
                with pdfplumber.open(temp_file.name) as pdf:
                    for page in pdf.pages:
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                # Handle encoding explicitly to avoid _ENC errors
                                text += page_text.encode('utf-8', errors='replace').decode('utf-8', errors='replace') + "\n\n"
                        except Exception as e:
                            st.warning(f"Error extracting text from page: {e}")
                            continue
            except Exception as e:
                st.error(f"Error extracting text from PDF: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            # Clean up
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass
                
            return text
    
    
    def extract_sections_from_text(self, text, section_patterns):
        """Extract sections using regular expressions and LLM backup."""
        # First attempt: Use regex pattern matching
        sections = {}
        text_lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in text_lines:
            matched = False
            for section, pattern in section_patterns.items():
                if re.search(fr'\b{pattern}\b', line, re.IGNORECASE) and len(line) < 100:
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = section
                    current_content = []
                    matched = True
                    break
            
            if not matched and current_section:
                current_content.append(line)
        
        # Don't forget the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # If regex failed to extract sections, use LLM as fallback
        if not sections:
            prompt = f"""
            Extract the main sections from this academic paper. Identify common sections 
            in economics papers (e.g., Abstract, Introduction, Literature Review, Theory/Model,
            Methodology, Results, Discussion, Conclusion).
            
            Return a JSON object where keys are section names and values are the section text.
            
            PAPER TEXT:
            {text[:50000]}
            """
            
            response = single_query(prompt)
            
            try:
                sections = json.loads(response)
            except:
                # If JSON parsing fails, use a simpler approach
                sections = {"Full Text": text}
        
        return sections
    
    def evaluate_section(self, section_name, section_text):
        """Evaluate a section using detailed criteria."""
        prompt = f"""
        You are a seasoned academic reviewer for top economics journals with expertise in financial economics, 
        econometrics, mathematics, statistics, and economic theory. Evaluate the following {section_name} section of an economics 
        research paper:

        {section_text[:8000]}

        Provide a comprehensive evaluation following this format:

        1. STRENGTHS AND WEAKNESSES: 
        - Strengths: [List 1-3 bullet points]
        - Weaknesses: [List 1-3 bullet points]

        2. ACTIONABLE SUGGESTIONS: 
        - [List 1-3 specific suggestions for improving this section]

        3. SCORES (1-5 scale):
        - Clarity: [Score from 1-5]
        - Depth: [Score from 1-5]
        - Relevance: [Score from 1-5] 
        - Technical Accuracy: [Score from 1-5]

        4. OVERALL SCORE: [Score from 1-5]

        Be specific, constructive, and focused on substantive improvements. Guidance on the scores: Clarity: Does the section present ideas logically and clearly? Depth: Does it demonstrate thorough understanding and analysis? Relevance: Does it contribute meaningfully to the paper's objectives? Technical Accuracy: Are methods, theories, and interpretations sound?
        """

        response = self.llm.conv_query(prompt)
        
        # Parse the response
        evaluation = {
            "raw_response": response,
            "strengths_and_weaknesses": "",
            "actionable_suggestions": "",
            "scores": {}
        }
        
        # Extract structured information
        if "STRENGTHS AND WEAKNESSES:" in response:
            parts = response.split("ACTIONABLE SUGGESTIONS:")
            if len(parts) > 1:
                evaluation["strengths_and_weaknesses"] = parts[0].split("STRENGTHS AND WEAKNESSES:")[1].strip()
                next_parts = parts[1].split("SCORES")
                if len(next_parts) > 1:
                    evaluation["actionable_suggestions"] = next_parts[0].strip()
        
        # More flexible score extraction
        # Try different patterns to extract scores
        score_patterns = [
            r"(\w+):\s*(\d+)\/5",  # Clarity: 4/5
            r"(\w+):\s*\[(\d+)\]",  # Clarity: [4]
            r"(\w+):\s*(\d+)",      # Clarity: 4
            r"(\w+)[^\w\d]*?(\d)"   # Clarity - 4
        ]
        
        found_scores = False
        for pattern in score_patterns:
            scores = re.findall(pattern, response)
            if scores:
                found_scores = True
                for name, score in scores:
                    if name.lower() not in ["overall", "overall score"]:  # Skip overall score for now
                        evaluation["scores"][name.lower()] = int(score)
        
        # Try to extract overall score with more flexible patterns
        overall_patterns = [
            r"OVERALL\s+SCORE.*?(\d+)\/5",
            r"OVERALL\s+SCORE.*?\[(\d+)\]",
            r"OVERALL\s+SCORE.*?(\d+)",
            r"OVERALL.*?(\d)"
        ]
        
        overall_score = None
        for pattern in overall_patterns:
            overall_match = re.search(pattern, response, re.IGNORECASE)
            if overall_match:
                overall_score = int(overall_match.group(1))
                break
        
        if overall_score:
            evaluation["scores"]["overall"] = overall_score
        elif evaluation["scores"]:
            # Calculate overall if not explicitly provided but other scores exist
            scores_values = list(evaluation["scores"].values())
            if scores_values:
                evaluation["scores"]["overall"] = round(sum(scores_values) / len(scores_values), 1)
        else:
            # If no scores were found, add default scores to avoid N/A
            if not found_scores:
                # This is a fallback - try to infer scores from the text
                if "excellent" in response.lower() or "strong" in response.lower():
                    default_score = 4
                elif "good" in response.lower() or "adequate" in response.lower():
                    default_score = 3
                elif "poor" in response.lower() or "weak" in response.lower():
                    default_score = 2
                else:
                    default_score = 3  # Neutral default
                    
                evaluation["scores"] = {
                    "clarity": default_score,
                    "depth": default_score,
                    "relevance": default_score,
                    "technical_accuracy": default_score,
                    "overall": default_score
                }
        
        return evaluation
    def generate_overall_assessment(self, sections, evaluations):
        """Generate an overall assessment of the manuscript with improved structure."""
        # Create a summary of the evaluations
        section_summaries = []
        for section_name, evaluation in evaluations.items():
            scores = evaluation.get("scores", {})
            overall_score = scores.get("overall", "N/A")
            section_summaries.append(f"{section_name}: {overall_score}/5")
        
        scores_summary = "\n".join(section_summaries)
        
        prompt = f"""
        You are providing an overall assessment of an economics research paper that has been evaluated section by section.
        
        The manuscript has been evaluated with the following section scores:
        
        {scores_summary}
        
        Based on these evaluations, provide a comprehensive assessment following EXACTLY this format:

        ## Key Strengths (3 most important)
        1. [First strength]
        2. [Second strength]
        3. [Third strength]
        
        ## Key Weaknesses (3 most important)
        1. [First weakness]
        2. [Second weakness]
        3. [Third weakness]
        
        ## Priority Improvements (3-5 most critical)
        1. [First priority]
        2. [Second priority]
        3. [Third priority]
        
        ## Publication Readiness
        [Assessment: Not ready, Needs major revisions, Needs minor revisions, or Ready]
        
        ## Additional Recommendations
        [2-3 sentences with specific advice to enhance the manuscript's contribution]
        
        Focus on substantive issues rather than formatting or minor concerns.
        """
        
        return self.llm.conv_query(prompt)

    def generate_pdf_report(self, sections, evaluations, overall_assessment):
        """Generate a PDF report from the evaluation results."""
        class PDF(FPDF):
            def header(self):
                self.set_font('Arial', 'B', 16)
                self.cell(0, 10, 'Manuscript Evaluation Report', 0, 1, 'C')
                self.set_font('Arial', 'I', 10)
                self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

            def chapter_title(self, title):
                self.set_font('Arial', 'B', 14)
                self.cell(0, 10, title, 0, 1, 'L')
                self.ln(2)

            def section_title(self, title):
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, title, 0, 1, 'L')
                self.ln(1)

            def body_text(self, text):
                self.set_font('Arial', '', 11)
                # Handle encoding issues and split text into lines
                text = text.encode('latin-1', 'replace').decode('latin-1')
                self.multi_cell(0, 6, text)
                self.ln(2)

            def bullet_point(self, text):
                self.set_font('Arial', '', 11)
                text = text.encode('latin-1', 'replace').decode('latin-1')
                self.cell(10, 6, '•', 0, 0)
                self.multi_cell(0, 6, text)

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()

        # Overall Assessment
        pdf.chapter_title('Overall Assessment')
        pdf.body_text(overall_assessment)
        pdf.ln(5)

        # Section Evaluations
        pdf.chapter_title('Section Evaluations')

        for section_name, evaluation in evaluations.items():
            scores = evaluation.get("scores", {})
            overall_score = scores.get("overall", "N/A")

            pdf.section_title(f'{section_name} - Score: {overall_score}/5')

            # Strengths and Weaknesses
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Strengths and Weaknesses:', 0, 1)
            pdf.body_text(evaluation.get("strengths_and_weaknesses", "N/A"))

            # Actionable Suggestions
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Actionable Suggestions:', 0, 1)
            pdf.body_text(evaluation.get("actionable_suggestions", "N/A"))

            # Detailed Scores
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, 'Detailed Scores:', 0, 1)
            for name, score in scores.items():
                if name != "overall":
                    pdf.bullet_point(f'{name.capitalize()}: {score}/5')

            pdf.ln(5)

        # Return PDF as bytes
        return pdf.output(dest='S').encode('latin-1')

    def display_results(self, sections, evaluations, overall_assessment):
        """Display the evaluation results in the Streamlit UI."""
        st.success("Evaluation complete!")
        
        # Overall assessment
        st.subheader("Overall Assessment")
        st.write(overall_assessment)
        
        # Section evaluations
        st.subheader("Section Evaluations")
        
        for section_name, evaluation in evaluations.items():
            scores = evaluation.get("scores", {})
            overall_score = scores.get("overall", "N/A")
            
            with st.expander(f"{section_name} - Score: {overall_score}/5"):
                st.markdown("**Strengths and Weaknesses**")
                st.write(evaluation.get("strengths_and_weaknesses", ""))
                
                st.markdown("**Actionable Suggestions**")
                st.write(evaluation.get("actionable_suggestions", ""))
                
                # Display detailed scores
                st.markdown("**Detailed Scores**")
                for name, score in scores.items():
                    if name != "overall":
                        st.write(f"- {name.capitalize()}: {score}/5")
        
        # Download options
        st.subheader("Download Results")

        # Prepare full report
        full_report = f"""# Manuscript Evaluation Report

## Overall Assessment
{overall_assessment}

## Section Evaluations
"""

        for section_name, evaluation in evaluations.items():
            scores = evaluation.get("scores", {})
            overall_score = scores.get("overall", "N/A")

            full_report += f"""
### {section_name} - Score: {overall_score}/5

**Strengths and Weaknesses**
{evaluation.get('strengths_and_weaknesses', '')}

**Actionable Suggestions**
{evaluation.get('actionable_suggestions', '')}

**Detailed Scores**
"""

            for name, score in scores.items():
                if name != "overall":
                    full_report += f"- {name.capitalize()}: {score}/5\n"

        # Create two columns for download buttons
        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                label="📄 Download as Markdown",
                data=full_report,
                file_name="manuscript_evaluation.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col2:
            # Generate PDF report
            try:
                pdf_data = self.generate_pdf_report(sections, evaluations, overall_assessment)
                st.download_button(
                    label="📕 Download as PDF",
                    data=pdf_data,
                    file_name="manuscript_evaluation.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error generating PDF: {e}")


# Wrapper class for compatibility with app.py
class SectionEvaluatorApp(SectionEvaluator):
    """Wrapper class to maintain compatibility with the app.py interface."""
    pass