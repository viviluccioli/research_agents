import streamlit as st
import json
import tempfile
import os
import re
from typing import Dict, List, Any
import pdfplumber
from utils import cm, single_query
import codecs

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
    
    def render_ui(self, files=None):
        """Render the Streamlit UI for the section evaluator."""
        st.subheader("Manuscript Section Evaluation")
        st.write("This tool analyzes each section of your manuscript, grades it, and provides improvement suggestions.")
        
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
        st.write("You can customize which sections to evaluate below:")
        selected_sections = {}
        for section, pattern in self.SECTION_PATTERNS.items():
            if st.checkbox(section, value=True, key=f"check_{section}"):
                selected_sections[section] = pattern
        
        if st.button("Evaluate Manuscript"):
            if not manuscript_file:
                st.error("Please select a manuscript file.")
                return
                
            with st.spinner("Extracting text from manuscript..."):
                # Extract sections from manuscript
                manuscript_text = self.extract_text_from_pdf(files[manuscript_file])
                manuscript_sections = self.extract_sections_from_text(manuscript_text, selected_sections)
            
            # Evaluate sections
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
        
        st.download_button(
            label="Download Full Evaluation",
            data=full_report,
            file_name="manuscript_evaluation.md",
            mime="text/markdown"
        )