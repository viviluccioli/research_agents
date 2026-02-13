# referee_checker.py
#not working becuase of API timeout? 
import streamlit as st
import json
import tempfile
import os
from typing import Dict, List, Any
import pdfplumber
from utils import cm, single_query

class RefereeReportChecker:
    """
    Workflow for analyzing referee reports and suggesting revisions
    to academic papers based on referee feedback.
    """
    
    def __init__(self, llm=cm):
        """Initialize with conversation manager from utils."""
        self.llm = llm
    
    def render_ui(self, files=None):
        """Render the Streamlit UI for the referee report checker."""
        st.subheader("Referee Report Analysis")
        st.write("This tool analyzes a provided referee report and suggests according revisions to your draft(s).")
        
        # Check if files are available
        if not files:
            st.info("Please upload referee reports and your manuscript using the document uploader above.")
            return
            
        # File selection
        manuscript_file = st.selectbox(
            "Select your manuscript",
            options=list(files.keys()),
            key="manuscript_selector"
        )
        
        referee_files = st.multiselect(
            "Select referee report(s)",
            options=list(files.keys()),
            key="referee_selector"
        )
        
        if st.button("Analyze Reports"):
            if not manuscript_file:
                st.error("Please select a manuscript file.")
                return
                
            if not referee_files:
                st.error("Please select at least one referee report.")
                return
                
            with st.spinner("Extracting text from files..."):
                # Extract text from manuscript
                manuscript_sections = self.extract_sections_from_pdf(
                    files[manuscript_file]
                )
                
                # Extract text from referee reports
                referee_texts = []
                for report_file in referee_files:
                    referee_text = self.extract_text_from_pdf(files[report_file])
                    referee_texts.append(referee_text)
            
            # Process the reports
            with st.spinner("Analyzing referee reports..."):
                # Summarize referee reports
                suggestions = self.summarize_referee_reports(referee_texts)
                st.session_state.suggestions = suggestions
                
                # Compare paper with suggestions
                comparison = self.compare_paper_and_suggestions(
                    manuscript_sections, suggestions
                )
                st.session_state.comparison = comparison
                
                # Edit paper with suggestions
                revised_sections = self.edit_paper_with_suggestions(
                    manuscript_sections, suggestions
                )
                st.session_state.revised_sections = revised_sections
            
            # Display results
            self.display_results(
                comparison, suggestions, manuscript_sections, revised_sections
            )
    
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
    
    def extract_sections_from_pdf(self, file_content):
        """Extract sections from a PDF manuscript."""
        full_text = self.extract_text_from_pdf(file_content)
        
        # Use LLM to identify sections
        prompt = f"""
        Extract the main sections from this academic paper. Output should be a JSON object
        where keys are section names (like "Introduction", "Methods", etc.) and values
        are the section text.
        
        PAPER TEXT:
        {full_text[:50000]}
        
        Return ONLY valid JSON.
        """
        
        response = single_query(prompt)
        
        try:
            sections = json.loads(response)
            if not isinstance(sections, dict):
                raise ValueError("Expected a dictionary of sections")
        except Exception as e:
            st.warning(f"Could not extract sections: {e}")
            sections = {"Full Text": full_text}
        
        return sections
    
    def summarize_referee_reports(self, referee_reports):
        """Summarizes referee reports into structured suggestions."""
        combined_reports = "\n\n".join(referee_reports)
        prompt = f"""
        You are an expert research editor. Summarize the following referee reports into
        concise, actionable revision steps for the authors.

        Output a JSON object with:
        - "major_revisions": list of key conceptual or methodological revisions
        - "minor_revisions": list of small corrections or clarifications
        - "section_specific_comments": dict mapping section names to lists of comments

        Return valid JSON only.

        REFEREE REPORTS:
        {combined_reports}
        """

        response = single_query(prompt)

        try:
            summary = json.loads(response)
            if not isinstance(summary, dict):
                raise ValueError("Invalid JSON structure.")
        except Exception as e:
            st.warning(f"Invalid JSON output: {e}")
            summary = {"major_revisions": [], "minor_revisions": [], "section_specific_comments": {}}

        return summary
    
    def compare_paper_and_suggestions(self, paper_sections, suggestions):
        """Holistically compares the paper with referee suggestions."""
        paper_text = "\n".join(paper_sections.values())
        prompt = f"""
        You are an economics editor at a top 5 economics journal. Given the following paper and a structured list of referee suggestions,
        produce a holistic summary of key mismatches between the paper and the feedback.
        Do NOT go section by section. Focus on how well the paper as a whole addresses the feedback.

        PAPER:
        {paper_text[:15000]}

        SUGGESTIONS:
        {json.dumps(suggestions, indent=2)}

        Provide a concise bullet-point list of gaps or areas needing major adjustment.
        """

        return self.llm.conv_query(prompt)
    
    def edit_paper_with_suggestions(self, paper_sections, suggestions):
        """Edits paper sections based on referee suggestions."""
        revised_sections = {}
        
        with st.expander("Section Revision Progress", expanded=False):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_sections = len(paper_sections)
            for i, (section_name, content) in enumerate(paper_sections.items()):
                status_text.text(f"Revising section: {section_name}")
                
                prompt = f"""
                Section: {section_name}

                Original text:
                {content}

                Referee suggestions (holistic and section-specific):
                {json.dumps(suggestions, indent=2)}

                Please revise this section in LaTeX format, using \\added{{}} for insertions and \\deleted{{}} for deletions.
                Append any editorial comments in orange at the end if necessary.
                """
                try:
                    revised_text = self.llm.conv_query(prompt)
                    revised_sections[section_name] = revised_text
                except Exception as e:
                    st.warning(f"Failed to revise section {section_name}: {e}")
                    revised_sections[section_name] = content
                
                progress_bar.progress((i + 1) / total_sections)
            
            status_text.text("All sections revised!")
        
        return revised_sections
    
    def display_results(self, comparison, suggestions, original_sections, revised_sections):
        """Display the results in the Streamlit UI."""
        st.success("Analysis complete!")
        
        # Summary of changes
        st.subheader("Summary of Suggested Changes")
        st.write(comparison)
        
        # Major revisions
        st.subheader("Major Revisions")
        for item in suggestions.get("major_revisions", []):
            st.write(f"- {item}")
        
        # Minor revisions
        st.subheader("Minor Revisions")
        for item in suggestions.get("minor_revisions", []):
            st.write(f"- {item}")
        
        # Section revisions
        st.subheader("Revised Sections")
        for section_name, revised_text in revised_sections.items():
            with st.expander(f"{section_name}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.text(original_sections.get(section_name, ""))
                with col2:
                    st.markdown("**Revised**")
                    st.text(revised_text)
        
        # Download options
        st.subheader("Download Results")
        
        # Prepare full report
        full_report = f"""# Referee Report Analysis

## Summary of Suggested Changes
{comparison}

## Major Revisions
{"".join(f"- {item}\n" for item in suggestions.get("major_revisions", []))}

## Minor Revisions
{"".join(f"- {item}\n" for item in suggestions.get("minor_revisions", []))}

## Revised Sections
{"".join(f"### {section}\n\n{text}\n\n" for section, text in revised_sections.items())}
"""
        
        st.download_button(
            label="Download Full Report",
            data=full_report,
            file_name="referee_report_analysis.md",
            mime="text/markdown"
        )

#Missing line by line editor, but I'm not sure I can output that in streamlit 