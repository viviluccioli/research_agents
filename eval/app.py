# app.py - Evaluation Agent with tabbed workflows
import streamlit as st
from referee import RefereeReportChecker
from section_eval_new import SectionEvaluator
from utils import cm

WORKFLOWS = {
    "Referee Report": RefereeReportChecker,
    "Section Evaluator": SectionEvaluator,
}

def _ensure_session_keys():
    """Create session keys that the workflows expect."""
    if "file_data" not in st.session_state:
        st.session_state.file_data = {}
    # Track which tab was active last (for LLM history clearing)
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = None

def _safe_render(workflow_instance, files):
    """
    Render the workflow UI with the shared files
    """
    try:
        workflow_instance.render_ui(files=files)
    except Exception as e:
        st.error(f"Error rendering workflow UI: {e}")

def main():
    st.set_page_config(layout="wide", page_title="Evaluation Agent")
    st.title("Evaluation Agent")
    
    # Use cleaner heading style
    st.markdown("This agent helps you evaluate and improve your academic work. Choose one of two workflows below to get started.")
    
    _ensure_session_keys()

    # ----------------- Shared File Uploader -----------------
    with st.expander("Document Uploader", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload manuscripts and/or referee reports for evaluation.",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True
        )
        if uploaded_files:
            count = 0
            for uploaded_file in uploaded_files:
                st.session_state.file_data[uploaded_file.name] = uploaded_file.getvalue()
                count += 1
            st.success(f"Successfully uploaded {count} file(s)")
            # Clear LLM history because content changed
            try:
                cm.clear_history()
            except Exception:
                pass

    # Show available files in compact form
    if st.session_state.file_data:
        st.info(f"Available files: {', '.join(list(st.session_state.file_data.keys()))}")

    # ----------------- Create visual tabs -----------------
    tab_labels = list(WORKFLOWS.keys())
    tab_objs = st.tabs(tab_labels)
    tab_map = dict(zip(tab_labels, tab_objs))

    # Make tabs larger
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.25rem;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    # ----------------- Render each tab -----------------
    for label, tab in tab_map.items():
        with tab:
            # Clear LLM history when switching tabs
            if st.session_state.active_tab != label:
                if st.session_state.active_tab is not None:
                    try:
                        cm.clear_history()
                    except Exception:
                        pass
                st.session_state.active_tab = label

            # Workflow session key (persist instance)
            key = f"{label.lower().replace(' ', '_')}_workflow"
            if key not in st.session_state:
                # Instantiate and store
                try:
                    st.session_state[key] = WORKFLOWS[label]()
                except Exception as e:
                    st.error(f"Failed to initialize {label} workflow: {e}")
                    continue

            instance = st.session_state[key]
            

            # Safe render with shared files
            _safe_render(instance, files=st.session_state.get("file_data", {}))

    # Footer
    st.markdown("---")
    st.caption("Choose a tab above to access different evaluation workflows.")

if __name__ == "__main__":
    main()