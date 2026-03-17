# Demo App Guide

## 🚀 Running the Demo

### Quick Start
```bash
cd /ofs/home/m1vcl00/FS-CASL/research_agents-main/eval
streamlit run app_demo.py
```

### What's Included

This demo showcases TWO evaluation systems side-by-side:

## 📁 Shared File Uploader (Top of Page)

**Features:**
- Upload multiple manuscripts (PDF, TXT, DOCX, LaTeX)
- Paste text directly (plain text or LaTeX source)
- Files available to Section Evaluator tab
- Session state management

**Location:** Above tabs, always accessible

---

## 📝 Tab 1: Section Evaluator

**Architecture Displayed:**
```
Paper Type Selection → Text Extraction → Section Detection
    ↓
Paper Type-Specific Criteria Mapping
    ↓
Weighted Section Evaluation
    ↓
Overall Scoring & Publication Readiness
```

**Features:**
- Uses uploaded files from shared uploader
- Automatic section detection with hierarchy
- Paper type-specific evaluation (Empirical/Theoretical/Policy)
- Weighted scoring with importance multipliers
- Download reports as PDF or Markdown

**Live Functionality:**
- ✅ **Fully functional** - Real evaluation of your papers!
- ✅ Real LLM evaluation
- ✅ PDF generation works
- ✅ Upload files and paste text
- ✅ All original features included

---

## ⚖️ Tab 2: Referee Report (Multi-Agent Debate)

**Architecture Displayed:**
```
Three Specialized Agents:
  🔢 Mathematician → Technical validity
  📚 Historian → Literature context
  🚀 Visionary → Innovation potential
        ↓
Multi-Round Debate:
  Round 1: Independent Evaluation
  Round 2A: Cross-Examination
  Round 2B: Final Amendments
  Round 3: Editor Decision
```

**Demo Features:**
- **Hardcoded demonstration** using real multi-agent output
- Styled agent personas with color-coded boxes
- Round-by-round debate display
- Final verdict summary table
- Editor's synthesized decision

**Current Status:**
- 🎭 **Demo mode** - Shows pre-generated output (doesn't use uploaded files)
- 📊 Displays actual multi-agent debate transcript from madoutput1.txt
- 🖼️ Shows final results table image
- ⏳ **Live evaluation coming soon** (framework in madexp.py)
- 💡 Files uploaded above are for Section Evaluator only

---

## 🎨 Visual Features

### Color-Coded Personas

| Agent | Color | Border |
|-------|-------|--------|
| 🔢 Mathematician | Light Blue | Blue |
| 📚 Historian | Light Purple | Purple |
| 🚀 Visionary | Light Orange | Orange |
| 🏛️ Editor | Light Green | Green |

### Verdict Styling

- ✅ **PASS** → Green, bold
- ❌ **FAIL** → Red, bold
- ⚠️ **REVISE** → Orange, bold

### Round Headers

Gradient purple headers for each debate round with clear titles.

---

## 📊 Architecture Diagrams

Both tabs start with ASCII architecture diagrams showing:

### Section Evaluator:
- Tree-based hierarchical structure
- Paper type differentiation
- Importance multipliers visualization
- Weighted scoring formula

### Referee Report:
- Multi-agent system overview
- Agent specializations
- Debate flow (Rounds 1 → 2A → 2B → 3)
- Hierarchy of Truth decision rules

---

## 🔄 Differences from Original app.py

| Feature | Original app.py | app_demo.py |
|---------|----------------|-------------|
| **File Upload** | ✅ Shared uploader | ✅ Shared uploader + paste text |
| **Section Eval** | ✅ Fully functional | ✅ Fully functional (same features) |
| **Referee Report** | Basic stub | 🎭 Beautiful demo with styled output |
| **Architecture Diagrams** | ❌ None | ✅ Both tabs (ASCII art) |
| **Styling** | Basic | ✅ Color-coded personas, styled boxes |
| **Debate Display** | ❌ None | ✅ Round-by-round with expanders |
| **Results Table** | ❌ None | ✅ Image display + markdown table |

---

## 📁 Files Used

### Demo Dependencies
```
app_demo.py                  # Main demo application
madoutput1.txt              # Multi-agent debate transcript (source)
mad1_table.png              # Results summary table image
section_eval/               # Section evaluation framework
utils.py                    # LLM utilities (MartinAI API)
```

### Documentation References
```
ARCHITECTURE.md             # Detailed system architecture
CRITERIA_REFERENCE.md       # Complete evaluation criteria
QUICK_REFERENCE.md          # Quick lookup guide
CRITERIA_MATRIX.md          # Criteria by paper type
```

---

## 🎯 Use Cases

### For Demos/Presentations
1. **Show Section Evaluator**: Upload a real paper, get live evaluation
2. **Show Referee Report**: Display the beautiful multi-agent debate demo
3. **Explain Architecture**: Use the ASCII diagrams at top of each tab

### For Development
1. **Section Eval**: Modify criteria in `section_eval/criteria/base.py`
2. **Referee Report**: Implement live version using `madexp.py` as template
3. **Styling**: Adjust CSS in `app_demo.py` markdown blocks

### For Users
1. Upload manuscript in Section Evaluator tab
2. Get detailed section-by-section feedback
3. Download professional PDF report
4. View referee demo to understand multi-agent evaluation concept

---

## 🔧 Customization

### Changing Personas

Edit the persona boxes in `app_demo.py`:
```python
<div class="persona-box mathematician-box">
    <h4>🔢 Mathematician</h4>
    <p><strong>Focus:</strong> Your custom focus</p>
    ...
</div>
```

### Adding More Demo Content

The demo pulls from `madoutput1.txt`. To show different debates:
1. Run `madexp.py` with a different paper
2. Copy output to a new file (e.g., `madoutput2.txt`)
3. Update the expanders in Tab 2 with new content

### Modifying Architecture Diagrams

Edit the markdown blocks at the top of each tab:
```python
st.markdown("""
    ```
    Your custom ASCII diagram here
    ```
""")
```

---

## 🚀 Future Enhancements

### For Referee Report Tab (Convert Demo → Live)

**Step 1:** Integrate `madexp.py` async pipeline
```python
from madexp import execute_debate_pipeline
```

**Step 2:** Add file upload to Referee tab
```python
uploaded_paper = st.file_uploader("Upload paper for referee evaluation")
```

**Step 3:** Run live evaluation on upload
```python
if st.button("Run Multi-Agent Evaluation"):
    results = await execute_debate_pipeline(uploaded_paper)
    display_results(results)  # Format like demo
```

**Step 4:** Stream results in real-time
```python
with st.spinner(f"Round {round_num} in progress..."):
    # Display as each agent responds
```

### Additional Features

- **Debate Replay**: Slider to step through rounds
- **Agent Comparison**: Side-by-side verdict comparison
- **Custom Prompts**: Let users modify agent instructions
- **Paper Library**: Save and compare multiple evaluations
- **Export Debates**: Download full transcript as PDF

---

## 📝 Notes

### Demo Mode vs. Live Mode

**Demo Mode (Current):**
- ✅ Fast, no API calls for referee report
- ✅ Perfect for presentations
- ✅ Shows ideal output format
- ⚠️ Static content

**Live Mode (Future):**
- ✅ Real-time paper evaluation
- ✅ Dynamic agent responses
- ✅ Unique feedback per paper
- ⚠️ Slower (LLM calls)
- ⚠️ API rate limits

### Why Two Apps?

- `app.py` → Original working version
- `app_demo.py` → Enhanced demo with:
  - Architecture diagrams
  - Styled referee demo
  - Better presentation formatting

Choose based on your needs:
- **For real work**: Use `app.py`
- **For demos**: Use `app_demo.py`
- **For development**: Modify either based on requirements

---

## 🆘 Troubleshooting

### Image Not Loading
```python
# Check image path
import os
os.path.exists("/ofs/home/m1vcl00/FS-CASL/research_agents-main/eval/mad1_table.png")
```

### Section Eval Not Working
```bash
# Verify utils.py has correct API configuration
grep "API_KEY" utils.py
```

### Styling Not Appearing
```python
# Ensure unsafe_allow_html=True
st.markdown(content, unsafe_allow_html=True)
```

---

## 📞 Support

For questions:
- **Architecture**: See `ARCHITECTURE.md`
- **Criteria**: See `CRITERIA_REFERENCE.md`
- **Quick Help**: See `QUICK_REFERENCE.md`

---

**Last Updated:** March 5, 2026
