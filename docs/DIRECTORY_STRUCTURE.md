# Directory Structure

## Current Organization (After Migration)

```
eval/
│
├── app_system/                          🆕 NEW: All production app files
│   ├── README.md                        📘 Documentation for app system
│   │
│   ├── app.py                           🎯 Main production app
│   ├── app_demo.py                      🎬 Demo 1 (Adjusted R² issue)
│   ├── app_demo2.py                     🎬 Demo 2 (standard errors issue)
│   │
│   ├── referee.py                       📋 Referee Report Checker UI
│   ├── multi_agent_debate.py            🤖 Multi-agent debate logic (Rounds 0-3)
│   ├── utils.py                         🔧 LLM utilities (single_query, ConversationManager)
│   │
│   ├── madoutput1.txt                   📄 Demo 1 debate transcript
│   ├── madouput2.txt                    📄 Demo 2 debate transcript
│   │
│   └── section_eval/                    📦 Section evaluator module
│       ├── __init__.py
│       ├── main.py                      Entry point
│       ├── evaluator.py                 Core evaluation logic
│       ├── utils.py                     Section eval LLM calls
│       ├── scoring.py                   Score calculations
│       ├── section_detection.py         Section identification
│       ├── text_extraction.py           PDF/text extraction
│       ├── hierarchy.py                 Section hierarchy
│       ├── criteria/                    📂 Evaluation criteria
│       │   ├── __init__.py
│       │   └── base.py                  Criteria definitions
│       └── prompts/                     📂 Prompt templates
│           ├── __init__.py
│           └── templates.py             System prompts
│
├── 📚 DOCUMENTATION FILES (11 files)
│   ├── ARCHITECTURE.md                  System architecture overview
│   ├── CRITERIA_MATRIX.md               Criteria by paper/section type
│   ├── CRITERIA_REFERENCE.md            Detailed criteria guide
│   ├── DEMO_README.md                   Demo apps guide
│   ├── EVALUATION_SYSTEM_IMPROVEMENTS.md Improvements documentation
│   ├── Evaluation_system_prompts.md     System prompts reference
│   ├── QUICK_REFERENCE.md               Quick start guide
│   ├── README_DOCS.md                   Documentation index
│   ├── SCORING_EXAMPLES_OLD_VS_NEW.md   Scoring comparison
│   ├── SYSTEM_PROMPTS_DOCUMENTATION.md  Complete prompts catalog
│   ├── TESTING_NEW_SYSTEM.md            Testing protocol
│   ├── APP_SYSTEM_MIGRATION.md          🆕 This migration guide
│   └── DIRECTORY_STRUCTURE.md           🆕 This file
│
├── 🧪 EXPERIMENTAL/LEGACY FILES
│   ├── madexp.py                        Original experimental debate
│   ├── madexp2.py                       Gemini Colab version
│   ├── madexp2_local.py                 Local Gemini version
│   ├── section_eval.py                  Legacy section evaluator v1
│   ├── section_eval_new.py              Legacy section evaluator v2
│   ├── section_eval_vivi_0223.py        Historical section evaluator
│   ├── section_eval_llm_vivi.py         Historical section evaluator
│   └── routing.py                       Routing utilities
│
├── 📊 ASSETS
│   └── mad1_table.png                   Demo visualization
│
├── 📁 DIRECTORIES
│   ├── changelog/                       Change history
│   ├── comparative results/             Results comparisons
│   └── __pycache__/                     Python cache
│
└── ⚙️ CONFIG
    └── pyproject.toml                   Python project config
```

## Before vs After

### Before (Messy):
```
eval/
├── app.py                    }
├── app_demo.py               }
├── app_demo2.py              } Mixed with everything else
├── referee.py                }
├── multi_agent_debate.py     }
├── utils.py                  }
├── section_eval/             }
├── madoutput1.txt            }
├── madouput2.txt             }
├── madexp.py
├── madexp2.py
├── madexp2_local.py
├── section_eval.py
├── section_eval_new.py
├── section_eval_vivi_0223.py
├── section_eval_llm_vivi.py
├── routing.py
├── [11 documentation files]
└── [other directories]
```

### After (Clean):
```
eval/
├── app_system/              ← All production app files here
│   ├── app.py
│   ├── app_demo.py
│   ├── app_demo2.py
│   ├── referee.py
│   ├── multi_agent_debate.py
│   ├── utils.py
│   ├── section_eval/
│   ├── madoutput1.txt
│   └── madouput2.txt
│
├── [11 documentation files]  ← Easy to find
├── [8 experimental files]    ← Clearly separated
└── [other directories]
```

## Key Benefits

1. ✅ **Separation of Concerns**: Production apps isolated from experiments
2. ✅ **Easy Navigation**: All app files in one directory
3. ✅ **Clear Purpose**: Each directory has a clear role
4. ✅ **Preserved Functionality**: All imports work unchanged
5. ✅ **Scalable**: Easy to add new features to app_system

## Running the Apps

```bash
# From eval/ directory (recommended)
streamlit run app_system/app.py
streamlit run app_system/app_demo.py
streamlit run app_system/app_demo2.py

# Or from app_system/ directory
cd app_system
streamlit run app.py
streamlit run app_demo.py
streamlit run app_demo2.py
```

## File Locations Quick Reference

| File Type | Location | Count |
|-----------|----------|-------|
| Production apps | `app_system/` | 3 apps |
| Core modules | `app_system/` | 3 files |
| Section evaluator | `app_system/section_eval/` | 1 directory |
| Demo data | `app_system/` | 2 files |
| Documentation | `eval/` (root) | 13 files |
| Experimental | `eval/` (root) | 8 files |
| Assets | `eval/` (root) | 1 file |

## Migration Summary

- **Date**: March 11, 2026
- **Files Moved**: 8 files + 1 directory (section_eval/)
- **Import Changes**: None (all relative imports preserved)
- **Functionality**: 100% preserved
- **Organization**: Significantly improved ✨
