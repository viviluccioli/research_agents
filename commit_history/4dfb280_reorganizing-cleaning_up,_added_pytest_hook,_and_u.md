# Commit: reorganizing/cleaning up, added pytest hook, and updated context/documentation for reproducibility

**Hash**: `4dfb2803de446d2d86d236b400c9245a75da6e51`
**Date**: 2026-05-06 13:03:46 -0400
**Author**: Viviana C. Luccioli

## Changes Summary

```
commit 4dfb2803de446d2d86d236b400c9245a75da6e51
Author: Viviana C. Luccioli <m1vcl00@salt2.rsma.frb.gov>
Date:   Wed May 6 13:03:46 2026 -0400

    reorganizing/cleaning up, added pytest hook, and updated context/documentation for reproducibility

 .claude/hooks/gen-commit-docs.sh                   | 71 ++++++++++-------
 .claude/hooks/run-tests-on-edit.sh                 | 88 ++++++++++++++++++++++
 .claude/settings.json                              |  6 +-
 .gitignore                                         |  1 +
 CLAUDE.md                                          | 63 ++++++++++++++++
 README.md                                          | 58 +++++++++++++-
 app_system/{ => _archives}/app_backup_20260420.py  |  0
 app_system/{ => _archives}/app_exp_4.py            |  0
 .../referee/engine_backup_20260420.py              |  0
 app_system/{ => _archives}/referee/engine_exp_4.py |  0
 .../referee/workflow_backup_20260420.py            |  0
 .../{ => _archives}/referee/workflow_exp_4.py      |  0
 app_system/{ => _archives}/run_app_exp_4.sh        |  0
 app_system/app-memo.py                             |  2 +-
 app_system/{ => docs}/EXPERIMENT_4_SUMMARY.md      |  0
 app_system/{ => docs}/README_EXP_4.md              |  0
 app_system/{ => docs}/TEST_EXP_4.md                |  0
 app_system/referee/memo/__init__.py                | 18 +++++
 app_system/referee/{ => memo}/memo_engine.py       |  2 +-
 app_system/referee/{ => memo}/memo_prompts.py      |  0
 pyproject.toml                                     |  4 +
 requirements.txt                                   |  1 +
 22 files changed, 280 insertions(+), 34 deletions(-)
```

## Full Diff

```diff
commit 4dfb2803de446d2d86d236b400c9245a75da6e51
Author: Viviana C. Luccioli <m1vcl00@salt2.rsma.frb.gov>
Date:   Wed May 6 13:03:46 2026 -0400

    reorganizing/cleaning up, added pytest hook, and updated context/documentation for reproducibility

diff --git a/.claude/hooks/gen-commit-docs.sh b/.claude/hooks/gen-commit-docs.sh
index 9c5ed1a..bed8a94 100755
--- a/.claude/hooks/gen-commit-docs.sh
+++ b/.claude/hooks/gen-commit-docs.sh
@@ -2,32 +2,47 @@
 # .claude/hooks/gen-commit-docs.sh
 # Hook to generate documentation on commit
 
-# Parse stdin to see if the command was a git commit
-INPUT=$(cat)
-COMMAND=$(echo "$INPUT" | jq -r '.command // empty')
-
-# Only act if it's a git commit
-if [[ "$COMMAND" == *"git commit"* ]]; then
-  
-  # Ensure the directory exists
-  mkdir -p "commit history"
-  
-  # Get last commit title (assuming commit already happened or staging)
-  COMMIT_TITLE=$(git log -1 --pretty=%s)
-  
-  # Sanitize title for filename
-  FILENAME="commit history/${COMMIT_TITLE// /-}.md"
-  
-  # Tell Claude to analyze changes and write them to the file
-  # Note: This requires Claude Code's internal API to generate content
-  git diff HEAD > /tmp/current_diff
-  
-  echo "Generating documentation for: $COMMIT_TITLE"
-  
-  # --- Logic to call Claude to write the documentation goes here ---
-  # Alternatively, use this hook to simply stage the diff for another process
-  git diff HEAD > "$FILENAME"
-fi
-
-# Exit 0 to allow the commit to proceed
+# Change to repo root
+cd /casl/home/m1vcl00/FS-CASL/research_agents || exit 1
+
+# Ensure the directory exists
+mkdir -p "commit_history"
+
+# Get last commit info
+COMMIT_HASH=$(git log -1 --pretty=%H)
+COMMIT_TITLE=$(git log -1 --pretty=%s)
+COMMIT_DATE=$(git log -1 --pretty=%ci)
+COMMIT_AUTHOR=$(git log -1 --pretty=%an)
+
+# Sanitize title for filename (remove special chars, limit length)
+SAFE_TITLE=$(echo "$COMMIT_TITLE" | tr '/' '-' | tr ' ' '_' | cut -c1-50)
+FILENAME="commit_history/${COMMIT_HASH:0:7}_${SAFE_TITLE}.md"
+
+# Get the actual committed changes (diff against parent)
+CHANGES=$(git show --stat "$COMMIT_HASH")
+DIFF=$(git show "$COMMIT_HASH")
+
+# Generate documentation
+cat > "$FILENAME" <<EOF
+# Commit: $COMMIT_TITLE
+
+**Hash**: \`$COMMIT_HASH\`
+**Date**: $COMMIT_DATE
+**Author**: $COMMIT_AUTHOR
+
+## Changes Summary
+
+\`\`\`
+$CHANGES
+\`\`\`
+
+## Full Diff
+
+\`\`\`diff
+$DIFF
+\`\`\`
+EOF
+
+echo "✅ Generated commit documentation: $FILENAME"
+
 exit 0
diff --git a/.claude/hooks/run-tests-on-edit.sh b/.claude/hooks/run-tests-on-edit.sh
new file mode 100755
index 0000000..4e647df
--- /dev/null
+++ b/.claude/hooks/run-tests-on-edit.sh
@@ -0,0 +1,88 @@
+#!/bin/bash
+# .claude/hooks/run-tests-on-edit.sh
+# Hook to run pytest on modified Python files
+
+# Change to repo root
+cd /casl/home/m1vcl00/FS-CASL/research_agents || exit 1
+
+# Activate virtual environment
+source venv/bin/activate 2>/dev/null || true
+
+# Read the tool result from stdin
+TOOL_RESULT=$(cat)
+
+# Extract the file path from the tool result
+FILE_PATH=$(echo "$TOOL_RESULT" | jq -r '.tool_input.file_path // .tool_response.filePath // empty')
+
+# Exit if no file path or not a Python file
+if [[ -z "$FILE_PATH" ]] || [[ "$FILE_PATH" != *.py ]]; then
+    exit 0
+fi
+
+# Skip if file is not in app_system/
+if [[ "$FILE_PATH" != *"app_system/"* ]]; then
+    exit 0
+fi
+
+# Skip test files themselves
+if [[ "$FILE_PATH" == *"/tests/"* ]] || [[ "$FILE_PATH" == test_*.py ]]; then
+    exit 0
+fi
+
+echo "🧪 Running tests for modified file: $FILE_PATH"
+
+# Extract module name from file path
+# E.g., app_system/referee/engine.py -> referee.engine
+MODULE_NAME=$(echo "$FILE_PATH" | sed 's|app_system/||' | sed 's|\.py$||' | sed 's|/|.|g')
+
+# Try to find related test file
+TEST_FILE=""
+if [[ "$FILE_PATH" == app_system/referee/* ]]; then
+    # For referee package, look for tests in app_system/tests/
+    BASE_NAME=$(basename "$FILE_PATH" .py)
+    TEST_FILE="app_system/tests/test_${BASE_NAME}.py"
+elif [[ "$FILE_PATH" == app_system/section_eval/* ]]; then
+    # For section_eval package, look for tests
+    BASE_NAME=$(basename "$FILE_PATH" .py)
+    TEST_FILE="app_system/tests/test_${BASE_NAME}.py"
+else
+    # Generic pattern
+    BASE_NAME=$(basename "$FILE_PATH" .py)
+    TEST_FILE="app_system/tests/test_${BASE_NAME}.py"
+fi
+
+# Run tests
+if [[ -f "$TEST_FILE" ]]; then
+    echo "  → Running specific tests: $TEST_FILE"
+    python -m pytest "$TEST_FILE" -v --tb=short 2>&1
+    TEST_EXIT=$?
+else
+    # If no specific test file, run quick smoke tests
+    echo "  → No specific test file found, running smoke tests"
+
+    # Try quick tests first
+    if [[ -f "app_system/tests/test_referee_quick.py" ]]; then
+        python -m pytest app_system/tests/test_referee_quick.py -v --tb=short 2>&1
+        TEST_EXIT=$?
+    else
+        echo "  ℹ️  No smoke tests found - skipping"
+        exit 0
+    fi
+
+    # If no tests collected, that's OK
+    if [[ $TEST_EXIT -eq 5 ]]; then
+        echo "  ℹ️  No tests found for this module - skipping"
+        exit 0
+    fi
+fi
+
+if [[ $TEST_EXIT -eq 0 ]]; then
+    echo "✅ All tests passed!"
+    exit 0
+else
+    echo "❌ Tests failed (exit code: $TEST_EXIT)"
+    echo "  ⚠️  Review test failures above"
+    # Exit 0 to not block Claude's response to user
+    # The test output above will be visible in the conversation
+    exit 0
+fi
diff --git a/.claude/settings.json b/.claude/settings.json
index 5fa753f..168019e 100644
--- a/.claude/settings.json
+++ b/.claude/settings.json
@@ -6,9 +6,9 @@
         "hooks": [
           {
             "type": "command",
-            "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | xargs -I{} bash -c 'if [[ \"{}\" == *.py ]]; then cd /casl/home/m1vcl00/FS-CASL/research_agents && python3 -m pytest app_system/tests/; fi'",
-            "statusMessage": "Running tests...",
-            "async": true
+            "command": "bash .claude/hooks/run-tests-on-edit.sh",
+            "statusMessage": "Running tests on modified file...",
+            "async": false
           }
         ]
       },
diff --git a/.gitignore b/.gitignore
index d3c08c4..9534867 100755
--- a/.gitignore
+++ b/.gitignore
@@ -1,6 +1,7 @@
 # Specific to this repo
 
 claude_extra_docs/
+commit_history/
 papers/*.txt
 app_system/tests/example_math_cleanup_integration.py
 app_system/.referee_cache/
diff --git a/CLAUDE.md b/CLAUDE.md
index b18f012..8e18245 100644
--- a/CLAUDE.md
+++ b/CLAUDE.md
@@ -76,6 +76,69 @@ Configuration is loaded by `app_system/config.py` which uses `python-dotenv`. Th
 
 **Never commit `.env` to git** - it's in `.gitignore`.
 
+## Claude Code Hooks & Commit History
+
+This repository uses Claude Code automation hooks to maintain code quality and documentation.
+
+### Automated Hooks
+
+Two hooks are configured in `.claude/settings.json`:
+
+1. **Test Runner** (`.claude/hooks/run-tests-on-edit.sh`)
+   - **Trigger**: Runs automatically when Python files in `app_system/` are edited via Write or Edit tools
+   - **Action**: Executes pytest on related test files
+   - **Behavior**: Non-blocking (shows output but doesn't halt Claude)
+   - **Skips**: Test files themselves, files outside `app_system/`
+
+2. **Commit Documentation Generator** (`.claude/hooks/gen-commit-docs.sh`)
+   - **Trigger**: Runs automatically after any `git commit` command
+   - **Action**: Generates a markdown file documenting the commit
+   - **Output Location**: `commit_history/{short_hash}_{sanitized_title}.md`
+   - **Content**: Commit hash, date, author, changes summary, and full diff
+
+### Commit History Archive
+
+**IMPORTANT FOR HANDOFF**: All git commits are automatically documented in the `commit_history/` directory.
+
+**Format**: Each file follows the pattern `{7-char-hash}_{commit-title}.md`
+
+**Contents**:
+- Commit metadata (hash, date, author)
+- Changes summary (`git show --stat`)
+- Full diff (`git show`)
+
+**Example**: `commit_history/de77b56_added_git_commit_hook.md`
+
+**Use Cases**:
+- Review what changed in a specific commit without running `git show`
+- Understand context behind major changes during handoff
+- Quick reference for Claude Code when understanding repository history
+- Searchable archive of all development decisions
+
+**Note**: The `commit_history/` directory is tracked in git, so all commit documentation is version-controlled and travels with the repository.
+
+### Hook Configuration
+
+To view or modify hook behavior, edit `.claude/settings.json`:
+
+```json
+{
+  "hooks": {
+    "PostToolUse": [
+      {
+        "matcher": "Write|Edit",
+        "hooks": [{"command": "bash .claude/hooks/run-tests-on-edit.sh"}]
+      },
+      {
+        "matcher": "Bash",
+        "filter": "git commit",
+        "hooks": [{"command": "bash .claude/hooks/gen-commit-docs.sh"}]
+      }
+    ]
+  }
+}
+```
+
 ## File Organization Rules
 
 The `app_system/` directory follows standard Python package organization:
diff --git a/README.md b/README.md
index 9e20c8d..aba8270 100644
--- a/README.md
+++ b/README.md
@@ -194,8 +194,64 @@ research_agents/
 ├── papers/                            # Sample papers for testing
 ├── requirements.txt                   # Python dependencies
 ├── README.md                          # This file
-└── CLAUDE.md                          # Guidance for Claude Code
+├── CLAUDE.md                          # Guidance for Claude Code
+│
+├── .claude/                           # Claude Code configuration & hooks
+│   ├── settings.json                 # Hook configuration
+│   ├── hooks/                        # Automation scripts
+│   │   ├── gen-commit-docs.sh       # Auto-generates commit documentation
+│   │   └── run-tests-on-edit.sh     # Auto-runs tests on file edits
+│   ├── rules/                        # Context-specific guidance for Claude
+│   ├── memory/                       # Claude Code persistent memory
+│   └── skills/                       # Custom Claude Code skills
+│
+└── commit_history/                    # 📝 Auto-generated commit documentation
+    └── {hash}_{title}.md             # One markdown file per git commit
+```
+
+---
+
+## Claude Code Hooks & Commit History
+
+**IMPORTANT FOR HANDOFF**: This repository uses automated hooks that run during development with Claude Code.
+
+### Automated Documentation
+
+Every git commit automatically generates a documentation file in `commit_history/`:
+
+**File Format**: `{7-char-hash}_{sanitized-commit-title}.md`
+
+**Contents**:
+- Commit metadata (hash, date, author)
+- Changes summary (files modified, insertions, deletions)
+- Full diff of all changes
+
+**Example**: After running `git commit -m "added git commit hook"`, the system automatically creates:
 ```
+commit_history/de77b56_added_git_commit_hook.md
+```
+
+**Why This Matters**: When you take over this repository, you can quickly review what changed in any commit by reading these markdown files instead of running `git show`. This is especially useful for understanding major architectural decisions or debugging issues.
+
+### Automated Testing
+
+When editing Python files in `app_system/`, Claude Code automatically runs relevant pytest tests to catch regressions immediately.
+
+**How It Works**:
+- Hook: `.claude/hooks/run-tests-on-edit.sh`
+- Trigger: Any Write or Edit operation on `*.py` files in `app_system/`
+- Action: Runs related test file (e.g., editing `engine.py` runs `test_engine.py`)
+- Falls back to smoke tests if no specific test exists
+
+### Hook Configuration
+
+All hooks are configured in `.claude/settings.json`. To modify hook behavior:
+
+```bash
+nano .claude/settings.json
+```
+
+See `CLAUDE.md` for detailed hook documentation.
 
 ---
 
diff --git a/app_system/app_backup_20260420.py b/app_system/_archives/app_backup_20260420.py
similarity index 100%
rename from app_system/app_backup_20260420.py
rename to app_system/_archives/app_backup_20260420.py
diff --git a/app_system/app_exp_4.py b/app_system/_archives/app_exp_4.py
similarity index 100%
rename from app_system/app_exp_4.py
rename to app_system/_archives/app_exp_4.py
diff --git a/app_system/referee/engine_backup_20260420.py b/app_system/_archives/referee/engine_backup_20260420.py
similarity index 100%
rename from app_system/referee/engine_backup_20260420.py
rename to app_system/_archives/referee/engine_backup_20260420.py
diff --git a/app_system/referee/engine_exp_4.py b/app_system/_archives/referee/engine_exp_4.py
similarity index 100%
rename from app_system/referee/engine_exp_4.py
rename to app_system/_archives/referee/engine_exp_4.py
diff --git a/app_system/referee/workflow_backup_20260420.py b/app_system/_archives/referee/workflow_backup_20260420.py
similarity index 100%
rename from app_system/referee/workflow_backup_20260420.py
rename to app_system/_archives/referee/workflow_backup_20260420.py
diff --git a/app_system/referee/workflow_exp_4.py b/app_system/_archives/referee/workflow_exp_4.py
similarity index 100%
rename from app_system/referee/workflow_exp_4.py
rename to app_system/_archives/referee/workflow_exp_4.py
diff --git a/app_system/run_app_exp_4.sh b/app_system/_archives/run_app_exp_4.sh
similarity index 100%
rename from app_system/run_app_exp_4.sh
rename to app_system/_archives/run_app_exp_4.sh
diff --git a/app_system/app-memo.py b/app_system/app-memo.py
index 2d3d770..57ef431 100644
--- a/app_system/app-memo.py
+++ b/app_system/app-memo.py
@@ -18,7 +18,7 @@ from io import BytesIO
 import pdfplumber
 
 from utils import cm
-from referee.memo_engine import execute_debate_pipeline, MEMO_SYSTEM_PROMPTS
+from referee.memo.memo_engine import execute_debate_pipeline, MEMO_SYSTEM_PROMPTS
 from referee._utils.summarizer import summarize_all_rounds
 
 # Import helper functions from archived full output UI (domain-agnostic)
diff --git a/app_system/EXPERIMENT_4_SUMMARY.md b/app_system/docs/EXPERIMENT_4_SUMMARY.md
similarity index 100%
rename from app_system/EXPERIMENT_4_SUMMARY.md
rename to app_system/docs/EXPERIMENT_4_SUMMARY.md
diff --git a/app_system/README_EXP_4.md b/app_system/docs/README_EXP_4.md
similarity index 100%
rename from app_system/README_EXP_4.md
rename to app_system/docs/README_EXP_4.md
diff --git a/app_system/TEST_EXP_4.md b/app_system/docs/TEST_EXP_4.md
similarity index 100%
rename from app_system/TEST_EXP_4.md
rename to app_system/docs/TEST_EXP_4.md
diff --git a/app_system/referee/memo/__init__.py b/app_system/referee/memo/__init__.py
new file mode 100644
index 0000000..1ea0a78
--- /dev/null
+++ b/app_system/referee/memo/__init__.py
@@ -0,0 +1,18 @@
+"""
+Memo Evaluation System
+
+A parallel system for evaluating policy memos using Multi-Agent Debate architecture
+with memo-specific analyst personas.
+
+Main components:
+- memo_engine: Debate orchestration for memo evaluation
+- memo_prompts: Memo-specific analyst personas
+"""
+
+from .memo_engine import execute_debate_pipeline
+from .memo_prompts import MEMO_SYSTEM_PROMPTS
+
+__all__ = [
+    'execute_debate_pipeline',
+    'MEMO_SYSTEM_PROMPTS',
+]
diff --git a/app_system/referee/memo_engine.py b/app_system/referee/memo/memo_engine.py
similarity index 99%
rename from app_system/referee/memo_engine.py
rename to app_system/referee/memo/memo_engine.py
index 7d64e64..79dbe77 100644
--- a/app_system/referee/memo_engine.py
+++ b/app_system/referee/memo/memo_engine.py
@@ -13,7 +13,7 @@ from typing import Dict, List, Optional
 from pathlib import Path
 
 from utils import single_query, count_tokens
-from referee.memo_prompts import (
+from .memo_prompts import (
     MEMO_SYSTEM_PROMPTS,
     MEMO_SELECTION_PROMPT,
     MEMO_TYPE_CONTEXTS,
diff --git a/app_system/referee/memo_prompts.py b/app_system/referee/memo/memo_prompts.py
similarity index 100%
rename from app_system/referee/memo_prompts.py
rename to app_system/referee/memo/memo_prompts.py
diff --git a/pyproject.toml b/pyproject.toml
index 845f64d..1b9d9fc 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,3 +1,7 @@
+[project]
+name = "research-agents"
+requires-python = ">=3.9"
+
 dependencies = [
     "streamlit>=1.11.0",
     "pandas>=1.3.0",
diff --git a/requirements.txt b/requirements.txt
index c435f35..c0b7ec9 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -14,3 +14,4 @@ filelock  # For thread-safe cache file locking
 thefuzz  # For fuzzy string matching (quote validation)
 python-Levenshtein  # Optional: speeds up thefuzz (recommended)
 sentence-transformers  # Optional: for semantic similarity in deduplication (requires ~500MB model download)
+pytest>=7.0.0  # For running tests
```
