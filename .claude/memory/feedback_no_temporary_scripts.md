---
name: No temporary Python scripts
description: Don't create temporary Python scripts that apply changes to other files - modify directly instead
type: feedback
---

Don't create temporary Python scripts for one-time changes (e.g., a script that applies a change to another script). Instead, make the changes directly using Edit.

**Why:** The user finds these temporary scripts cluttering the repository. They serve no long-term purpose and need manual cleanup.

**How to apply:** When asked to make changes to code (rename variables, refactor, etc.), use Edit directly on the target files instead of generating a temporary migration/change script. Only create scripts if:
- The user explicitly asks for a reusable script
- The change requires complex logic that needs to be reviewed before execution
- The script serves a legitimate automation purpose beyond this one-time change
