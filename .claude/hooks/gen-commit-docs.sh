#!/bin/bash
# .claude/hooks/gen-commit-docs.sh
# Hook to generate documentation on commit

# Parse stdin to see if the command was a git commit
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.command // empty')

# Only act if it's a git commit
if [[ "$COMMAND" == *"git commit"* ]]; then
  
  # Ensure the directory exists
  mkdir -p "commit history"
  
  # Get last commit title (assuming commit already happened or staging)
  COMMIT_TITLE=$(git log -1 --pretty=%s)
  
  # Sanitize title for filename
  FILENAME="commit history/${COMMIT_TITLE// /-}.md"
  
  # Tell Claude to analyze changes and write them to the file
  # Note: This requires Claude Code's internal API to generate content
  git diff HEAD > /tmp/current_diff
  
  echo "Generating documentation for: $COMMIT_TITLE"
  
  # --- Logic to call Claude to write the documentation goes here ---
  # Alternatively, use this hook to simply stage the diff for another process
  git diff HEAD > "$FILENAME"
fi

# Exit 0 to allow the commit to proceed
exit 0
