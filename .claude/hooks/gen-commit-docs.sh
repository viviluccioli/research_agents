#!/bin/bash
# .claude/hooks/gen-commit-docs.sh
# Hook to generate documentation on commit

# Change to repo root
cd /casl/home/m1vcl00/FS-CASL/research_agents || exit 1

# Ensure the directory exists
mkdir -p "commit_history"

# Get last commit info
COMMIT_HASH=$(git log -1 --pretty=%H)
COMMIT_TITLE=$(git log -1 --pretty=%s)
COMMIT_DATE=$(git log -1 --pretty=%ci)
COMMIT_AUTHOR=$(git log -1 --pretty=%an)

# Sanitize title for filename (remove special chars, limit length)
SAFE_TITLE=$(echo "$COMMIT_TITLE" | tr '/' '-' | tr ' ' '_' | cut -c1-50)
FILENAME="commit_history/${COMMIT_HASH:0:7}_${SAFE_TITLE}.md"

# Get the actual committed changes (diff against parent)
CHANGES=$(git show --stat "$COMMIT_HASH")
DIFF=$(git show "$COMMIT_HASH")

# Generate documentation
cat > "$FILENAME" <<EOF
# Commit: $COMMIT_TITLE

**Hash**: \`$COMMIT_HASH\`
**Date**: $COMMIT_DATE
**Author**: $COMMIT_AUTHOR

## Changes Summary

\`\`\`
$CHANGES
\`\`\`

## Full Diff

\`\`\`diff
$DIFF
\`\`\`
EOF

echo "✅ Generated commit documentation: $FILENAME"

exit 0
