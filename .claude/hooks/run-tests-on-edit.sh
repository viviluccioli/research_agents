#!/bin/bash
# .claude/hooks/run-tests-on-edit.sh
# Hook to run pytest on modified Python files

# Change to repo root
cd /casl/home/m1vcl00/FS-CASL/research_agents || exit 1

# Activate virtual environment
source venv/bin/activate 2>/dev/null || true

# Read the tool result from stdin
TOOL_RESULT=$(cat)

# Extract the file path from the tool result
FILE_PATH=$(echo "$TOOL_RESULT" | jq -r '.tool_input.file_path // .tool_response.filePath // empty')

# Exit if no file path or not a Python file
if [[ -z "$FILE_PATH" ]] || [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Skip if file is not in app_system/
if [[ "$FILE_PATH" != *"app_system/"* ]]; then
    exit 0
fi

# Skip test files themselves
if [[ "$FILE_PATH" == *"/tests/"* ]] || [[ "$FILE_PATH" == test_*.py ]]; then
    exit 0
fi

echo "🧪 Running tests for modified file: $FILE_PATH"

# Extract module name from file path
# E.g., app_system/referee/engine.py -> referee.engine
MODULE_NAME=$(echo "$FILE_PATH" | sed 's|app_system/||' | sed 's|\.py$||' | sed 's|/|.|g')

# Try to find related test file
TEST_FILE=""
if [[ "$FILE_PATH" == app_system/referee/* ]]; then
    # For referee package, look for tests in app_system/tests/
    BASE_NAME=$(basename "$FILE_PATH" .py)
    TEST_FILE="app_system/tests/test_${BASE_NAME}.py"
elif [[ "$FILE_PATH" == app_system/section_eval/* ]]; then
    # For section_eval package, look for tests
    BASE_NAME=$(basename "$FILE_PATH" .py)
    TEST_FILE="app_system/tests/test_${BASE_NAME}.py"
else
    # Generic pattern
    BASE_NAME=$(basename "$FILE_PATH" .py)
    TEST_FILE="app_system/tests/test_${BASE_NAME}.py"
fi

# Run tests
if [[ -f "$TEST_FILE" ]]; then
    echo "  → Running specific tests: $TEST_FILE"
    python -m pytest "$TEST_FILE" -v --tb=short 2>&1
    TEST_EXIT=$?
else
    # If no specific test file, run quick smoke tests
    echo "  → No specific test file found, running smoke tests"

    # Try quick tests first
    if [[ -f "app_system/tests/test_referee_quick.py" ]]; then
        python -m pytest app_system/tests/test_referee_quick.py -v --tb=short 2>&1
        TEST_EXIT=$?
    else
        echo "  ℹ️  No smoke tests found - skipping"
        exit 0
    fi

    # If no tests collected, that's OK
    if [[ $TEST_EXIT -eq 5 ]]; then
        echo "  ℹ️  No tests found for this module - skipping"
        exit 0
    fi
fi

if [[ $TEST_EXIT -eq 0 ]]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Tests failed (exit code: $TEST_EXIT)"
    echo "  ⚠️  Review test failures above"
    # Exit 0 to not block Claude's response to user
    # The test output above will be visible in the conversation
    exit 0
fi
