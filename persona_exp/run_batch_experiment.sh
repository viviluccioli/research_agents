#!/bin/bash
# Batch Experiment Runner
# Run persona selection experiments on multiple papers

set -e  # Exit on error

# Configuration
NUM_RUNS=10
RESULTS_DIR="results"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Persona Selection Batch Experiment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if papers directory is provided
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Usage: $0 <papers_directory> [num_runs]${NC}"
    echo ""
    echo "Example:"
    echo "  $0 ../papers 10"
    echo "  $0 /path/to/papers 15"
    echo ""
    echo "This will run the experiment on all PDF/TXT/TEX files in the directory."
    exit 1
fi

PAPERS_DIR="$1"

# Override NUM_RUNS if provided
if [ $# -ge 2 ]; then
    NUM_RUNS="$2"
fi

# Check if directory exists
if [ ! -d "$PAPERS_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $PAPERS_DIR${NC}"
    exit 1
fi

# Count papers
PAPER_COUNT=$(find "$PAPERS_DIR" -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.tex" \) | wc -l)

if [ "$PAPER_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No papers found in $PAPERS_DIR${NC}"
    echo "Looking for: *.pdf, *.txt, *.tex"
    exit 1
fi

echo "Papers directory: $PAPERS_DIR"
echo "Number of papers: $PAPER_COUNT"
echo "Runs per paper: $NUM_RUNS"
echo "Results directory: $RESULTS_DIR"
echo ""

# Activate virtual environment if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source ../venv/bin/activate
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Counter for progress
CURRENT=0

# Find and process all papers
find "$PAPERS_DIR" -type f \( -name "*.pdf" -o -name "*.txt" -o -name "*.tex" \) | while read -r paper; do
    CURRENT=$((CURRENT + 1))

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Paper $CURRENT of $PAPER_COUNT${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "File: $(basename "$paper")"
    echo ""

    # Run experiment
    if python run_persona_selection_experiment.py "$paper" --runs "$NUM_RUNS"; then
        echo -e "${GREEN}✓ Completed: $(basename "$paper")${NC}"
    else
        echo -e "${RED}✗ Failed: $(basename "$paper")${NC}"
    fi

    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Batch experiment complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Results saved in: $RESULTS_DIR/"
echo ""
echo "To view all results:"
echo "  ls -lh $RESULTS_DIR/"
echo ""
echo "To analyze consistency across papers:"
echo "  grep 'unique_combinations' $RESULTS_DIR/*_metadata.json"
