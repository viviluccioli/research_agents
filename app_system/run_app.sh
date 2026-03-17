#!/bin/bash
# Launch script for Evaluation Agent app
# Disables file watcher to avoid inotify limit issues

echo "🚀 Starting Evaluation Agent..."
echo "📍 Running from: $(pwd)"
echo ""

python3 -m streamlit run app.py \
    --server.fileWatcherType none \
    --server.port 8501 \
    --server.address localhost

# Alternative ports if 8501 is taken:
# --server.port 8502
