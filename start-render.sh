#!/bin/bash

# Render-optimized start script
# Dependencies are already installed by Build Command, so start immediately

echo "============================================================"
echo "Parkinson's Disease Prediction System"
echo "Starting on Render..."
echo "============================================================"

# Set environment for full mode
export USE_LIGHT_MODE=0

echo "✓ Full mode enabled (filename-based predictions)"
echo "✓ Starting server immediately..."
echo ""

# Start the WSGI application immediately
python wsgi.py