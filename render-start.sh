#!/bin/bash

# Render deployment script - dependencies are already installed by build command
echo "Starting Parkinson's Disease Prediction System on Render..."

# Set environment variables for full mode
export USE_LIGHT_MODE=0

# Start the application immediately
python wsgi.py