#!/bin/bash

# Parkinson's Disease Prediction System - Start Script (Full mode)
# Uses all ML libraries (NumPy, sklearn, PyTorch, OpenCV, Librosa, etc.)

echo "============================================================"
echo "Parkinson's Disease Prediction System"
echo "Automated Setup & Start (Full mode)"
echo "============================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_info() { echo -e "${YELLOW}ℹ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }

# Step 1: Check Python
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
print_success "Found $(python3 --version)"
echo ""

# Step 2: Create/Activate Virtual Environment
print_info "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

source venv/bin/activate
print_success "Virtual environment activated"
echo ""

# Step 3: Install full dependencies
print_info "Installing dependencies (full mode - this may take a while)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ $? -eq 0 ]; then
    print_success "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi
echo ""

# Step 4: Stop any existing server
print_info "Checking for running server..."
pkill -f "waitress-serve.*wsgi:app" 2>/dev/null
pkill -f "wsgi:app" 2>/dev/null
pkill -f "wsgi.py" 2>/dev/null
sleep 2
echo ""

# Step 5: Start the server (full mode)
export USE_LIGHT_MODE=0
print_success "Starting server (full mode - all ML libraries enabled)..."
echo ""
echo "============================================================"
echo "  URL: http://localhost:8000"
echo "  Mode: Full (with ML libraries and file processing)"
echo "============================================================"
echo ""
print_info "Press Ctrl+C to stop the server"
echo ""

python wsgi.py

echo ""
print_info "Server stopped"