#!/bin/bash
# VPBank StreamGuard - Fraud Detection Demo
# Linux/Mac Shell Script to run the Streamlit demo

echo "========================================"
echo "VPBank StreamGuard - Fraud Detection"
echo "Interactive Demo Dashboard"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Checking Python installation..."
python3 --version

echo ""
echo "[2/3] Installing required packages..."
echo "This may take a few minutes on first run..."
pip3 install streamlit numpy pandas scikit-learn boto3 plotly --quiet

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install required packages"
    exit 1
fi

echo ""
echo "[3/3] Starting Streamlit demo..."
echo ""
echo "The demo will open in your default web browser."
echo "If it doesn't open automatically, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the demo."
echo "========================================"
echo ""

# Run Streamlit
streamlit run streamlit_app.py
