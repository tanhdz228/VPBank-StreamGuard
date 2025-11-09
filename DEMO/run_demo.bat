@echo off
REM VPBank StreamGuard - Fraud Detection Demo
REM Windows Batch Script to run the Streamlit demo

echo ========================================
echo VPBank StreamGuard - Fraud Detection
echo Interactive Demo Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version

echo.
echo [2/3] Installing required packages...
echo This may take a few minutes on first run...
pip install streamlit numpy pandas scikit-learn boto3 plotly --quiet

if errorlevel 1 (
    echo ERROR: Failed to install required packages
    pause
    exit /b 1
)

echo.
echo [3/3] Starting Streamlit demo...
echo.
echo The demo will open in your default web browser.
echo If it doesn't open automatically, navigate to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the demo.
echo ========================================
echo.

REM Run Streamlit
streamlit run streamlit_app.py

pause
