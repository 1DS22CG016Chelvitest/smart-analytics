@echo off
title Smart Customer Analytics Assistant
color 0A

echo.
echo  ================================================
echo   Smart Customer Analytics Assistant
echo   Setting up environment...
echo  ================================================
echo.

python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo  [ERROR] Python not found.
    echo  Download Python 3.9+ from https://www.python.org/downloads/
    echo  Tick "Add Python to PATH" during install.
    pause & exit /b 1
)
echo  [OK] Python found.

IF NOT EXIST "venv\" (
    echo  [1/3] Creating virtual environment...
    python -m venv venv --without-pip
    echo  [1/3] Done.
) ELSE (
    echo  [1/3] Virtual environment already exists.
)

echo  [2/3] Activating...
call venv\Scripts\activate.bat

echo  [3/3] Installing dependencies (first run takes 3-5 mins)...
python -m ensurepip --upgrade >nul 2>&1
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo  [3/3] Done.

echo.
echo  ================================================
echo   Launching at: http://localhost:8501
echo   Open this in Chrome or Edge.
echo   Press Ctrl+C here to stop the app.
echo  ================================================
echo.

streamlit run app.py
pause
