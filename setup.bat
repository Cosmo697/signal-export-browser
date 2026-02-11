@echo off
setlocal
title Signal Export Browser - Setup
echo ============================================
echo   Signal Export Browser - First-Time Setup
echo ============================================
echo.

REM --- Check Python is installed ---
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo.
    echo   Download Python 3.10+ from https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM --- Check Python version ---
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%

REM --- Create virtual environment if needed ---
if not exist .venv (
    echo.
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM --- Activate venv ---
call .venv\Scripts\activate.bat

REM --- Install / upgrade dependencies ---
echo.
echo Installing dependencies...
python -m pip install --upgrade pip >nul 2>nul
python -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed.

REM --- Build the database if main.jsonl exists and signal.db does not ---
if exist main.jsonl (
    if not exist signal.db (
        echo.
        echo Building database from main.jsonl...
        python build_signal_db.py --input main.jsonl --out signal.db
        if %ERRORLEVEL% neq 0 (
            echo [WARNING] Database build had errors. You can rebuild from the GUI.
        ) else (
            echo Database built successfully.
        )
    ) else (
        echo Database already exists. To rebuild, delete signal.db and run setup again, or use the Build tab in the GUI.
    )
) else (
    echo.
    echo [NOTE] No main.jsonl found in this folder.
    echo   Copy your signal-export output (main.jsonl and files/ folder) here,
    echo   then run setup.bat again, or build from the GUI's Build tab.
)

REM --- Check for ffmpeg (optional) ---
echo.
where ffmpeg >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [NOTE] ffmpeg not found. Video/audio thumbnails will show placeholders.
    echo   Optional: install with  winget install ffmpeg  or from https://ffmpeg.org
) else (
    echo ffmpeg found - video/audio thumbnails enabled.
)

echo.
echo ============================================
echo   Setup complete! Launching the app...
echo ============================================
echo.
python signal_gui.py
pause
