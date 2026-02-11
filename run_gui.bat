@echo off
setlocal
title Signal Export Browser
if not exist .venv (
    echo Virtual environment not found. Running setup first...
    call setup.bat
    exit /b
)
call .venv\Scripts\activate.bat
python signal_gui.py
pause
