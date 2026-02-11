#!/usr/bin/env bash
set -e
echo "============================================"
echo "  Signal Export Browser - First-Time Setup"
echo "============================================"
echo

# --- Check Python is installed ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo
    echo "  macOS:  brew install python"
    echo "  Ubuntu: sudo apt install python3 python3-venv python3-tk"
    echo "  Fedora: sudo dnf install python3 python3-tkinter"
    echo
    exit 1
fi

PYVER=$(python3 --version 2>&1)
echo "Found $PYVER"

# --- Create virtual environment if needed ---
if [ ! -d ".venv" ]; then
    echo
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# --- Activate venv ---
source .venv/bin/activate

# --- Install / upgrade dependencies ---
echo
echo "Installing dependencies..."
python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install -r requirements.txt
echo "Dependencies installed."

# --- Build the database if main.jsonl exists and signal.db does not ---
if [ -f "main.jsonl" ]; then
    if [ ! -f "signal.db" ]; then
        echo
        echo "Building database from main.jsonl..."
        python build_signal_db.py --input main.jsonl --out signal.db
        echo "Database built successfully."
    else
        echo "Database already exists. To rebuild, delete signal.db and run setup again, or use the Build tab in the GUI."
    fi
else
    echo
    echo "[NOTE] No main.jsonl found in this folder."
    echo "  Copy your signal-export output (main.jsonl and files/ folder) here,"
    echo "  then run setup.sh again, or build from the GUI's Build tab."
fi

# --- Check for ffmpeg (optional) ---
echo
if ! command -v ffmpeg &>/dev/null; then
    echo "[NOTE] ffmpeg not found. Video/audio thumbnails will show placeholders."
    echo "  Optional: brew install ffmpeg (macOS) or sudo apt install ffmpeg (Linux)"
else
    echo "ffmpeg found - video/audio thumbnails enabled."
fi

echo
echo "============================================"
echo "  Setup complete! Launching the app..."
echo "============================================"
echo
python signal_gui.py
