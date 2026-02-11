# Signal Export Browser

A desktop GUI for browsing, searching, and exporting your Signal message history from a [signal-export](https://github.com/carderne/signal-export) JSONL backup.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## Features

- **Full-text search** across all conversations with search-as-you-type
- **Media gallery** with dynamic grid layout, thumbnails for images/video/audio, and media type filtering
- **Theming system** with stock presets (Signal Dark, Terminal, Solarized, etc.), live editing, import/export, and persistent settings
- **Attachment viewer** — click to open any attachment; right-click to copy path/filename
- **HTML export** with inline thumbnails, themed styling, and full metadata
- **DB builder** — converts `main.jsonl` into a searchable SQLite database with FTS5, speaker names, and attachment resolution
- **Filters** — filter by "has attachments", "has links", date range, and more
- **Thumbnail caching** — persistent `.thumbcache/` directory with ffmpeg frame extraction for video and cover art for audio
- **DB status panel** showing conversation and message statistics

## Requirements

- **Python 3.10+** (with tkinter — included in standard Windows/macOS installers)
- **Pillow** (installed automatically by the setup script)
- **ffmpeg** *(optional)* — enables video frame and audio cover art thumbnails. Without it, placeholder icons are shown instead.

## Quick Start (Windows)

1. **Export your Signal data** using [signal-export](https://github.com/carderne/signal-export) to get a `main.jsonl` file and `files/` directory.

2. **Clone this repo** into your export folder (or copy the app files there):
   ```
   git clone https://github.com/Cosmo697/signal-export-browser.git
   ```

3. **Run the setup script** — double-click `setup.bat` or run it in a terminal:
   ```
   setup.bat
   ```
   This creates a virtual environment, installs dependencies, builds the database, and launches the GUI.

4. **On subsequent launches**, just double-click `run_gui.bat`.

## Quick Start (macOS / Linux)

```bash
# Make setup executable and run it
chmod +x setup.sh
./setup.sh
```

On subsequent launches:
```bash
source .venv/bin/activate
python signal_gui.py
```

## File Structure

```
your-signal-export/
├── main.jsonl              # Signal export data (your data, not committed)
├── files/                  # Exported media files (your data, not committed)
├── build_signal_db.py      # DB builder script
├── signal_gui.py           # Main GUI application
├── requirements.txt        # Python dependencies
├── setup.bat               # Windows one-click setup
├── setup.sh                # macOS/Linux setup
├── run_gui.bat             # Windows launcher
├── .gitignore
└── README.md
```

**Generated at runtime (git-ignored):**
- `signal.db` — SQLite database built from your export
- `.thumbcache/` — cached thumbnail images
- `app_state.json` — window state, last DB path, etc.
- `theme.json` — your active theme settings

## Usage

### Building the Database

The setup script builds the DB automatically. To rebuild manually:

```bash
python build_signal_db.py --input main.jsonl --out signal.db
```

Or use the **Build** tab inside the GUI.

### Searching Messages

Type in the search box and results appear automatically (debounced). Use the filter checkboxes for "Has attachments" and "Has links" to narrow results.

### Media Gallery

1. Go to the **Gallery** tab
2. Enter a Chat ID (shown in search results) and click **Load**
3. Use the **Type** dropdown to filter by image, video, audio, doc, or file
4. Columns adjust dynamically to your window width

### Themes

Open **Settings → Theme** to choose from stock presets or create your own. Themes are saved to `theme.json` and restored on next launch.

### Exporting

Select a conversation and use the **Export** tab to generate a styled HTML file with inline thumbnails and full message metadata.

## Optional: ffmpeg

For video/audio thumbnails, install ffmpeg:

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or `winget install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg` (or your distro's package manager)

The app works fine without ffmpeg — you'll just see placeholder icons instead of preview frames.

## License

MIT
