#!/usr/bin/env python3
from __future__ import annotations
"""
Signal Export Browser v3 (GUI)

All-in-one:
- Conversation list pane with hit counts
- Hits list per conversation
- Thread preview with context window (±N messages)
- Full thread window with Prev/Next hit navigation
- Search term highlighting in preview and full thread
- Media gallery per chat (image grid)
- Export search results (CSV/HTML)
- Export threads (HTML/Markdown)
- Inline thumbnails and [open] links for attachments (Pillow)

Requirements: Pillow
"""

import base64 as _b64
import csv
import html
import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import traceback
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

from PIL import Image, ImageTk, ImageDraw, ImageFont

# Split-out helpers/constants (keeps this file focused on the app)
from gui_utils import (
    _best_bw_for_bg,
    _bind_mousewheel,
    _contrast_ratio,
    _fmt_ts_local_iso,
    _fmt_ts_short,
    _fts5_available,
    _fts_terms,
    _hex_to_rgb,
    _human_size,
    _insert_body_with_links,
    _is_hex_color,
    _lift_or_drop,
    _normalize_date,
    _rgb_to_hex,
    _safe_open_path,
    _shift_toward,
    _table_exists,
)
from stop_words import STOP_WORDS as _STOP_WORDS

# Light/Dark mode is implemented directly in this file.

import build_signal_db

# Bump this value to invalidate the thumbnail cache key.
_THUMB_CACHE_VERSION = 3

@dataclass
class SearchParams:
    q: str
    recipient: str | None
    after: str | None
    before: str | None
    limit: int
    attach_kind: str  # any|image|video|audio|doc|file
    inout: str        # any|IN|OUT
    has_attachments: bool = False
    has_links: bool = False


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Signal Export Browser v3")
        self.root.geometry("1380x840")

        self._state_path = Path.cwd() / "app_state.json"
        self._cache_state_path = Path.cwd() / "cache_state.json"

        # Light/Dark mode toggle (no custom themes)
        self.dark_mode = tk.BooleanVar(value=False)
        self.theme: Dict[str, Any] = self._palette_for_mode(False)

        self._open_thread_text_widgets: List[tk.Text] = []
        self._style = ttk.Style(self.root)
        self._menubar: Optional[tk.Menu] = None
        self._settings_menu: Optional[tk.Menu] = None

        self.input_jsonl = tk.StringVar(value=str(Path.cwd() / "main.jsonl"))
        self.output_db = tk.StringVar(value=str(Path.cwd() / "signal.db"))
        self.store_raw_e164 = tk.BooleanVar(value=False)

        # Prefer rebuilt DBs (Windows may lock signal.db while GUI is open)
        preferred_db = Path.cwd() / "signal_fixed.db"
        default_db = preferred_db if preferred_db.exists() else (Path.cwd() / "signal.db")
        self.db_path = tk.StringVar(value=str(default_db))
        self.q = tk.StringVar(value="")
        self.recipient = tk.StringVar(value="")
        self.after = tk.StringVar(value="")
        self.before = tk.StringVar(value="")
        self.limit = tk.IntVar(value=500)
        self.context_n = tk.IntVar(value=40)

        self.attach_kind = tk.StringVar(value="any")
        self.inout = tk.StringVar(value="any")

        # Extra filters
        self.filter_has_attachments = tk.BooleanVar(value=False)
        self.filter_has_links = tk.BooleanVar(value=False)

        # Search-as-you-type debounce state
        self._search_debounce_id: Optional[str] = None

        # In-memory PhotoImage LRU cache (capped at 600 entries)
        self._photo_cache: OrderedDict[str, ImageTk.PhotoImage] = OrderedDict()
        self._photo_cache_max = 600

        self._build_lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._has_fts5: bool = False
        self._has_link_col: bool = False

        self._search_rows: List[Dict[str, Any]] = []
        self._chat_hits: List[Dict[str, Any]] = []
        self._selected_chat_id: Optional[int] = None
        self._selected_hit_rowids: List[int] = []
        self._hit_idx: int = -1
        self._current_terms: List[str] = []

        # Restore last state (UI fields + light/dark mode) if available
        self._load_app_state()

        # Apply persisted mode selection
        self.theme = self._palette_for_mode(bool(self.dark_mode.get()))

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._make_ui()
        self._apply_mode_now()

    # ---------- Light/Dark mode + high-contrast styling ----------

    @staticmethod
    def _palette_light() -> Dict[str, Any]:
        # High-contrast light palette
        return {
            "app_bg": "#f3f4f6",
            "panel_bg": "#ffffff",
            "widget_bg": "#ffffff",
            "widget_fg": "#111827",
            "text_bg": "#ffffff",
            "text_fg": "#111827",
            "meta_fg": "#374151",
            "you_fg": "#0b5cad",
            "them_fg": "#166534",
            "link_fg": "#0b5cad",
            "accent": "#0b5cad",
            "select_bg": "#0b5cad",
            "select_fg": "#ffffff",
            # Strong highlight fills with forced readable text
            "hit_bg": "#ffea00",
            "term_bg": "#00d5ff",
        }

    @staticmethod
    def _palette_dark() -> Dict[str, Any]:
        # High-contrast dark palette
        return {
            "app_bg": "#0b0d10",
            "panel_bg": "#0f1318",
            "widget_bg": "#151b22",
            "widget_fg": "#e5e7eb",
            "text_bg": "#0f1318",
            "text_fg": "#f3f4f6",
            "meta_fg": "#cbd5e1",
            "you_fg": "#7dd3fc",
            "them_fg": "#86efac",
            "link_fg": "#60a5fa",
            "accent": "#60a5fa",
            "select_bg": "#2563eb",
            "select_fg": "#ffffff",
            # Strong highlight fills with forced readable text
            "hit_bg": "#ffcc00",
            "term_bg": "#00e5ff",
        }

    def _palette_for_mode(self, dark: bool) -> Dict[str, Any]:
        return dict(self._palette_dark() if dark else self._palette_light())

    def _apply_mode_now(self) -> None:
        # ttk + menus + all text widgets + open thread windows
        self._apply_palette_to_ttk()
        self._apply_palette_to_menu()

        for wname in ("preview_txt", "build_log"):
            try:
                w = getattr(self, wname)
                if isinstance(w, tk.Text):
                    self._apply_palette_to_text(w)
            except Exception:
                pass

        # Stats dashboard canvas (tk widget; needs manual bg)
        try:
            c = getattr(self, "_stats_canvas", None)
            if isinstance(c, tk.Canvas):
                c.configure(
                    bg=self.theme.get("panel_bg", "#ffffff"),
                    highlightthickness=0,
                    bd=0,
                )
        except Exception:
            pass

        # Open thread windows
        try:
            alive: List[tk.Text] = []
            for t in self._open_thread_text_widgets:
                try:
                    if t.winfo_exists():
                        self._apply_palette_to_text(t)
                        alive.append(t)
                except Exception:
                    continue
            self._open_thread_text_widgets = alive
        except Exception:
            pass

        # Root background (helps on some platforms)
        try:
            self.root.configure(bg=self.theme.get("app_bg", "#0b0d10"))
        except Exception:
            pass

    def _apply_palette_to_ttk(self) -> None:
        # Force a theme where color maps work reliably.
        try:
            if "clam" in self._style.theme_names():
                self._style.theme_use("clam")
        except Exception:
            pass

        app_bg = self.theme.get("app_bg", "#f3f4f6")
        panel_bg = self.theme.get("panel_bg", "#ffffff")
        widget_bg = self.theme.get("widget_bg", panel_bg)
        fg = self.theme.get("widget_fg", "#111827")
        sel_bg = self.theme.get("select_bg", "#0b5cad")
        sel_fg = self.theme.get("select_fg", "#ffffff")

        s = self._style
        try:
            s.configure(".", background=panel_bg, foreground=fg)
        except Exception:
            pass
        s.configure("TFrame", background=panel_bg)
        s.configure("TLabelframe", background=panel_bg, foreground=fg)
        s.configure("TLabelframe.Label", background=panel_bg, foreground=fg)
        s.configure("TLabel", background=panel_bg, foreground=fg)

        s.configure("TButton", background=widget_bg, foreground=fg, padding=(10, 5))
        s.map(
            "TButton",
            background=[("pressed", sel_bg), ("active", sel_bg)],
            foreground=[("pressed", sel_fg), ("active", sel_fg)],
        )

        s.configure("TEntry", fieldbackground=widget_bg, foreground=fg)
        s.configure("TCombobox", fieldbackground=widget_bg, foreground=fg)
        try:
            s.map(
                "TCombobox",
                fieldbackground=[("readonly", widget_bg)],
                foreground=[("readonly", fg)],
                selectbackground=[("readonly", sel_bg)],
                selectforeground=[("readonly", sel_fg)],
            )
        except Exception:
            pass

        s.configure("TNotebook", background=app_bg)
        s.configure("TNotebook.Tab", background=widget_bg, foreground=fg, padding=(12, 6))
        s.map(
            "TNotebook.Tab",
            background=[("selected", panel_bg), ("active", sel_bg)],
            foreground=[("selected", fg), ("active", sel_fg)],
        )

        s.configure(
            "Treeview",
            background=panel_bg,
            fieldbackground=panel_bg,
            foreground=fg,
            rowheight=22,
        )
        s.configure("Treeview.Heading", background=widget_bg, foreground=fg)
        s.map("Treeview", background=[("selected", sel_bg)], foreground=[("selected", sel_fg)])

        # Dashboard bars
        try:
            s.configure(
                "Horizontal.TProgressbar",
                background=self.theme.get("accent", sel_bg),
                troughcolor=widget_bg,
            )
        except Exception:
            pass

    def _apply_palette_to_menu(self) -> None:
        bg = self.theme.get("panel_bg", "#ffffff")
        fg = self.theme.get("widget_fg", "#111827")
        abg = self.theme.get("select_bg", "#0b5cad")
        afg = self.theme.get("select_fg", "#ffffff")

        def _apply(m: Optional[tk.Menu]) -> None:
            if m is None:
                return
            try:
                m.configure(background=bg, foreground=fg, activebackground=abg, activeforeground=afg)
            except Exception:
                return
            try:
                end = m.index("end")
                if end is None:
                    return
                for i in range(end + 1):
                    try:
                        sub = m.entrycget(i, "menu")
                        if sub:
                            _apply(m.nametowidget(sub))
                    except Exception:
                        continue
            except Exception:
                pass

        _apply(getattr(self, "_menubar", None))
        _apply(getattr(self, "_settings_menu", None))

    def _apply_palette_to_text(self, txt: tk.Text) -> None:
        bg = self.theme.get("text_bg", "#ffffff")
        fg = self.theme.get("text_fg", "#111827")
        sel_bg = self.theme.get("select_bg", "#0b5cad")
        sel_fg = self.theme.get("select_fg", "#ffffff")
        meta = self.theme.get("meta_fg", "#374151")
        you = self.theme.get("you_fg", "#0b5cad")
        them = self.theme.get("them_fg", "#166534")
        link = self.theme.get("link_fg", "#0b5cad")
        hit_bg = self.theme.get("hit_bg", "#ffea00")
        term_bg = self.theme.get("term_bg", "#00d5ff")

        # Force readable text in highlighted regions.
        hit_fg = "#000000"
        term_fg = "#000000"

        try:
            txt.configure(
                background=bg,
                foreground=fg,
                insertbackground=fg,
                selectbackground=sel_bg,
                selectforeground=sel_fg,
                inactiveselectbackground=sel_bg,
            )
        except Exception:
            pass

        txt.tag_configure("meta", foreground=meta)
        txt.tag_configure("you", foreground=you)
        txt.tag_configure("them", foreground=them)
        txt.tag_configure("hit", background=hit_bg, foreground=hit_fg)
        txt.tag_configure("term", background=term_bg, foreground=term_fg)
        txt.tag_configure("link", foreground=link, underline=True)
        txt.tag_configure("pv_link", foreground=link, underline=True)

        # Keep highlighted text readable even when it's also a link.
        try:
            txt.tag_lower("link")
            txt.tag_lower("pv_link")
            txt.tag_raise("hit")
            txt.tag_raise("term")
        except Exception:
            pass

    def _on_toggle_dark_mode(self) -> None:
        self.theme = self._palette_for_mode(bool(self.dark_mode.get()))
        self._apply_mode_now()

    def _thumb_cache_dir(self) -> Path:
        d = Path.cwd() / ".thumbcache"
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return d

    # --- In-memory PhotoImage cache (keyed by (abs_path, w, h, kind)) ---

    def _photo_cache_key(self, ap: str, kind: str, max_size: Tuple[int, int]) -> str:
        try:
            st = os.stat(ap)
            return f"{ap}|{st.st_mtime_ns}|{st.st_size}|{kind}|{max_size[0]}x{max_size[1]}|v={_THUMB_CACHE_VERSION}"
        except Exception:
            return f"{ap}|{kind}|{max_size[0]}x{max_size[1]}|v={_THUMB_CACHE_VERSION}"

    def _clear_photo_cache(self) -> None:
        """Flush the in-memory PhotoImage cache."""
        self._photo_cache.clear()

    def _photo_cache_put(self, key: str, photo: ImageTk.PhotoImage) -> None:
        """Insert into LRU cache with eviction when over capacity."""
        self._photo_cache[key] = photo
        self._photo_cache.move_to_end(key)
        while len(self._photo_cache) > self._photo_cache_max:
            self._photo_cache.popitem(last=False)

    def _batch_fetch_attachments(self, message_rowids: List[int]) -> Dict[int, List[sqlite3.Row]]:
        """Fetch attachments for many messages in one query, grouped by message_rowid."""
        if not message_rowids or self._conn is None:
            return {}
        result: Dict[int, List[sqlite3.Row]] = {}
        # SQLite variable limit is 999 by default; batch in chunks
        for i in range(0, len(message_rowids), 900):
            chunk = message_rowids[i:i + 900]
            ph = ",".join(["?"] * len(chunk))
            rows = self._conn.execute(
                f"SELECT message_rowid, file_name, abs_path, mime, size_bytes, width, height, duration_ms, kind "
                f"FROM attachments WHERE message_rowid IN ({ph}) ORDER BY message_rowid, id ASC",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                mid = int(r["message_rowid"])
                result.setdefault(mid, []).append(r)
        return result

    def _rebuild_thumb_cache(self, log_func=None, conn: Optional[sqlite3.Connection] = None) -> None:
        """Delete the .thumbcache/ directory and rebuild thumbnails for all known attachments.

        *conn* – optional explicit DB connection (e.g. thread-local).  Falls
        back to ``self._conn`` when omitted.
        """
        cache_dir = self._thumb_cache_dir()
        try:
            shutil.rmtree(str(cache_dir), ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self._clear_photo_cache()

        db = conn or self._conn
        if db is None:
            if log_func:
                log_func("No DB connected — disk cache cleared only.")
            return

        rows = db.execute(
            "SELECT abs_path, kind, file_name, duration_ms FROM attachments WHERE abs_path != '' AND kind IN ('image','video','audio')"
        ).fetchall()
        total = len(rows)
        if log_func:
            log_func(f"Rebuilding thumbnails for {total} attachments…")
        for i, r in enumerate(rows):
            ap = r["abs_path"] or ""
            if not ap or not os.path.exists(ap):
                continue
            # Just generate the disk-cache entry; ignore the PhotoImage
            self._attachment_thumbnail_photo(ap, r["kind"] or "file", r["file_name"] or "", (240, 180),
                                             duration_ms=r["duration_ms"])
            if log_func and (i + 1) % 200 == 0:
                log_func(f"  {i+1}/{total}…")
        if log_func:
            log_func(f"Thumbnail cache rebuilt: {total} items processed.")

    def _placeholder_thumbnail(self, kind: str, file_name: str, size: Tuple[int, int]) -> Image.Image:
        w, h = size
        bg = self.theme.get("widget_bg", "#222222")
        fg = self.theme.get("widget_fg", "#ffffff")
        accent = self.theme.get("accent", fg)
        im = Image.new("RGB", (max(64, int(w)), max(48, int(h))), bg)
        draw = ImageDraw.Draw(im)
        # Accent stripe
        try:
            draw.rectangle([0, 0, im.size[0], 6], fill=accent)
        except Exception:
            pass

        k = (kind or "file").upper()
        ext = ""
        try:
            ext = Path(file_name or "").suffix.upper().lstrip(".")
        except Exception:
            ext = ""
        label = k if not ext else f"{k} ({ext})"

        # Centered label
        try:
            tw, th = draw.textbbox((0, 0), label)[2:4]
            x = (im.size[0] - tw) // 2
            y = (im.size[1] - th) // 2
        except Exception:
            x, y = 10, 10
        try:
            draw.text((x, y), label, fill=fg)
        except Exception:
            pass
        return im

    def _try_ffmpeg_extract_frame(self, ap: str, out_path: Path) -> bool:
        ff = shutil.which("ffmpeg")
        if not ff:
            return False

        # Best-effort: pick a frame around 1s in.
        attempts = [
            [ff, "-hide_banner", "-loglevel", "error", "-ss", "00:00:01", "-i", ap, "-frames:v", "1", "-y", str(out_path)],
            [ff, "-hide_banner", "-loglevel", "error", "-i", ap, "-ss", "00:00:01", "-frames:v", "1", "-y", str(out_path)],
            [ff, "-hide_banner", "-loglevel", "error", "-ss", "00:00:00", "-i", ap, "-frames:v", "1", "-y", str(out_path)],
        ]
        for cmd in attempts:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                    return True
            except Exception:
                continue
        return False

    def _try_ffmpeg_extract_cover(self, ap: str, out_path: Path) -> bool:
        ff = shutil.which("ffmpeg")
        if not ff:
            return False
        # Extract embedded cover art if present.
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-i", ap, "-map", "0:v:0", "-frames:v", "1", "-y", str(out_path)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            return False

    def _try_ffmpeg_render_waveform(self, ap: str, out_path: Path, size: Tuple[int, int]) -> bool:
        """Render a simple waveform thumbnail for audio files using ffmpeg.

        This is a fallback when an audio file has no embedded cover art.
        Requires ffmpeg to be installed.
        """
        ff = shutil.which("ffmpeg")
        if not ff:
            return False

        w = max(64, int(size[0]))
        h = max(48, int(size[1]))

        bg_hex = self.theme.get("widget_bg", "#222222")
        accent_hex = self.theme.get("accent", "#00a8fc")
        # Use 0xRRGGBB to avoid any parsing issues with '#'.
        bg_ff = "0x" + (str(bg_hex).strip().lstrip("#") if bg_hex else "222222")
        accent_ff = "0x" + (str(accent_hex).strip().lstrip("#") if accent_hex else "00a8fc")

        # Prefer ffmpeg's showwavespic, which summarizes the FULL audio stream into
        # the target width.
        filter_complex = (
            f"color=c={bg_ff}:s={w}x{h}[bg];"
            f"[0:a]showwavespic=s={w}x{h}:colors={accent_ff}[fg];"
            f"[bg][fg]overlay=format=auto"
        )
        cmd = [
            ff,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            ap,
            "-filter_complex",
            filter_complex,
            "-frames:v",
            "1",
            "-y",
            str(out_path),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                return True

            import array as _array

            # Fallback: decode the FULL audio stream at a very low sample rate so
            # memory stays bounded, then render a waveform in Python.
            cmd2 = [
                ff,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                ap,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "800",
                "-f",
                "s16le",
                "pipe:1",
            ]
            r2 = subprocess.run(cmd2, capture_output=True)
            if r2.returncode != 0 or not r2.stdout:
                return False

            raw = r2.stdout
            raw = raw[: (len(raw) // 2) * 2]  # int16 alignment
            if not raw:
                return False
            samples = _array.array("h")
            samples.frombytes(raw)
            if not samples:
                return False

            max_abs = max((abs(int(v)) for v in samples), default=0)
            if max_abs <= 0:
                return False

            try:
                bg = _hex_to_rgb(bg_hex)
            except Exception:
                bg = (34, 34, 34)
            try:
                accent = _hex_to_rgb(accent_hex)
            except Exception:
                accent = (0, 168, 252)

            im = Image.new("RGB", (w, h), bg)
            draw = ImageDraw.Draw(im)
            mid = h // 2
            pad = max(6, h // 10)
            scale = max(1, mid - pad)

            step = max(1, len(samples) // w)
            for x in range(w):
                i0 = x * step
                i1 = min(len(samples), i0 + step)
                if i0 >= i1:
                    break
                seg = samples[i0:i1]
                amp = 0
                for v in seg:
                    av = abs(int(v))
                    if av > amp:
                        amp = av
                y = int((amp / max_abs) * scale)
                if y <= 0:
                    continue
                draw.line((x, mid - y, x, mid + y), fill=accent)

            try:
                im.save(str(out_path), "JPEG", quality=85)
            except Exception:
                return False
            return out_path.exists() and out_path.stat().st_size > 0
        except Exception:
            return False

    def _attachment_thumbnail_photo(self, ap: str, kind: str, file_name: str, max_size: Tuple[int, int],
                                     duration_ms: Optional[int] = None) -> Optional[ImageTk.PhotoImage]:
        """Return a PhotoImage thumbnail for media attachments.

        - images: use Pillow directly (cached on disk)
        - video: use ffmpeg for a frame if available; else placeholder
        - audio: try ffmpeg embedded cover; else placeholder
        - duration_ms: if provided, overlays a duration badge on video/audio thumbs
        """
        if not ap or not os.path.exists(ap):
            return None

        kind = (kind or "file").lower()
        is_media = kind in ("image", "video", "audio")
        if not is_media:
            return None

        # In-memory cache check
        mem_key = self._photo_cache_key(ap, kind, max_size)
        if duration_ms:
            mem_key += f"|dur={duration_ms}"
        cached = self._photo_cache.get(mem_key)
        if cached is not None:
            self._photo_cache.move_to_end(mem_key)
            return cached

        cache_dir = self._thumb_cache_dir()
        try:
            st = os.stat(ap)
            key_src = f"{ap}|{st.st_mtime_ns}|{st.st_size}|{kind}|{max_size[0]}x{max_size[1]}|v={_THUMB_CACHE_VERSION}".encode("utf-8", errors="ignore")
        except Exception:
            key_src = f"{ap}|{kind}|{max_size[0]}x{max_size[1]}|v={_THUMB_CACHE_VERSION}".encode("utf-8", errors="ignore")
        key = hashlib.sha1(key_src).hexdigest()

        # Persistent disk cache for ALL types (image/video/audio)
        cache_img = cache_dir / f"{kind}_{key}.jpg"

        try:
            if cache_img.exists() and cache_img.stat().st_size > 0:
                im = Image.open(str(cache_img))
            elif kind == "image":
                im = Image.open(ap)
                try:
                    im = im.convert("RGB")
                except Exception:
                    pass
                im.thumbnail(max_size)
                # Save to disk cache
                try:
                    im.save(str(cache_img), "JPEG", quality=85)
                except Exception:
                    pass
            else:
                ok = False
                if kind == "video":
                    ok = self._try_ffmpeg_extract_frame(ap, cache_img)
                elif kind == "audio":
                    ok = self._try_ffmpeg_extract_cover(ap, cache_img)
                    if not ok:
                        ok = self._try_ffmpeg_render_waveform(ap, cache_img, max_size)
                if not ok:
                    im = self._placeholder_thumbnail(kind, file_name, max_size)
                    im.thumbnail(max_size)
                    # Duration overlay for placeholders too
                    if duration_ms and kind in ("video", "audio"):
                        im = self._draw_duration_overlay(im, duration_ms)
                    photo = ImageTk.PhotoImage(im)
                    self._photo_cache_put(mem_key, photo)
                    return photo

                im = Image.open(str(cache_img))

            # Normalize and size
            try:
                im = im.convert("RGB")
            except Exception:
                pass
            im.thumbnail(max_size)

            # Duration overlay for video/audio
            if duration_ms and kind in ("video", "audio"):
                im = self._draw_duration_overlay(im, duration_ms)

            photo = ImageTk.PhotoImage(im)
            self._photo_cache_put(mem_key, photo)
            return photo
        except Exception:
            try:
                im = self._placeholder_thumbnail(kind, file_name, max_size)
                im.thumbnail(max_size)
                if duration_ms and kind in ("video", "audio"):
                    im = self._draw_duration_overlay(im, duration_ms)
                photo = ImageTk.PhotoImage(im)
                self._photo_cache_put(mem_key, photo)
                return photo
            except Exception:
                return None

    @staticmethod
    def _format_duration(ms: int) -> str:
        """Format milliseconds as M:SS or H:MM:SS."""
        total_s = max(0, int(ms) // 1000)
        h = total_s // 3600
        m = (total_s % 3600) // 60
        s = total_s % 60
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _draw_duration_overlay(self, im: Image.Image, duration_ms: int) -> Image.Image:
        """Draw a semi-transparent duration badge on the bottom-right of an image."""
        if not duration_ms or duration_ms <= 0:
            return im
        im = im.copy()
        draw = ImageDraw.Draw(im)
        label = self._format_duration(duration_ms)
        try:
            bbox = draw.textbbox((0, 0), label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * 7, 12
        pad_x, pad_y = 6, 3
        x = im.size[0] - tw - pad_x * 2 - 4
        y = im.size[1] - th - pad_y * 2 - 4
        # Semi-transparent background
        try:
            overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
            draw_ov = ImageDraw.Draw(overlay)
            draw_ov.rounded_rectangle([x, y, x + tw + pad_x * 2, y + th + pad_y * 2], radius=4, fill=(0, 0, 0, 180))
            im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(im)
        except Exception:
            draw.rectangle([x, y, x + tw + pad_x * 2, y + th + pad_y * 2], fill="#000000")
        draw.text((x + pad_x, y + pad_y), label, fill="#ffffff")
        return im

    def _load_app_state(self) -> None:
        try:
            import json

            if not self._state_path.exists():
                return
            st = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(st, dict):
                return

            geom = st.get("geometry")
            if isinstance(geom, str) and geom:
                try:
                    self.root.geometry(geom)
                except Exception:
                    pass

            # Light/Dark mode (current)
            dm = st.get("dark_mode")
            if isinstance(dm, bool):
                self.dark_mode.set(dm)
            else:
                # Back-compat: infer mode from older saved theme dicts (when present)
                t = st.get("theme")
                if isinstance(t, dict):
                    try:
                        bg_hex = str(t.get("text_bg") or t.get("app_bg") or "").strip() or "#ffffff"
                        r, g, b = _hex_to_rgb(bg_hex)
                        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
                        self.dark_mode.set(lum < 0.5)
                    except Exception:
                        pass

            # UI vars
            for key, var in (
                ("input_jsonl", self.input_jsonl),
                ("output_db", self.output_db),
                ("db_path", self.db_path),
                ("q", self.q),
                ("recipient", self.recipient),
                ("after", self.after),
                ("before", self.before),
                ("attach_kind", self.attach_kind),
                ("inout", self.inout),
            ):
                val = st.get(key)
                if isinstance(val, str):
                    var.set(val)

            for key, var in (
                ("limit", self.limit),
                ("context_n", self.context_n),
            ):
                val = st.get(key)
                if isinstance(val, int):
                    var.set(val)

            val = st.get("store_raw_e164")
            if isinstance(val, bool):
                self.store_raw_e164.set(val)

            for key, var in (
                ("filter_has_attachments", self.filter_has_attachments),
                ("filter_has_links", self.filter_has_links),
            ):
                val = st.get(key)
                if isinstance(val, bool):
                    var.set(val)

            # Stash tab index/chat ids to restore after UI is built
            self._restore_tab_index = st.get("tab_index") if isinstance(st.get("tab_index"), int) else None
            self._restore_gallery_chat = st.get("gallery_chat") if isinstance(st.get("gallery_chat"), str) else ""
        except Exception:
            return

    def _save_app_state(self) -> None:
        try:
            import json

            st: Dict[str, Any] = {
                "geometry": self.root.winfo_geometry(),
                "tab_index": int(self.nb.index(self.nb.select())) if hasattr(self, "nb") else 0,
                "input_jsonl": self.input_jsonl.get(),
                "output_db": self.output_db.get(),
                "store_raw_e164": bool(self.store_raw_e164.get()),
                "db_path": self.db_path.get(),
                "q": self.q.get(),
                "recipient": self.recipient.get(),
                "after": self.after.get(),
                "before": self.before.get(),
                "limit": int(self.limit.get()),
                "context_n": int(self.context_n.get()),
                "attach_kind": self.attach_kind.get(),
                "inout": self.inout.get(),
                "filter_has_attachments": bool(self.filter_has_attachments.get()),
                "filter_has_links": bool(self.filter_has_links.get()),
                "gallery_chat": self._gallery_selected_chat_id.get() if hasattr(self, "_gallery_selected_chat_id") else "",
                "dark_mode": bool(self.dark_mode.get()),
            }
            self._state_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        except Exception:
            return

    def _on_close(self) -> None:
        self._save_app_state()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _attachment_context_menu(self, event, abs_path: str) -> None:
        """Show a right-click context menu for an attachment with copy options."""
        if not abs_path:
            return
        menu = tk.Menu(self.root, tearoff=0)
        fn = os.path.basename(abs_path)
        menu.add_command(label=f"Copy filename: {fn}", command=lambda: self._clipboard_copy(fn))
        menu.add_command(label="Copy full path", command=lambda: self._clipboard_copy(abs_path))
        menu.add_command(label="Copy folder path", command=lambda: self._clipboard_copy(os.path.dirname(abs_path)))
        menu.add_separator()
        menu.add_command(label="Open file", command=lambda: _safe_open_path(abs_path))
        menu.add_command(label="Open containing folder", command=lambda: _safe_open_path(os.path.dirname(abs_path)))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _clipboard_copy(self, text: str) -> None:
        """Copy text to the system clipboard."""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
        except Exception:
            pass

    def _speaker_names_for_message_ids(self, message_ids: List[int]) -> Dict[int, str]:
        if self._conn is None or not message_ids:
            return {}
        placeholders = ",".join(["?"] * len(message_ids))
        q = f"SELECT message_rowid, authorName, recipientName, outgoing FROM messages_enriched WHERE message_rowid IN ({placeholders})"
        author_rows = self._conn.execute(q, tuple(message_ids)).fetchall()
        out: Dict[int, str] = {}
        for r in author_rows:
            mid = int(r["message_rowid"])
            if int(r["outgoing"] or 0) == 1:
                out[mid] = "You"
            else:
                nm = (r["authorName"] or "").strip()
                if not nm:
                    nm = (r["recipientName"] or "").strip()
                out[mid] = nm or "Them"
        return out

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)

        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label="Open DB...", command=lambda: (self._pick_db_in(), self._on_connect()))
        filem.add_separator()
        filem.add_command(label="Exit", command=self.root.destroy)

        settings = tk.Menu(menubar, tearoff=0)
        settings.add_checkbutton(
            label="Dark mode",
            onvalue=True,
            offvalue=False,
            variable=self.dark_mode,
            command=self._on_toggle_dark_mode,
        )

        helpm = tk.Menu(menubar, tearoff=0)

        def about() -> None:
            messagebox.showinfo(
                "About",
                "Signal Export Browser\n\n"
                "- Browse/search Signal exports\n"
                "- Media gallery + exports\n"
                "- Stats + word frequency\n\n"
                "Tip: Use Settings → Dark mode to toggle.",
                parent=self.root,
            )

        helpm.add_command(label="About", command=about)

        menubar.add_cascade(label="File", menu=filem)
        menubar.add_cascade(label="Settings", menu=settings)
        menubar.add_cascade(label="Help", menu=helpm)
        self.root.config(menu=menubar)
        self._menubar = menubar
        self._settings_menu = settings
        self._apply_palette_to_menu()

    def _make_ui(self) -> None:
        self._build_menubar()
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True)

        build_tab = ttk.Frame(self.nb)
        search_tab = ttk.Frame(self.nb)
        gallery_tab = ttk.Frame(self.nb)
        export_tab = ttk.Frame(self.nb)
        stats_tab = ttk.Frame(self.nb)

        self.nb.add(build_tab, text="Build DB")
        self.nb.add(search_tab, text="Search & Threads")
        self.nb.add(gallery_tab, text="Media Gallery")
        self.nb.add(export_tab, text="Exports")
        self.nb.add(stats_tab, text="Stats")

        self._build_build_tab(build_tab)
        self._build_search_tab(search_tab)
        self._build_gallery_tab(gallery_tab)
        self._build_exports_tab(export_tab)
        self._build_stats_tab(stats_tab)

        # Lazy gallery refresh: only load thumbnails when switching to Gallery tab
        self._gallery_needs_refresh = False
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Restore last selected tab / gallery chat
        try:
            if hasattr(self, "_restore_tab_index") and self._restore_tab_index is not None:
                idx = int(self._restore_tab_index)
                if 0 <= idx < self.nb.index("end"):
                    self.nb.select(idx)
        except Exception:
            pass

        try:
            if hasattr(self, "_restore_gallery_chat") and isinstance(self._restore_gallery_chat, str) and self._restore_gallery_chat:
                self._gallery_selected_chat_id.set(self._restore_gallery_chat)
        except Exception:
            pass

    # ---------- Build tab ----------

    def _build_build_tab(self, parent: ttk.Frame) -> None:
        frm = ttk.Frame(parent, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0
        ttk.Label(frm, text="Input main.jsonl").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_jsonl, width=110).grid(row=row, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Browse...", command=self._pick_jsonl).grid(row=row, column=2, sticky="e")
        row += 1

        ttk.Label(frm, text="Output DB").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.output_db, width=110).grid(row=row, column=1, sticky="we", padx=6, pady=(8, 0))
        ttk.Button(frm, text="Browse...", command=self._pick_db_out).grid(row=row, column=2, sticky="e", pady=(8, 0))
        row += 1

        ttk.Checkbutton(frm, text="Store raw phone numbers (E.164) in DB (off by default)", variable=self.store_raw_e164).grid(
            row=row, column=1, sticky="w", pady=(10, 0)
        )
        row += 1

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=row, column=1, sticky="w", pady=(12, 0))
        ttk.Button(btn_row, text="Build Database", command=self._start_build).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Rebuild DB (same paths)", command=self._rebuild_db).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text="Use DB for Search", command=self._use_built_db).pack(side=tk.LEFT, padx=8)
        row += 1

        # --- Global cache row ---
        cache_row = ttk.Frame(frm)
        cache_row.grid(row=row, column=0, columnspan=3, sticky="we", pady=(12, 0))
        ttk.Button(cache_row, text="\u26a1 Build All Cache", command=self._build_all_cache).pack(side=tk.LEFT)
        ttk.Button(cache_row, text="Clear Cache", command=self._clear_cache).pack(side=tk.LEFT, padx=8)
        self._cache_status_var = tk.StringVar(value="")
        ttk.Label(cache_row, textvariable=self._cache_status_var, foreground="gray").pack(side=tk.LEFT, padx=12)
        self._refresh_cache_status()
        row += 1

        ttk.Separator(frm).grid(row=row, column=0, columnspan=3, sticky="we", pady=12)
        row += 1

        ttk.Label(frm, text="Log").grid(row=row, column=0, sticky="nw")

        log_container = ttk.Frame(frm)
        log_container.grid(row=row, column=1, columnspan=2, sticky="nsew")
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

        self.build_log = tk.Text(log_container, height=26, wrap="word")
        log_vsb = ttk.Scrollbar(log_container, orient="vertical", command=self.build_log.yview)
        self.build_log.configure(yscrollcommand=log_vsb.set)

        self.build_log.grid(row=0, column=0, sticky="nsew")
        log_vsb.grid(row=0, column=1, sticky="ns")
        log_container.columnconfigure(0, weight=1)
        log_container.rowconfigure(0, weight=1)

        _bind_mousewheel(self.build_log)

    def _log(self, msg: str) -> None:
        self.build_log.insert(tk.END, msg + "\n")
        self.build_log.see(tk.END)

    # ---------- Cache state persistence ----------

    def _load_cache_state(self) -> Dict[str, Any]:
        try:
            import json
            if self._cache_state_path.exists():
                return json.loads(self._cache_state_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_cache_state(self, updates: Dict[str, Any]) -> None:
        import json
        st = self._load_cache_state()
        st.update(updates)
        try:
            self._cache_state_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _refresh_cache_status(self) -> None:
        """Update the cache status label from cache_state.json."""
        st = self._load_cache_state()
        parts: List[str] = []
        db_ts = st.get("db_built")
        if db_ts:
            parts.append(f"DB built {db_ts}")
        thumb_ts = st.get("thumbs_built")
        if thumb_ts:
            parts.append(f"Thumbnails built {thumb_ts}")
        if parts:
            self._cache_status_var.set("Cache: " + " | ".join(parts))
        else:
            self._cache_status_var.set("Cache: not built")

    def _build_all_cache(self) -> None:
        """Build DB + thumbnail cache in one go. Persistent until cleared."""
        if not self._build_lock.acquire(blocking=False):
            messagebox.showwarning("Build running", "A build is already running.")
            return

        in_path = Path(self.input_jsonl.get()).expanduser()
        out_path = Path(self.output_db.get()).expanduser()
        store_raw = bool(self.store_raw_e164.get())

        if not in_path.exists():
            self._build_lock.release()
            messagebox.showerror("Missing file", f"Input file not found:\n{in_path}")
            return

        self._log("=" * 72)
        self._log("BUILD ALL CACHE — Starting")
        self._log(f"Input:  {in_path}")
        self._log(f"Output: {out_path}")
        self._log("=" * 72)

        def worker() -> None:
            from datetime import datetime
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # --- Step 1: Build DB ---
                def log_cb(m: str) -> None:
                    self.root.after(0, lambda: self._log(m))

                # Disconnect if same DB is open
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None

                self.root.after(0, lambda: self._cache_status_var.set("Building database…"))
                build_signal_db.build_db(in_path, out_path, store_raw, log=log_cb)
                self.root.after(0, lambda: self._log("DB build complete."))
                self._save_cache_state({"db_built": now_str, "db_path": str(out_path)})

                # --- Step 2: Build thumbnail cache ---
                # Open a thread-local DB connection for the thumbnail query
                # (self._conn lives on the main thread and can't be used here).
                thumb_conn = sqlite3.connect(str(out_path))
                thumb_conn.row_factory = sqlite3.Row
                try:
                    self.root.after(0, lambda: self._cache_status_var.set("Building thumbnail cache…"))
                    self.root.after(0, lambda: self._log("Building thumbnail cache…"))
                    self._rebuild_thumb_cache(log_func=log_cb, conn=thumb_conn)
                    thumb_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._save_cache_state({"thumbs_built": thumb_ts})
                finally:
                    thumb_conn.close()

                # Now connect on the main thread so the GUI is ready
                def _auto_connect() -> None:
                    self.db_path.set(str(out_path))
                    self._connect()
                self.root.after(0, _auto_connect)

                self.root.after(0, lambda: self._log("BUILD ALL CACHE — Complete"))
                self.root.after(0, lambda: self._refresh_cache_status())
                self.root.after(0, lambda: messagebox.showinfo("Cache Built", "Database and thumbnail cache built successfully.\nCache persists until you clear or rebuild it."))
            except Exception:
                tb = traceback.format_exc()
                self.root.after(0, lambda: self._log("Build All Cache failed:\n" + tb))
                self.root.after(0, lambda: messagebox.showerror("Build failed", "Build All Cache failed. See Log."))
            finally:
                self.root.after(0, lambda: self._refresh_cache_status())
                self._build_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    def _clear_cache(self) -> None:
        """Clear thumbnail cache and optionally the DB. Resets cache_state.json."""
        choice = messagebox.askyesnocancel(
            "Clear Cache",
            "Clear the persistent cache?\n\n"
            "YES  = Clear thumbnails + DB\n"
            "NO   = Clear thumbnails only\n"
            "CANCEL = Do nothing",
        )
        if choice is None:
            return

        # Always clear thumbnails + in-memory photo cache
        cache_dir = self._thumb_cache_dir()
        try:
            shutil.rmtree(str(cache_dir), ignore_errors=True)
        except Exception:
            pass
        self._clear_photo_cache()
        self._log("Thumbnail cache cleared.")
        updates: Dict[str, Any] = {"thumbs_built": None}

        if choice:  # YES — also remove DB
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                self.status_var.set("Disconnected")
            db_path = Path(self.output_db.get()).expanduser()
            if db_path.exists():
                try:
                    db_path.unlink()
                    self._log(f"Deleted DB: {db_path}")
                except Exception as e:
                    self._log(f"Could not delete DB: {e}")
            updates["db_built"] = None
            updates["db_path"] = None

        # Reset cache state
        self._save_cache_state(updates)
        self._refresh_cache_status()
        self._log("Cache cleared.")

    def _rebuild_db(self) -> None:
        """Rebuild the DB using the same input/output paths. Convenience shortcut."""
        if not messagebox.askyesno("Rebuild DB", "Rebuild the database from scratch using current input/output paths?"):
            return
        # Close current connection if it points to the output DB
        outp = Path(self.output_db.get()).expanduser()
        if self._conn is not None:
            try:
                cur_db = Path(self.db_path.get()).resolve()
                if cur_db == outp.resolve():
                    self._conn.close()
                    self._conn = None
                    self.status_var.set("Disconnected for rebuild…")
            except Exception:
                pass
        self._start_build()

    def _pick_jsonl(self) -> None:
        p = filedialog.askopenfilename(title="Select main.jsonl", filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")])
        if p:
            self.input_jsonl.set(p)

    def _pick_db_out(self) -> None:
        p = filedialog.asksaveasfilename(title="Save DB as", defaultextension=".db", filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")])
        if p:
            self.output_db.set(p)

    def _use_built_db(self) -> None:
        db = self.output_db.get()
        if not db or not Path(db).expanduser().exists():
            messagebox.showerror("Missing DB", f"DB not found:\n{db}")
            return
        self.db_path.set(db)
        if self._connect():
            self.nb.select(1)  # Switch to Search & Threads tab

    def _start_build(self) -> None:
        if not self._build_lock.acquire(blocking=False):
            messagebox.showwarning("Build running", "A build is already running.")
            return

        in_path = Path(self.input_jsonl.get()).expanduser()
        out_path = Path(self.output_db.get()).expanduser()
        store_raw = bool(self.store_raw_e164.get())

        if not in_path.exists():
            self._build_lock.release()
            messagebox.showerror("Missing file", f"Input file not found:\n{in_path}")
            return

        self._log("=" * 72)
        self._log("Starting build")
        self._log(f"Input:  {in_path}")
        self._log(f"Output: {out_path}")
        self._log(f"Store raw E.164: {'yes' if store_raw else 'no'}")
        self._log("=" * 72)

        def worker() -> None:
            try:
                def log_cb(m: str) -> None:
                    self.root.after(0, lambda: self._log(m))

                build_signal_db.build_db(in_path, out_path, store_raw, log=log_cb)
                from datetime import datetime as _dt
                self._save_cache_state({"db_built": _dt.now().strftime("%Y-%m-%d %H:%M:%S"), "db_path": str(out_path)})
                self.root.after(0, lambda: self._log("Build complete"))
                self.root.after(0, lambda: self._refresh_cache_status())
                self.root.after(0, lambda: messagebox.showinfo("Build complete", f"DB created:\n{out_path}"))
            except Exception:
                tb = traceback.format_exc()
                self.root.after(0, lambda: self._log("Build failed:\n" + tb))
                self.root.after(0, lambda: messagebox.showerror("Build failed", "Build failed. See Log for details."))
            finally:
                self._build_lock.release()

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Connect ----------

    def _connect(self) -> bool:
        p = Path(self.db_path.get()).expanduser()
        if not p.exists():
            messagebox.showerror("Missing DB", f"DB not found:\n{p}")
            return False
        try:
            if self._conn is not None:
                self._conn.close()
            conn = sqlite3.connect(str(p))
            conn.row_factory = sqlite3.Row
            # Performance pragmas
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8 MB
            self._has_fts5 = _fts5_available(conn) and _table_exists(conn, "messages_fts")
            # Check for has_link column (added in newer DB builds)
            try:
                conn.execute("SELECT has_link FROM messages LIMIT 1")
                self._has_link_col = True
            except sqlite3.OperationalError:
                self._has_link_col = False
            self._conn = conn
            mode = "FTS5" if self._has_fts5 else "LIKE"
            ffmpeg = "yes" if shutil.which("ffmpeg") else "no"

            # DB stats
            stats_parts = [f"Connected: {p.name}", f"mode={mode}", f"ffmpeg={ffmpeg}"]
            try:
                msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                att_count = conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0]
                att_resolved = conn.execute("SELECT COUNT(*) FROM attachments WHERE abs_path != ''").fetchone()[0]
                chat_count = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
                db_size = _human_size(p.stat().st_size)
                pct = f"{att_resolved*100/att_count:.0f}%" if att_count > 0 else "n/a"
                stats_parts.extend([
                    f"chats={chat_count:,}",
                    f"msgs={msg_count:,}",
                    f"atts={att_count:,} ({pct} resolved)",
                    f"size={db_size}",
                ])
            except Exception:
                pass
            self.status_var.set(" | ".join(stats_parts))
            return True
        except Exception as e:
            messagebox.showerror("Connect failed", str(e))
            return False

    # ---------- Search tab ----------

    def _build_search_tab(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X)

        ttk.Label(top, text="DB").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.db_path, width=80).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(top, text="Browse...", command=self._pick_db_in).grid(row=0, column=2, sticky="e")
        ttk.Button(top, text="Connect", command=self._on_connect).grid(row=0, column=3, sticky="e", padx=(6, 0))

        ttk.Label(top, text="Query").grid(row=1, column=0, sticky="w", pady=(10, 0))
        q_ent = ttk.Entry(top, textvariable=self.q, width=80)
        q_ent.grid(row=1, column=1, sticky="we", padx=6, pady=(10, 0))
        ttk.Button(top, text="Search", command=self._run_search).grid(row=1, column=2, sticky="e", pady=(10, 0))
        ttk.Button(top, text="Clear", command=self._clear_search).grid(row=1, column=3, sticky="e", padx=(6, 0), pady=(10, 0))

        ttk.Label(top, text="Recipient contains").grid(row=2, column=0, sticky="w", pady=(10, 0))
        recip_ent = ttk.Entry(top, textvariable=self.recipient, width=28)
        recip_ent.grid(row=2, column=1, sticky="w", padx=6, pady=(10, 0))

        ttk.Label(top, text="After").grid(row=2, column=2, sticky="e", pady=(10, 0))
        after_ent = ttk.Entry(top, textvariable=self.after, width=14)
        after_ent.grid(row=2, column=3, sticky="w", padx=(6, 0), pady=(10, 0))

        ttk.Label(top, text="Before").grid(row=3, column=2, sticky="e", pady=(8, 0))
        before_ent = ttk.Entry(top, textvariable=self.before, width=14)
        before_ent.grid(row=3, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(top, text="Limit").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=10, to=20000, textvariable=self.limit, width=10).grid(row=3, column=1, sticky="w", padx=6, pady=(8, 0))

        ttk.Label(top, text="Attach kind").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(top, textvariable=self.attach_kind, values=["any", "image", "video", "audio", "doc", "file"], width=12, state="readonly").grid(
            row=4, column=1, sticky="w", padx=6, pady=(8, 0)
        )

        ttk.Label(top, text="Dir").grid(row=4, column=2, sticky="e", pady=(8, 0))
        ttk.Combobox(top, textvariable=self.inout, values=["any", "IN", "OUT"], width=12, state="readonly").grid(
            row=4, column=3, sticky="w", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(top, text="Context ±N").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(top, from_=0, to=5000, textvariable=self.context_n, width=10).grid(row=5, column=1, sticky="w", padx=6, pady=(8, 0))

        # Extra filters row
        ttk.Checkbutton(top, text="Has attachments", variable=self.filter_has_attachments).grid(row=5, column=2, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Checkbutton(top, text="Has links", variable=self.filter_has_links).grid(row=5, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        top.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Not connected")
        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(6, 6))

        panes = ttk.PanedWindow(outer, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(panes, padding=(0, 0, 8, 0))
        mid = ttk.Frame(panes, padding=(0, 0, 8, 0))
        right = ttk.Frame(panes)

        panes.add(left, weight=2)
        panes.add(mid, weight=3)
        panes.add(right, weight=5)

        ttk.Label(left, text="Conversations").pack(anchor="w")
        self.chat_tree = ttk.Treeview(left, columns=("recipient", "hits", "chatId"), show="headings", height=20)
        self.chat_tree.heading("recipient", text="Recipient")
        self.chat_tree.heading("hits", text="Hits")
        self.chat_tree.heading("chatId", text="chatId")
        self.chat_tree.column("recipient", width=220, anchor="w")
        self.chat_tree.column("hits", width=60, anchor="e")
        self.chat_tree.column("chatId", width=70, anchor="e")
        self.chat_tree.pack(fill=tk.BOTH, expand=True)
        self.chat_tree.bind("<<TreeviewSelect>>", self._on_chat_selected)
        _bind_mousewheel(self.chat_tree)

        ttk.Label(mid, text="Hits in selected conversation").pack(anchor="w")
        self.hit_tree = ttk.Treeview(mid, columns=("date", "dir", "snippet", "rowid"), show="headings", height=20)
        self.hit_tree.heading("date", text="Date (Local)")
        self.hit_tree.heading("dir", text="Dir")
        self.hit_tree.heading("snippet", text="Snippet")
        self.hit_tree.heading("rowid", text="RowID")
        self.hit_tree.column("date", width=170, anchor="w")
        self.hit_tree.column("dir", width=50, anchor="center")
        self.hit_tree.column("snippet", width=420, anchor="w")
        self.hit_tree.column("rowid", width=70, anchor="e")
        self.hit_tree.pack(fill=tk.BOTH, expand=True)
        self.hit_tree.bind("<<TreeviewSelect>>", self._on_hit_selected)
        self.hit_tree.bind("<Double-1>", self._open_thread_from_selected_hit)
        _bind_mousewheel(self.hit_tree)

        nav = ttk.Frame(mid)
        nav.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(nav, text="Open Thread", command=self._open_thread_from_selected_hit).pack(side=tk.LEFT)
        ttk.Button(nav, text="Prev Hit", command=self._prev_hit).pack(side=tk.LEFT, padx=6)
        ttk.Button(nav, text="Next Hit", command=self._next_hit).pack(side=tk.LEFT)

        ttk.Label(right, text="Thread preview (context window)").pack(anchor="w")
        preview_container = ttk.Frame(right)
        preview_container.pack(fill=tk.BOTH, expand=True)
        self.preview_txt = tk.Text(preview_container, wrap="word")
        pv_vsb = ttk.Scrollbar(preview_container, orient="vertical", command=self.preview_txt.yview)
        self.preview_txt.configure(yscrollcommand=pv_vsb.set)
        self.preview_txt.grid(row=0, column=0, sticky="nsew")
        pv_vsb.grid(row=0, column=1, sticky="ns")
        preview_container.columnconfigure(0, weight=1)
        preview_container.rowconfigure(0, weight=1)

        self._apply_palette_to_text(self.preview_txt)
        self.preview_txt.tag_configure(
            "pv_link",
            foreground=self.theme.get("link_fg", "#8a2be2"),
            underline=True,
        )

        self.preview_txt.configure(state="disabled")
        _bind_mousewheel(self.preview_txt)

        # UX: Enter triggers a search from any search field
        for w in (q_ent, recip_ent, after_ent, before_ent):
            w.bind("<Return>", lambda _e: self._run_search())

        # Search-as-you-type: debounced trace on query var (400ms)
        self.q.trace_add("write", self._on_query_changed)

    def _pick_db_in(self) -> None:
        p = filedialog.askopenfilename(title="Select signal.db", filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")])
        if p:
            self.db_path.set(p)

    def _on_query_changed(self, *_args) -> None:
        """Debounced search-as-you-type: schedule search 400ms after last keystroke."""
        if self._search_debounce_id is not None:
            self.root.after_cancel(self._search_debounce_id)
        self._search_debounce_id = self.root.after(400, self._debounced_search)

    def _debounced_search(self) -> None:
        self._search_debounce_id = None
        q = (self.q.get() or "").strip()
        if len(q) == 1:
            return  # Don't search single chars
        # Allow empty query when filters are set (collected in _collect_params)
        self._run_search(silent=True)

    def _on_connect(self) -> None:
        self._connect()

    def _clear_search(self) -> None:
        self._search_rows = []
        self._chat_hits = []
        self._selected_chat_id = None
        self._selected_hit_rowids = []
        self._hit_idx = -1
        self._current_terms = []
        for t in (self.chat_tree, self.hit_tree):
            for item in t.get_children():
                t.delete(item)
        self.preview_txt.configure(state="normal")
        self.preview_txt.delete("1.0", tk.END)
        self.preview_txt.configure(state="disabled")
        self.export_status.set("")

    def _collect_params(self, silent: bool = False) -> Optional[SearchParams]:
        q = (self.q.get() or "").strip()
        recipient = (self.recipient.get() or "").strip() or None
        has_att = bool(self.filter_has_attachments.get())
        has_lnk = bool(self.filter_has_links.get())
        ak = self.attach_kind.get()
        io = self.inout.get()
        # Allow empty query when at least one filter is active
        if not q and not recipient and not has_att and not has_lnk and ak == "any" and io == "any":
            if not silent:
                messagebox.showwarning("Missing query", "Enter a query or set at least one filter.")
            return None
        after = _normalize_date(self.after.get())
        before = _normalize_date(self.before.get())
        if self.after.get().strip() and not after:
            if not silent:
                messagebox.showwarning("Bad date", "After must be YYYY-MM-DD (or ISO).")
            return None
        if self.before.get().strip() and not before:
            if not silent:
                messagebox.showwarning("Bad date", "Before must be YYYY-MM-DD (or ISO).")
            return None
        limit = int(self.limit.get() or 500)
        limit = max(10, min(limit, 20000))
        return SearchParams(
            q=q, recipient=recipient, after=after, before=before, limit=limit,
            attach_kind=ak, inout=io,
            has_attachments=has_att,
            has_links=has_lnk,
        )

    def _run_search(self, silent: bool = False) -> None:
        if self._conn is None and not self._connect():
            return
        assert self._conn is not None
        params = self._collect_params(silent=silent)
        if not params:
            return

        self._current_terms = _fts_terms(params.q)
        binds: Dict[str, Any] = {"limit": params.limit}

        recip_filter_sql = ""
        if params.recipient:
            recip_filter_sql = " AND recipientName LIKE :recipient "
            binds["recipient"] = f"%{params.recipient}%"

        date_filters = ""
        if params.after:
            date_filters += " AND date(e.dateSentMs / 1000, 'unixepoch', 'localtime') >= date(:after) "
            binds["after"] = params.after
        if params.before:
            date_filters += " AND date(e.dateSentMs / 1000, 'unixepoch', 'localtime') <= date(:before) "
            binds["before"] = params.before

        dir_filter_sql = ""
        if params.inout in ("IN", "OUT"):
            dir_filter_sql = " AND e.outgoing = 1 " if params.inout == "OUT" else " AND e.outgoing = 0 "

        kind_filter_sql = ""
        if params.attach_kind != "any":
            kind_filter_sql = " AND EXISTS (SELECT 1 FROM attachments a WHERE a.message_rowid = e.message_rowid AND a.kind = :kind) "
            binds["kind"] = params.attach_kind

        # New filters
        has_att_sql = ""
        if params.has_attachments:
            has_att_sql = " AND EXISTS (SELECT 1 FROM attachments a2 WHERE a2.message_rowid = e.message_rowid) "

        has_link_sql = ""
        if params.has_links:
            # Use indexed has_link column if available, fall back to LIKE
            if self._has_link_col:
                has_link_sql = " AND e.has_link = 1 "
            else:
                has_link_sql = " AND (e.body LIKE '%http://%' OR e.body LIKE '%https://%') "

        if params.q:
            # --- Text query present: use FTS5 or LIKE ---
            if self._has_fts5:
                sql = f"""
                SELECT
                    e.message_rowid AS rowid,
                    e.chatId AS chatId,
                    e.dateSentMs AS dateSentMs,
                    e.dateSentIso AS dateSentIso,
                    COALESCE(e.recipientName, '(unknown)') AS recipientName,
                    CASE WHEN e.outgoing = 1 THEN 'OUT' ELSE 'IN' END AS dir,
                    e.outgoing AS outgoing,
                    e.body AS body,
                    f.att_names AS att_names
                FROM messages_fts f
                JOIN messages_enriched e ON e.message_rowid = f.rowid
                WHERE messages_fts MATCH :q
                  {recip_filter_sql}
                  {date_filters}
                  {dir_filter_sql}
                  {kind_filter_sql}
                  {has_att_sql}
                  {has_link_sql}
                ORDER BY e.dateSentMs DESC
                LIMIT :limit
                """
                binds["q"] = params.q
            else:
                sql = f"""
                SELECT
                    e.message_rowid AS rowid,
                    e.chatId AS chatId,
                    e.dateSentMs AS dateSentMs,
                    e.dateSentIso AS dateSentIso,
                    COALESCE(e.recipientName, '(unknown)') AS recipientName,
                    CASE WHEN e.outgoing = 1 THEN 'OUT' ELSE 'IN' END AS dir,
                    e.outgoing AS outgoing,
                    e.body AS body,
                    COALESCE((
                        SELECT group_concat(a.file_name, ' ')
                        FROM attachments a
                        WHERE a.message_rowid = e.message_rowid AND a.file_name != ''
                    ), '') AS att_names
                FROM messages_enriched e
                WHERE (
                    (e.body IS NOT NULL AND lower(e.body) LIKE :q)
                    OR EXISTS (SELECT 1 FROM attachments a WHERE a.message_rowid = e.message_rowid AND lower(a.file_name) LIKE :q)
                )
                  {recip_filter_sql}
                  {date_filters}
                  {dir_filter_sql}
                  {kind_filter_sql}
                  {has_att_sql}
                  {has_link_sql}
                ORDER BY e.dateSentMs DESC
                LIMIT :limit
                """
                binds["q"] = f"%{params.q.lower()}%"
        else:
            # --- No text query: filter-only browse ---
            sql = f"""
            SELECT
                e.message_rowid AS rowid,
                e.chatId AS chatId,
                e.dateSentMs AS dateSentMs,
                e.dateSentIso AS dateSentIso,
                COALESCE(e.recipientName, '(unknown)') AS recipientName,
                CASE WHEN e.outgoing = 1 THEN 'OUT' ELSE 'IN' END AS dir,
                e.outgoing AS outgoing,
                e.body AS body,
                COALESCE((
                    SELECT group_concat(a.file_name, ' ')
                    FROM attachments a
                    WHERE a.message_rowid = e.message_rowid AND a.file_name != ''
                ), '') AS att_names
            FROM messages_enriched e
            WHERE 1=1
              {recip_filter_sql}
              {date_filters}
              {dir_filter_sql}
              {kind_filter_sql}
              {has_att_sql}
              {has_link_sql}
            ORDER BY e.dateSentMs DESC
            LIMIT :limit
            """

        try:
            self._search_rows = [dict(r) for r in self._conn.execute(sql, binds).fetchall()]
        except Exception as e:
            if not silent:
                messagebox.showerror("Search failed", str(e))
            return

        by_chat: Dict[int, Dict[str, Any]] = {}
        for r in self._search_rows:
            cid = int(r["chatId"])
            ent = by_chat.get(cid)
            if not ent:
                by_chat[cid] = {"chatId": cid, "recipientName": r.get("recipientName") or "(unknown)", "hits": 1}
            else:
                ent["hits"] += 1

        self._chat_hits = sorted(by_chat.values(), key=lambda x: (-x["hits"], x["recipientName"]))
        self._render_chat_list()
        self._render_hits_for_chat(None)
        self.export_status.set(f"Search results ready: {len(self._search_rows)} messages across {len(self._chat_hits)} chats")

    def _render_chat_list(self) -> None:
        for item in self.chat_tree.get_children():
            self.chat_tree.delete(item)
        for ch in self._chat_hits:
            self.chat_tree.insert("", tk.END, values=(ch["recipientName"], ch["hits"], ch["chatId"]))

    def _on_chat_selected(self, _evt=None) -> None:
        sel = self.chat_tree.selection()
        if not sel:
            return
        vals = self.chat_tree.item(sel[0], "values")
        if not vals or len(vals) < 3:
            return
        chat_id = int(vals[2])
        self._render_hits_for_chat(chat_id)
        self._gallery_selected_chat_id.set(str(chat_id))
        # Only load gallery thumbnails if the Gallery tab is currently visible
        if self.nb.index(self.nb.select()) == 2:  # Gallery tab index
            self._refresh_gallery_for_chat(chat_id)
        else:
            self._gallery_needs_refresh = True

    def _render_hits_for_chat(self, chat_id: Optional[int]) -> None:
        self._selected_chat_id = chat_id
        for item in self.hit_tree.get_children():
            self.hit_tree.delete(item)
        self.preview_txt.configure(state="normal")
        self.preview_txt.delete("1.0", tk.END)
        self.preview_txt.configure(state="disabled")
        self._selected_hit_rowids = []
        self._hit_idx = -1

        if chat_id is None:
            return

        hits = [r for r in self._search_rows if int(r["chatId"]) == chat_id]
        hits.sort(key=lambda r: r.get("dateSentMs") or 0, reverse=True)
        self._selected_hit_rowids = [int(r["rowid"]) for r in hits]

        for r in hits:
            snippet = (r.get("body") or "").strip()
            if not snippet:
                snippet = f"[attachments] {(r.get('att_names') or '').strip()}"
            snippet = snippet.replace("\n", " ")
            if len(snippet) > 260:
                snippet = snippet[:260] + "…"
            self.hit_tree.insert(
                "",
                tk.END,
                values=(_fmt_ts_short(r.get("dateSentIso", "")), r.get("dir", ""), snippet, r.get("rowid", "")),
            )

        if self._selected_hit_rowids:
            self._hit_idx = 0
            self._render_thread_preview(self._selected_hit_rowids[0])

    def _selected_hit_rowid(self) -> Optional[int]:
        sel = self.hit_tree.selection()
        if sel:
            vals = self.hit_tree.item(sel[0], "values")
            try:
                return int(vals[3])
            except Exception:
                return None
        if 0 <= self._hit_idx < len(self._selected_hit_rowids):
            return self._selected_hit_rowids[self._hit_idx]
        return None

    def _prev_hit(self) -> None:
        if not self._selected_hit_rowids:
            return
        self._hit_idx = max(0, self._hit_idx - 1)
        self._render_thread_preview(self._selected_hit_rowids[self._hit_idx])

    def _next_hit(self) -> None:
        if not self._selected_hit_rowids:
            return
        self._hit_idx = min(len(self._selected_hit_rowids) - 1, self._hit_idx + 1)
        self._render_thread_preview(self._selected_hit_rowids[self._hit_idx])

    def _on_hit_selected(self, _evt=None) -> None:
        rid = self._selected_hit_rowid()
        if rid is not None:
            self._render_thread_preview(rid)

    def _open_thread_from_selected_hit(self, _evt=None) -> None:
        rid = self._selected_hit_rowid()
        if rid is None:
            return
        self._open_thread_window(rid)

    # ---------- Preview ----------

    def _load_context_window(self, rowid: int, n: int) -> List[Dict[str, Any]]:
        assert self._conn is not None
        # Step 1: locate the hit's chat and timestamp (indexed lookup)
        hit = self._conn.execute(
            "SELECT chatId, dateSentMs FROM messages WHERE rowid = :rowid", {"rowid": rowid}
        ).fetchone()
        if not hit:
            return []
        chat_id = hit["chatId"]
        hit_ms = hit["dateSentMs"]

        # Step 2: count how many rows in this chat come before the hit (using chatId+date index)
        pos = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chatId = :cid AND (dateSentMs < :ms OR (dateSentMs = :ms AND rowid < :rid))",
            {"cid": chat_id, "ms": hit_ms, "rid": rowid},
        ).fetchone()[0]

        offset = max(0, pos - n)
        limit = 2 * n + 1

        # Step 3: fetch only the context window from this one chat
        q = """
        SELECT m.rowid AS rowid, m.outgoing, m.dateSentIso, m.dateSentMs, COALESCE(m.body,'') AS body
        FROM messages m
        WHERE m.chatId = :cid
        ORDER BY m.dateSentMs ASC, m.rowid ASC
        LIMIT :lim OFFSET :off
        """
        return [dict(r) for r in self._conn.execute(q, {"cid": chat_id, "lim": limit, "off": offset}).fetchall()]

    def _highlight_term_in_text(self, txt: tk.Text, term: str) -> None:
        if not term:
            return
        content = txt.get("1.0", tk.END)
        for m in re.finditer(re.escape(term), content, flags=re.IGNORECASE):
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            txt.tag_add("term", start, end)

    def _render_thread_preview(self, rowid: int) -> None:
        if self._conn is None:
            return
        ctx = int(self.context_n.get() or 0)
        ctx = max(0, min(ctx, 5000))
        rows = self._load_context_window(rowid=rowid, n=ctx)

        self.preview_txt.configure(state="normal")
        self.preview_txt.delete("1.0", tk.END)

        # Keep image references alive for inline thumbnails
        self._preview_img_refs: List[Any] = []
        link_idx_ref: List[int] = [0]

        msg_ids = [int(m["rowid"]) for m in rows]
        authors = self._speaker_names_for_message_ids(msg_ids)
        atts_by_msg = self._batch_fetch_attachments(msg_ids)

        def add_link(label: str, path: str) -> None:
            s = self.preview_txt.index(tk.END)
            self.preview_txt.insert(tk.END, label)
            e = self.preview_txt.index(tk.END)
            self.preview_txt.tag_add("pv_link", s, e)
            tag = f"pv_link_{link_idx_ref[0]}"
            link_idx_ref[0] += 1
            self.preview_txt.tag_add(tag, s, e)
            self.preview_txt.tag_bind(tag, "<Button-1>", lambda _e, p=path: _safe_open_path(p))
            self.preview_txt.tag_bind(tag, "<Button-3>", lambda _e, p=path: self._attachment_context_menu(_e, p))

        ranges: Dict[int, Tuple[str, str]] = {}
        for m in rows:
            mid = int(m["rowid"])
            who = authors.get(mid) or ("You" if m["outgoing"] else "Them")
            tag = "you" if m["outgoing"] else "them"
            ts = _fmt_ts_short(m["dateSentIso"] or "")
            body = m["body"] or ""

            s = self.preview_txt.index(tk.END)
            self.preview_txt.insert(tk.END, f"[{ts}] ", ("meta",))
            self.preview_txt.insert(tk.END, f"{who}: ", (tag,))
            _insert_body_with_links(self.preview_txt, body, link_idx_ref, "pv_link")
            e = self.preview_txt.index(tk.END)
            ranges[mid] = (s, e)

            atts = atts_by_msg.get(mid, [])
            if atts:
                self.preview_txt.insert(tk.END, "\n")
                for a in atts:
                    fn = a["file_name"] or "(file)"
                    ap = a["abs_path"] or ""
                    kind = a["kind"] or "file"
                    meta = []
                    if a["mime"]:
                        meta.append(a["mime"])
                    if a["size_bytes"]:
                        meta.append(_human_size(a["size_bytes"]))
                    if a["width"] and a["height"]:
                        meta.append(f"{a['width']}x{a['height']}")
                    if a["duration_ms"]:
                        meta.append(f"{int(a['duration_ms'])/1000:.1f}s")
                    meta_str = " | ".join(meta)

                    if kind in ("image", "video", "audio") and ap and os.path.exists(ap):
                        tkimg = self._attachment_thumbnail_photo(ap, kind, fn, (320, 240), duration_ms=a["duration_ms"])
                        if tkimg is not None:
                            self._preview_img_refs.append(tkimg)
                            self.preview_txt.insert(tk.END, "  ")
                            self.preview_txt.image_create(tk.END, image=tkimg)
                            self.preview_txt.insert(tk.END, f"\n  {fn}", ("meta",))
                            if meta_str:
                                self.preview_txt.insert(tk.END, f" ({meta_str})", ("meta",))
                            self.preview_txt.insert(tk.END, "  ")
                            if ap:
                                add_link("[open]", ap)
                            self.preview_txt.insert(tk.END, "\n")
                            continue

                    self.preview_txt.insert(tk.END, f"  [{kind}] {fn}", ("meta",))
                    if meta_str:
                        self.preview_txt.insert(tk.END, f" ({meta_str})", ("meta",))
                    self.preview_txt.insert(tk.END, "  ")
                    if ap:
                        add_link("[open]", ap)
                    self.preview_txt.insert(tk.END, "\n")

            self.preview_txt.insert(tk.END, "\n\n")

        if rowid in ranges:
            s, e = ranges[rowid]
            self.preview_txt.tag_add("hit", s, e)
            self.preview_txt.see(s)

        for term in self._current_terms:
            self._highlight_term_in_text(self.preview_txt, term)

        self.preview_txt.configure(state="disabled")

    # ---------- Full thread window ----------

    def _open_thread_window(self, hit_rowid: int) -> None:
        if self._conn is None:
            return

        info = self._conn.execute(
            "SELECT chatId, COALESCE(recipientName,'(unknown)') AS recipientName FROM messages_enriched WHERE message_rowid = ?",
            (hit_rowid,),
        ).fetchone()
        if not info:
            messagebox.showerror("Thread open failed", "Hit not found.")
            return

        chat_id = int(info["chatId"])
        recipient_name = info["recipientName"]

        hit_rowids = sorted({int(r["rowid"]) for r in self._search_rows if int(r["chatId"]) == chat_id})
        self._selected_hit_rowids = hit_rowids
        self._hit_idx = hit_rowids.index(hit_rowid) if hit_rowid in hit_rowids else -1

        msgs = [dict(r) for r in self._conn.execute(
            """
            SELECT m.rowid AS rowid, m.dateSentIso, m.dateSentMs, m.outgoing, COALESCE(m.body,'') AS body
            FROM messages m
            WHERE m.chatId = ?
            ORDER BY m.dateSentMs ASC, m.rowid ASC
            """,
            (chat_id,),
        ).fetchall()]

        win = tk.Toplevel(self.root)
        win.title(f"Thread: {recipient_name} (chatId={chat_id})")
        win.geometry("1060x780")

        top = ttk.Frame(win, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text=f"Recipient: {recipient_name}").pack(side=tk.LEFT)
        ttk.Label(top, text=f"  chatId={chat_id}").pack(side=tk.LEFT)

        btns = ttk.Frame(win, padding=(10, 0, 10, 10))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Prev Hit", command=lambda: self._thread_nav(win, -1)).pack(side=tk.LEFT)
        ttk.Button(btns, text="Next Hit", command=lambda: self._thread_nav(win, +1)).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Find in Thread (Ctrl+F)", command=lambda: self._thread_find_dialog(win)).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Export Thread HTML", command=lambda: self._export_thread_html(chat_id)).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Export Thread MD", command=lambda: self._export_thread_md(chat_id)).pack(side=tk.RIGHT, padx=6)

        container = ttk.Frame(win, padding=(10, 0, 10, 10))
        container.pack(fill=tk.BOTH, expand=True)
        txt = tk.Text(container, wrap="word")
        vsb = ttk.Scrollbar(container, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        _bind_mousewheel(txt)

        self._apply_palette_to_text(txt)
        txt.tag_configure(
            "link",
            foreground=self.theme.get("link_fg", "#8a2be2"),
            underline=True,
        )

        # Track open thread windows so theme updates propagate live.
        self._open_thread_text_widgets.append(txt)

        def _prune_open_thread_texts() -> None:
            alive: List[tk.Text] = []
            for t in self._open_thread_text_widgets:
                try:
                    if t is txt:
                        continue
                    if t.winfo_exists():
                        alive.append(t)
                except Exception:
                    continue
            self._open_thread_text_widgets = alive

        win.bind("<Destroy>", lambda e: _prune_open_thread_texts() if e.widget is win else None)

        win._img_refs = []  # type: ignore[attr-defined]
        msg_ids = [int(m["rowid"]) for m in msgs]
        atts_by_msg = self._batch_fetch_attachments(msg_ids)
        authors = self._speaker_names_for_message_ids(msg_ids)

        ranges: Dict[int, Tuple[str, str]] = {}
        link_idx_ref: List[int] = [0]

        def add_link(label: str, path: str) -> None:
            s = txt.index(tk.END)
            txt.insert(tk.END, label)
            e = txt.index(tk.END)
            txt.tag_add("link", s, e)
            tag = f"link_{link_idx_ref[0]}"
            link_idx_ref[0] += 1
            txt.tag_add(tag, s, e)
            txt.tag_bind(tag, "<Button-1>", lambda _e, p=path: _safe_open_path(p))
            txt.tag_bind(tag, "<Button-3>", lambda _e, p=path: self._attachment_context_menu(_e, p))

        for m in msgs:
            mid = int(m["rowid"])
            who = authors.get(mid) or ("You" if m["outgoing"] else recipient_name)
            wtag = "you" if m["outgoing"] else "them"
            ts = _fmt_ts_short(m["dateSentIso"] or "")
            body = m["body"] or ""

            s = txt.index(tk.END)
            txt.insert(tk.END, f"[{ts}] ", ("meta",))
            txt.insert(tk.END, f"{who}: ", (wtag,))
            _insert_body_with_links(txt, body, link_idx_ref, "link")
            e = txt.index(tk.END)
            ranges[mid] = (s, e)

            atts = atts_by_msg.get(mid, [])
            if atts:
                txt.insert(tk.END, "\n")
                for a in atts:
                    fn = a["file_name"] or "(file)"
                    ap = a["abs_path"] or ""
                    kind = a["kind"] or "file"
                    meta = []
                    if a["mime"]:
                        meta.append(a["mime"])
                    if a["size_bytes"]:
                        meta.append(_human_size(a["size_bytes"]))
                    if a["width"] and a["height"]:
                        meta.append(f"{a['width']}x{a['height']}")
                    if a["duration_ms"]:
                        meta.append(f"{int(a['duration_ms'])/1000:.1f}s")
                    meta_str = " | ".join(meta)

                    if kind in ("image", "video", "audio") and ap and os.path.exists(ap):
                        tkimg = self._attachment_thumbnail_photo(ap, kind, fn, (560, 420), duration_ms=a["duration_ms"])
                        if tkimg is not None:
                            win._img_refs.append(tkimg)  # type: ignore[attr-defined]
                            txt.insert(tk.END, "  ")
                            txt.image_create(tk.END, image=tkimg)
                            txt.insert(tk.END, f"\n  {fn}", ("meta",))
                            if meta_str:
                                txt.insert(tk.END, f" ({meta_str})", ("meta",))
                            txt.insert(tk.END, "  ")
                            add_link("[open]", ap)
                            txt.insert(tk.END, "\n")
                            continue

                    txt.insert(tk.END, f"  [{kind}] {fn}", ("meta",))
                    if meta_str:
                        txt.insert(tk.END, f" ({meta_str})", ("meta",))
                    txt.insert(tk.END, "  ")
                    if ap:
                        add_link("[open]", ap)
                    txt.insert(tk.END, "\n")

            txt.insert(tk.END, "\n")

        if hit_rowid in ranges:
            s, e = ranges[hit_rowid]
            txt.tag_add("hit", s, e)
            txt.see(s)

        for term in self._current_terms:
            self._highlight_term_in_text(txt, term)

        txt.configure(state="disabled")
        win._thread_txt = txt  # type: ignore[attr-defined]
        win._thread_ranges = ranges  # type: ignore[attr-defined]
        win.bind("<Control-f>", lambda _e: self._thread_find_dialog(win))

    def _thread_nav(self, win: tk.Toplevel, delta: int) -> None:
        if not self._selected_hit_rowids:
            return
        if self._hit_idx < 0:
            self._hit_idx = 0
        else:
            self._hit_idx = max(0, min(len(self._selected_hit_rowids) - 1, self._hit_idx + delta))
        rid = self._selected_hit_rowids[self._hit_idx]
        txt = getattr(win, "_thread_txt", None)
        ranges = getattr(win, "_thread_ranges", None)
        if txt is None or ranges is None:
            return
        if rid in ranges:
            txt.configure(state="normal")
            txt.tag_remove("hit", "1.0", tk.END)
            s, e = ranges[rid]
            txt.tag_add("hit", s, e)
            txt.see(s)
            txt.configure(state="disabled")

    def _thread_find_dialog(self, win: tk.Toplevel) -> None:
        txt = getattr(win, "_thread_txt", None)
        if txt is None:
            return

        dlg = tk.Toplevel(win)
        dlg.title("Find in Thread")
        dlg.geometry("420x120")
        qv = tk.StringVar(value="")

        ttk.Label(dlg, text="Find (case-insensitive)").pack(anchor="w", padx=10, pady=(10, 0))
        ent = ttk.Entry(dlg, textvariable=qv)
        ent.pack(fill=tk.X, padx=10, pady=6)
        ent.focus_set()

        state = {"last": "1.0"}

        def find_next() -> None:
            needle = qv.get().strip()
            if not needle:
                return
            txt.configure(state="normal")
            txt.tag_remove("term", "1.0", tk.END)
            idx = txt.search(needle, state["last"], nocase=True, stopindex=tk.END)
            if not idx:
                state["last"] = "1.0"
                txt.configure(state="disabled")
                return
            end = f"{idx}+{len(needle)}c"
            txt.tag_add("term", idx, end)
            txt.see(idx)
            state["last"] = end
            txt.configure(state="disabled")

        ttk.Button(dlg, text="Find Next", command=find_next).pack(pady=(0, 10))
        dlg.bind("<Return>", lambda _e: find_next())

    # ---------- Gallery tab ----------

    def _build_gallery_tab(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Chat ID").pack(side=tk.LEFT)
        self._gallery_selected_chat_id = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self._gallery_selected_chat_id, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load", command=self._gallery_load_current).pack(side=tk.LEFT)
        ttk.Button(top, text="Clear", command=self._clear_gallery).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Type").pack(side=tk.LEFT, padx=(12, 0))
        self._gallery_kind_filter = tk.StringVar(value="all")
        kind_cb = ttk.Combobox(top, textvariable=self._gallery_kind_filter,
                               values=["all", "image", "video", "audio", "doc", "file"],
                               width=8, state="readonly")
        kind_cb.pack(side=tk.LEFT, padx=6)
        # Use a variable trace so the filter refresh works reliably across Tk
        # versions/platforms (<<ComboboxSelected>> can be inconsistent).
        self._gallery_kind_filter.trace_add(
            "write",
            lambda *_a: self._refresh_gallery_for_chat(self._safe_int(self._gallery_selected_chat_id.get())),
        )

        ttk.Button(top, text="Rebuild Thumbnails", command=self._start_rebuild_thumbnails).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="Click to open. Right-click for copy.").pack(side=tk.LEFT, padx=10)

        self.gallery_canvas = tk.Canvas(outer)
        self.gallery_scroll = ttk.Scrollbar(outer, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=self.gallery_scroll.set)
        self.gallery_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.gallery_inner = ttk.Frame(self.gallery_canvas)
        self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")
        # Throttle scrollregion updates — only recalculate when inner frame size changes
        self._gallery_inner_size: tuple = (0, 0)
        self.gallery_inner.bind("<Configure>", self._on_gallery_inner_configure)

        self._gallery_img_refs: List[Any] = []
        self._gallery_thumb_size = (240, 180)

        # Pixel-aligned scroll for smoother rendering; 17 px ≈ one text line
        self.gallery_canvas.configure(yscrollincrement=17)
        _bind_mousewheel(self.gallery_canvas, speed=3)
        _bind_mousewheel(self.gallery_inner, scroll_widget=self.gallery_canvas, speed=3)

        # Reflow on resize (debounced)
        self.gallery_canvas.bind("<Configure>", self._on_gallery_canvas_resize)
        self._gallery_last_cols: int = 0
        self._gallery_cached_rows: List[Any] = []
        self._gallery_resize_after_id: Optional[str] = None

    def _gallery_load_current(self) -> None:
        self._refresh_gallery_for_chat(self._safe_int(self._gallery_selected_chat_id.get()))

    def _gallery_bind_scroll_recursive(self, widget: Any) -> None:
        """Bind mousewheel on every descendant so scrolling works even when hovering over thumbnails."""
        _bind_mousewheel(widget, scroll_widget=self.gallery_canvas, speed=3)
        for child in widget.winfo_children():
            self._gallery_bind_scroll_recursive(child)

    def _on_gallery_inner_configure(self, _event: Any) -> None:
        """Update scrollregion only when the inner frame's size actually changes."""
        new_size = (self.gallery_inner.winfo_reqwidth(), self.gallery_inner.winfo_reqheight())
        if new_size != self._gallery_inner_size:
            self._gallery_inner_size = new_size
            self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))

    def _on_gallery_canvas_resize(self, event=None) -> None:
        """Re-layout the gallery grid when the canvas width changes (debounced)."""
        if self._gallery_resize_after_id:
            self.gallery_canvas.after_cancel(self._gallery_resize_after_id)
        self._gallery_resize_after_id = self.gallery_canvas.after(150, self._gallery_do_resize)

    def _gallery_do_resize(self) -> None:
        """Debounced callback — actually re-layout the gallery grid."""
        self._gallery_resize_after_id = None
        if not self._gallery_cached_rows:
            return
        canvas_w = self.gallery_canvas.winfo_width()
        thumb_w = self._gallery_thumb_size[0]
        cell_w = thumb_w + 24  # padding
        new_cols = max(1, canvas_w // cell_w)
        if new_cols != self._gallery_last_cols:
            self._layout_gallery_grid(self._gallery_cached_rows, new_cols)

    def _clear_gallery(self) -> None:
        for w in self.gallery_inner.winfo_children():
            w.destroy()
        self._gallery_img_refs = []
        self._gallery_cached_rows = []
        self._gallery_last_cols = 0

    def _start_rebuild_thumbnails(self) -> None:
        """Rebuild the thumbnail cache in a background thread with status feedback."""
        if not messagebox.askyesno("Rebuild Thumbnails", "Delete and rebuild all cached thumbnails?\nThis may take a while for large exports."):
            return
        self.export_status.set("Rebuilding thumbnail cache…")

        def worker() -> None:
            def log_cb(msg: str) -> None:
                self.root.after(0, lambda: self.export_status.set(msg))
            self._rebuild_thumb_cache(log_func=log_cb)
            from datetime import datetime as _dt
            self._save_cache_state({"thumbs_built": _dt.now().strftime("%Y-%m-%d %H:%M:%S")})
            self.root.after(0, lambda: self.export_status.set("Thumbnail cache rebuilt."))
            self.root.after(0, lambda: self._refresh_cache_status())
            # Refresh gallery if a chat is loaded
            try:
                chat_id = self._safe_int(self._gallery_selected_chat_id.get())
                if chat_id:
                    self.root.after(100, lambda: self._refresh_gallery_for_chat(chat_id))
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _safe_int(self, s: str) -> Optional[int]:
        try:
            return int(s)
        except Exception:
            return None

    def _on_tab_changed(self, _evt=None) -> None:
        """Lazy-load gallery when user switches to the Gallery tab."""
        try:
            if self.nb.index(self.nb.select()) == 2 and self._gallery_needs_refresh:
                self._gallery_needs_refresh = False
                cid = self._safe_int(self._gallery_selected_chat_id.get())
                if cid:
                    self._refresh_gallery_for_chat(cid)
        except Exception:
            pass

    def _refresh_gallery_for_chat(self, chat_id: Optional[int]) -> None:
        if self._conn is None and not self._connect():
            return
        assert self._conn is not None
        if not chat_id:
            return

        self._clear_gallery()

        kind_filter = self._gallery_kind_filter.get()
        if kind_filter == "all":
            kind_sql = ""
            binds: tuple = (chat_id,)
        else:
            kind_sql = " AND a.kind = ?"
            binds = (chat_id, kind_filter)

        rows = self._conn.execute(
            f"""
            SELECT a.abs_path, a.file_name, a.kind, a.mime, a.size_bytes, a.width, a.height, a.duration_ms, m.dateSentIso
            FROM attachments a
            JOIN messages m ON m.rowid = a.message_rowid
            WHERE m.chatId = ? AND a.abs_path != ''{kind_sql}
            ORDER BY m.dateSentMs DESC, a.id DESC
            LIMIT 500
            """,
            binds,
        ).fetchall()

        self._gallery_cached_rows = rows

        # Calculate dynamic column count from canvas width
        canvas_w = self.gallery_canvas.winfo_width()
        thumb_w, thumb_h = self._gallery_thumb_size
        cell_w = thumb_w + 24
        cols = max(1, canvas_w // cell_w) if canvas_w > 1 else 4
        self._layout_gallery_grid(rows, cols)

    def _layout_gallery_grid(self, rows: List[Any], cols: int) -> None:
        """(Re-)build the gallery grid with the given column count."""
        self._gallery_last_cols = cols
        # Destroy old widgets
        for w in self.gallery_inner.winfo_children():
            w.destroy()
        self._gallery_img_refs = []

        thumb_w, thumb_h = self._gallery_thumb_size

        for idx, r in enumerate(rows):
            ap = r["abs_path"]
            fn = r["file_name"] or "(file)"
            kind = r["kind"] or "file"
            ts = _fmt_ts_short(r["dateSentIso"] or "")
            row = idx // cols
            col = idx % cols

            cell = ttk.Frame(self.gallery_inner, padding=6)
            cell.grid(row=row, column=col, sticky="n")

            if ap and os.path.exists(ap):
                if kind in ("image", "video", "audio"):
                    tkimg = self._attachment_thumbnail_photo(ap, kind, fn, (thumb_w, thumb_h), duration_ms=r["duration_ms"])
                    if tkimg is not None:
                        self._gallery_img_refs.append(tkimg)
                        lbl = ttk.Label(cell, image=tkimg)
                        lbl.pack()
                        lbl.bind("<Button-1>", lambda _e, p=ap: _safe_open_path(p))
                        lbl.bind("<Button-3>", lambda _e, p=ap: self._attachment_context_menu(_e, p))
                    else:
                        lbl = ttk.Label(cell, text=f"[{kind}]\n{fn}", wraplength=thumb_w, justify="center")
                        lbl.pack(fill=tk.BOTH, expand=True)
                        lbl.bind("<Button-1>", lambda _e, p=ap: _safe_open_path(p))
                        lbl.bind("<Button-3>", lambda _e, p=ap: self._attachment_context_menu(_e, p))
                else:
                    lbl = ttk.Label(cell, text=f"[{kind}]\n{fn}", wraplength=thumb_w, justify="center")
                    lbl.pack(fill=tk.BOTH, expand=True)
                    lbl.bind("<Button-1>", lambda _e, p=ap: _safe_open_path(p))
                    lbl.bind("<Button-3>", lambda _e, p=ap: self._attachment_context_menu(_e, p))
            else:
                ttk.Label(cell, text="[missing]").pack()

            if kind in ("image", "video", "audio"):
                ttk.Label(cell, text=fn, wraplength=thumb_w).pack()
            ttk.Label(cell, text=ts, foreground=self.theme.get("meta_fg", "#666666")).pack()

        for c in range(cols):
            self.gallery_inner.columnconfigure(c, weight=1)

        # Bind scroll on all child widgets so hovering over thumbnails doesn't eat the wheel
        self._gallery_bind_scroll_recursive(self.gallery_inner)

    # ---------- Exports tab ----------

    def _build_exports_tab(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="Exports").pack(anchor="w")
        ttk.Label(outer, text="Export your current search results or a full chat thread.").pack(anchor="w", pady=(0, 8))

        row = ttk.Frame(outer)
        row.pack(fill=tk.X, pady=8)

        ttk.Button(row, text="Export Search Results CSV", command=self._export_search_csv).pack(side=tk.LEFT)
        ttk.Button(row, text="Export Search Results HTML", command=self._export_search_html).pack(side=tk.LEFT, padx=6)

        row2 = ttk.Frame(outer)
        row2.pack(fill=tk.X, pady=8)
        ttk.Label(row2, text="Chat ID for thread export").pack(side=tk.LEFT)
        self.export_chat_id = tk.StringVar(value="")
        ttk.Entry(row2, textvariable=self.export_chat_id, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(row2, text="Export Thread HTML", command=lambda: self._export_thread_html(self._safe_int(self.export_chat_id.get()))).pack(side=tk.LEFT)
        ttk.Button(row2, text="Export Thread MD", command=lambda: self._export_thread_md(self._safe_int(self.export_chat_id.get()))).pack(side=tk.LEFT, padx=6)

        self.export_status = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.export_status).pack(anchor="w", pady=(12, 0))

    # ---------- Stats tab ----------

    def _build_stats_tab(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(outer)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Refresh Stats", command=self._refresh_stats).pack(side=tk.LEFT)
        ttk.Button(top, text="Export Stats", command=self._export_stats).pack(side=tk.LEFT, padx=(8, 0))
        self._stats_status = tk.StringVar(value="Click Refresh Stats after connecting to a database.")
        ttk.Label(top, textvariable=self._stats_status).pack(side=tk.LEFT, padx=12)

        # Scrollable dashboard area (replaces monospaced Text report)
        container = ttk.Frame(outer)
        container.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self._stats_canvas = tk.Canvas(container, bg=self.theme.get("panel_bg", "#ffffff"), highlightthickness=0, bd=0)
        self._stats_scroll = ttk.Scrollbar(container, orient="vertical", command=self._stats_canvas.yview)
        self._stats_canvas.configure(yscrollcommand=self._stats_scroll.set)

        self._stats_canvas.grid(row=0, column=0, sticky="nsew")
        self._stats_scroll.grid(row=0, column=1, sticky="ns")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._stats_inner = ttk.Frame(self._stats_canvas)
        self._stats_canvas.create_window((0, 0), window=self._stats_inner, anchor="nw")
        self._stats_inner_size: tuple = (0, 0)
        self._stats_inner.bind("<Configure>", self._on_stats_inner_configure)

        self._stats_canvas.configure(yscrollincrement=17)
        _bind_mousewheel(self._stats_canvas, speed=3)
        _bind_mousewheel(self._stats_inner, scroll_widget=self._stats_canvas, speed=3)

        # Keep text report for export
        self._stats_report_text: str = ""
        self._stats_cloud_ref: Optional[ImageTk.PhotoImage] = None  # prevent GC
        self._stats_cloud_pil: Optional[Image.Image] = None  # PIL copy for export

        # KPI font (uses default font family)
        try:
            self._stats_kpi_font = tkfont.nametofont("TkDefaultFont").copy()
            self._stats_kpi_font.configure(size=int(self._stats_kpi_font.cget("size")) + 4, weight="bold")
        except Exception:
            self._stats_kpi_font = None

    def _on_stats_inner_configure(self, _event: Any) -> None:
        """Update stats scrollregion when inner frame size changes."""
        try:
            new_size = (self._stats_inner.winfo_reqwidth(), self._stats_inner.winfo_reqheight())
            if new_size != getattr(self, "_stats_inner_size", (0, 0)):
                self._stats_inner_size = new_size
                self._stats_canvas.configure(scrollregion=self._stats_canvas.bbox("all"))
        except Exception:
            pass

    def _render_stats_dashboard(self, data: Dict[str, Any]) -> None:
        """Render computed stats into the dashboard UI."""
        inner = getattr(self, "_stats_inner", None)
        if inner is None:
            return

        for w in inner.winfo_children():
            w.destroy()

        th = self.theme

        def _kpi(parent: ttk.Frame, title: str, value: str, subtitle: str = "") -> ttk.Labelframe:
            lf = ttk.Labelframe(parent, text=title, padding=10)
            v = ttk.Label(lf, text=value)
            kpi_font = getattr(self, "_stats_kpi_font", None)
            if kpi_font is not None:
                try:
                    v.configure(font=kpi_font)
                except Exception:
                    pass
            v.pack(anchor="w")
            if subtitle:
                ttk.Label(lf, text=subtitle, foreground=th.get("meta_fg", "#666666")).pack(anchor="w", pady=(4, 0))
            return lf

        def _bar_list(parent: ttk.Frame, title: str, items: list[tuple[str, int]], max_rows: int | None = None) -> None:
            lf = ttk.Labelframe(parent, text=title, padding=10)
            lf.pack(fill=tk.BOTH, expand=True)
            if not items:
                ttk.Label(lf, text="No data").pack(anchor="w")
                return
            mx = max((c for _l, c in items), default=1) or 1
            rows = items if max_rows is None else items[:max_rows]
            for lbl, cnt in rows:
                row = ttk.Frame(lf)
                row.pack(fill=tk.X, pady=2)
                ttk.Label(row, text=lbl, width=10).pack(side=tk.LEFT)
                pb = ttk.Progressbar(row, orient="horizontal", mode="determinate", maximum=mx, value=cnt)
                pb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
                ttk.Label(row, text=f"{cnt:,}", width=10, anchor="e").pack(side=tk.RIGHT)

        def _table(parent: ttk.Frame, title: str, cols: list[tuple[str, str, int]], rows: list[tuple[Any, ...]], height: int = 8) -> None:
            lf = ttk.Labelframe(parent, text=title, padding=10)
            lf.pack(fill=tk.BOTH, expand=True)
            if not rows:
                ttk.Label(lf, text="No data").pack(anchor="w")
                return
            wrap = ttk.Frame(lf)
            wrap.pack(fill=tk.BOTH, expand=True)
            col_ids = [c[0] for c in cols]
            tv = ttk.Treeview(wrap, columns=col_ids, show="headings", height=height)
            vs = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
            tv.configure(yscrollcommand=vs.set)
            tv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            vs.pack(side=tk.RIGHT, fill=tk.Y)
            for cid, heading, w in cols:
                tv.heading(cid, text=heading)
                tv.column(cid, width=w, anchor="w")
            for r in rows:
                tv.insert("", tk.END, values=tuple(r))
            _bind_mousewheel(tv)

        # --- KPI row ---
        kpi_row = ttk.Frame(inner)
        kpi_row.pack(fill=tk.X, pady=(0, 10))
        for i in range(4):
            kpi_row.columnconfigure(i, weight=1)

        ov = data.get("overview", {})
        _kpi(kpi_row, "Messages", ov.get("msg_count", "0"), f"Sent {ov.get('outgoing', '0')} · Recv {ov.get('incoming', '0')}").grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        _kpi(kpi_row, "Chats", ov.get("chat_count", "0"), f"Recipients {ov.get('recip_count', '0')}").grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        _kpi(kpi_row, "Attachments", ov.get("att_count", "0"), f"Resolved {ov.get('att_resolved', '0')} ({ov.get('att_pct', 'n/a')})").grid(row=0, column=2, sticky="nsew", padx=(0, 8))
        _kpi(kpi_row, "Database", ov.get("db_size", "?"), ov.get("db_name", "")).grid(row=0, column=3, sticky="nsew")

        # --- Date range + Extras ---
        mid = ttk.Frame(inner)
        mid.pack(fill=tk.X, pady=(0, 10))
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)

        dr = data.get("date_range")
        dr_lf = ttk.Labelframe(mid, text="Date Range", padding=10)
        dr_lf.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        if dr:
            ttk.Label(dr_lf, text=f"First: {dr.get('first','')}").pack(anchor="w")
            ttk.Label(dr_lf, text=f"Last:  {dr.get('last','')}").pack(anchor="w")
            if dr.get("span_days") is not None:
                ttk.Label(dr_lf, text=f"Span:  {dr.get('span_days'):,} days").pack(anchor="w")
                if dr.get("avg_per_day") is not None:
                    ttk.Label(dr_lf, text=f"Avg:   {dr.get('avg_per_day'):.1f} msgs/day").pack(anchor="w")
        else:
            ttk.Label(dr_lf, text="No dated messages found").pack(anchor="w")

        ex = data.get("extras")
        ex_lf = ttk.Labelframe(mid, text="Extras", padding=10)
        ex_lf.grid(row=0, column=1, sticky="nsew")
        if ex:
            for k, v in ex:
                ttk.Label(ex_lf, text=f"{k}: {v}").pack(anchor="w")
        else:
            ttk.Label(ex_lf, text="—").pack(anchor="w")

        # --- Activity bars ---
        bars = ttk.Frame(inner)
        bars.pack(fill=tk.X, pady=(0, 10))
        bars.columnconfigure(0, weight=1)
        bars.columnconfigure(1, weight=1)

        left = ttk.Frame(bars)
        right = ttk.Frame(bars)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew")
        _bar_list(left, "Busiest Months", data.get("monthly", []), max_rows=12)
        _bar_list(right, "Messages by Hour (Local)", data.get("hod", []), max_rows=None)

        bars2 = ttk.Frame(inner)
        bars2.pack(fill=tk.X, pady=(0, 10))
        bars2.columnconfigure(0, weight=1)
        bars2.columnconfigure(1, weight=1)
        left2 = ttk.Frame(bars2)
        right2 = ttk.Frame(bars2)
        left2.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right2.grid(row=0, column=1, sticky="nsew")
        _bar_list(left2, "Messages by Day of Week", data.get("dow", []), max_rows=7)
        _bar_list(right2, "Messages by Year", data.get("yearly", []), max_rows=None)

        # --- Tables ---
        _table(inner, "Top Conversations", [("name", "Name", 320), ("cnt", "Messages", 100), ("chat", "Chat ID", 80)], data.get("top_chats", []), height=10)
        _table(inner, "Attachments by Type", [("kind", "Kind", 120), ("cnt", "Files", 80), ("size", "Total Size", 120)], data.get("att_kinds", []), height=8)
        _table(inner, "Top MIME Types", [("mime", "MIME", 360), ("cnt", "Files", 80)], data.get("top_mime", []), height=8)
        _table(inner, "Largest Files", [("size", "Size", 100), ("kind", "Kind", 90), ("name", "Name", 420)], data.get("big_files", []), height=8)

        _table(inner, "Top Shared Domains", [("dom", "Domain", 420), ("cnt", "Count", 80)], data.get("top_domains", []), height=8)
        _table(inner, "Top Emojis", [("emoji", "Emoji", 120), ("cnt", "Count", 80)], data.get("top_emoji", []), height=6)

        # --- Word cloud ---
        wc = ttk.Labelframe(inner, text="Word Cloud", padding=10)
        wc.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        cloud = data.get("cloud_img")
        if cloud is not None:
            lbl = ttk.Label(wc, image=cloud)
            lbl.pack(anchor="center")
        else:
            ttk.Label(wc, text="No word cloud data").pack(anchor="w")

    def _generate_word_cloud(self, word_freq: list[tuple[str, int]], width: int = 800, height: int = 400) -> Optional[ImageTk.PhotoImage]:
        """Generate a word cloud image from (word, count) pairs using PIL only."""
        if not word_freq:
            return None
        try:
            bg = self.theme.get("app_bg", "#1e1e2e")
            # Parse bg to RGB tuple
            bg_rgb = tuple(int(bg.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            bg_rgb = (30, 30, 46)

        img = Image.new("RGB", (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)

        # Colour palette for words
        palette = [
            "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8", "#cba6f7",
            "#fab387", "#94e2d5", "#89dceb", "#f5c2e7", "#74c7ec",
            "#b4befe", "#eba0ac",
        ]

        max_cnt = word_freq[0][1] if word_freq else 1
        # Use up to 80 words
        words = word_freq[:80]

        import random as _rng
        _rng.seed(42)  # deterministic layout

        # Occupied rectangles for collision avoidance
        occupied: list[tuple[int, int, int, int]] = []

        def _overlaps(x1: int, y1: int, x2: int, y2: int) -> bool:
            for ox1, oy1, ox2, oy2 in occupied:
                if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
                    return True
            return False

        for idx, (word, cnt) in enumerate(words):
            # Font size proportional to frequency (log scale looks better)
            import math
            ratio = math.log1p(cnt) / math.log1p(max_cnt)
            font_size = max(12, int(10 + ratio * 48))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()

            color = palette[idx % len(palette)]
            color_rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

            bbox = font.getbbox(word)
            tw, th = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

            # Try random placements
            placed = False
            for _ in range(200):
                x = _rng.randint(4, max(5, width - tw - 4))
                y = _rng.randint(4, max(5, height - th - 4))
                if not _overlaps(x, y, x + tw + 4, y + th + 4):
                    draw.text((x, y), word, fill=color_rgb, font=font)
                    occupied.append((x - 2, y - 2, x + tw + 6, y + th + 6))
                    placed = True
                    break
            if not placed:
                break  # canvas full

        self._stats_cloud_pil = img  # keep PIL copy for export
        return ImageTk.PhotoImage(img)

    def _refresh_stats(self) -> None:
        if self._conn is None and not self._connect():
            return
        assert self._conn is not None
        self._stats_status.set("Loading stats…")
        self.root.update_idletasks()

        conn = self._conn
        lines: list[str] = []

        def _q1(sql: str) -> Any:
            try:
                return conn.execute(sql).fetchone()[0]
            except Exception:
                return None

        def _qall(sql: str) -> list:
            try:
                return conn.execute(sql).fetchall()
            except Exception:
                return []

        # --- Overview ---
        db_path = Path(self.db_path.get()).expanduser()
        db_size = _human_size(db_path.stat().st_size) if db_path.exists() else "?"
        msg_count = _q1("SELECT COUNT(*) FROM messages") or 0
        chat_count = _q1("SELECT COUNT(*) FROM chats") or 0
        att_count = _q1("SELECT COUNT(*) FROM attachments") or 0
        att_resolved = _q1("SELECT COUNT(*) FROM attachments WHERE abs_path != ''") or 0
        recip_count = _q1("SELECT COUNT(*) FROM recipients") or 0
        outgoing = _q1("SELECT COUNT(*) FROM messages WHERE outgoing = 1") or 0
        incoming = msg_count - outgoing

        lines.append("═══════════════ DATABASE OVERVIEW ═══════════════")
        lines.append(f"  Database file:       {db_path.name}  ({db_size})")
        lines.append(f"  Conversations:       {chat_count:,}")
        lines.append(f"  Recipients:          {recip_count:,}")
        lines.append(f"  Total messages:      {msg_count:,}")
        lines.append(f"    Sent (outgoing):   {outgoing:,}  ({outgoing*100/msg_count:.1f}%)" if msg_count else "    Sent: 0")
        lines.append(f"    Received (incoming):{incoming:,}  ({incoming*100/msg_count:.1f}%)" if msg_count else "    Received: 0")
        lines.append(f"  Total attachments:   {att_count:,}")
        att_pct = f"{att_resolved*100/att_count:.1f}%" if att_count else "n/a"
        lines.append(f"    Resolved (on disk): {att_resolved:,}  ({att_pct})")
        lines.append("")

        # --- Date range (LOCAL TIME) ---
        first_date = _q1("""
            SELECT MIN(datetime(dateSentMs / 1000, 'unixepoch', 'localtime'))
            FROM messages WHERE dateSentMs IS NOT NULL
        """)
        last_date = _q1("""
            SELECT MAX(datetime(dateSentMs / 1000, 'unixepoch', 'localtime'))
            FROM messages WHERE dateSentMs IS NOT NULL
        """)
        if first_date and last_date:
            lines.append("═══════════════ DATE RANGE ═══════════════")
            lines.append(f"  First message:  {str(first_date)[:19]}")
            lines.append(f"  Last message:   {str(last_date)[:19]}")
            try:
                d1 = datetime.fromisoformat(str(first_date))
                d2 = datetime.fromisoformat(str(last_date))
                span = (d2 - d1).days
                lines.append(f"  Span:           {span:,} days  ({span/365.25:.1f} years)")
                if msg_count and span > 0:
                    lines.append(f"  Avg msgs/day:   {msg_count/span:.1f}")
            except Exception:
                pass
            lines.append("")

        # --- Messages per month (top 12, LOCAL TIME) ---
        monthly = _qall("""
            SELECT strftime('%Y-%m', dateSentMs / 1000, 'unixepoch', 'localtime') AS month,
                   COUNT(*) AS cnt
            FROM messages WHERE dateSentMs IS NOT NULL
            GROUP BY month ORDER BY cnt DESC LIMIT 12
        """)
        if monthly:
            lines.append("═══════════════ BUSIEST MONTHS (top 12) ═══════════════")
            max_cnt = monthly[0][1] if monthly else 1
            for r in monthly:
                bar_len = int(r[1] / max_cnt * 30)
                lines.append(f"  {r[0]}  {'█' * bar_len}  {r[1]:,}")
            lines.append("")

        # --- Top 15 conversations by message count ---
        top_chats = _qall("""
            SELECT e.chatId, COALESCE(e.recipientName, '(unknown)') AS name,
                   COUNT(*) AS cnt
            FROM messages_enriched e
            GROUP BY e.chatId
            ORDER BY cnt DESC LIMIT 15
        """)
        if top_chats:
            lines.append("═══════════════ TOP 15 CONVERSATIONS ═══════════════")
            max_cnt = top_chats[0][2] if top_chats else 1
            for r in top_chats:
                bar_len = int(r[2] / max_cnt * 30)
                lines.append(f"  {r[1][:28]:<28}  {'█' * bar_len}  {r[2]:,}  (chat {r[0]})")
            lines.append("")

        # --- Attachment breakdown by kind ---
        att_kinds = _qall("""
            SELECT COALESCE(kind, '(none)') AS k, COUNT(*) AS cnt,
                   SUM(COALESCE(size_bytes, 0)) AS total_bytes
            FROM attachments GROUP BY k ORDER BY cnt DESC
        """)
        if att_kinds:
            lines.append("═══════════════ ATTACHMENTS BY TYPE ═══════════════")
            for r in att_kinds:
                sz = _human_size(r[2]) if r[2] else "0 B"
                lines.append(f"  {r[0]:<10}  {r[1]:>7,} files   {sz:>10}")
            total_size = sum(r[2] or 0 for r in att_kinds)
            lines.append(f"  {'TOTAL':<10}  {att_count:>7,} files   {_human_size(total_size):>10}")
            lines.append("")

        # --- Top 10 file types by count ---
        top_mime = _qall("""
            SELECT COALESCE(mime, '(unknown)') AS m, COUNT(*) AS cnt
            FROM attachments WHERE mime IS NOT NULL AND mime != ''
            GROUP BY m ORDER BY cnt DESC LIMIT 10
        """)
        if top_mime:
            lines.append("═══════════════ TOP 10 MIME TYPES ═══════════════")
            for r in top_mime:
                lines.append(f"  {r[0]:<40}  {r[1]:,}")
            lines.append("")

        # --- Largest attachments ---
        big_files = _qall("""
            SELECT file_name, size_bytes, kind, mime
            FROM attachments WHERE size_bytes IS NOT NULL
            ORDER BY size_bytes DESC LIMIT 10
        """)
        if big_files:
            lines.append("═══════════════ LARGEST FILES ═══════════════")
            for r in big_files:
                fn = (r[0] or "(unnamed)")[:40]
                lines.append(f"  {_human_size(r[1]):>10}  {r[2] or '?':<8}  {fn}")
            lines.append("")

        # --- Day of week distribution ---
        dow = _qall("""
            SELECT
                CASE CAST(strftime('%w', dateSentMs / 1000, 'unixepoch', 'localtime') AS INTEGER)
                    WHEN 0 THEN 'Sun' WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue'
                    WHEN 3 THEN 'Wed' WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri'
                    WHEN 6 THEN 'Sat' END AS day_name,
                COUNT(*) AS cnt
            FROM messages WHERE dateSentMs IS NOT NULL
            GROUP BY strftime('%w', dateSentMs / 1000, 'unixepoch', 'localtime')
            ORDER BY strftime('%w', dateSentMs / 1000, 'unixepoch', 'localtime')
        """)
        if dow:
            lines.append("═══════════════ MESSAGES BY DAY OF WEEK ═══════════════")
            max_cnt = max(r[1] for r in dow) if dow else 1
            for r in dow:
                bar_len = int(r[1] / max_cnt * 30)
                lines.append(f"  {r[0]}  {'█' * bar_len}  {r[1]:,}")
            lines.append("")

        # --- Hour of day distribution (LOCAL TIME) ---
        # Use epoch milliseconds and SQLite's 'localtime' modifier so this reflects
        # the user's machine timezone (including DST where applicable).
        hod = _qall("""
            SELECT CAST(strftime('%H', dateSentMs / 1000, 'unixepoch', 'localtime') AS INTEGER) AS hr,
                   COUNT(*) AS cnt
            FROM messages WHERE dateSentMs IS NOT NULL
            GROUP BY hr ORDER BY hr
        """)
        if hod:
            lines.append("═══════════════ MESSAGES BY HOUR (LOCAL) ═══════════════")
            max_cnt = max(r[1] for r in hod) if hod else 1
            for r in hod:
                bar_len = int(r[1] / max_cnt * 30)
                lines.append(f"  {r[0]:02d}:00  {'█' * bar_len}  {r[1]:,}")
            lines.append("")

        # --- Messages with links ---
        if self._has_link_col:
            link_count = _q1("SELECT COUNT(*) FROM messages WHERE has_link = 1") or 0
        else:
            link_count = _q1("SELECT COUNT(*) FROM messages WHERE body LIKE '%http://%' OR body LIKE '%https://%'") or 0
        if msg_count:
            lines.append("═══════════════ EXTRAS ═══════════════")
            lines.append(f"  Messages containing links:  {link_count:,}  ({link_count*100/msg_count:.1f}%)")
            msgs_with_att = _q1("SELECT COUNT(DISTINCT message_rowid) FROM attachments") or 0
            lines.append(f"  Messages with attachments:  {msgs_with_att:,}  ({msgs_with_att*100/msg_count:.1f}%)")
            avg_len = _q1("SELECT AVG(LENGTH(body)) FROM messages WHERE body IS NOT NULL AND body != ''")
            if avg_len:
                lines.append(f"  Avg message length:         {avg_len:.0f} chars")
            longest = _q1("SELECT MAX(LENGTH(body)) FROM messages")
            if longest:
                lines.append(f"  Longest message:            {longest:,} chars")
            empty = _q1("SELECT COUNT(*) FROM messages WHERE body IS NULL OR body = ''") or 0
            lines.append(f"  Empty messages (media-only): {empty:,}  ({empty*100/msg_count:.1f}%)")
            lines.append("")

        # --- Who texts first (conversation openers) ---
        # A message is a "conversation opener" if it's the first message in a chat after 4+ hour gap
        try:
            opener_rows = _qall("""
                WITH ordered AS (
                    SELECT chatId, outgoing, dateSentMs,
                           LAG(dateSentMs) OVER (PARTITION BY chatId ORDER BY dateSentMs) AS prev_ms
                    FROM messages WHERE dateSentMs IS NOT NULL
                )
                SELECT
                    CASE WHEN outgoing = 1 THEN 'You' ELSE 'Them' END AS who,
                    COUNT(*) AS cnt
                FROM ordered
                WHERE prev_ms IS NULL OR (dateSentMs - prev_ms) > 14400000
                GROUP BY outgoing
            """)
            if opener_rows:
                lines.append("═══════════════ WHO TEXTS FIRST ═══════════════")
                lines.append("  (first message after 4+ hour gap)")
                total_openers = sum(r[1] for r in opener_rows)
                for r in opener_rows:
                    pct = r[1] * 100 / total_openers if total_openers else 0
                    lines.append(f"  {r[0]:<6}  {r[1]:,}  ({pct:.1f}%)")
                lines.append("")
        except Exception:
            pass

        # --- Longest streaks (consecutive days with messages) ---
        try:
            day_rows = _qall("""
                SELECT DISTINCT date(dateSentMs / 1000, 'unixepoch', 'localtime') AS d
                FROM messages WHERE dateSentMs IS NOT NULL
                ORDER BY d
            """)
            if day_rows:
                from datetime import timedelta
                days_list = [datetime.fromisoformat(r[0]).date() for r in day_rows if r[0]]
                best_streak = cur_streak = 1
                best_start = cur_start = days_list[0] if days_list else None
                best_end = cur_end = best_start
                for i in range(1, len(days_list)):
                    if (days_list[i] - days_list[i - 1]).days == 1:
                        cur_streak += 1
                        cur_end = days_list[i]
                    else:
                        if cur_streak > best_streak:
                            best_streak = cur_streak
                            best_start = cur_start
                            best_end = cur_end
                        cur_streak = 1
                        cur_start = days_list[i]
                        cur_end = days_list[i]
                if cur_streak > best_streak:
                    best_streak = cur_streak
                    best_start = cur_start
                    best_end = cur_end
                lines.append("═══════════════ STREAKS & ACTIVITY ═══════════════")
                lines.append(f"  Days with messages:  {len(days_list):,}")
                lines.append(f"  Longest streak:      {best_streak} consecutive days")
                if best_start and best_end:
                    lines.append(f"    ({best_start.isoformat()} → {best_end.isoformat()})")
                lines.append("")
        except Exception:
            pass

        # --- Single-pass body scan for domains, emoji, and word counts ---
        # (Avoids 3-4 separate full table scans of messages)
        domain_counts: dict[str, int] = {}
        emoji_counts: dict[str, int] = {}
        your_word_counts: dict[str, int] = {}
        their_word_counts: dict[str, int] = {}
        top_combined: list[tuple[str, int]] = []
        try:
            url_re = re.compile(r'https?://([^/\s?#]+)')
            emoji_re = re.compile(
                '['
                '\U0001F600-\U0001F64F'  # emoticons
                '\U0001F300-\U0001F5FF'  # symbols & pictographs
                '\U0001F680-\U0001F6FF'  # transport
                '\U0001F1E0-\U0001F1FF'  # flags
                '\U00002702-\U000027B0'
                '\U0001F900-\U0001F9FF'  # supplemental
                '\U0001FA00-\U0001FA6F'  # chess/extended-A
                '\U0001FA70-\U0001FAFF'  # extended-A cont
                '\U00002600-\U000026FF'  # misc
                '\U0000FE00-\U0000FE0F'  # variation selectors
                '\U0000200D'             # ZWJ
                '\U00002764'             # heart
                ']+'
            )
            word_re = re.compile(r"[a-zA-Z'\u2019]+")

            cursor = conn.execute("SELECT body, outgoing FROM messages WHERE body IS NOT NULL AND body != ''")
            while True:
                batch = cursor.fetchmany(2000)
                if not batch:
                    break
                for body_text, outgoing in batch:
                    # URLs / domains
                    for m in url_re.finditer(body_text):
                        dom = m.group(1).lower()
                        domain_counts[dom] = domain_counts.get(dom, 0) + 1
                    # Emojis
                    for m in emoji_re.finditer(body_text):
                        e = m.group()
                        emoji_counts[e] = emoji_counts.get(e, 0) + 1
                    # Words (split by outgoing)
                    tgt = your_word_counts if outgoing else their_word_counts
                    for w in word_re.findall(body_text):
                        wl = w.lower().replace("'", "").replace("\u2019", "")
                        if len(wl) >= 3 and wl not in _STOP_WORDS:
                            tgt[wl] = tgt.get(wl, 0) + 1
        except Exception:
            pass

        # --- Top shared domains ---
        if domain_counts:
            top_domains = sorted(domain_counts.items(), key=lambda x: -x[1])[:15]
            if top_domains:
                lines.append("═══════════════ TOP SHARED DOMAINS ═══════════════")
                for dom, cnt in top_domains:
                    lines.append(f"  {dom:<40}  {cnt:,}")
                lines.append("")

        # --- Emoji usage ---
        if emoji_counts:
            top_emoji = sorted(emoji_counts.items(), key=lambda x: -x[1])[:20]
            if top_emoji:
                lines.append("═══════════════ TOP 20 EMOJIS ═══════════════")
                total_emoji = sum(emoji_counts.values())
                lines.append(f"  Unique emojis used: {len(emoji_counts):,}   Total emoji uses: {total_emoji:,}")
                for i in range(0, len(top_emoji), 5):
                    chunk = top_emoji[i:i+5]
                    parts = [f"{e} ×{c:,}" for e, c in chunk]
                    lines.append("  " + "   ".join(parts))
                lines.append("")

        # --- Most used words (split: You / Them / Combined) ---
        try:
            def _render_top(counts, n=40):
                top = sorted(counts.items(), key=lambda x: -x[1])[:n]
                if not top:
                    return top
                mx = top[0][1]
                for w, c in top:
                    bar_len = int(c / mx * 25)
                    lines.append(f"  {w:<20}  {'█' * bar_len}  {c:,}")
                return top

            # Combined (for word cloud)
            all_counts: dict[str, int] = {}
            for wl, c in your_word_counts.items():
                all_counts[wl] = all_counts.get(wl, 0) + c
            for wl, c in their_word_counts.items():
                all_counts[wl] = all_counts.get(wl, 0) + c

            if your_word_counts:
                lines.append("═══════════════ YOUR MOST USED WORDS (top 40) ═══════════════")
                _render_top(your_word_counts)
                lines.append("")

            if their_word_counts:
                lines.append("═══════════════ THEIR MOST USED WORDS (top 40) ═══════════════")
                _render_top(their_word_counts)
                lines.append("")

            top_combined = sorted(all_counts.items(), key=lambda x: -x[1])[:40]
            if top_combined:
                lines.append("═══════════════ COMBINED MOST USED WORDS (top 40) ═══════════════")
                mx = top_combined[0][1]
                for w, c in top_combined:
                    bar_len = int(c / mx * 25)
                    lines.append(f"  {w:<20}  {'█' * bar_len}  {c:,}")
                lines.append("")

                # --- Word Cloud (combined) ---
                lines.append("═══════════════ WORD CLOUD ═══════════════")
                lines.append("")
                lines.append("<<WORDCLOUD>>")
                lines.append("")
        except Exception:
            pass

        # --- Yearly message volume (LOCAL TIME) ---
        yearly = _qall("""
            SELECT strftime('%Y', dateSentMs / 1000, 'unixepoch', 'localtime') AS yr,
                   COUNT(*) AS cnt
            FROM messages WHERE dateSentMs IS NOT NULL
            GROUP BY yr ORDER BY yr
        """)
        if yearly and len(yearly) > 1:
            lines.append("═══════════════ MESSAGES BY YEAR ═══════════════")
            max_cnt = max(r[1] for r in yearly)
            for r in yearly:
                bar_len = int(r[1] / max_cnt * 30) if max_cnt else 0
                lines.append(f"  {r[0]}  {'█' * bar_len}  {r[1]:,}")
            lines.append("")

        report = "\n".join(lines)

        # Keep text report for export (Text widget previously omitted images anyway)
        self._stats_report_text = report.replace("<<WORDCLOUD>>", "").strip()

        # Prepare dashboard data
        data: Dict[str, Any] = {
            "overview": {
                "db_name": db_path.name,
                "db_size": db_size,
                "msg_count": f"{msg_count:,}",
                "chat_count": f"{chat_count:,}",
                "recip_count": f"{recip_count:,}",
                "att_count": f"{att_count:,}",
                "att_resolved": f"{att_resolved:,}",
                "att_pct": att_pct,
                "outgoing": f"{outgoing:,}",
                "incoming": f"{incoming:,}",
            },
            "monthly": [(r[0], int(r[1])) for r in monthly] if monthly else [],
            "dow": [(r[0], int(r[1])) for r in dow] if dow else [],
            "hod": [(f"{int(r[0]):02d}:00", int(r[1])) for r in hod] if hod else [],
            "yearly": [(r[0], int(r[1])) for r in yearly] if yearly else [],
            "top_chats": [(r[1], int(r[2]), int(r[0])) for r in top_chats] if top_chats else [],
            "att_kinds": [(r[0], int(r[1]), _human_size(r[2] or 0)) for r in att_kinds] if att_kinds else [],
            "top_mime": [(r[0], int(r[1])) for r in top_mime] if top_mime else [],
            "big_files": [(_human_size(r[1] or 0), r[2] or "?", (r[0] or "(unnamed)")) for r in big_files] if big_files else [],
            "top_domains": [(d, int(c)) for d, c in (sorted(domain_counts.items(), key=lambda x: -x[1])[:15] if domain_counts else [])],
            "top_emoji": [(e, int(c)) for e, c in (sorted(emoji_counts.items(), key=lambda x: -x[1])[:20] if emoji_counts else [])],
        }

        if first_date and last_date:
            dr: Dict[str, Any] = {"first": str(first_date)[:19], "last": str(last_date)[:19]}
            try:
                d1 = datetime.fromisoformat(str(first_date))
                d2 = datetime.fromisoformat(str(last_date))
                span_days = (d2 - d1).days
                dr["span_days"] = span_days
                if msg_count and span_days > 0:
                    dr["avg_per_day"] = msg_count / span_days
            except Exception:
                pass
            data["date_range"] = dr

        extras_pairs: list[tuple[str, str]] = []
        if msg_count:
            try:
                extras_pairs.append(("Links", f"{link_count:,} ({(link_count * 100 / msg_count):.1f}%)"))
            except Exception:
                extras_pairs.append(("Links", f"{link_count:,}"))
            msgs_with_att = _q1("SELECT COUNT(DISTINCT message_rowid) FROM attachments") or 0
            try:
                extras_pairs.append(("Msgs w/ attachments", f"{msgs_with_att:,} ({(msgs_with_att * 100 / msg_count):.1f}%)"))
            except Exception:
                extras_pairs.append(("Msgs w/ attachments", f"{msgs_with_att:,}"))
            if opener_rows:
                total_openers = sum(r[1] for r in opener_rows)
                for who, cnt in opener_rows:
                    pct = cnt * 100 / total_openers if total_openers else 0
                    extras_pairs.append((f"Opener: {who}", f"{cnt:,} ({pct:.1f}%)"))
            data["extras"] = extras_pairs

        # Word cloud image for dashboard
        cloud_img = None
        try:
            cloud_img = self._generate_word_cloud(top_combined, width=900, height=420)
            if cloud_img:
                self._stats_cloud_ref = cloud_img
        except Exception:
            cloud_img = None
        data["cloud_img"] = cloud_img

        self._render_stats_dashboard(data)
        self._stats_status.set(f"Stats loaded — {msg_count:,} messages, {chat_count:,} chats")

    # ---------- Export Stats ----------

    def _export_stats(self) -> None:
        """Export the current Stats tab content as HTML or plain text."""
        txt = (getattr(self, "_stats_report_text", "") or "").strip()
        if not txt:
            messagebox.showwarning("No stats", "Refresh Stats first.")
            return

        p = filedialog.asksaveasfilename(
            title="Export Stats",
            defaultextension=".html",
            filetypes=[("HTML", "*.html"), ("Text", "*.txt")],
        )
        if not p:
            return

        is_html = p.lower().endswith(".html") or p.lower().endswith(".htm")

        if is_html:
            self._export_stats_html(p, txt)
        else:
            self._export_stats_txt(p, txt)

    def _export_stats_txt(self, path: str, txt: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            self._stats_status.set(f"Stats exported to {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _export_stats_html(self, path: str, txt: str) -> None:
        th = self.theme
        bg = th.get("text_bg", "#ffffff")
        fg = th.get("text_fg", "#000000")
        accent = th.get("accent", fg)
        font = th.get("mono_family", "Consolas")
        size = th.get("mono_size", 10)

        cloud_b64 = ""

        # Try to re-generate word cloud as PNG bytes for the export
        cloud_html = ""
        try:
            from io import BytesIO as _BytesIO
            import base64 as _b64
            # Rebuild cloud from the _generate_word_cloud method internals
            # The simplest way: just grab the word cloud data from the text widget
            # Actually re-generate if we have the combined word data
            # We stored top_combined on the instance if available
            if hasattr(self, '_stats_cloud_pil') and self._stats_cloud_pil is not None:
                buf = _BytesIO()
                self._stats_cloud_pil.save(buf, format="PNG")
                cloud_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
                cloud_html = f'<div style="text-align:center;margin:16px 0"><img src="data:image/png;base64,{cloud_b64}" style="max-width:100%" alt="Word Cloud"></div>'
        except Exception:
            pass

        import html as _html

        # Convert text lines into themed HTML
        html_lines: list[str] = []
        for line in txt.split("\n"):
            if line.startswith("═"):
                # Section header
                title = line.replace("═", "").strip()
                html_lines.append(f'<h2 style="color:{accent};border-bottom:2px solid {accent};padding-bottom:4px;margin-top:24px">{_html.escape(title)}</h2>')
            else:
                html_lines.append(f'<pre style="margin:0;line-height:1.5">{_html.escape(line)}</pre>')

        # Insert word cloud image after the "WORD CLOUD" header
        if cloud_html:
            for i, hl in enumerate(html_lines):
                if "WORD CLOUD" in hl:
                    html_lines.insert(i + 1, cloud_html)
                    break

        body = "\n".join(html_lines)

        doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Signal Stats Export</title>
<style>
  body {{
    background: {bg};
    color: {fg};
    font-family: '{font}', 'Consolas', 'Courier New', monospace;
    font-size: {size}pt;
    padding: 24px;
    max-width: 960px;
    margin: 0 auto;
  }}
  h2 {{ font-size: {size + 3}pt; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; }}
</style>
</head>
<body>
<h1 style="color:{accent}">Signal Export — Stats</h1>
{body}
</body>
</html>"""

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc)
            self._stats_status.set(f"Stats exported to {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc))

    def _export_search_csv(self) -> None:
        if not self._search_rows:
            messagebox.showwarning("No results", "Run a search first.")
            return
        p = filedialog.asksaveasfilename(title="Export CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        fields = ["rowid", "chatId", "dateSentIso", "recipientName", "dir", "body", "att_names"]
        try:
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in self._search_rows:
                    row = {k: r.get(k) for k in fields}
                    # Export dates in local time.
                    row["dateSentIso"] = _fmt_ts_local_iso(r.get("dateSentIso", ""), r.get("dateSentMs"))
                    w.writerow(row)
            self.export_status.set(f"Saved: {p}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_search_html(self) -> None:
        if not self._search_rows:
            messagebox.showwarning("No results", "Run a search first.")
            return
        p = filedialog.asksaveasfilename(title="Export HTML", defaultextension=".html", filetypes=[("HTML", "*.html")])
        if not p:
            return

        def esc(x: Any) -> str:
            return html.escape(str(x or ""))

        rows = self._search_rows[:5000]
        out = ["<!doctype html><meta charset='utf-8'><title>Signal Search Export</title>"]
        bg = self.theme.get("text_bg", "#ffffff")
        fg = self.theme.get("text_fg", "#000000")
        panel_bg = self.theme.get("panel_bg", bg)
        meta = self.theme.get("meta_fg", "#666666")
        link = self.theme.get("link_fg", "#8a2be2")
        border = self.theme.get("widget_bg", "#dddddd")
        sel_bg = self.theme.get("select_bg", "#0b5cad")
        sel_fg = self.theme.get("select_fg", "#ffffff")

        out.append(
            "<style>"
            ":root{"
            f"--bg:{bg};--fg:{fg};--panel:{panel_bg};--meta:{meta};--link:{link};--border:{border};--selbg:{sel_bg};--selfg:{sel_fg};"
            "}"
            "body{font-family:system-ui,Segoe UI,Arial;margin:20px;background:var(--bg);color:var(--fg)}"
            "a{color:var(--link)}"
            "table{border-collapse:collapse;width:100%}"
            "th,td{border:1px solid var(--border);padding:6px;font-size:12px;vertical-align:top}"
            "th{background:var(--panel)}"
            "tr:nth-child(even) td{background:var(--panel)}"
            "::selection{background:var(--selbg);color:var(--selfg)}"
            "</style>"
        )
        out.append(f"<h2>Search export</h2><p>Query: <b>{esc(self.q.get())}</b> Results: {len(self._search_rows)}</p>")
        out.append("<table><tr><th>Date</th><th>Recipient</th><th>Dir</th><th>Text</th><th>Attachments</th><th>chatId</th><th>rowid</th></tr>")
        for r in rows:
            dt_local = _fmt_ts_local_iso(r.get("dateSentIso", ""), r.get("dateSentMs"))
            out.append("<tr>"
                       f"<td>{esc(dt_local)}</td>"
                       f"<td>{esc(r.get('recipientName'))}</td>"
                       f"<td>{esc(r.get('dir'))}</td>"
                       f"<td>{esc(r.get('body'))}</td>"
                       f"<td>{esc(r.get('att_names'))}</td>"
                       f"<td>{esc(r.get('chatId'))}</td>"
                       f"<td>{esc(r.get('rowid'))}</td>"
                       "</tr>")
        out.append("</table>")
        try:
            Path(p).write_text("\n".join(out), encoding="utf-8")
            self.export_status.set(f"Saved: {p}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_thread_html(self, chat_id: Optional[int]) -> None:
        if self._conn is None and not self._connect():
            return
        if not chat_id:
            messagebox.showwarning("Missing chatId", "Enter a chatId.")
            return
        assert self._conn is not None
        p = filedialog.asksaveasfilename(title="Export Thread HTML", defaultextension=".html", filetypes=[("HTML", "*.html")])
        if not p:
            return

        recipient = self._conn.execute("SELECT COALESCE(r.name,'(unknown)') AS name FROM chats c LEFT JOIN recipients r ON r.id=c.recipientId WHERE c.id=?", (chat_id,)).fetchone()
        rname = recipient["name"] if recipient else "(unknown)"

        msgs = self._conn.execute(
            "SELECT m.rowid, m.dateSentIso, m.dateSentMs, m.outgoing, COALESCE(m.body,'') AS body FROM messages m WHERE m.chatId=? ORDER BY m.dateSentMs ASC, m.rowid ASC",
            (chat_id,),
        ).fetchall()

        def esc(x: Any) -> str:
            return html.escape(str(x or ""))

        out = ["<!doctype html><meta charset='utf-8'>", f"<title>Thread Export {esc(rname)}</title>"]
        bg = self.theme.get("text_bg", "#ffffff")
        fg = self.theme.get("text_fg", "#000000")
        meta = self.theme.get("meta_fg", "#666666")
        you = self.theme.get("you_fg", "#0b5cad")
        them = self.theme.get("them_fg", "#1a7f37")
        link = self.theme.get("link_fg", "#8a2be2")
        border = self.theme.get("widget_bg", "#eeeeee")
        panel = self.theme.get("panel_bg", bg)
        sel_bg = self.theme.get("select_bg", "#0b5cad")
        sel_fg = self.theme.get("select_fg", "#ffffff")

        out.append(
            "<style>"
            ":root{"
            f"--bg:{bg};--fg:{fg};--panel:{panel};--meta:{meta};--you:{you};--them:{them};--link:{link};--border:{border};--selbg:{sel_bg};--selfg:{sel_fg};"
            "}"
            "body{font-family:system-ui,Segoe UI,Arial;margin:20px;background:var(--bg);color:var(--fg)}"
            "a{color:var(--link)}"
            ".msg{padding:10px;border-bottom:1px solid var(--border)}"
            ".meta{color:var(--meta);font-size:12px}"
            ".you{color:var(--you)}"
            ".them{color:var(--them)}"
            ".att{margin-top:6px;font-size:12px}"
            ".thumb{max-width:320px;max-height:240px;border-radius:4px;margin:4px 0}"
            "code{white-space:pre-wrap}"
            "::selection{background:var(--selbg);color:var(--selfg)}"
            "</style>"
        )
        out.append(f"<h2>Thread: {esc(rname)} (chatId={chat_id})</h2>")
        msg_ids = [int(m["rowid"]) for m in msgs]
        atts_by_msg = self._batch_fetch_attachments(msg_ids)
        authors = self._speaker_names_for_message_ids(msg_ids)

        for m in msgs:
            mid = int(m["rowid"])
            who = authors.get(mid) or ("You" if m["outgoing"] else rname)
            cls = "you" if m["outgoing"] else "them"
            ts_local = _fmt_ts_local_iso(m["dateSentIso"] or "", m["dateSentMs"])
            out.append(f"<div class='msg'><div class='meta'>{esc(ts_local)} <span class='{cls}'>{who}</span></div><div><code>{esc(m['body'])}</code></div>")
            atts = atts_by_msg.get(int(m["rowid"]), [])
            if atts:
                out.append("<div class='att'>")
                for a in atts:
                    ap = a["abs_path"] or ""
                    fn = a["file_name"] or "(file)"
                    kind = a["kind"] or "file"
                    href = ("file:///" + ap.replace("\\", "/")) if ap else ""

                    # Metadata line
                    meta_parts = [f"[{esc(kind)}]"]
                    if a["mime"]:
                        meta_parts.append(esc(a["mime"]))
                    if a["size_bytes"]:
                        meta_parts.append(_human_size(a["size_bytes"]))
                    if a["width"] and a["height"]:
                        meta_parts.append(f"{a['width']}x{a['height']}")
                    if a["duration_ms"]:
                        meta_parts.append(self._format_duration(a["duration_ms"]))
                    meta_line = " &middot; ".join(meta_parts)

                    # Inline base64 thumbnail for images
                    thumb_html = ""
                    if kind in ("image", "video", "audio") and ap and os.path.exists(ap):
                        try:
                            cache_dir = self._thumb_cache_dir()
                            st = os.stat(ap)
                            key_src = f"{ap}|{st.st_mtime_ns}|{st.st_size}|{kind}|320x240".encode("utf-8", errors="ignore")
                            key = hashlib.sha1(key_src).hexdigest()
                            cache_img = cache_dir / f"{kind}_{key}.jpg"
                            if cache_img.exists() and cache_img.stat().st_size > 0:
                                b64 = _b64.b64encode(cache_img.read_bytes()).decode("ascii")
                                thumb_html = f"<br><img class='thumb' src='data:image/jpeg;base64,{b64}' alt='{esc(fn)}'/>"
                            elif kind == "image":
                                # Generate thumbnail on the fly
                                im = Image.open(ap)
                                im = im.convert("RGB")
                                im.thumbnail((320, 240))
                                buf = BytesIO()
                                im.save(buf, "JPEG", quality=80)
                                im.save(buf, "JPEG", quality=80)
                                b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
                                thumb_html = f"<br><img class='thumb' src='data:image/jpeg;base64,{b64}' alt='{esc(fn)}'/>"
                        except Exception:
                            pass

                    out.append(f"<div>{thumb_html}<br><a href='{esc(href)}'>{esc(fn)}</a> <span class='meta'>{meta_line}</span></div>")
                out.append("</div>")
            out.append("</div>")

        try:
            Path(p).write_text("\n".join(out), encoding="utf-8")
            self.export_status.set(f"Saved: {p}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    def _export_thread_md(self, chat_id: Optional[int]) -> None:
        if self._conn is None and not self._connect():
            return
        if not chat_id:
            messagebox.showwarning("Missing chatId", "Enter a chatId.")
            return
        assert self._conn is not None
        p = filedialog.asksaveasfilename(title="Export Thread Markdown", defaultextension=".md", filetypes=[("Markdown", "*.md")])
        if not p:
            return

        recipient = self._conn.execute("SELECT COALESCE(r.name,'(unknown)') AS name FROM chats c LEFT JOIN recipients r ON r.id=c.recipientId WHERE c.id=?", (chat_id,)).fetchone()
        rname = recipient["name"] if recipient else "(unknown)"

        msgs = self._conn.execute(
            "SELECT m.rowid, m.dateSentIso, m.dateSentMs, m.outgoing, COALESCE(m.body,'') AS body FROM messages m WHERE m.chatId=? ORDER BY m.dateSentMs ASC, m.rowid ASC",
            (chat_id,),
        ).fetchall()

        lines = [f"# Thread: {rname} (chatId={chat_id})", ""]
        msg_ids = [int(m["rowid"]) for m in msgs]
        atts_by_msg = self._batch_fetch_attachments(msg_ids)
        authors = self._speaker_names_for_message_ids(msg_ids)
        for m in msgs:
            mid = int(m["rowid"])
            who = authors.get(mid) or ("You" if m["outgoing"] else rname)
            ts_local = _fmt_ts_local_iso(m["dateSentIso"] or "", m["dateSentMs"])
            lines.append(f"## {ts_local} {who}")
            if m["body"]:
                lines.append(m["body"])
            atts = atts_by_msg.get(int(m["rowid"]), [])
            if atts:
                lines.append("")
                lines.append("**Attachments:**")
                for a in atts:
                    ap = a["abs_path"] or ""
                    fn = a["file_name"] or "(file)"
                    kind = a["kind"] or "file"
                    meta_parts = [kind]
                    if a["mime"]:
                        meta_parts.append(a["mime"])
                    if a["size_bytes"]:
                        meta_parts.append(_human_size(a["size_bytes"]))
                    if a["width"] and a["height"]:
                        meta_parts.append(f"{a['width']}x{a['height']}")
                    if a["duration_ms"]:
                        meta_parts.append(self._format_duration(a["duration_ms"]))
                    meta_str = " · ".join(meta_parts)
                    lines.append(f"- {fn} ({meta_str}) `{ap}`")
            lines.append("")
        try:
            Path(p).write_text("\n".join(lines), encoding="utf-8")
            self.export_status.set(f"Saved: {p}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
