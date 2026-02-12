"""GUI/DB helper functions for Signal Export Browser.

This module holds the pure helpers (formatting, parsing, color math, small
utilities) so `signal_gui.py` can focus on the Tkinter application code.

Naming is kept compatible with the existing codebase.
"""

from __future__ import annotations

import os
import re
import sys
import webbrowser
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

import sqlite3
import tkinter as tk


def _normalize_date(s: str) -> str | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt.date().isoformat()
    except Exception:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return s
    return None


def _fmt_ts_short(iso: str) -> str:
    """Format an ISO timestamp as YYMMDD-HH:MM:SS in local time (24h)."""
    s = (iso or "").strip()
    if not s:
        return ""
    try:
        # Support common forms: ...+00:00 or ...Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%y%m%d-%H:%M:%S")
    except Exception:
        return iso


def _fmt_ts_local_iso(iso: str, ms: Any = None) -> str:
    """Format a message timestamp as a local-time ISO string (seconds precision).

    Prefers *ms* (epoch milliseconds) when provided; falls back to parsing *iso*.
    """
    if ms is not None:
        try:
            dt = datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).astimezone()
            return dt.isoformat(timespec="seconds")
        except Exception:
            pass

    s = (iso or "").strip()
    if not s:
        return ""
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.isoformat(timespec="seconds") if hasattr(dt, "isoformat") else str(dt)
    except Exception:
        return iso


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_test USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_test")
        return True
    except sqlite3.OperationalError:
        return False


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,),
    ).fetchone()
    return r is not None


def _safe_open_path(path: str) -> None:
    if not path:
        return
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", path])
            return
        import subprocess

        subprocess.Popen(["xdg-open", path])
        return
    except Exception:
        pass
    try:
        webbrowser.open(f"file:///{path}")
    except Exception:
        pass


def _bind_mousewheel(widget: tk.Widget, scroll_widget: Optional[tk.Widget] = None, speed: int = 1) -> None:
    """Enable mouse wheel scrolling when the cursor is over widget."""

    target = scroll_widget or widget

    def _yview_scroll(delta_lines: int) -> None:
        try:
            target.yview_scroll(delta_lines, "units")  # type: ignore[attr-defined]
        except Exception:
            pass

    def on_mousewheel(event) -> str:
        delta = getattr(event, "delta", 0) or 0
        if delta:
            step = -1 if delta > 0 else 1
            lines = step * max(1, int(abs(delta) / 120)) * speed
            _yview_scroll(lines)
            return "break"

        num = getattr(event, "num", None)
        if num == 4:
            _yview_scroll(-1 * speed)
            return "break"
        if num == 5:
            _yview_scroll(1 * speed)
            return "break"
        return "break"

    widget.bind("<MouseWheel>", on_mousewheel)
    widget.bind("<Button-4>", on_mousewheel)
    widget.bind("<Button-5>", on_mousewheel)


def _human_size(n: Optional[int]) -> str:
    if not n:
        return ""
    n = int(n)
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n/1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n/1024/1024:.1f} MB"
    return f"{n/1024/1024/1024:.1f} GB"


def _fts_terms(q: str) -> List[str]:
    q = q.strip()
    if not q:
        return []
    parts = re.findall(r'"([^"]+)"|(\S+)', q)
    tokens: List[str] = []
    for a, b in parts:
        t = a or b
        if not t:
            continue
        t = t.strip()
        if t.upper() in ("OR", "AND", "NOT", "NEAR"):
            continue
        t = re.sub(r"[^0-9A-Za-z_]+", "", t)
        if t:
            tokens.append(t)
    seen = set()
    out: List[str] = []
    for t in tokens:
        tl = t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        out.append(t)
    return out[:20]


_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

# Matches http / https URLs in message bodies.
_URL_RE = re.compile(r'https?://[^\s<>\"\')\]]+')


def _insert_body_with_links(
    txt: "tk.Text",
    body: str,
    link_idx_ref: List[int],
    link_tag: str = "link",
) -> None:
    """Insert *body* into *txt*, turning URLs into clickable hyperlinks."""

    last = 0
    for m in _URL_RE.finditer(body):
        if m.start() > last:
            txt.insert(tk.END, body[last : m.start()])
        url = m.group(0)
        while url and url[-1] in (".", ",", ";", ":", "!", "?", ")", "]"):
            url = url[:-1]
        s = txt.index(tk.END)
        txt.insert(tk.END, url)
        e = txt.index(tk.END)
        txt.tag_add(link_tag, s, e)
        tag = f"{link_tag}_{link_idx_ref[0]}"
        link_idx_ref[0] += 1
        txt.tag_add(tag, s, e)
        txt.tag_bind(tag, "<Button-1>", lambda _e, u=url: webbrowser.open(u))
        stripped = body[m.start() + len(url) : m.end()]
        if stripped:
            txt.insert(tk.END, stripped)
        last = m.end()
    if last < len(body):
        txt.insert(tk.END, body[last:])


def _is_hex_color(s: Any) -> bool:
    return isinstance(s, str) and bool(_HEX_COLOR_RE.match(s.strip()))


def _hex_to_rgb(hexv: str) -> Tuple[int, int, int]:
    h = hexv.strip().lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance(hexv: str) -> float:
    r, g, b = _hex_to_rgb(hexv)
    rl = _srgb_to_linear(r)
    gl = _srgb_to_linear(g)
    bl = _srgb_to_linear(b)
    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl


def _contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    l1 = _relative_luminance(fg_hex)
    l2 = _relative_luminance(bg_hex)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _shift_toward(hexv: str, toward: str, amount: float) -> str:
    amount = max(0.0, min(1.0, float(amount)))
    r, g, b = _hex_to_rgb(hexv)
    tr, tg, tb = _hex_to_rgb(toward)
    nr = int(round(r + (tr - r) * amount))
    ng = int(round(g + (tg - g) * amount))
    nb = int(round(b + (tb - b) * amount))
    return _rgb_to_hex(nr, ng, nb)


def _lift_or_drop(hexv: str, amount: float) -> str:
    try:
        lum = _relative_luminance(hexv)
    except Exception:
        return hexv
    if lum < 0.5:
        return _shift_toward(hexv, "#ffffff", amount)
    return _shift_toward(hexv, "#000000", amount)


def _best_bw_for_bg(bg_hex: str) -> str:
    try:
        c_black = _contrast_ratio("#000000", bg_hex)
        c_white = _contrast_ratio("#ffffff", bg_hex)
        return "#000000" if c_black >= c_white else "#ffffff"
    except Exception:
        return "#000000"
