"""Theming support for Signal Export Browser.

This module extracts theming-related methods from the main GUI so
`signal_gui.py` can stay focused on the application flow and UI.

The code is intentionally kept close to the original (same method names) to
minimize refactor risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import tkinter.font as tkfont

from gui_utils import (
    _best_bw_for_bg,
    _bind_mousewheel,
    _contrast_ratio,
    _is_hex_color,
    _lift_or_drop,
)


class ThemeSupport:
    # --- Attributes provided by App (typing-only; set by App.__init__) ---
    root: tk.Tk
    theme: Dict[str, Any]
    active_theme_name: str
    custom_themes: Dict[str, Dict[str, Any]]
    _themes_path: Path
    _style: ttk.Style
    _menubar: Optional[tk.Menu]
    _settings_menu: Optional[tk.Menu]
    preview_txt: tk.Text
    build_log: tk.Text
    _open_thread_text_widgets: List[tk.Text]

    # Optional UI pieces used defensively
    gallery_canvas: Any
    _gallery_selected_chat_id: Any

    # Methods implemented by App
    def _safe_int(self, v: Any) -> int:  # pragma: no cover
        raise NotImplementedError

    def _refresh_gallery_for_chat(self, chat_id: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def _resolve_font_spec(self, spec: str, fallback: str) -> str:
        """Resolve a theme font spec into an installed font family.

        Supports a simple fallback list separated by '|', e.g.
        'Roboto|Segoe UI|Arial'. Returns the first installed family.
        """
        spec = (spec or "").strip()
        if not spec:
            return fallback
        if "|" not in spec:
            return spec
        try:
            installed = set(tkfont.families(self.root))
        except Exception:
            installed = set()
        for cand in [p.strip() for p in spec.split("|") if p.strip()]:
            if not installed or cand in installed:
                return cand
        return fallback

    def _default_theme(self) -> Dict[str, Any]:
        return {
            "font_family": "Segoe UI",
            # Back-compat: font_size is treated as UI size.
            "font_size": 10,
            "ui_size": 10,
            "text_size": 10,
            "heading_size": 11,
            "mono_family": "Consolas",
            "mono_size": 10,
            "app_bg": "#f3f3f3",
            "panel_bg": "#ffffff",
            "widget_bg": "#ffffff",
            "widget_fg": "#000000",
            "select_bg": "#0b5cad",
            "select_fg": "#ffffff",
            "accent": "#0b5cad",
            "text_bg": "#ffffff",
            "text_fg": "#000000",
            "meta_fg": "#666666",
            "you_fg": "#0b5cad",
            "them_fg": "#1a7f37",
            "hit_bg": "#fff2a8",
            "term_bg": "#d8f3ff",
            "link_fg": "#8a2be2",
        }

    def _normalize_theme(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        base = dict(self._default_theme())
        if isinstance(raw, dict):
            base.update(raw)

        def _as_int(v: Any, default: int) -> int:
            if isinstance(v, int):
                return v
            if isinstance(v, str):
                try:
                    return int(v.strip())
                except Exception:
                    return default
            return default

        def _as_str(v: Any, default: str) -> str:
            if isinstance(v, str) and v.strip():
                return v.strip()
            return default

        # Fonts (support fallback specs separated by '|')
        base["font_family"] = self._resolve_font_spec(
            _as_str(base.get("font_family"), "Segoe UI"),
            "Segoe UI",
        )
        base["mono_family"] = self._resolve_font_spec(
            _as_str(base.get("mono_family"), "Consolas"),
            "Consolas",
        )

        # Sizes (keep font_size as UI size for older saved themes)
        ui_size = _as_int(base.get("ui_size"), _as_int(base.get("font_size"), 10))
        base["font_size"] = ui_size
        base["ui_size"] = ui_size
        base["text_size"] = _as_int(base.get("text_size"), ui_size)
        base["heading_size"] = _as_int(base.get("heading_size"), max(7, ui_size + 1))
        base["mono_size"] = _as_int(base.get("mono_size"), ui_size)

        # Colors: validate/normalize
        for k in (
            "app_bg",
            "panel_bg",
            "widget_bg",
            "widget_fg",
            "select_bg",
            "select_fg",
            "accent",
            "text_bg",
            "text_fg",
            "meta_fg",
            "you_fg",
            "them_fg",
            "hit_bg",
            "term_bg",
            "link_fg",
        ):
            v = base.get(k)
            if not _is_hex_color(v):
                base[k] = self._default_theme().get(k)
            else:
                base[k] = str(v).lower()

        # Derived depth: only if explicit values are missing/empty in raw
        def _missing(key: str) -> bool:
            if not isinstance(raw, dict):
                return False
            v = raw.get(key)
            return v is None or (isinstance(v, str) and not v.strip())

        if _missing("panel_bg"):
            base["panel_bg"] = _lift_or_drop(base["app_bg"], 0.06)
        if _missing("widget_bg"):
            base["widget_bg"] = _lift_or_drop(base["panel_bg"], 0.05)
        if _missing("text_bg"):
            base["text_bg"] = base["panel_bg"]

        if _missing("widget_fg"):
            base["widget_fg"] = _best_bw_for_bg(base["widget_bg"])
        if _missing("text_fg"):
            base["text_fg"] = _best_bw_for_bg(base["text_bg"])

        # Ensure select_fg has reasonable contrast against select_bg if not specified
        if _missing("select_fg"):
            base["select_fg"] = _best_bw_for_bg(base["select_bg"])

        return base

    def _theme_presets(self) -> Dict[str, Dict[str, Any]]:
        base = self._default_theme()
        return {
            # Stock presets: popular app-inspired palettes.
            # Keep an equal number of light and dark themes.
            "GitHub Light": {
                **base,
                "font_family": "-apple-system|Segoe UI|Inter|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#f6f8fa",
                "panel_bg": "#ffffff",
                "widget_bg": "#ffffff",
                "widget_fg": "#24292f",
                "select_bg": "#0969da",
                "select_fg": "#ffffff",
                "accent": "#0969da",
                "text_bg": "#ffffff",
                "text_fg": "#24292f",
                "meta_fg": "#57606a",
                "you_fg": "#0969da",
                "them_fg": "#1a7f37",
                "hit_bg": "#fff8c5",
                "term_bg": "#ddf4ff",
                "link_fg": "#0969da",
            },
            "Google Light": {
                **base,
                "font_family": "Roboto|Segoe UI|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#f1f3f4",
                "panel_bg": "#ffffff",
                "widget_bg": "#ffffff",
                "widget_fg": "#202124",
                "select_bg": "#1a73e8",
                "select_fg": "#ffffff",
                "accent": "#1a73e8",
                "text_bg": "#ffffff",
                "text_fg": "#202124",
                "meta_fg": "#5f6368",
                "you_fg": "#1a73e8",
                "them_fg": "#34a853",
                "hit_bg": "#fff8e1",
                "term_bg": "#e8f0fe",
                "link_fg": "#1a73e8",
            },
            "Slack Light": {
                **base,
                "font_family": "Lato|Segoe UI|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#f8f8f8",
                "panel_bg": "#ffffff",
                "widget_bg": "#ffffff",
                "widget_fg": "#1d1c1d",
                "select_bg": "#4a154b",
                "select_fg": "#ffffff",
                "accent": "#4a154b",
                "text_bg": "#ffffff",
                "text_fg": "#1d1c1d",
                "meta_fg": "#616061",
                "you_fg": "#4a154b",
                "them_fg": "#2eb67d",
                "hit_bg": "#fff3c4",
                "term_bg": "#e6f7ff",
                "link_fg": "#36c5f0",
            },
            "Apple Notes Light": {
                **base,
                "font_family": "SF Pro Text|Segoe UI|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#f5f2ea",
                "panel_bg": "#fffdf8",
                "widget_bg": "#fffdf8",
                "widget_fg": "#1f2328",
                "select_bg": "#ffd60a",
                "select_fg": "#1f2328",
                "accent": "#ff9f0a",
                "text_bg": "#fffdf8",
                "text_fg": "#1f2328",
                "meta_fg": "#6e6e73",
                "you_fg": "#0a84ff",
                "them_fg": "#34c759",
                "hit_bg": "#ffefb0",
                "term_bg": "#e9f5ff",
                "link_fg": "#0a84ff",
            },
            "GitHub Dark": {
                **base,
                "font_family": "-apple-system|Segoe UI|Inter|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#0d1117",
                "panel_bg": "#161b22",
                "widget_bg": "#21262d",
                "widget_fg": "#c9d1d9",
                "select_bg": "#1f6feb",
                "select_fg": "#ffffff",
                "accent": "#1f6feb",
                "text_bg": "#0d1117",
                "text_fg": "#c9d1d9",
                "meta_fg": "#8b949e",
                "you_fg": "#58a6ff",
                "them_fg": "#3fb950",
                "hit_bg": "#5a3b00",
                "term_bg": "#003049",
                "link_fg": "#a371f7",
            },
            "Discord Dark": {
                **base,
                "font_family": "gg sans|Whitney|Segoe UI|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#1e1f22",
                "panel_bg": "#2b2d31",
                "widget_bg": "#313338",
                "widget_fg": "#dbdee1",
                "select_bg": "#5865f2",
                "select_fg": "#ffffff",
                "accent": "#5865f2",
                "text_bg": "#2b2d31",
                "text_fg": "#dbdee1",
                "meta_fg": "#949ba4",
                "you_fg": "#5865f2",
                "them_fg": "#3ba55d",
                "hit_bg": "#4f3b00",
                "term_bg": "#22303c",
                "link_fg": "#00a8fc",
            },
            "Spotify Dark": {
                **base,
                "font_family": "Circular Spotify|Circular|Segoe UI|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#121212",
                "panel_bg": "#181818",
                "widget_bg": "#242424",
                "widget_fg": "#ffffff",
                "select_bg": "#1db954",
                "select_fg": "#000000",
                "accent": "#1db954",
                "text_bg": "#121212",
                "text_fg": "#ffffff",
                "meta_fg": "#b3b3b3",
                "you_fg": "#1db954",
                "them_fg": "#1ed760",
                "hit_bg": "#3a2a00",
                "term_bg": "#003322",
                "link_fg": "#1db954",
            },
            "X Dark": {
                **base,
                "font_family": "TwitterChirp|Segoe UI|Arial",
                "mono_family": "Cascadia Mono|Consolas|Menlo|Courier New",
                "app_bg": "#000000",
                "panel_bg": "#0a0a0a",
                "widget_bg": "#16181c",
                "widget_fg": "#e7e9ea",
                "select_bg": "#1d9bf0",
                "select_fg": "#ffffff",
                "accent": "#1d9bf0",
                "text_bg": "#000000",
                "text_fg": "#e7e9ea",
                "meta_fg": "#71767b",
                "you_fg": "#1d9bf0",
                "them_fg": "#00ba7c",
                "hit_bg": "#2f2a00",
                "term_bg": "#001d2a",
                "link_fg": "#1d9bf0",
            },
        }

    def _load_custom_themes(self) -> None:
        try:
            import json

            if not self._themes_path.exists():
                self.custom_themes = {}
                return
            data = json.loads(self._themes_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                themes = data.get("themes")
                if isinstance(themes, dict):
                    cleaned: Dict[str, Dict[str, Any]] = {}
                    for name, theme in themes.items():
                        if not isinstance(name, str) or not isinstance(theme, dict):
                            continue
                        t: Dict[str, Any] = {}
                        for k, v in theme.items():
                            if isinstance(k, str) and isinstance(v, (str, int)):
                                t[k] = v
                        if t:
                            cleaned[name] = t
                    self.custom_themes = cleaned
        except Exception:
            self.custom_themes = {}

    def _save_custom_themes(self) -> None:
        try:
            import json

            payload = {"themes": self.custom_themes}
            self._themes_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("Save themes failed", str(e))

    def _is_stock_theme(self, name: str) -> bool:
        return name in self._theme_presets()

    def _set_active_theme_by_name(self, name: str) -> None:
        presets = self._theme_presets()
        if name in presets:
            self.active_theme_name = name
            self.theme = self._normalize_theme(presets[name])
            self._apply_theme_now()
            return
        if name in self.custom_themes:
            self.active_theme_name = name
            self.theme = self._normalize_theme(self.custom_themes[name])
            self._apply_theme_now()

    def _apply_theme_to_ttk(self) -> None:
        try:
            if "clam" in self._style.theme_names():
                self._style.theme_use("clam")
        except Exception:
            pass

        # Fonts (use named Tk fonts so existing widgets update live)
        ui_font = "TkDefaultFont"
        heading_font = "TkHeadingFont"

        app_bg = self.theme.get("app_bg", "#f3f3f3")
        panel_bg = self.theme.get("panel_bg", "#ffffff")
        widget_bg = self.theme.get("widget_bg", panel_bg)
        fg = self.theme.get("widget_fg", "#000000")
        select_bg = self.theme.get("select_bg", self.theme.get("accent", "#0b5cad"))
        select_fg = self.theme.get("select_fg", "#ffffff")
        accent = self.theme.get("accent", select_bg)

        # Base
        self.root.configure(background=app_bg)
        self._style.configure(".", background=app_bg, foreground=fg, font=ui_font)
        self._style.configure("TFrame", background=app_bg)
        self._style.configure("TLabelframe", background=app_bg)
        self._style.configure("TLabelframe.Label", background=app_bg, foreground=fg)
        self._style.configure("TLabel", background=app_bg, foreground=fg, font=ui_font)

        # Buttons
        self._style.configure("TButton", background=panel_bg, foreground=fg, font=ui_font)
        self._style.map(
            "TButton",
            # Make Accent visibly affect the UI (hover/pressed buttons).
            background=[("active", accent), ("pressed", accent)],
            foreground=[("active", select_fg), ("pressed", select_fg), ("disabled", "#888888")],
        )

        # Entries/Combobox
        self._style.configure("TEntry", fieldbackground=widget_bg, foreground=fg, font=ui_font)
        self._style.map(
            "TEntry",
            fieldbackground=[("disabled", panel_bg), ("readonly", widget_bg)],
            foreground=[("disabled", "#888888")],
        )
        self._style.configure("TCombobox", fieldbackground=widget_bg, background=widget_bg, foreground=fg, font=ui_font)
        self._style.map(
            "TCombobox",
            fieldbackground=[("readonly", widget_bg)],
            background=[("readonly", widget_bg)],
            foreground=[("disabled", "#888888")],
        )

        # Notebook
        self._style.configure("TNotebook", background=app_bg)
        self._style.configure("TNotebook.Tab", background=panel_bg, foreground=fg, font=ui_font)
        self._style.map(
            "TNotebook.Tab",
            background=[("selected", widget_bg)],
            foreground=[("selected", fg)],
        )

        # Treeview
        self._style.configure(
            "Treeview",
            background=widget_bg,
            fieldbackground=widget_bg,
            foreground=fg,
            font=ui_font,
        )
        self._style.configure("Treeview.Heading", background=panel_bg, foreground=fg, font=heading_font)
        self._style.map(
            "Treeview",
            background=[("selected", select_bg)],
            foreground=[("selected", select_fg)],
        )

        # Scrollbar
        self._style.configure("TScrollbar", background=panel_bg, troughcolor=app_bg)

    def _apply_theme_to_menu(self) -> None:
        if not self._menubar or not self._settings_menu:
            return
        bg = self.theme.get("panel_bg", "#ffffff")
        fg = self.theme.get("widget_fg", "#000000")
        abg = self.theme.get("select_bg", self.theme.get("accent", "#0b5cad"))
        afg = self.theme.get("select_fg", "#ffffff")
        try:
            self._menubar.configure(background=bg, foreground=fg, activebackground=abg, activeforeground=afg, font="TkMenuFont")
            self._settings_menu.configure(background=bg, foreground=fg, activebackground=abg, activeforeground=afg, font="TkMenuFont")
        except Exception:
            pass

    def _apply_theme_to_text(self, txt: tk.Text) -> None:
        bg = self.theme.get("text_bg", "#ffffff")
        fg = self.theme.get("text_fg", "#000000")
        try:
            txt.configure(background=bg, foreground=fg, insertbackground=fg, font="TkTextFont")
        except Exception:
            pass
        txt.tag_configure("meta", foreground=self.theme.get("meta_fg", "#666666"))
        txt.tag_configure("you", foreground=self.theme.get("you_fg", "#0b5cad"))
        txt.tag_configure("them", foreground=self.theme.get("them_fg", "#1a7f37"))
        txt.tag_configure("hit", background=self.theme.get("hit_bg", "#fff2a8"))
        txt.tag_configure("term", background=self.theme.get("term_bg", "#d8f3ff"))

    def _apply_theme_fonts(self) -> None:
        def _as_int(v: Any, default: int) -> int:
            if isinstance(v, int):
                return v
            if isinstance(v, str):
                try:
                    return int(v.strip())
                except Exception:
                    return default
            return default

        def _as_str(v: Any, default: str) -> str:
            if isinstance(v, str) and v.strip():
                return v.strip()
            return default

        ui_family = _as_str(self.theme.get("font_family"), "Segoe UI")
        ui_size = _as_int(self.theme.get("ui_size"), _as_int(self.theme.get("font_size"), 10))
        text_size = _as_int(self.theme.get("text_size"), ui_size)
        heading_size = _as_int(self.theme.get("heading_size"), max(7, ui_size + 1))
        mono_family = _as_str(self.theme.get("mono_family"), "Consolas")
        mono_size = _as_int(self.theme.get("mono_size"), ui_size)

        try:
            tkfont.nametofont("TkDefaultFont").configure(family=ui_family, size=ui_size)
        except Exception:
            pass
        try:
            tkfont.nametofont("TkTextFont").configure(family=ui_family, size=text_size)
        except Exception:
            pass
        try:
            tkfont.nametofont("TkMenuFont").configure(family=ui_family, size=ui_size)
        except Exception:
            pass
        try:
            tkfont.nametofont("TkHeadingFont").configure(family=ui_family, size=heading_size, weight="bold")
        except Exception:
            pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=mono_size)
        except Exception:
            pass

    def _apply_theme_now(self) -> None:
        # Ensure theme is always safe/complete before applying.
        self.theme = self._normalize_theme(self.theme)
        # Fonts first so ttk + Text widgets pick them up.
        self._apply_theme_fonts()
        # Apply ttk/global colors first (most of the UI)
        self._apply_theme_to_ttk()
        self._apply_theme_to_menu()

        link_fg = self.theme.get("link_fg", "#8a2be2")

        try:
            self._apply_theme_to_text(self.preview_txt)
            self.preview_txt.tag_configure("pv_link", foreground=link_fg, underline=True)
        except Exception:
            pass

        try:
            self._apply_theme_to_text(self.build_log)
            self.build_log.configure(font="TkFixedFont")
        except Exception:
            pass

        try:
            panel_bg = self.theme.get("panel_bg", "#ffffff")
            if hasattr(self, "gallery_canvas"):
                self.gallery_canvas.configure(background=panel_bg, highlightthickness=0)
        except Exception:
            pass

        # Propagate to already-open thread windows
        alive: List[tk.Text] = []
        for txt in list(self._open_thread_text_widgets):
            try:
                if not txt.winfo_exists():
                    continue
                self._apply_theme_to_text(txt)
                txt.tag_configure("link", foreground=link_fg, underline=True)
                alive.append(txt)
            except Exception:
                continue
        self._open_thread_text_widgets = alive

        # Refresh gallery timestamps if a chat is loaded
        try:
            if hasattr(self, "_gallery_selected_chat_id"):
                chat_id = self._safe_int(self._gallery_selected_chat_id.get())
                if chat_id:
                    self._refresh_gallery_for_chat(chat_id)
        except Exception:
            pass

    def _reset_theme(self) -> None:
        self.theme = self._normalize_theme(self._default_theme())
        self.active_theme_name = ""
        self._apply_theme_now()

    def _open_theme_dialog(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Theme")
        dlg.geometry("760x760")
        dlg.minsize(720, 520)
        dlg.resizable(True, True)
        dlg.transient(self.root)

        theme_work = dict(self._normalize_theme(self.theme))

        presets = self._theme_presets()
        stock_names = list(presets.keys())
        custom_names = sorted(self.custom_themes.keys(), key=lambda s: s.lower())

        theme_combo_values = ["(Current)"] + [f"Stock: {n}" for n in stock_names] + [f"Custom: {n}" for n in custom_names]

        contrast_var = tk.StringVar(value="")

        def update_contrast() -> None:
            try:
                t = self._normalize_theme(theme_work)
                w = _contrast_ratio(t["widget_fg"], t["widget_bg"])
                tx = _contrast_ratio(t["text_fg"], t["text_bg"])
                sel = _contrast_ratio(t["select_fg"], t["select_bg"])

                def mark(x: float) -> str:
                    return "OK" if x >= 4.5 else "LOW"

                contrast_var.set(
                    f"Contrast (higher is better):  Widget {w:.2f} ({mark(w)})   Text {tx:.2f} ({mark(tx)})   Selection {sel:.2f} ({mark(sel)})"
                )
            except Exception:
                contrast_var.set("Contrast: n/a")

        def apply_work() -> None:
            norm = self._normalize_theme(theme_work)
            theme_work.clear()
            theme_work.update(norm)
            self.theme = dict(norm)
            self._apply_theme_now()
            # Refresh UI elements driven by theme_work
            for k, sw in swatches.items():
                try:
                    sw.configure(background=theme_work.get(k, "#ffffff"))
                except Exception:
                    pass
            ui_family_var.set(str(theme_work.get("font_family", "Segoe UI")))
            ui_size_var.set(str(theme_work.get("ui_size", theme_work.get("font_size", 10))))
            text_size_var.set(str(theme_work.get("text_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
            heading_size_var.set(str(theme_work.get("heading_size", max(7, int(theme_work.get("ui_size", theme_work.get("font_size", 10))) + 1))))
            mono_family_var.set(str(theme_work.get("mono_family", "Consolas")))
            mono_size_var.set(str(theme_work.get("mono_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
            update_contrast()

        def apply_named(name: str) -> None:
            if name.startswith("Stock: "):
                nm = name[len("Stock: "):]
                if nm not in presets:
                    return
                theme_work.clear()
                theme_work.update(self._normalize_theme(presets[nm]))
                self.active_theme_name = nm
            elif name.startswith("Custom: "):
                nm = name[len("Custom: "):]
                if nm not in self.custom_themes:
                    return
                theme_work.clear()
                theme_work.update(self._normalize_theme(self.custom_themes[nm]))
                self.active_theme_name = nm
            else:
                return
            for k, sw in swatches.items():
                sw.configure(background=theme_work.get(k, "#ffffff"))
            ui_family_var.set(str(theme_work.get("font_family", "Segoe UI")))
            ui_size_var.set(str(theme_work.get("ui_size", theme_work.get("font_size", 10))))
            text_size_var.set(str(theme_work.get("text_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
            heading_size_var.set(str(theme_work.get("heading_size", max(7, int(theme_work.get("ui_size", theme_work.get("font_size", 10))) + 1))))
            mono_family_var.set(str(theme_work.get("mono_family", "Consolas")))
            mono_size_var.set(str(theme_work.get("mono_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
            apply_work()

        def pick(key: str) -> None:
            initial = theme_work.get(key, "")
            _rgb, hexv = colorchooser.askcolor(color=initial or None, parent=dlg)
            if not hexv:
                return
            theme_work[key] = hexv
            swatches[key].configure(background=hexv)
            apply_work()

        # Scrollable body so long option lists are not cut off.
        body = ttk.Frame(dlg)
        body.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(body, highlightthickness=0)
        vbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        frm = ttk.Frame(canvas, padding=12)
        win_id = canvas.create_window((0, 0), window=frm, anchor="nw")

        def _on_frm_configure(_e=None) -> None:
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(e=None) -> None:
            try:
                canvas.itemconfigure(win_id, width=canvas.winfo_width())
            except Exception:
                pass

        frm.bind("<Configure>", _on_frm_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        _bind_mousewheel(canvas, speed=3)
        _bind_mousewheel(frm, scroll_widget=canvas, speed=3)

        preset_row = ttk.Frame(frm)
        preset_row.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 10))
        ttk.Label(preset_row, text="Theme").pack(side=tk.LEFT)
        preset_combo = ttk.Combobox(preset_row, values=theme_combo_values, state="readonly", width=34)
        preset_combo.pack(side=tk.LEFT, padx=8)
        preset_combo.set("(Current)")
        preset_combo.bind("<<ComboboxSelected>>", lambda _e: apply_named(preset_combo.get()))

        save_row = ttk.Frame(frm)
        save_row.grid(row=1, column=0, columnspan=3, sticky="we", pady=(0, 10))
        ttk.Label(save_row, text="Save as").pack(side=tk.LEFT)
        name_var = tk.StringVar(value="")
        ttk.Entry(save_row, textvariable=name_var, width=26).pack(side=tk.LEFT, padx=8)

        io_row = ttk.Frame(frm)
        io_row.grid(row=2, column=0, columnspan=3, sticky="we", pady=(0, 10))

        def export_theme() -> None:
            try:
                import json

                p = filedialog.asksaveasfilename(
                    parent=dlg,
                    defaultextension=".json",
                    filetypes=[("JSON", "*.json")],
                    title="Export theme as JSON",
                )
                if not p:
                    return
                norm = self._normalize_theme(theme_work)
                Path(p).write_text(json.dumps(norm, indent=2), encoding="utf-8")
            except Exception as e:
                messagebox.showerror("Export failed", str(e), parent=dlg)

        def import_theme() -> None:
            try:
                import json

                p = filedialog.askopenfilename(
                    parent=dlg,
                    filetypes=[("JSON", "*.json"), ("All files", "*")],
                    title="Import theme JSON",
                )
                if not p:
                    return
                data = json.loads(Path(p).read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("theme"), dict):
                    data = data.get("theme")
                if not isinstance(data, dict):
                    raise ValueError("Theme JSON must be an object/dict")
                theme_work.clear()
                theme_work.update(self._normalize_theme(data))
                for k, sw in swatches.items():
                    sw.configure(background=theme_work.get(k, "#ffffff"))
                ui_family_var.set(str(theme_work.get("font_family", "Segoe UI")))
                ui_size_var.set(str(theme_work.get("ui_size", theme_work.get("font_size", 10))))
                text_size_var.set(str(theme_work.get("text_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
                heading_size_var.set(str(theme_work.get("heading_size", max(7, int(theme_work.get("ui_size", theme_work.get("font_size", 10))) + 1))))
                mono_family_var.set(str(theme_work.get("mono_family", "Consolas")))
                mono_size_var.set(str(theme_work.get("mono_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
                apply_work()
            except Exception as e:
                messagebox.showerror("Import failed", str(e), parent=dlg)

        ttk.Button(io_row, text="Import...", command=import_theme).pack(side=tk.LEFT)
        ttk.Button(io_row, text="Export...", command=export_theme).pack(side=tk.LEFT, padx=8)

        # Fonts
        try:
            families = sorted(set(tkfont.families()))
        except Exception:
            families = []

        ui_family_var = tk.StringVar(value=str(theme_work.get("font_family", "Segoe UI")))
        ui_size_var = tk.StringVar(value=str(theme_work.get("ui_size", theme_work.get("font_size", 10))))
        text_size_var = tk.StringVar(value=str(theme_work.get("text_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))
        heading_size_var = tk.StringVar(value=str(theme_work.get("heading_size", max(7, int(theme_work.get("ui_size", theme_work.get("font_size", 10))) + 1))))
        mono_family_var = tk.StringVar(value=str(theme_work.get("mono_family", "Consolas")))
        mono_size_var = tk.StringVar(value=str(theme_work.get("mono_size", theme_work.get("ui_size", theme_work.get("font_size", 10)))))

        def _apply_font_vars() -> None:
            theme_work["font_family"] = (ui_family_var.get() or "").strip() or "Segoe UI"
            try:
                ui_sz = int((ui_size_var.get() or "").strip())
                theme_work["ui_size"] = ui_sz
                theme_work["font_size"] = ui_sz
            except Exception:
                pass
            try:
                theme_work["text_size"] = int((text_size_var.get() or "").strip())
            except Exception:
                pass
            try:
                theme_work["heading_size"] = int((heading_size_var.get() or "").strip())
            except Exception:
                pass
            theme_work["mono_family"] = (mono_family_var.get() or "").strip() or "Consolas"
            try:
                theme_work["mono_size"] = int((mono_size_var.get() or "").strip())
            except Exception:
                pass
            apply_work()

        Spin = getattr(ttk, "Spinbox", tk.Spinbox)

        font_row = ttk.Frame(frm)
        font_row.grid(row=3, column=0, columnspan=3, sticky="we", pady=(0, 6))

        ttk.Label(font_row, text="Font").grid(row=0, column=0, sticky="w")
        ui_family = ttk.Combobox(font_row, values=families, textvariable=ui_family_var, width=26)
        ui_family.grid(row=0, column=1, sticky="w", padx=(8, 6))
        ui_family.bind("<<ComboboxSelected>>", lambda _e: _apply_font_vars())
        ui_family.bind("<FocusOut>", lambda _e: _apply_font_vars())

        ttk.Label(font_row, text="UI").grid(row=0, column=2, sticky="e", padx=(8, 0))
        ui_size = Spin(font_row, from_=7, to=32, textvariable=ui_size_var, width=5)
        ui_size.grid(row=0, column=3, sticky="w", padx=(6, 0))
        ui_size.bind("<Return>", lambda _e: _apply_font_vars())
        ui_size.bind("<FocusOut>", lambda _e: _apply_font_vars())

        ttk.Label(font_row, text="Text").grid(row=0, column=4, sticky="e", padx=(8, 0))
        text_size = Spin(font_row, from_=7, to=32, textvariable=text_size_var, width=5)
        text_size.grid(row=0, column=5, sticky="w", padx=(6, 0))
        text_size.bind("<Return>", lambda _e: _apply_font_vars())
        text_size.bind("<FocusOut>", lambda _e: _apply_font_vars())

        ttk.Label(font_row, text="Mono").grid(row=1, column=0, sticky="w", pady=(6, 0))
        mono_family = ttk.Combobox(font_row, values=families, textvariable=mono_family_var, width=26)
        mono_family.grid(row=1, column=1, sticky="w", padx=(8, 6), pady=(6, 0))
        mono_family.bind("<<ComboboxSelected>>", lambda _e: _apply_font_vars())
        mono_family.bind("<FocusOut>", lambda _e: _apply_font_vars())

        ttk.Label(font_row, text="Mono").grid(row=1, column=2, sticky="e", padx=(8, 0), pady=(6, 0))
        mono_size = Spin(font_row, from_=7, to=32, textvariable=mono_size_var, width=5)
        mono_size.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(6, 0))
        mono_size.bind("<Return>", lambda _e: _apply_font_vars())
        mono_size.bind("<FocusOut>", lambda _e: _apply_font_vars())

        ttk.Label(font_row, text="Head").grid(row=1, column=4, sticky="e", padx=(8, 0), pady=(6, 0))
        heading_size = Spin(font_row, from_=7, to=40, textvariable=heading_size_var, width=5)
        heading_size.grid(row=1, column=5, sticky="w", padx=(6, 0), pady=(6, 0))
        heading_size.bind("<Return>", lambda _e: _apply_font_vars())
        heading_size.bind("<FocusOut>", lambda _e: _apply_font_vars())

        for c in range(6):
            font_row.columnconfigure(c, weight=1 if c == 1 else 0)

        contrast_row = ttk.Frame(frm)
        contrast_row.grid(row=4, column=0, columnspan=3, sticky="we", pady=(0, 10))
        ttk.Label(contrast_row, textvariable=contrast_var).pack(side=tk.LEFT)

        rows = [
            ("app_bg", "App background (window)"),
            ("panel_bg", "Panel background (tabs/frames)"),
            ("widget_bg", "Control background (entries/lists)"),
            ("widget_fg", "UI text (labels/buttons)"),
            ("select_bg", "Selection background (selected rows/menus)"),
            ("select_fg", "Selection text"),
            ("accent", "Accent (button hover/pressed, thumbnails)"),
            ("text_bg", "Message text background"),
            ("text_fg", "Message text color"),
            ("meta_fg", "Meta text (timestamps)"),
            ("you_fg", "You text"),
            ("them_fg", "Them text"),
            ("hit_bg", "Hit highlight (search results)"),
            ("term_bg", "Term highlight (in messages)"),
            ("link_fg", "Link color"),
        ]

        swatches: Dict[str, tk.Label] = {}
        for i, (key, label) in enumerate(rows, start=5):
            ttk.Label(frm, text=label).grid(row=i, column=0, sticky="w", pady=6)
            sw = tk.Label(frm, width=12, relief="groove", background=theme_work.get(key, "#ffffff"))
            sw.grid(row=i, column=1, sticky="w", padx=10)
            swatches[key] = sw
            ttk.Button(frm, text="Choose...", command=lambda k=key: pick(k)).grid(row=i, column=2, sticky="e")

        frm.columnconfigure(0, weight=1)

        btns = ttk.Frame(dlg, padding=(12, 0, 12, 12))
        btns.pack(fill=tk.X)

        def apply_only() -> None:
            apply_work()

        def save_named() -> None:
            nm = (name_var.get() or "").strip()
            if not nm:
                messagebox.showwarning("Missing name", "Enter a theme name to save.")
                return
            if self._is_stock_theme(nm):
                messagebox.showerror("Protected", "That name is a stock theme and cannot be overwritten. Choose a different name.")
                return
            norm = self._normalize_theme(theme_work)
            theme_work.clear()
            theme_work.update(norm)
            self.custom_themes[nm] = dict(norm)
            self._save_custom_themes()
            self.active_theme_name = nm
            messagebox.showinfo("Saved", f"Saved custom theme: {nm}")

        def apply_and_save() -> None:
            apply_work()
            # Keep legacy behavior: if a name is provided, save into custom themes.
            if (name_var.get() or "").strip():
                save_named()
            dlg.destroy()

        ttk.Button(btns, text="Apply", command=apply_only).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save as", command=save_named).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Save & Close", command=apply_and_save).pack(side=tk.LEFT, padx=8)
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side=tk.RIGHT)

        update_contrast()
