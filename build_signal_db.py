#!/usr/bin/env python3
from __future__ import annotations
"""
Signal Export Browser v5 - build_signal_db.py (pointer+digest aware)

Your export layout shows media stored in a content-addressed tree:
  files/00 ... files/ff

And JSON shows attachments as:
  standardMessage.attachments: [{"pointer": {...}}]

Signal-style "pointer" objects commonly include:
- contentType
- size
- digest (base64)
- fileName (optional)

This builder resolves attachment files using:
1) pointer.relativePath / pointer.path if present
2) pointer.fileName if present
3) pointer.digest (base64) -> sha256 hex -> files/<first2>/<hex> (or <hex>.*)
4) fallback recursive scan by basename (cached)

It also improves speaker naming:
- Adds authorName by joining messages.authorId -> recipients.id (when possible)
- Adds recipientName from chat recipient as before

GUI should display:
- outgoing messages as "Me" (or your own name if you set it in GUI)
- incoming messages in 1:1 as recipientName
- group messages as authorName when available
"""

import argparse
import base64
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple


def _ms_to_iso(ms: Any) -> str:
    try:
        ms_int = int(ms)
        return datetime.fromtimestamp(ms_int / 1000, tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_test USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_test")
        return True
    except sqlite3.OperationalError:
        return False


def _pick_int(d: dict, keys: Iterable[str]) -> Optional[int]:
    for k in keys:
        v = d.get(k)
        try:
            if v is None:
                continue
            return int(v)
        except Exception:
            continue
    return None


def _mime_to_ext(mime: str) -> str:
    m = (mime or "").lower().strip()
    common = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tif",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/flac": ".flac",
        "audio/mp4": ".m4a",
        "application/pdf": ".pdf",
    }
    return common.get(m, "")


def _classify(hint: str) -> str:
    low = (hint or "").lower()
    if low.startswith("image/") or low.endswith((".png",".jpg",".jpeg",".webp",".gif",".bmp",".tif",".tiff")):
        return "image"
    if low.startswith("video/") or low.endswith((".mp4",".mov",".mkv",".webm",".avi",".m4v")):
        return "video"
    if low.startswith("audio/") or low.endswith((".mp3",".wav",".flac",".m4a",".aac",".ogg")):
        return "audio"
    if low.endswith((".pdf",".doc",".docx",".txt",".md",".rtf")):
        return "doc"
    return "file"


def _extract_attachments(chat_item: dict) -> list[dict]:
    sm = chat_item.get("standardMessage") or {}
    out: list[dict] = []
    a = sm.get("attachments")
    if isinstance(a, list):
        out.extend([x for x in a if isinstance(x, dict)])
    # quoted attachments may include thumbnails only
    q = sm.get("quote")
    if isinstance(q, dict):
        qa = q.get("attachments")
        if isinstance(qa, list):
            out.extend([{"quoteAttachment": x} for x in qa if isinstance(x, dict)])
    # dedupe by stable json
    seen = set()
    ded = []
    for att in out:
        key = json.dumps(att, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        ded.append(att)
    return ded


def _build_basename_index(export_base: Path, log: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    by_name: Dict[str, str] = {}
    for p in export_base.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name in ("main.jsonl",) or name.endswith((".db", ".db-wal", ".db-shm", ".tmp")):
            continue
        by_name.setdefault(name, str(p.resolve()))
    if log:
        log(f"Indexed files by basename: {len(by_name):,}")
    return by_name


def _resolve_rel_or_name(export_base: Path, rel_or_name: str, by_name: Dict[str, str]) -> str:
    if not rel_or_name:
        return ""
    rel_norm = rel_or_name.replace("\\", os.sep).replace("/", os.sep)
    cand = (export_base / rel_norm).resolve()
    if cand.exists():
        return str(cand)
    bn = os.path.basename(rel_norm).lower()
    return by_name.get(bn, "")


def _digest_b64_to_hex(digest_b64: str) -> str:
    try:
        # normalize padding
        s = digest_b64.strip()
        missing = (-len(s)) % 4
        if missing:
            s += "=" * missing
        raw = base64.b64decode(s)
        return raw.hex()
    except Exception:
        return ""


def _sha256_b64_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return base64.b64encode(h.digest()).decode("ascii")


def _build_files_by_size_index(export_base: Path, log: Optional[Callable[[str], None]] = None) -> DefaultDict[int, List[str]]:
    """Index only files under export_base/files/** by byte size."""
    by_size: DefaultDict[int, List[str]] = defaultdict(list)
    files_root = export_base / "files"
    if not files_root.exists():
        return by_size

    n = 0
    for p in files_root.rglob("*"):
        if not p.is_file():
            continue
        try:
            sz = int(p.stat().st_size)
        except Exception:
            continue
        by_size[sz].append(str(p.resolve()))
        n += 1

    if log:
        log(f"Indexed files/ by size: {n:,} files")
    return by_size


def _resolve_by_plaintext_hash(files_by_size: DefaultDict[int, List[str]], plaintext_hash_b64: str, size_bytes: Optional[int]) -> str:
    """Resolve a local file by matching SHA-256(base64) against locatorInfo.plaintextHash.

    Uses size_bytes to narrow candidates (fast) and hashes only those candidates.
    """
    if not plaintext_hash_b64:
        return ""
    if not size_bytes:
        return ""

    cands = files_by_size.get(int(size_bytes)) or []
    if not cands:
        return ""
    if len(cands) == 1:
        # Still validate hash to avoid false positives when size collides.
        try:
            if _sha256_b64_file(Path(cands[0])) == plaintext_hash_b64:
                return cands[0]
        except Exception:
            return ""
        return ""

    for p in cands:
        try:
            if _sha256_b64_file(Path(p)) == plaintext_hash_b64:
                return p
        except Exception:
            continue
    return ""


def _resolve_by_digest(export_base: Path, digest_hex: str) -> str:
    """
    Try content-addressed layout:
      files/<first2>/<hex>
    Also handle possible extensions:
      files/<first2>/<hex>.<ext>
    """
    if not digest_hex or len(digest_hex) < 8:
        return ""
    sub = digest_hex[:2].lower()
    base = export_base / "files" / sub
    if not base.exists():
        return ""

    # exact match first
    cand = (base / digest_hex).resolve()
    if cand.exists():
        return str(cand)

    # try any extension
    try:
        for p in base.glob(digest_hex + ".*"):
            if p.is_file():
                return str(p.resolve())
    except Exception:
        pass
    return ""


def build_db(input_jsonl: Path, out_db: Path, store_raw_e164: bool = False, log: Optional[Callable[[str], None]] = None) -> None:
    def _log(msg: str) -> None:
        if log:
            log(msg)

    input_jsonl = input_jsonl.resolve()
    export_base_dir = input_jsonl.parent.resolve()

    by_name = _build_basename_index(export_base_dir, log=_log)
    files_by_size = _build_files_by_size_index(export_base_dir, log=_log)

    tmp_db = out_db.with_suffix(out_db.suffix + ".tmp")
    if tmp_db.exists():
        try:
            tmp_db.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(str(tmp_db))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

        CREATE TABLE recipients (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT,
            e164 TEXT,
            e164_sha256 TEXT,
            aci TEXT
        );

        CREATE TABLE chats (
            id INTEGER PRIMARY KEY,
            recipientId TEXT,
            expirationTimerMs INTEGER
        );

        CREATE TABLE messages (
            rowid INTEGER PRIMARY KEY,
            chatId INTEGER,
            authorId TEXT,
            outgoing INTEGER,
            dateSentMs INTEGER,
            dateSentIso TEXT,
            body TEXT
        );

        CREATE TABLE attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_rowid INTEGER NOT NULL,
            rel_path TEXT,
            abs_path TEXT,
            file_name TEXT,
            mime TEXT,
            size_bytes INTEGER,
            width INTEGER,
            height INTEGER,
            duration_ms INTEGER,
            kind TEXT,
            extra_json TEXT,
            FOREIGN KEY(message_rowid) REFERENCES messages(rowid)
        );

        CREATE INDEX idx_messages_chatId_date ON messages(chatId, dateSentMs);
        CREATE INDEX idx_messages_dateSentIso ON messages(dateSentIso);
        CREATE INDEX idx_messages_outgoing ON messages(outgoing);
        CREATE INDEX idx_attachments_message_rowid ON attachments(message_rowid);
        CREATE INDEX idx_attachments_kind ON attachments(kind);
        CREATE INDEX idx_attachments_abs_path ON attachments(abs_path) WHERE abs_path != '';
        CREATE INDEX idx_attachments_msg_kind ON attachments(message_rowid, kind);
        CREATE INDEX idx_chats_recipientId ON chats(recipientId);
        CREATE INDEX idx_recipients_name ON recipients(name);
        """
    )

    cur.execute("INSERT INTO meta(key, value) VALUES (?, ?)", ("export_base_dir", str(export_base_dir)))

    rec_count = chat_count = msg_count = att_count = resolved = 0
    _log(f"Reading JSONL: {input_jsonl}")
    _log(f"Export base dir: {export_base_dir}")

    with input_jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "recipient" in obj:
                r = obj["recipient"]
                rid = r.get("id")
                if not rid:
                    continue
                rtype = "unknown"
                name = ""
                e164 = ""
                aci = ""
                if "contact" in r:
                    rtype = "contact"
                    c = r.get("contact") or {}
                    name = c.get("profileGivenName") or c.get("profileName") or ""
                    e164 = c.get("e164") or ""
                    aci = c.get("aci") or ""
                elif "group" in r:
                    rtype = "group"
                    g = r.get("group") or {}
                    snap = g.get("snapshot") or {}
                    t = snap.get("title") or {}
                    name = t.get("title") or ""
                e164_sha = _sha256(e164) if e164 else ""
                e164_store = e164 if store_raw_e164 else ""
                cur.execute(
                    "INSERT OR REPLACE INTO recipients(id, type, name, e164, e164_sha256, aci) VALUES (?, ?, ?, ?, ?, ?)",
                    (rid, rtype, name, e164_store, e164_sha, aci),
                )
                rec_count += 1

            elif "chat" in obj:
                c = obj["chat"]
                cid = c.get("id")
                if cid is None:
                    continue
                cur.execute(
                    "INSERT OR REPLACE INTO chats(id, recipientId, expirationTimerMs) VALUES (?, ?, ?)",
                    (cid, c.get("recipientId"), c.get("expirationTimerMs")),
                )
                chat_count += 1

            elif "chatItem" in obj:
                ci = obj["chatItem"]
                sm = ci.get("standardMessage") or {}
                text = (sm.get("text") or {}).get("body")
                if not isinstance(text, str):
                    text = ""
                atts = _extract_attachments(ci)
                if not text and not atts:
                    continue

                date_ms = ci.get("dateSent")
                cur.execute(
                    "INSERT INTO messages(chatId, authorId, outgoing, dateSentMs, dateSentIso, body) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        ci.get("chatId"),
                        ci.get("authorId"),
                        1 if ci.get("outgoing") else 0,
                        int(date_ms) if date_ms is not None else None,
                        _ms_to_iso(date_ms),
                        text,
                    ),
                )
                msg_rowid = cur.lastrowid
                msg_count += 1

                for att in atts:
                    pointer = att.get("pointer") if isinstance(att.get("pointer"), dict) else None
                    mime = ""
                    size_b = None
                    w = None
                    h = None
                    dur = None
                    relp = ""
                    file_name = ""
                    digest_hex = ""
                    plaintext_hash_b64 = ""

                    if pointer:
                        mime = pointer.get("contentType") or pointer.get("mimeType") or ""
                        if not isinstance(mime, str):
                            mime = ""
                        size_b = _pick_int(pointer, ("size", "byteSize", "fileSize"))
                        w = _pick_int(pointer, ("width", "w"))
                        h = _pick_int(pointer, ("height", "h"))
                        dur = _pick_int(pointer, ("durationMs", "duration", "duration_ms"))
                        relp = pointer.get("relativePath") or pointer.get("path") or ""
                        if not isinstance(relp, str):
                            relp = ""
                        file_name = pointer.get("fileName") or pointer.get("filename") or pointer.get("name") or ""
                        if not isinstance(file_name, str):
                            file_name = ""
                        digest_b64 = pointer.get("digest") or ""
                        if isinstance(digest_b64, str) and digest_b64:
                            digest_hex = _digest_b64_to_hex(digest_b64)

                        # Newer exports store the local file hash and size inside locatorInfo
                        loc = pointer.get("locatorInfo")
                        if isinstance(loc, dict) and loc:
                            if size_b is None:
                                size_b = _pick_int(loc, ("size", "byteSize", "fileSize"))
                            ph = loc.get("plaintextHash")
                            if isinstance(ph, str) and ph:
                                plaintext_hash_b64 = ph.strip()

                    if "quoteAttachment" in att and isinstance(att["quoteAttachment"], dict):
                        qa = att["quoteAttachment"]
                        if not mime:
                            ct = qa.get("contentType")
                            if isinstance(ct, str):
                                mime = ct

                    abs_path = ""
                    if relp:
                        abs_path = _resolve_rel_or_name(export_base_dir, relp, by_name)
                    if not abs_path and file_name:
                        abs_path = _resolve_rel_or_name(export_base_dir, file_name, by_name)
                    if not abs_path and digest_hex:
                        abs_path = _resolve_by_digest(export_base_dir, digest_hex)

                    # If we have plaintextHash (base64 of sha256(local file)) + size, resolve via files/ tree.
                    if not abs_path and plaintext_hash_b64 and size_b:
                        abs_path = _resolve_by_plaintext_hash(files_by_size, plaintext_hash_b64, size_b)

                    # last resort: try digest hex as basename with extension from mime
                    if not abs_path and digest_hex:
                        ext = _mime_to_ext(mime)
                        if ext:
                            abs_path = _resolve_rel_or_name(export_base_dir, digest_hex + ext, by_name)

                    if abs_path:
                        resolved += 1
                        if not file_name:
                            file_name = Path(abs_path).name

                    kind = _classify(mime or file_name)

                    cur.execute(
                        """
                        INSERT INTO attachments(message_rowid, rel_path, abs_path, file_name, mime, size_bytes, width, height, duration_ms, kind, extra_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (msg_rowid, relp, abs_path, file_name, mime, size_b, w, h, dur, kind, json.dumps(att, ensure_ascii=False)),
                    )
                    att_count += 1

            if i % 20000 == 0:
                conn.commit()
                _log(f"Processed {i:,} lines… recipients~{rec_count:,} chats~{chat_count:,} messages~{msg_count:,} attachments~{att_count:,} resolved~{resolved:,}")

    conn.commit()
    _log(f"Inserted: recipients~{rec_count:,} chats~{chat_count:,} messages~{msg_count:,} attachments~{att_count:,} resolved~{resolved:,}")

    # Improved view: add authorName (join recipients on authorId)
    cur.executescript(
        """
        CREATE VIEW messages_enriched AS
        SELECT
            m.rowid AS message_rowid,
            m.chatId,
            m.authorId,
            ra.name AS authorName,
            m.outgoing,
            m.dateSentMs,
            m.dateSentIso,
            m.body,
            rr.name AS recipientName,
            rr.type AS recipientType
        FROM messages m
        LEFT JOIN chats c ON c.id = m.chatId
        LEFT JOIN recipients rr ON rr.id = c.recipientId
        LEFT JOIN recipients ra ON ra.id = m.authorId;
        """
    )

    if _fts5_available(conn):
        cur.executescript(
            """
            CREATE VIRTUAL TABLE messages_fts
            USING fts5(
                body,
                att_names,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        cur.execute(
            """
            INSERT INTO messages_fts(rowid, body, att_names)
            SELECT
                m.rowid,
                COALESCE(m.body,'') AS body,
                COALESCE((
                    SELECT group_concat(a.file_name, ' ')
                    FROM attachments a
                    WHERE a.message_rowid = m.rowid AND a.file_name != ''
                ), '') AS att_names
            FROM messages m;
            """
        )
        conn.commit()

    conn.commit()
    conn.close()

    try:
        if out_db.exists():
            out_db.unlink()
        tmp_db.replace(out_db)
        _log(f"Saved DB: {out_db}")
    except PermissionError as e:
        raise PermissionError(f"Could not replace {out_db}. Close any app using it and retry.") from e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store-raw-e164", action="store_true")
    args = ap.parse_args()
    build_db(Path(args.input), Path(args.out), bool(args.store_raw_e164), log=print)


if __name__ == "__main__":
    main()
