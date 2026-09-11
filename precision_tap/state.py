"""Persistent alert state (SQLite) — de-duplication, delivery retries, audit trail.

A live loop must never re-ping the same Tap 1 after a restart, and must never lose
an alert because Telegram was briefly unreachable. Both are handled here:

``alerts``       one row per (symbol, event, zone, bar-date) — ``INSERT OR IGNORE``
                 makes the whole scanner idempotent.
``sent``         0 = queued for retry, 1 = delivered, 2 = failed permanently.
``zones``        last-known state per (symbol, zone) — powers "what is still live?"
                 queries and a warm-start context for the next scan.
``runs``         one row per scan cycle (for the heartbeat message and dashboards).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger("precision_tap.state")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    key         TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    event       TEXT NOT NULL,
    zone_id     INTEGER,
    bar_date    TEXT,
    bar_time    TEXT,
    level       REAL,
    price       REAL,
    stop        REAL,
    sent        INTEGER NOT NULL DEFAULT 0,   -- 0 queued, 1 delivered, 2 failed
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    payload     TEXT,                          -- full event json (for resend + audit)
    message     TEXT,                          -- rendered text (for resend)
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_symbol_idx ON alerts(symbol);
CREATE INDEX IF NOT EXISTS alerts_sent_idx   ON alerts(sent, created_at);

CREATE TABLE IF NOT EXISTS zones (
    symbol      TEXT NOT NULL,
    zone_id     INTEGER NOT NULL,
    born_date   TEXT,
    top         REAL, bot REAL, entry REAL, entry0 REAL, stop REAL,
    state       INTEGER, taps INTEGER, departed INTEGER,
    first_tap_date TEXT, last_tap_date TEXT, confirm_date TEXT,
    updated_at  TEXT,
    PRIMARY KEY (symbol, zone_id)
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    mode        TEXT,
    symbols     INTEGER DEFAULT 0,
    usable      INTEGER DEFAULT 0,
    alerts      INTEGER DEFAULT 0,
    queued      INTEGER DEFAULT 0,
    errors      INTEGER DEFAULT 0,
    note        TEXT
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class AlertRow:
    key: str
    symbol: str
    event: str
    message: str
    payload: Dict[str, Any]
    attempts: int = 0
    sent: int = 0

    @property
    def is_tap1(self) -> bool:
        return self.event == "tap1"


class StateStore:
    """Small, dependency-free persistence layer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                pass
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ── alerts ───────────────────────────────────────────────────────────
    def seen(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM alerts WHERE key = ?", (key,)).fetchone()
        return row is not None

    def record_alert(self, key: str, *, symbol: str, event: str, zone_id: Optional[int] = None,
                     bar_date: str = "", bar_time: str = "", level: float = float("nan"),
                     price: float = float("nan"), stop: float = float("nan"),
                     payload: Optional[Dict[str, Any]] = None, message: str = "",
                     sent: bool = False, error: str = "") -> bool:
        """Insert a new alert row. Returns True if it is new (i.e. should be sent)."""
        now = utcnow()
        vals = (key, symbol, event, zone_id, bar_date, bar_time,
                _f(level), _f(price), _f(stop), 1 if sent else 0, 1 if sent else 0,
                error if not sent else "", json.dumps(payload or {}, separators=(",", ":"), default=str),
                message, now, now)
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO alerts (key, symbol, event, zone_id, bar_date, bar_time,"
                " level, price, stop, sent, attempts, error, payload, message, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", vals)
            self._conn.commit()
        return bool(cur.rowcount)

    def mark(self, key: str, *, sent: bool, error: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE alerts SET sent = ?, error = ?, attempts = attempts + 1, updated_at = ?"
                " WHERE key = ?", (1 if sent else 0, error[:400], utcnow(), key))
            self._conn.commit()

    def pending(self, limit: int = 20, max_attempts: int = 6) -> List[AlertRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE sent = 0 AND attempts < ?"
                " ORDER BY created_at ASC LIMIT ?", (max_attempts, int(limit))).fetchall()
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            out.append(AlertRow(key=r["key"], symbol=r["symbol"], event=r["event"],
                                message=r["message"] or "", payload=payload,
                                attempts=r["attempts"], sent=r["sent"]))
        return out

    def give_up(self, keys: Sequence[str]) -> None:
        if not keys:
            return
        with self._lock:
            self._conn.executemany("UPDATE alerts SET sent = 2, updated_at = ? WHERE key = ?",
                                   [(utcnow(), k) for k in keys])
            self._conn.commit()

    def recent_alerts(self, limit: int = 50, symbol: Optional[str] = None,
                      since: Optional[str] = None) -> List[Dict[str, Any]]:
        q, args = "SELECT * FROM alerts", []
        wh: List[str] = []
        if symbol:
            wh.append("symbol = ?")
            args.append(symbol)
        if since:
            wh.append("created_at >= ?")
            args.append(since)
        if wh:
            q += " WHERE " + " AND ".join(wh)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d.get("payload") or "{}")
            except Exception:
                pass
            out.append(d)
        return out

    def alert_counts(self, days: int = 7) -> int:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).replace(microsecond=0).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) c FROM alerts WHERE created_at >= ? AND event IN "
                "('tap1','tap','confirmed')", (since,)).fetchone()
        return int(row["c"] if row else 0)

    # ── zones ────────────────────────────────────────────────────────────
    def upsert_zones(self, symbol: str, zones: Iterable[Any], *, dates: Sequence[Any] = ()) -> None:
        now = utcnow()

        def date_of(bar: int) -> str:
            try:
                return str(dates[bar])[:10] if dates and 0 <= bar < len(dates) else ""
            except Exception:
                return ""

        rows = []
        for z in zones:
            rows.append((symbol, z.zid, date_of(z.born), _f(z.top), _f(z.bot), _f(z.entry), _f(z.entry0),
                         _f(z.stop), z.state, z.taps, 1 if z.departed else 0,
                         date_of(z.tap_bars[0]) if z.tap_bars else "",
                         date_of(z.tap_bars[-1]) if z.tap_bars else "",
                         date_of(z.confirm_bar) if z.confirm_bar >= 0 else "", now))
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO zones (symbol, zone_id, born_date, top, bot, entry, entry0, stop,"
                " state, taps, departed, first_tap_date, last_tap_date, confirm_date, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._conn.commit()

    def live_zones(self, symbol: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        q = "SELECT * FROM zones WHERE state >= 0"
        args: List[Any] = []
        if symbol:
            q += " AND symbol = ?"
            args.append(symbol)
        q += " ORDER BY updated_at DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, args).fetchall()]

    # ── runs ─────────────────────────────────────────────────────────────
    def start_run(self, mode: str, symbols: int) -> int:
        with self._lock:
            cur = self._conn.execute("INSERT INTO runs (started_at, mode, symbols) VALUES (?,?,?)",
                                     (utcnow(), mode, int(symbols)))
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def finish_run(self, run_id: int, *, usable: int = 0, alerts: int = 0, queued: int = 0,
                   errors: int = 0, note: str = "") -> None:
        if not run_id:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, usable = ?, alerts = ?, queued = ?, errors = ?,"
                " note = ? WHERE id = ?", (utcnow(), int(usable), int(alerts), int(queued),
                                           int(errors), note[:500], run_id))
            self._conn.commit()

    def last_run(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def purge(self, days: int = 90) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).replace(microsecond=0).isoformat()
        with self._lock:
            cur = self._conn.execute("DELETE FROM alerts WHERE created_at < ?", (cutoff,))
            self._conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))
            self._conn.commit()
        return int(cur.rowcount or 0)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _f(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f
