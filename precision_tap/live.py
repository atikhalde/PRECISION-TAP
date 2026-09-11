"""Always-on scheduling for the live market.

Two triggers per trading day:

* **intraday polling** every ``live.intraday_poll_minutes`` while the session is
  open — this is what catches an intrabar touch of the OB during the trading day;
* **scheduled scans** at ``live.scan_times`` (default 10:00 / 15:15 / 16:15 local
  exchange time: mid-session, first minutes after the close, and after the EOD
  data settles).  A scheduled scan outside market hours runs in closed-bar mode,
  which is the authoritative "yesterday tapped my OB" check.

Plus an optional daily heartbeat so you always know the loop is alive, a graceful
SIGINT/SIGTERM shutdown, and a restart-safe "already fired" record (SQLite).
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from .params import ScanConfig
from .scanner import Scanner, _market_open

log = logging.getLogger("precision_tap.live")


def _local_now(tzname: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tzname))
    except Exception:
        return datetime.now().astimezone()


def _parse_hhmm(text: str) -> Optional[Tuple[int, int]]:
    """Parse ``"HH:MM"``; return ``None`` for anything malformed.

    Returning ``(0, 0)`` — as this used to — would silently turn a typo in
    ``live.scan_times`` / ``live.heartbeat_daily_time`` into a job scheduled for
    midnight, i.e. a scan that fires at the one moment of the day when there is
    never a fresh bar to look at.
    """
    try:
        hh, mm = str(text).strip().split(":")
        hh_i, mm_i = int(hh), int(mm)
    except Exception:
        log.warning("ignoring malformed time %r (want HH:MM)", text)
        return None
    if not (0 <= hh_i <= 23 and 0 <= mm_i <= 59):
        log.warning("ignoring out-of-range time %r (want HH:MM)", text)
        return None
    return hh_i, mm_i


@dataclass
class LiveLoop:
    cfg: ScanConfig
    scanner: Scanner
    store: Any = None
    heartbeat: bool = True
    max_cycles: int = 0                 # 0 = forever (handy for tests/CI)
    on_scan: Optional[Callable[[Any], None]] = None
    stop_after: Optional[float] = None  # monotonic deadline (tests)
    _stop: bool = field(default=False, repr=False)
    _last_poll: Optional[float] = field(default=None, repr=False)
    _fired: Dict[str, str] = field(default_factory=dict, repr=False)
    _last_day: str = ""
    _cycles: int = 0

    # ── scheduling ───────────────────────────────────────────────────────
    def _due(self, now: datetime) -> Optional[str]:
        """Which scan (if any) is due right now? Returns a tag or None."""
        live = self.cfg.live
        key = now.date().isoformat()
        if self._last_day != key:                      # new day: reset the fired set
            self._fired = {}
            self._last_day = key
        day_ok = now.strftime("%a").lower()[:3] in {d.lower()[:3] for d in live.trading_days}
        if day_ok:
            for t in (live.scan_times or []):
                parsed = _parse_hhmm(t)
                if parsed is None:
                    continue
                hh, mm = parsed
                due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                tag = f"scheduled@{t}"
                if now >= due and self._fired.get(tag) != key:
                    self._fired[tag] = key
                    return tag
            if live.intraday_poll_minutes and _market_open(live.market_timezone, now,
                                                            (live.session_open, live.session_close)):
                if self._poll_due(live.intraday_poll_minutes):
                    return "intraday"
        hb = str(live.heartbeat_daily_time or "").strip()
        if self.heartbeat and hb:
            parsed = _parse_hm(hb)
            if parsed is not None:
                hh, mm = parsed
                due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if now >= due and self._fired.get("heartbeat") != key:
                    self._fired["heartbeat"] = key
                    return "heartbeat"
        return None

    def _poll_due(self, minutes: int) -> bool:
        """Has an intraday poll interval elapsed?

        ``_last_poll`` starts at ``None`` (not ``0.0``): ``time.monotonic()`` has
        an arbitrary origin, so "0" only *accidentally* means "long ago" — on a
        freshly booted host it can read as "just now" and swallow the first poll.
        """
        if self._last_poll is None:
            return True
        return time.monotonic() - self._last_poll >= max(0.0, float(minutes) * 60.0)

    def _sleep_seconds(self, now: datetime) -> float:
        live = self.cfg.live
        cand: List[float] = []
        if live.intraday_poll_minutes and _market_open(live.market_timezone, now,
                                                        (live.session_open, live.session_close)):
            if self._last_poll is None:
                cand.append(0.0)
            else:
                cand.append(max(20.0, live.intraday_poll_minutes * 60
                                - (time.monotonic() - self._last_poll)))
        for t in (live.scan_times or []):
            parsed = _parse_hhmm(t)
            if parsed is None:
                continue
            hh, mm = parsed
            due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if due <= now:
                due += timedelta(days=1)
            cand.append(max(20.0, (due - now).total_seconds()))
        hb = str(live.heartbeat_daily_time or "").strip()
        if self.heartbeat and hb:
            parsed = _parse_hm(hb)
            if parsed is not None:
                # The heartbeat used to be checked only when some *other* wake
                # fired, so with scans at 15:35/16:10 a 15:45 heartbeat went out
                # after 17:10.  Waking for it keeps "the loop is alive" on time.
                hh, mm = parsed
                due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
                if due <= now:
                    due += timedelta(days=1)
                cand.append(max(20.0, (due - now).total_seconds()))
        if not cand:
            cand = [30 * 60.0]
        return float(min(min(cand), 3600.0))

    # ── loop ─────────────────────────────────────────────────────────────
    def install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            log.info("received signal %s — finishing the current cycle then exiting", signum)
            self._stop = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):               # not the main thread
                pass

    def run(self) -> int:
        live = self.cfg.live
        self.install_signal_handlers()
        log.info("live loop started · tz=%s · poll=%s min · scans=%s · universe=%s",
                 live.market_timezone, live.intraday_poll_minutes, live.scan_times,
                 len(self.scanner.universe()))
        if not getattr(self.scanner.dispatcher, "deliverable", False):
            log.warning("no working telegram transport — alerts will be logged, not sent "
                        "(check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env)")
        if live.run_on_startup:
            self._scan("startup")
        while not self._stop:
            if self.max_cycles and self._cycles >= self.max_cycles:
                break
            if self.stop_after and time.monotonic() > self.stop_after:
                log.info("stop_after deadline reached")
                break
            now = _local_now(live.market_timezone)
            tag = self._due(now)
            if tag == "heartbeat":
                self._send_heartbeat(now)
            elif tag:
                self._scan(tag)
            time.sleep(self._sleep_seconds(_local_now(live.market_timezone)))
        log.info("live loop stopped after %d cycle(s)", self._cycles)
        return 0

    def _scan(self, tag: str):
        """Run one cycle. Returns the :class:`ScanReport` (or ``None`` on crash).

        The mode follows the **market**, not the trigger: a scan scheduled for
        11:30 or 13:30 lands squarely inside the session, and running it in
        closed-bar mode would (a) skip the intraday rebuild of today's bar and
        (b) read the previous session's close as if it were the live price —
        so every mid-session tap would be missed, and a stale tap would be
        re-reported. ``--once`` (the cron entry point) was hit hardest: its tag
        is neither ``intraday`` nor ``startup``, so it could never go live.
        """
        live = self.cfg.live
        open_now = _market_open(live.market_timezone, session=(live.session_open, live.session_close))
        mode = "live" if open_now else "eod"
        self._last_poll = time.monotonic()
        self._cycles += 1
        log.info("scan cycle: %s (mode=%s, market %s)", tag, mode, "OPEN" if open_now else "CLOSED")
        try:
            rep = self.scanner.scan(live=(mode == "live"), progress=False)
        except Exception as exc:                        # never die on a bad cycle
            log.exception("scan cycle failed: %s", exc)
            if self.store is not None:
                try:
                    self.store.finish_run(0, note=f"cycle {tag} failed: {exc}"[:200])
                except Exception:
                    pass
            return None
        try:
            self.scanner.dispatcher.retry_pending()
        except Exception as exc:
            log.debug("retry pass failed: %s", exc)
        log.info("cycle %s → %s", tag, rep.summary_line)
        # The summary line counts *found* signals, not delivered ones — a cycle
        # that matched three taps and delivered none used to log exactly the same
        # line as a healthy one, which is how "the scanner runs fine but Telegram
        # is silent" stays invisible for a whole session.
        d = rep.dispatch
        if d is not None and not isinstance(d, dict):
            log.info("cycle %s → delivery %s", tag, d)
            if getattr(d, "undelivered", 0):
                log.error("cycle %s: %d alert(s) did NOT reach Telegram — %s",
                          tag, d.undelivered,
                          " · ".join(dict.fromkeys(getattr(d, "errors", []) or []))[:300] or d)
        elif d is not None:
            log.info("cycle %s → %s", tag, d)
        else:
            log.info("cycle %s → nothing to send", tag)
        for note in rep.notes:
            if note.startswith(("DELIVERY PROBLEM", "NOT SENT", "FEED PROBLEM",
                                "no usable symbols", "nothing to send")):
                log.warning("cycle %s: %s", tag, note)
        if self.on_scan is not None:
            try:
                self.on_scan(rep)
            except Exception:
                log.debug("on_scan hook failed", exc_info=True)
        return rep

    def _send_heartbeat(self, now: datetime) -> None:
        if not self.heartbeat or self.scanner.telegram is None:
            return
        counts = {}
        if self.store is not None:
            try:
                counts["alerts_7d"] = self.store.alert_counts(7)
            except Exception:
                pass
        text = ("💓 Precision Tap alive · " + now.strftime("%Y-%m-%d %H:%M ") +
                f"{self.cfg.live.market_timezone} · universe "
                f"{len(self.scanner.universe())} · cycles {self._cycles}" +
                (f" · alerts/7d {counts.get('alerts_7d')}" if counts else ""))
        try:
            self.scanner.dispatcher.broadcast(text)
        except Exception as exc:
            log.debug("heartbeat failed: %s", exc)


def _parse_hm(text: str) -> Optional[Tuple[int, int]]:
    return _parse_hhmm(text)
