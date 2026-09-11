"""Live scanner — the piece that runs on the market.

Per cycle it:

1. loads the universe,
2. pulls daily OHLCV (plus, during a session, today's *forming* bar rebuilt from
   intraday data — the running high/low is what a "price touched the OB" alert needs),
3. replays the Pine state machine,
4. keeps events from the last ``alerts.recent_bars`` bar(s),
5. filters (liquidity, price band, zone age), dedupes (SQLite), renders, and
   dispatches to Telegram — queueing anything that could not be delivered.

``scan(live=True)``  → intraday mode: taps can fire on the running bar ("Once Per Bar").
``scan(live=False)`` → end-of-day mode: everything evaluated on closed bars only.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .alerts import (AlertDispatcher, build_buttons, dedupe_key, event_name, event_rank,
                     passes_filters, render_message)
from .chart import render_chart
from .data import LIVE_PROVIDERS, Bars, DataSource, read_universe
from .engine import EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, EngineResult, Event, Zone, run_engine
from .params import AlertConfig, DataConfig, Params, ScanConfig, TelegramConfig

log = logging.getLogger("precision_tap.scanner")


@dataclass
class SymbolScan:
    symbol: str
    ok: bool = False
    error: str = ""
    bars: int = 0
    last_date: str = ""
    price: float = float("nan")
    change_pct: float = float("nan")
    atr: float = float("nan")
    rvol: float = float("nan")
    live: bool = False
    events: List[Event] = field(default_factory=list)
    nearest_zone: Optional[Zone] = None
    nearest_entry: float = float("nan")
    nearest_stop: float = float("nan")
    dist_atr: float = float("nan")
    zone_state: str = ""
    taps: int = 0
    age: int = -1
    stale: bool = False
    events_closed: List[Any] = field(default_factory=list)
    mintick: float = 0.0
    result: Optional[EngineResult] = None
    df: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    source_error: str = ""

    @property
    def actionable(self) -> List[Event]:
        return [e for e in self.events if e.kind in (EV_TAP, EV_APPROACH, EV_CONFIRMED, EV_INVALID)]


@dataclass
class ScanReport:
    started_at: str = ""
    finished_at: str = ""
    mode: str = "eod"
    universe: int = 0
    usable: int = 0
    errors: int = 0
    symbols_with_zones: int = 0
    events_total: int = 0
    alerts: List[Tuple[SymbolScan, Event]] = field(default_factory=list)
    rows: List[SymbolScan] = field(default_factory=list)
    dispatch: Any = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at, "finished_at": self.finished_at, "mode": self.mode,
            "universe": self.universe, "usable": self.usable, "errors": self.errors,
            "symbols_with_zones": self.symbols_with_zones, "events_total": self.events_total,
            "alerts": [
                {"symbol": s.symbol, "event": event_name(e), "tap_no": e.tap_no,
                 "bar": e.bar, "date": str(e.ts), "level": e.level, "price": e.price,
                 "entry": e.zone.entry if e.zone else None, "stop": e.zone.stop if e.zone else None,
                 "top": e.zone.top if e.zone else None, "bot": e.zone.bot if e.zone else None,
                 "intrabar": e.intrabar, "reason": e.detail.get("reason", "")}
                for s, e in self.alerts],
            "notes": self.notes,
        }

    def console_table(self, limit: int = 40) -> str:
        head = f"{'symbol':<15}{'last':>12}{'chg%':>8}{'entry':>12}{'stop':>12}{'ΔATR':>7}" \
               f"{'age':>5}{'taps':>6}  state"
        out = [head, "-" * len(head)]
        rows = [r for r in self.rows if r.ok and r.nearest_zone is not None]
        rows.sort(key=lambda r: abs(r.dist_atr) if math.isfinite(r.dist_atr) else 9e9)
        for r in rows[:limit]:
            out.append(f"{r.symbol:<15}{r.price:>12,.2f}{r.change_pct:>8.2f}"
                       f"{r.nearest_entry:>12,.2f}{r.nearest_stop:>12,.2f}"
                       f"{r.dist_atr:>7.2f}{r.age:>5}{r.taps:>6}  {r.zone_state}")
        if len(rows) > limit:
            out.append(f"… {len(rows) - limit} more")
        if not rows:
            out.append("(no live precision zones in this universe)")
        return "\n".join(out)

    @property
    def summary_line(self) -> str:
        kinds: Dict[str, int] = {}
        for _, e in self.alerts:
            kinds[event_name(e)] = kinds.get(event_name(e), 0) + 1
        acc = " ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "no alerts"
        return (f"universe={self.universe} usable={self.usable} errors={self.errors} "
                f"zones={self.symbols_with_zones} | {acc}")


class Scanner:
    def __init__(self, cfg: ScanConfig, *, store=None, telegram: Optional[object] = None,
                 dry_run: bool = False, out_dir: Optional[Path] = None):
        self.cfg = cfg
        self.store = store
        self.dry_run = dry_run
        self.out_dir = Path(out_dir or cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.source = DataSource(cfg.data, live=False)
        self.telegram = telegram
        self._tg_own = False
        if self.telegram is None and not dry_run:
            from .telegram import TelegramClient
            self.telegram = TelegramClient(cfg.telegram)
            self._tg_own = True
        # An unconfigured client is worse than no client: `dispatch` would hand it
        # every alert, the send would raise, and the alert would land in the retry
        # queue where it is retried (and dropped) forever — the scanner looks
        # alive while nothing is ever delivered.  Refuse to build that path.
        if self.telegram is not None and not getattr(self.telegram, "configured", True):
            if not dry_run:
                log.warning("telegram token/chat not configured — alerts will be logged only "
                            "(set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env)")
            self.telegram = None
        self.dispatcher = AlertDispatcher(cfg.alert, self.telegram, store, cfg.params,
                                         render_charts=(self._render_chart if cfg.alert.chart else None),
                                         dry_run=dry_run or self.telegram is None)

    def close(self) -> None:
        if self._tg_own and self.telegram is not None:
            try:
                sess = getattr(self.telegram, "_session", None)
                if sess is not None:
                    sess.close()
            except Exception:
                pass

    # ── universe ─────────────────────────────────────────────────────────
    def universe(self, symbols: Optional[Sequence[str]] = None) -> List[str]:
        if symbols:
            return [s.upper() for s in symbols if str(s).strip()]
        d = self.cfg.data
        if d.universe:
            return [s.upper() for s in d.universe]
        return read_universe(d.universe_file or None)

    # ── one symbol ───────────────────────────────────────────────────────
    def scan_symbol(self, symbol: str, *, live: bool, end: Optional[str] = None) -> SymbolScan:
        """Fetch + evaluate one symbol (used by tests and ad-hoc `scan --symbol`)."""
        bars: Bars = self.source.get(symbol, end=end)
        if not bars.ok:
            st = SymbolScan(symbol=symbol, error=bars.error or "no data")
            return st
        return self._scan_frame(symbol, bars, live=bool(live and bars.live),
                                historical=bool(end))

    # ── full cycle ───────────────────────────────────────────────────────
    def scan(self, symbols: Optional[Sequence[str]] = None, *, live: Optional[bool] = None,
             end: Optional[str] = None, send: bool = True, workers: Optional[int] = None,
             progress: bool = True) -> ScanReport:
        """One full cycle over the universe. Returns a :class:`ScanReport`."""
        cfg = self.cfg
        t_start = time.time()
        if live is None:
            live = _market_open(cfg.live.market_timezone, session=(cfg.live.session_open,
                                                                   cfg.live.session_close)) and not end
        self.source.live = bool(live)
        self._frames: Dict[str, pd.DataFrame] = {}
        uni = self.universe(symbols)
        rep = ScanReport(started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         mode="live" if live and not end else "eod", universe=len(uni))
        run_id = self.store.start_run(rep.mode, len(uni)) if self.store else 0

        src = DataSource(cfg.data, live=bool(live and not end))
        frames: Dict[str, Bars] = src.get_many(uni, workers=workers or cfg.live.max_workers,
                                              progress=progress)
        self._frames = {k: v.df for k, v in frames.items() if v.ok}
        items: List[Tuple[SymbolScan, Event, Dict[str, Any]]] = []
        for sym in uni:
            b = frames.get(sym)
            if b is None or not b.ok:
                rep.errors += 1
                continue
            st = self._scan_frame(sym, b, live=bool(live and not end) and bool(b.live),
                                  historical=bool(end))
            rep.rows.append(st)
            if not st.ok:
                rep.errors += 1
                continue
            if st.stale and self.cfg.alert.skip_stale_bars:
                rep.notes.append(f"{sym}: last bar {st.last_date} is not today's session — skipped")
                continue
            rep.usable += 1
            if st.nearest_zone is not None:
                rep.symbols_with_zones += 1
            rep.events_total += len(st.actionable)
            for ev in self._recent_events(st, live=live and not end):
                ctx = self._context(st, ev)
                ok, why = passes_filters(ev, ctx, cfg.alert,
                                         already_seen=(lambda e: self.store.seen(dedupe_key(e, cfg.alert)))
                                         if self.store else None)
                if ok:
                    items.append((st, ev, ctx))
                else:
                    log.debug("skip %s %s: %s", sym, event_name(ev), why)
        # highest-value signal first, then alphabetical — the same order Telegram sees
        items.sort(key=lambda ic: (event_rank(ic[1]), ic[1].symbol))
        rep.alerts = [(st, ev) for st, ev, _ in items]
        if send and items:
            rep.dispatch = self.dispatcher.dispatch([(ev, ctx) for _, ev, ctx in items])
        elif items:
            rep.dispatch = {"would_send": len(items)}
        rep.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rep.notes.append(f"{time.time() - t_start:.1f}s")
        if self.store and run_id:
            self.store.finish_run(run_id, usable=rep.usable, alerts=len(items),
                                  queued=getattr(rep.dispatch, "queued", 0), errors=rep.errors,
                                  note=rep.summary_line)
        self._write_report(rep)
        return rep

    def _params_for(self, symbol: str) -> Params:
        """Resolve ``syminfo.mintick`` per symbol (NSE/BSE tick = 0.05, US = 0.01)."""
        cache = getattr(self, "_pmap", None)
        if cache is None:
            cache = self._pmap = {}
        if symbol in cache:
            return cache[symbol]
        p = self.cfg.params
        tick = None
        if hasattr(self.source, "tick_for"):
            try:
                tick = self.source.tick_for(self.source.fetch_symbol(symbol))
            except ValueError:
                tick = None                      # DataSource.get() already reported the error
        if tick is not None and tick != p.mintick:
            p = p.replace(mintick=tick)
        cache[symbol] = p
        return p

    def _tracks_live_market(self) -> bool:
        """Is this run reading a feed that follows the exchange clock?

        Only then does "the newest bar is not a recent session" mean anything.
        A csv/synthetic/backtest replay is frozen by definition.
        """
        return self.cfg.data.provider in LIVE_PROVIDERS

    def _scan_frame(self, symbol: str, bars: Bars, *, live: bool,
                    historical: bool = False) -> SymbolScan:
        p = self._params_for(symbol)
        st = SymbolScan(symbol=symbol)
        df = bars.df
        if len(df) < max(self.cfg.data.min_bars, p.warmup):
            st.error = f"only {len(df)} bars"
            return st
        st.ok = True
        st.bars = len(df)
        st.live = live
        st.stale = (not historical and self._tracks_live_market()
                    and _is_stale_bar(df.index[-1], self.cfg.live.market_timezone,
                                      strict=bool(live)))
        res = run_engine(df, p, symbol=symbol, intrabar_last=live, zone_cap=10 ** 6)
        st.result, st.df, st.events = res, df, res.events
        # Closed-bar parity pass.  On the forming bar the engine deliberately
        # creates no zone and confirms no defence (that is what makes an intraday
        # tap non-repainting), so `alerts.match_indicator_100` replays the same
        # frame with the last bar *closed* and keeps the events the indicator
        # would already have printed — including every `confirmed`, which can
        # never come from the intrabar pass.
        if live and self.cfg.alert.match_indicator_100:
            closed = run_engine(df, p, symbol=symbol, intrabar_last=False, zone_cap=10 ** 6)
            st.events_closed = [e for e in closed.events if not e.intrabar]
        A = res.arrays
        i = len(df) - 1
        st.price = float(A["close"][i])
        st.atr = float(A["atr"][i]) if np.isfinite(A["atr"][i]) else float("nan")
        st.rvol = float(A["rvol"][i])
        if i:
            st.change_pct = float((A["close"][i] / A["close"][i - 1] - 1.0) * 100.0)
        st.last_date = str(df.index[-1])[:10]
        ne, ns = res.flags["nearest_entry"][i], res.flags["nearest_stop"][i]
        if np.isfinite(ne):
            st.nearest_entry, st.nearest_stop = float(ne), float(ns)
            st.dist_atr = (st.price - st.nearest_entry) / st.atr if st.atr else float("nan")
            st.nearest_zone = min((z for z in res.zones if z.state >= 0),
                                  key=lambda z: abs(st.price - z.entry), default=None)
            if st.nearest_zone is not None:
                z = st.nearest_zone
                st.zone_state, st.taps, st.age = z.state_name, z.taps, i - z.born
        st.meta = dict(getattr(bars, "meta", {}) or {})
        st.source_error = getattr(bars, "error", "") or ""
        return st

    # ── helpers ──────────────────────────────────────────────────────────
    def _recent_events(self, st: SymbolScan, *, live: bool) -> List[Event]:
        """Events inside the alert window.

        Live mode: taps/approaches on the *forming* bar (== TradingView "Once Per
        Bar"), plus closed-bar events on the last completed bar from the parity pass
        (== "Once Per Bar Close"). EOD mode: closed bars only, so the alert set is
        identical to the indicator's confirmed signal set.
        """
        res = st.result
        if res is None or not (st.events or st.events_closed):
            return []
        n = len(res.index)
        back = max(1, int(self.cfg.alert.recent_bars))
        out: List[Event] = []
        if live:
            out = [e for e in st.events if e.bar >= n - 1]
            out = [e for e in out if e.kind in (EV_TAP, EV_APPROACH, EV_INVALID)]
            prev = [e for e in (st.events_closed or []) if n - 1 - back <= e.bar < n - 1]
            seen = {(e.kind, e.bar, e.zid) for e in out}
            out.extend(e for e in prev if (e.kind, e.bar, e.zid) not in seen)
        else:
            out = [e for e in st.events if e.bar >= n - back]
        return out

    def _context(self, st: SymbolScan, ev: Event) -> Dict[str, Any]:
        res, df = st.result, st.df
        meta = getattr(st, "meta", {}) or {}
        ctx: Dict[str, Any] = {
            "price": st.price, "change_pct": st.change_pct, "atr": st.atr, "rvol": st.rvol,
            "timeframe": self.cfg.data.interval, "last_bar": st.last_date,
            "exchange": (meta.get("fullExchangeName") or meta.get("exchangeName")
                         or self.cfg.alert.default_exchange),
            "currency": meta.get("currency", "USD"), "zone_method": self.cfg.params.zone_method,
            "live": st.live, "bars": st.bars,
        }
        if ev.zone is not None:
            z = ev.zone
            ctx["entry"] = z.entry
            ctx["origin_date"] = str(df.index[z.origin])[:10] if df is not None and z.origin < len(df) else ""
        vol = df["volume"].tail(21).to_numpy(float) if df is not None else np.array([])
        px = df["close"].tail(21).to_numpy(float) if df is not None else np.array([])
        if len(vol) >= 5:
            ctx["avg_dollar_volume"] = float(np.median(vol[1:] * px[1:]))
        return ctx

    def _render_chart(self, ev: Event, ctx: Dict[str, Any]) -> Optional[str]:
        df = getattr(self, "_frames", {}).get(ev.symbol)
        if df is None:
            bars = self.source.get(ev.symbol, use_cache=True)
            df = bars.df if bars.ok else None
        if df is None or len(df) < 20:
            return None
        out = self.out_dir / "charts" / f"{ev.symbol}_{ev.kind}_{int(time.time())}.png"
        return render_chart(df, ev.zone and [ev.zone] or [], ev, out_path=out,
                            symbol=ev.symbol, bars=self.cfg.alert.chart_bars,
                            timeframe=self.cfg.data.interval,
                            title_extra=f"{event_name(ev).upper()} {str(ev.ts)[:10]}")

    def _write_report(self, rep: ScanReport) -> None:
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self.out_dir / f"scan_{rep.mode}_{stamp}.json"
            path.write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8")
            lines = [f"# Precision Tap · {rep.mode} scan {stamp}", "",
                     "```", rep.console_table(self.cfg.alert.top_zones if hasattr(self.cfg.alert, "top_zones") else 40), "```",
                     "", rep.summary_line, ""]
            if rep.alerts:
                lines.append("## Alerts")
                for st, ev in rep.alerts:
                    z = ev.zone
                    tail = (f" — zone {_lvl(z.top)}/{_lvl(z.bot)}, buy {_lvl(z.entry)}, "
                            f"stop {_lvl(z.stop)}") if z is not None else ""
                    lines.append(f"- **{ev.symbol}** {event_name(ev)} @ {_lvl(ev.level)}{tail}")
            md = self.out_dir / f"scan_{rep.mode}_{stamp}.md"
            md.write_text("\n".join(lines), encoding="utf-8")
            rep.notes.append(str(md))
        except Exception as exc:                        # reporting must never break a scan
            log.debug("report write failed: %s", exc)

    def watchlist(self, symbols: Optional[Sequence[str]] = None, *, live: Optional[bool] = None,
                  limit: int = 40) -> ScanReport:
        """Non-alerting scan used by the heartbeat: nearest live level per symbol."""
        rep = self.scan(symbols, live=live, send=False)
        rep.rows = rep.rows[:limit]
        return rep


def _lvl(v: Any) -> str:
    try:
        f = float(v)
        return f"{f:,.2f}" if math.isfinite(f) else "—"
    except (TypeError, ValueError):
        return "—"


_MARKET_HOURS = {
    "Asia/Kolkata": ((9, 15), (15, 30)),      # NSE / BSE continuous session
    "Asia/Calcutta": ((9, 15), (15, 30)),
    "America/New_York": ((9, 30), (16, 0)),
    "America/Nassau": ((9, 30), (16, 0)),
    "Europe/London": ((8, 0), (16, 30)),
    "Asia/Tokyo": ((9, 0), (15, 0)),
}


def _market_open(tzname: str = "Asia/Kolkata", now: Optional[datetime] = None,
                 session: Optional[Tuple[str, str]] = None) -> bool:
    """Cheap session check: weekday + regular hours (NSE 09:15–15:30 by default).

    Used only to decide intraday-vs-EOD polling; holidays self-correct because the
    stale-bar guard refuses to alert when the last bar is not today's session.
    """
    try:
        from zoneinfo import ZoneInfo
        now = now or datetime.now(ZoneInfo(tzname))
    except Exception:
        now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    if session and session[0] and session[1]:
        try:
            oh, om = (int(x) for x in str(session[0]).split(":"))
            ch, cm = (int(x) for x in str(session[1]).split(":"))
            rng = ((oh, om), (ch, cm))
        except Exception:
            rng = _MARKET_HOURS.get(tzname, ((9, 15), (15, 30)))
    else:
        rng = _MARKET_HOURS.get(tzname, ((9, 15), (15, 30)))
    if rng[0] == rng[1]:
        return 1 <= now.hour <= 23
    mins = now.hour * 60 + now.minute
    return (rng[0][0] * 60 + rng[0][1]) <= mins < (rng[1][0] * 60 + rng[1][1])


def _session_now(tzname: str):
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tzname))
    except Exception:
        return datetime.now().astimezone()


def _is_stale_bar(bar_ts, tzname: str, *, strict: bool = False) -> bool:
    """True when the newest bar is too old to alert on.

    ``strict`` (intraday) — the bar must be *today's* session. A delayed feed that
    still shows yesterday's close must not fire a "price is tapping the OB right
    now" alert; the ``skip_stale_bars`` guard exists exactly for that.

    Not strict (closed-bar scan) — the most recent *completed* session is enough,
    so a Monday-morning scan still alerts on Friday's close, and a Saturday run
    still reports Friday. Only a feed that has fallen several sessions behind is
    treated as stale (holidays are absorbed by the slack).
    """
    try:
        bar = pd.Timestamp(bar_ts)
        if bar.tzinfo is not None:
            try:
                bar = bar.tz_convert(tzname)
            except Exception:
                pass
            bar = bar.tz_localize(None)
        bday = bar.date()
    except Exception:
        return False
    now = _session_now(tzname).date()
    if bday >= now:                                  # today (or a tz-shifted stamp)
        return False
    if strict:
        return not _same_session(bar_ts, tzname)
    try:
        return int(np.busday_count(bday, now)) > 3   # ≥4 sessions behind == broken feed
    except Exception:
        return (now - bday).days > 6


def _same_session(bar_ts, tzname: str) -> bool:
    """Is the last bar from today's session?  (holiday / stale-feed guard)

    Daily bars are date-stamped, so a naive index is read as the exchange session
    date; a tz-aware one is converted.  Both the exchange-local date and the UTC
    date are accepted so a UTC-stamped feed at 16:00Z (21:30 IST) is not mistaken
    for a stale bar.
    """
    try:
        from zoneinfo import ZoneInfo
        dates = {datetime.now(ZoneInfo(tzname)).date()}
    except Exception:
        dates = {datetime.now().date()}
    dates.add(datetime.now(timezone.utc).date())
    try:
        bar = pd.Timestamp(bar_ts)
        if bar.tzinfo is not None:
            try:
                bar = bar.tz_convert(tzname)
            except Exception:
                pass
            bar = bar.tz_localize(None)
        return bar.date() in dates
    except Exception:
        return True
