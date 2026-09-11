"""Alert rendering, filtering and dispatch.

The scanner hands this module a list of engine ``Event``s plus a small
``context`` dict per symbol (live price, exchange, liquidity, ATR…).  This module:

* applies the user filters (liquidity, price band, zone age, per-day dedupe),
* renders a compact, monospaced Telegram message (HTML / MarkdownV2 / plain),
* builds inline buttons (chart + quote links),
* delegates delivery, and stores everything in :mod:`precision_tap.state` so an
  outage becomes a retry instead of a lost signal.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .engine import EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, Event, Zone
from .params import AlertConfig
from .telegram import TelegramClient, escape_markdownv2, strip_html

log = logging.getLogger("precision_tap.alerts")

RULE_LINE = "─" * 30

EMOJI = {
    EV_TAP: "🎯",
    EV_APPROACH: "👀",
    EV_CONFIRMED: "🛡",
    EV_NEW: "🟦",
    EV_INVALID: "❌",
}
TITLE = {
    "tap1": "PRECISION OB · TAP 1",
    EV_TAP: "PRECISION OB · REPEAT TAP",
    EV_APPROACH: "APPROACHING PRECISION OB",
    EV_CONFIRMED: "OB DEFENCE CONFIRMED",
    EV_NEW: "NEW PRECISION OB",
    EV_INVALID: "PRECISION OB INVALIDATED",
}


def _num(v: Any, nd: int = 2, dash: str = "—") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(f):
        return dash
    if abs(f) >= 1000:
        return f"{f:,.{nd}f}"
    return f"{f:.{nd}f}"


def _pct(v: Any, nd: int = 2, dash: str = "—") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    return dash if not math.isfinite(f) else f"{f:+.{nd}f}%"


def event_name(ev: Event) -> str:
    if ev.kind == EV_TAP:
        return "tap1" if ev.tap_no == 1 else "tap"
    return ev.kind


def event_rank(ev: Event) -> int:
    return {"tap1": 0, "tap": 1, "confirmed": 2, "approach": 3, "new_ob": 4, "invalidated": 5}.get(
        event_name(ev), 9)


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def passes_filters(ev: Event, ctx: Dict[str, Any], cfg: AlertConfig, *,
                   already_seen=None) -> Tuple[bool, str]:
    name = event_name(ev)
    allow = set(cfg.events) | ({"invalidated"} if cfg.include_invalidations else set())
    if name == "tap" and "tap" not in allow:
        return False, "repeat tap not enabled"
    if name not in allow:
        return False, f"{name} not enabled"
    px = float(ctx.get("price") or ev.price or 0.0)
    if cfg.min_price and px < cfg.min_price:
        return False, f"price {px:g} < min_price {cfg.min_price:g}"
    if cfg.max_price and px > cfg.max_price:
        return False, f"price {px:g} > max_price {cfg.max_price:g}"
    adv = float(ctx.get("avg_dollar_volume") or 0.0)
    if cfg.min_liquidity_dollar_volume and adv < cfg.min_liquidity_dollar_volume:
        return False, f"illiquid (${adv/1e6:.1f}M/day < ${cfg.min_liquidity_dollar_volume/1e6:.0f}M)"
    z = ev.zone
    if z is not None and cfg.max_age_bars and (ev.bar - z.born) > cfg.max_age_bars:
        return False, f"zone age {ev.bar - z.born} > max_age_bars {cfg.max_age_bars}"
    if already_seen is not None and already_seen(ev):
        return False, "already alerted"
    return True, ""


def dedupe_key(ev: Event, cfg: AlertConfig) -> str:
    day = ev.ts.date().isoformat() if ev.ts is not None else str(ev.bar)
    if cfg.once_per_symbol_per_day and ev.kind in (EV_TAP, EV_APPROACH):
        return f"{ev.symbol}|{event_name(ev)}|{day}"
    return f"{ev.symbol}|{event_name(ev)}|{ev.zid}|{day}"


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_lines(ev: Event, ctx: Dict[str, Any], cfg: AlertConfig, params) -> List[str]:
    z: Optional[Zone] = ev.zone
    name = event_name(ev)
    price = float(ctx.get("price") or ev.price or (z.entry if z else 0.0))
    lines: List[str] = []
    head = f"{EMOJI.get(ev.kind, '•')} {TITLE.get(name, TITLE.get(ev.kind, ev.kind))}"
    lines.append(head)
    sub = f"{ev.symbol} · {ctx.get('exchange') or cfg.default_exchange} · {ctx.get('timeframe', '1d')}"
    stamp = ev.ts.strftime("%Y-%m-%d") if ev.ts is not None else ""
    live = " · LIVE" if ev.intrabar else ""
    lines.append(f"{sub} · {stamp}{live}")
    lines.append(RULE_LINE)

    if z is None:
        return lines
    entry_now = float(ctx.get("entry") or z.entry)
    # On a tap the head-line level must be the one that was actually touched —
    # `raise_after_first_tap` lifts the zone's pre-order *on the tap bar*, so
    # `zone.entry` is already the NEXT level by the time we render (the Pine
    # label makes the same distinction: "TAP n" at the touched price, then
    # "Next <raised>"). Everything else (approach, confirm, invalidation) is
    # about the level that is live right now.
    entry = float(ev.level) if (ev.kind == EV_TAP and math.isfinite(ev.level)) else entry_now
    stop = float(z.stop)
    risk = entry - stop
    targets: List[str] = []
    for k in (1.0, 2.0, 3.0):
        targets.append(_num(entry + risk * k))
    lines.append(f"📌 Price        {_num(price)}"
                 + (f"  ({_pct(ctx.get('change_pct'))})" if ctx.get("change_pct") is not None else ""))
    if ev.kind == EV_TAP:
        lines.append(f"🟢 Buy limit    {_num(entry)}   ← Tap {ev.tap_no} trigger")
    else:
        lines.append(f"🟢 Buy limit    {_num(entry)}")
    lines.append(f"🔴 Stop         {_num(stop)}   ({_pct(-risk / entry * 100 if entry else math.nan)})")
    lines.append(f"⬜ OB zone       {_num(z.top)} → {_num(z.bot)}")
    lines.append(f"🎯 Targets      R1 {targets[0]} · R2 {targets[1]} · R3 {targets[2]}")
    dist = (price - entry) / risk if risk else math.nan
    lines.append(f"📏 Dist to lvl  {_num((price - entry), 2)} ({_num(dist)} R)")
    lines.append(RULE_LINE)

    facts: List[str] = []
    age = ev.bar - z.born
    facts.append(f"zone age {age} bars")
    facts.append(f"taps {z.taps}/{params.max_touches}")
    if z.departed:
        facts.append("departure ✔")
    if z.adaptive:
        facts.append(f"next pre-order {_num(entry_now)} · raised after Tap 1"
                     if ev.kind == EV_TAP else f"level raised after Tap 1 → {_num(z.entry)}")
    if ev.kind == EV_TAP and getattr(params, "require_sweep", False) \
            and ev.detail.get("swept") is not None:
        facts.append("sweep ✔" if ev.detail.get("swept") else "no sweep")
    rvol = ctx.get("rvol", ev.detail.get("rvol"))
    if rvol is not None:
        facts.append(f"RVOL {_num(rvol, 1)}×")
    atr_now = ctx.get("atr")
    if atr_now:
        facts.append(f"ATR {_num(atr_now)} ({_num(atr_now / price * 100 if price else math.nan, 1)}%)")
    lines.append(" · ".join(facts))

    origin = ctx.get("origin_date") or ""
    if origin:
        lines.append(f"origin candle {origin} · {ctx.get('zone_method', params.zone_method)}")
    if ev.kind == EV_APPROACH:
        thr = entry + float(ctx.get("atr") or 0.0) * params.approach_atr
        lines.append(f"waiting for a touch of {_num(thr)}")
    if ev.kind == EV_TAP:
        need = []
        if params.confirm_bars > 0:
            need.append(f"close > {_num(z.top)} within {params.confirm_bars} bars")
            need.append(f"RVOL ≥ {_num(params.confirm_rvol, 1)}×, CLV ≥ {_num(params.confirm_clv, 2)}")
        lines.append("defence: " + ", ".join(need) if need else "defence window disabled")
        lines.append("no retest of a defended level is guaranteed — size for the stop")
    if ev.kind == EV_CONFIRMED:
        lines.append(f"micro-BOS + close above the OB → defence confirmed (rvol {_num(rvol, 1)}×)")
    if ev.kind == EV_INVALID:
        lines.append(f"reason: {ev.detail.get('reason', 'invalid')} · zone state → dead")
    if ev.kind == EV_NEW:
        lines.append(f"fresh zone; taps enabled after {params.min_age} bars "
                     f"(needs +{_num(params.require_departure, 1)}×ATR above {_num(z.top)})")
    lines.append(RULE_LINE)
    lines.append("not investment advice · Precision Tap scanner")
    return lines


def render_message(ev: Event, ctx: Dict[str, Any], cfg: AlertConfig, params,
                   *, parse_mode: str = "HTML") -> str:
    lines = render_lines(ev, ctx, cfg, params)
    mode = str(parse_mode or "HTML").strip().upper()
    if mode == "PLAIN":
        return "\n".join(lines)
    if mode in ("MARKDOWNV2", "MARKDOWN"):
        return "\n".join(escape_markdownv2(ln) for ln in lines)
    out: List[str] = []
    for i, ln in enumerate(lines):
        esc = (ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        if i == 0:
            out.append(f"<b>{esc}</b>")
        elif i == 1:
            out.append(f"<i>{esc}</i>")
        else:
            out.append(esc)
    return "\n".join(out)


def tradingview_symbol(symbol: str, exchange: str = "") -> Tuple[str, str]:
    """`TCS.NS` -> (`NSE`, `TCS`); `RELIANCE.BO` -> (`BSE`, `RELIANCE`)."""
    sym = str(symbol).upper()
    exch = (exchange or "").strip().upper()
    for suffix, default in ((".NS", "NSE"), (".BO", "BSE")):
        if sym.endswith(suffix):
            sym = sym[: -len(suffix)]
            exch = exch or default
            break
    return (exch or "NSE"), sym


def build_buttons(ev: Event, ctx: Dict[str, Any], cfg: AlertConfig) -> Dict[str, Any]:
    exch, tv_sym = tradingview_symbol(ev.symbol, str(ctx.get("exchange") or cfg.default_exchange))
    tv = cfg.link_template.format(exchange=exch, symbol=tv_sym, yf_symbol=ev.symbol)
    # TradingView wants the bare symbol (NSE:TCS) but Yahoo needs the suffixed
    # one — /quote/TCS resolves to a US listing, not the NSE stock.
    quote = f"https://finance.yahoo.com/quote/{ev.symbol}"
    return {"inline_keyboard": [[{"text": "📈 Chart", "url": tv},
                                 {"text": "🔎 Quote", "url": quote}]]}


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DispatchResult:
    sent: int = 0
    queued: int = 0
    skipped: int = 0
    failed: int = 0
    messages: List[str] = None  # type: ignore[assignment]
    errors: List[str] = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.messages is None:
            self.messages = []
        if self.errors is None:
            self.errors = []

    def __str__(self) -> str:
        out = (f"sent={self.sent} queued={self.queued} skipped={self.skipped} failed={self.failed}")
        if self.errors:
            out += " | " + " · ".join(dict.fromkeys(self.errors))
        return out

    @property
    def undelivered(self) -> int:
        """Alerts that were found but never reached Telegram."""
        return int(self.queued) + int(self.failed)


class AlertDispatcher:
    """Turns events into delivered (or queued) Telegram messages."""

    def __init__(self, cfg: AlertConfig, telegram, store, params, *,
                 render_charts=None, dry_run: bool = False):
        self.cfg = cfg
        self.tg = telegram
        self.store = store
        self.params = params
        self.dry_run = dry_run
        self.render_charts = render_charts          # callable(ev, ctx) -> png path | None

    # ── delivery ─────────────────────────────────────────────────────────
    @property
    def deliverable(self) -> bool:
        """Can this dispatcher actually put a message on Telegram right now?

        Everything else (quiet mode, ``--no-send``, a missing/unconfigured
        client) degrades to *log-only*: the alert is still rendered, printed and
        recorded, instead of being dropped into a retry queue that can never
        drain.
        """
        if self.cfg.quiet_log_only or self.dry_run:
            return False
        if self.tg is None or not getattr(self.tg, "configured", True):
            return False
        return not self.tg.dry_run

    def dispatch(self, items: Sequence[Tuple[Event, Dict[str, Any]]]) -> DispatchResult:
        res = DispatchResult()
        ordered = sorted(items, key=lambda ic: (event_rank(ic[0]), ic[0].symbol))
        if self.cfg.daily_limit:
            ordered = ordered[: int(self.cfg.daily_limit)]
        for ev, ctx in ordered:
            key = dedupe_key(ev, self.cfg)
            name = event_name(ev)
            text = render_message(ev, ctx, self.cfg, self.params,
                                   parse_mode=getattr(self.tg.cfg, "parse_mode", "HTML")
                                   if self.tg is not None else "HTML")
            payload = {**ev.to_dict(), "ctx": {k: v for k, v in ctx.items() if k != "df"}}
            is_new = True
            if self.store is not None:
                is_new = self.store.record_alert(
                    key, symbol=ev.symbol, event=name, zone_id=ev.zid,
                    bar_date=(ev.ts.date().isoformat() if ev.ts is not None else ""),
                    bar_time=str(ev.ts or ""), level=ev.level, price=ev.price,
                    stop=(ev.zone.stop if ev.zone else None), payload=payload, message=text)
            if not is_new:
                res.skipped += 1
                continue
            photo = None
            if self.cfg.chart and self.render_charts is not None:
                try:
                    photo = self.render_charts(ev, ctx)
                except Exception as exc:                # a bad chart must never eat a signal
                    log.debug("chart render failed for %s: %s", ev.symbol, exc)
            if not self.deliverable:
                res.skipped += 1
                log.info("alert (log-only) %s %s\n%s", ev.symbol, name, text)
                res.messages.append({"symbol": ev.symbol, "kind": name, "text": text,
                                     "ok": False, "why": "no telegram transport"})
                if self.store is not None:
                    # recorded above purely for de-duplication — do not leave it
                    # in the retry queue, there is nothing to retry it with
                    self.store.give_up([key])
                continue
            markup = build_buttons(ev, ctx, self.cfg)
            results = []
            try:
                if photo:
                    results = self.tg.send_photo(text, photo, reply_markup=markup)
                    if all(not r.ok for r in results) or not results:
                        results = self.tg.send_text(text, reply_markup=markup)
                else:
                    results = self.tg.send_text(text, reply_markup=markup)
            except Exception as exc:                    # network down, token revoked, …
                log.warning("telegram unavailable (%s) — alert queued for retry", exc)
                if self.store is not None:
                    self.store.mark(key, sent=False, error=str(exc))
                res.queued += 1
                res.errors.append(f"{ev.symbol}: {exc}"[:220])
                res.messages.append({"symbol": ev.symbol, "kind": name, "text": text})
                continue
            ok = bool(results) and all(r.ok for r in results)
            err = "; ".join(r.error for r in results if r.error)[:400]
            # A 401/400/403 will not heal by waiting: retrying it six times only
            # hides it.  Park it as given-up (so it is *not* deduped away from a
            # later, fixed run) and say so loudly.
            permanent = bool(results) and any(getattr(r, "permanent", False) for r in results)
            if self.store is not None:
                self.store.mark(key, sent=ok, error="" if ok else err)
                if not ok and permanent:
                    self.store.give_up([key])
            res.sent += 1 if ok else 0
            res.failed += 0 if ok else 1
            if not ok:
                res.errors.append(f"{ev.symbol}: {err or 'telegram rejected the message'}"[:220])
                if permanent:
                    log.error("telegram permanently rejected the alert for %s — %s "
                              "(fix the token/chat id; nothing will be retried)", ev.symbol, err)
            res.messages.append({"symbol": ev.symbol, "kind": name, "text": text, "ok": ok})
        return res

    def retry_pending(self, limit: int = 10) -> int:
        """Re-send alerts that were queued during an outage."""
        if self.store is None or self.cfg.quiet_log_only:
            return 0
        if not self.deliverable:
            return 0
        sent = 0
        for row in self.store.pending(limit=limit):
            try:
                if row.attempts >= 6:
                    self.store.give_up([row.key])
                    continue
                res = self.tg.send_text(row.message)
                ok = bool(res) and all(r.ok for r in res)
            except Exception as exc:
                log.debug("retry failed for %s: %s", row.key, exc)
                ok = False
            self.store.mark(row.key, sent=bool(ok))
            sent += 1 if ok else 0
        if sent:
            log.info("retried %d queued alert(s)", sent)
        return sent

    def broadcast(self, text: str, *, markup: Optional[Dict[str, Any]] = None) -> bool:
        """Heartbeats / scan summaries — never deduped, never queued."""
        if self.dry_run:
            log.info("[dry-run] broadcast:\n%s", text)
            return True
        if self.tg is None:
            log.info(text)
            return False
        try:
            res = self.tg.send_text(text, reply_markup=markup)
            return bool(res) and all(r.ok for r in res)
        except Exception as exc:
            log.warning("broadcast failed: %s", exc)
            return False
