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

from .engine import (EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, ST_DEAD,
                     Event, Zone)
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
    # A tap whose own bar closed through the stop is not an actionable signal:
    # the level failed at the same moment it was touched, so the rendered
    # "Buy limit / Stop / R1–R3" block describes a trade that was already dead
    # on arrival — and the invalidation that proves it is off by default
    # (`include_invalidations: false`), so the user would see the buy alert and
    # never the failure.  The engine still records the tap exactly as Pine's
    # `anyTap` does; this is an alert-layer decision only.
    if cfg.skip_dead_on_arrival and ev.kind == EV_TAP and z is not None \
            and z.state == ST_DEAD and z.dead_bar == ev.bar:
        return False, f"level failed on the same bar ({z.dead_reason or 'invalid'})"
    if z is not None and cfg.max_age_bars and (ev.bar - z.born) > cfg.max_age_bars:
        return False, f"zone age {ev.bar - z.born} > max_age_bars {cfg.max_age_bars}"
    if already_seen is not None and already_seen(ev):
        return False, "already alerted"
    return True, ""


def _bar_stamp(ev: Event) -> str:
    """The bar's wall-clock stamp, which is what makes two events the same event.

    The *timestamp*, not the bar index: every scan cycle refetches the history,
    so the index of the same bar moves around with the length of the frame while
    its timestamp does not.  A frame with no timestamps (a hand-built one) falls
    back to the index.
    """
    ts = ev.ts
    if ts is None:
        return str(ev.bar)
    try:
        return ts.isoformat()
    except AttributeError:
        return str(ts)


def dedupe_key(ev: Event, cfg: AlertConfig) -> str:
    """Identity of one alert, as stored in the state ledger.

    ``state.seen`` compares these, so the key decides which events are *one*
    signal and which are two:

    * tap/approach — one message per symbol per session under
      ``once_per_symbol_per_day``.  A level can be touched repeatedly and the
      user asked to hear about the level once, so the day is the identity.
    * new_ob / confirmed / invalidated — a state change, keyed by the zone *and*
      the exact bar.  A block really can confirm twice in one session (tap →
      confirm → retap → confirm) and both confirmation bars can share a date on
      an intraday timeframe; keying those by ``(zone, day)`` would swallow the
      second one, i.e. send less than the indicator's
      ``DEFENCE CONFIRMED`` alertcondition fired.
    """
    stamp = _bar_stamp(ev)
    if cfg.once_per_symbol_per_day and ev.kind in (EV_TAP, EV_APPROACH):
        return f"{ev.symbol}|{event_name(ev)}|{stamp[:10]}"
    return f"{ev.symbol}|{event_name(ev)}|{ev.zid}|{stamp}"


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def defence_rule(params, *, top: Optional[float] = None) -> str:
    """The indicator's ``defence`` expression as a one-line checklist.

    ``INDICATOR.txt``:

    .. code-block:: pine

        pending  = state == 1 and tapBar >= 0 and bar_index - tapBar <= confirmBars
        microBOS = close > ta.highest(high[1], confirmBOSLen)
        defence  = barstate.isconfirmed and pending and close > open
                   and clv >= confirmCLV and rvol >= confirmRVOL and close > top and microBOS

    Same gates, same order.  The Tap alert quotes this with the configured
    *thresholds* (so the reader knows what the 🛡 message will require); the
    🛡 message quotes it with the defence bar's *values*.  A gate the
    indicator checks but this string omits would be a silent difference
    between the chat and the chart, so keep the two lists identical.
    """
    if params.confirm_bars > 0:
        pending = f"closed bar within {params.confirm_bars} bars of the tap"
    else:
        pending = "closed bar, tap bar only (confirmBars = 0)"
    parts = [pending, "close > open"]
    if top is not None:
        parts.append(f"close > {_num(top)}")
    parts.append(f"CLV ≥ {_num(params.confirm_clv, 2)}")
    parts.append(f"RVOL ≥ {_num(params.confirm_rvol, 1)}×")
    parts.append(f"close > {params.confirm_bos_len}-bar high")
    return " · ".join(parts)


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
    if ev.kind == EV_CONFIRMED:
        # A defence confirmation is a *state change*, not a level to buy: the
        # generic block below advertises "Buy limit / Stop / R1–R3 / Dist to
        # lvl" for a price that has already left the zone, so it reads like a
        # late Tap message.  It gets its own layout instead.
        lines.extend(_confirmed_body(ev, ctx, cfg, params))
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
        # The follow-up 🛡 alert is the *same rule* being satisfied, so the Tap
        # message quotes the whole rule — thresholds, in the indicator's order.
        lines.append("defence: " + defence_rule(params, top=z.top))
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


def _confirmed_body(ev: Event, ctx: Dict[str, Any], cfg: AlertConfig, params) -> List[str]:
    """Body of the 🛡 OB DEFENCE CONFIRMED message.

    The Tap alert is an *order*: buy here, stop there, targets.  This one is a
    *verdict* on that order — the level held, buyers stepped in and price
    closed back above the block with volume and a micro break of structure.
    So the block answers a different set of questions:

    * which tap was defended, at what price, and how many bars it took;
    * the indicator's ``defence`` rule **gate by gate** — ``barstate.isconfirmed``,
      the ``pending`` window, ``close > open``, ``clv >= confirmCLV``,
      ``rvol >= confirmRVOL``, ``close > top`` and ``microBOS`` — each printed
      with the defence bar's own numbers, so the alert can be diffed against
      the chart instead of taken on trust;
    * where the stop still is and how far price has already travelled in R
      (the trade is *on*, and it may already be at R1 by the time this prints).

    Nothing here is a new entry — a defended level does not owe a retest.
    """
    z: Zone = ev.zone
    price = float(ctx.get("price") or ev.price or z.top)
    close = float(ev.price) if math.isfinite(ev.price) else price
    stop = float(z.stop)
    tap_bar = int(ev.detail.get("tap_bar", z.tap_bars[-1] if z.tap_bars else -1))
    # The level that was *defended* is the one the last tap touched.  Tap 1 is
    # always at the creation level (`raise_after_first_tap` lifts the pre-order
    # only after that touch); every later tap is at the live `entry`.  The
    # engine snapshots both on the event, because a confirmed block can be
    # tapped again *after* the defence bar and `zone.taps`/`zone.entry` would
    # then describe that later bar, not the one this message is about.
    taps = int(ev.detail.get("taps_at_event", z.taps))
    tapped_at = float(ev.detail.get("entry_at_event",
                                    float(z.entry0) if z.taps <= 1 else float(z.entry)))
    risk = tapped_at - stop
    gain = close - tapped_at
    r_now = gain / risk if risk else math.nan
    bars_to_confirm = (ev.bar - tap_bar) if tap_bar >= 0 else None
    # the *defence bar's* RVOL/CLV (engine detail), not whatever the newest bar shows
    rvol = ev.detail.get("rvol", ctx.get("rvol"))
    clv = ev.detail.get("clv")
    bos = ev.detail.get("micro_bos")
    lines: List[str] = []
    lines.append(f"📌 Price        {_num(price)}"
                 + (f"  ({_pct(ctx.get('change_pct'))})" if ctx.get("change_pct") is not None else ""))
    tap_lbl = f"Tap {taps}" if taps else "tap"
    lines.append(f"✅ Defended     {_num(tapped_at)}   ← {tap_lbl} held")
    lines.append(f"⬜ OB zone       {_num(z.top)} → {_num(z.bot)}")
    lines.append(f"🔴 Stop         {_num(stop)}   ({_pct(-(close - stop) / close * 100 if close else math.nan)} from close)")
    lines.append(f"📈 vs OB top    +{_num(close - z.top)}   "
                 f"({_pct((close / z.top - 1.0) * 100 if z.top else math.nan)} above {_num(z.top)})")
    targets = [_num(tapped_at + risk * k) for k in (1.0, 2.0, 3.0)]
    lines.append(f"🎯 Targets      R1 {targets[0]} · R2 {targets[1]} · R3 {targets[2]}")
    lines.append(f"📏 Open P&L     {_num(gain)} ({_num(r_now)} R from {tap_lbl})")
    lines.append(RULE_LINE)

    # ── the indicator's `defence` expression, gate by gate ───────────────
    # INDICATOR.txt:
    #   pending  = state == 1 and tapBar >= 0 and bar_index - tapBar <= confirmBars
    #   microBOS = close > ta.highest(high[1], confirmBOSLen)
    #   defence  = barstate.isconfirmed and pending and close > open
    #              and clv >= confirmCLV and rvol >= confirmRVOL and close > top and microBOS
    # Every gate is printed, in that order, with the defence bar's own numbers —
    # so the alert can be checked against the chart instead of taken on trust.
    d_open = ev.detail.get("open")
    d_top = ev.detail.get("zone_top", z.top)
    d_bos = ev.detail.get("bos_ref")
    if d_bos is None and bos is not None and math.isfinite(float(bos)):
        d_bos = close - float(bos)                 # micro_bos is the margin over the window high
    closed = bool(ev.detail.get("closed_bar", True))
    confirmed_bars = ev.detail.get("bars_since_tap", bars_to_confirm)
    window = int(ev.detail.get("confirm_window", params.confirm_bars))

    first: List[str] = []
    if confirmed_bars is None:
        first.append("confirmed")
    elif int(confirmed_bars) == 0:
        first.append(f"confirmed on the tap bar (window {window})")
    else:
        unit = "bar" if int(confirmed_bars) == 1 else "bars"
        first.append(f"confirmed {int(confirmed_bars)} {unit} after the tap (window {window})")
    first.append("closed bar ✔" if closed else "closed bar ✗")
    first.append(f"close {_num(close)} > open {_num(d_open)} ✔"
                 if d_open is not None and math.isfinite(float(d_open)) else "close > open ✔")
    lines.append(" · ".join(first))

    second: List[str] = []
    second.append(f"CLV {_num(clv, 2)} (≥ {_num(params.confirm_clv, 2)}) ✔" if clv is not None
                  else f"CLV ≥ {_num(params.confirm_clv, 2)} ✔")
    second.append(f"RVOL {_num(rvol, 1)}× (≥ {_num(params.confirm_rvol, 1)}) ✔" if rvol is not None
                  else f"RVOL ≥ {_num(params.confirm_rvol, 1)}× ✔")
    second.append(f"close > OB top {_num(d_top)} ✔")
    bos_txt = f"micro-BOS close {_num(close)} > {params.confirm_bos_len}-bar high"
    bos_txt += f" {_num(d_bos)} ✔" if d_bos is not None and math.isfinite(float(d_bos)) \
        else " ✔"
    if bos is not None and math.isfinite(float(bos)):
        bos_txt += f" (by {_num(abs(float(bos)))})"
    second.append(bos_txt)
    lines.append(" · ".join(second))
    age = ev.bar - z.born
    lines.append(f"zone age {age} bars · taps {taps}/{params.max_touches} · state → confirmed")
    origin = ctx.get("origin_date") or ""
    if origin:
        lines.append(f"origin candle {origin} · {ctx.get('zone_method', params.zone_method)}")
    if z.adaptive and math.isfinite(z.entry):
        lines.append(f"next pre-order {_num(z.entry)} if price revisits the block")
    lines.append("defence confirmed — manage the open trade; this is not a fresh entry")
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
    log_only: int = 0      # subset of ``skipped``: found, rendered, nothing to send with
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
        if self.log_only:
            # the difference between "already alerted" and "no transport" is the
            # whole story of a silent chat, so it must be readable at a glance
            out += f" log-only={self.log_only}"
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
            if not self.deliverable:
                # Rendered text, no picture: drawing a chart nobody will receive
                # costs a matplotlib pass *and* an extra provider fetch per alert
                # (`_render_chart` re-reads the frame when the cycle did not keep
                # it), which in a dry cycle over the full market is minutes of
                # work and hundreds of files for nothing.
                res.skipped += 1
                res.log_only += 1        # not a de-duplication: nothing could be sent
                log.info("alert (log-only) %s %s\n%s", ev.symbol, name, text)
                res.messages.append({"symbol": ev.symbol, "kind": name, "text": text,
                                     "ok": False, "why": "no telegram transport"})
                if self.store is not None:
                    # recorded above purely for de-duplication — do not leave it
                    # in the retry queue, there is nothing to retry it with
                    self.store.give_up([key])
                continue
            photo = None
            if self.cfg.chart and self.render_charts is not None:
                try:
                    photo = self.render_charts(ev, ctx)
                except Exception as exc:                # a bad chart must never eat a signal
                    log.debug("chart render failed for %s: %s", ev.symbol, exc)
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
            # A row left at QUEUED by a cycle that died before it could mark the
            # result counts as "already alerted" for the rest of the day — and
            # with no transport this pass can never clear it.  Hand those keys
            # back (``GIVEN_UP`` is not "seen") so the first cycle with a working
            # bot still gets them, and say how many were parked.
            stuck = self.store.pending(limit=200)
            if stuck:
                self.store.give_up([r.key for r in stuck])
                log.warning("%d queued alert(s) released without a Telegram transport — "
                            "set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID to actually send them",
                            len(stuck))
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
