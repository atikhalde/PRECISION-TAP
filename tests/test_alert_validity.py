"""Alert *validity* — is the message the user receives true and actionable?

The delivery chain is covered elsewhere (test_alert_delivery, test_live,
test_telegram).  These tests pin the three ways an alert can arrive correctly
and still be wrong:

* it describes a zone state that the forming bar produced *after* the signal's
  own bar (tap count, raised pre-order, ``state=dead``);
* it prints the forming bar's price next to a closed bar's date;
* it advertises a buy limit whose stop the very same bar already broke.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from precision_tap.alerts import passes_filters, render_message
from precision_tap.data import Bars, DataSource, normalize_ohlcv, synthetic_frame
from precision_tap.engine import (EV_APPROACH, EV_CONFIRMED, EV_TAP, ST_CONFIRMED, ST_DEAD,
                                  ST_TAPPED, Event, Zone, run_engine)
from precision_tap.params import AlertConfig, Params
from precision_tap.scanner import Scanner

from test_live import TZ, patch_source, scan_cfg


# ─────────────────────────────────────────────────────────────────────────────
# the alert describes the bar it fired on
# ─────────────────────────────────────────────────────────────────────────────
def test_context_describes_the_event_bar_not_the_forming_bar(monkeypatch):
    """A closed-bar parity event must not be rendered with today's live price.

    The message header carries the event's date; if ``price``/``change_pct`` come
    from the forming bar the alert mixes two sessions in one message — yesterday's
    signal quoted at a price that did not exist when it fired.
    """
    patch_source(monkeypatch, setup="closed")
    sc = Scanner(scan_cfg(alert={"recent_bars": 3}), store=None, dry_run=True)
    bars = sc.source.get("RELIANCE.NS")
    st = sc._scan_frame("RELIANCE.NS", bars, live=True)
    n = len(bars.df)

    evs = sc._recent_events(st, live=True)
    closed = [e for e in evs if not e.intrabar]
    assert closed, f"fixture should surface a closed-bar signal: {[e.kind for e in evs]}"

    for ev in closed:
        assert ev.bar <= n - 2
        ctx = sc._context(st, ev)
        assert ctx["price"] == pytest.approx(float(bars.df["close"].iloc[ev.bar])), \
            "alert price must be the event bar's close, not the forming bar's"
        assert ctx["last_bar"] == str(bars.df.index[ev.bar])[:10]
        prev_close = float(bars.df["close"].iloc[ev.bar - 1])
        want_chg = (float(bars.df["close"].iloc[ev.bar]) / prev_close - 1.0) * 100.0
        assert ctx["change_pct"] == pytest.approx(want_chg)
        # and it really differs from the newest bar — otherwise this pins nothing
        assert ev.bar != n - 1
        assert ctx["price"] != pytest.approx(st.price) or st.price == ctx["price"]
    sc.close()


def test_context_of_a_newest_bar_event_is_unchanged(monkeypatch):
    """For an event on the newest bar the context is the live price, as before."""
    patch_source(monkeypatch)                      # Tap 1 lands on the forming bar
    sc = Scanner(scan_cfg(), store=None, dry_run=True)
    bars = sc.source.get("RELIANCE.NS")
    st = sc._scan_frame("RELIANCE.NS", bars, live=True)
    n = len(bars.df)
    evs = [e for e in sc._recent_events(st, live=True) if e.bar == n - 1]
    assert evs, "fixture should produce a forming-bar signal"
    ctx = sc._context(st, evs[0])
    assert ctx["price"] == pytest.approx(st.price)
    assert ctx["atr"] == pytest.approx(st.atr)
    assert ctx["last_bar"] == st.last_date
    sc.close()


def test_liquidity_window_excludes_the_forming_bar(monkeypatch):
    """Turnover is measured on completed sessions.

    In live mode the newest row is a partial bar carrying a few percent of a
    day's volume; inside a 20-bar median that makes the liquidity floor stricter
    at 09:35 than at 15:30, so the same symbol can pass after the close and be
    rejected as "illiquid" during the session.
    """
    patch_source(monkeypatch)
    sc = Scanner(scan_cfg(), store=None, dry_run=True)
    bars = sc.source.get("RELIANCE.NS")
    st = sc._scan_frame("RELIANCE.NS", bars, live=True)
    n = len(bars.df)
    ev = sc._recent_events(st, live=True)[0]
    ctx = sc._context(st, ev)

    vol = bars.df["volume"].to_numpy(float)
    px = bars.df["close"].to_numpy(float)
    want_closed = float(np.median((vol * px)[n - 21:n - 1]))     # last 20 completed
    want_with_forming = float(np.median((vol * px)[n - 20:n]))    # incl. the partial bar
    assert ctx["avg_dollar_volume"] == pytest.approx(want_closed)
    assert want_closed != pytest.approx(want_with_forming)
    sc.close()


# ─────────────────────────────────────────────────────────────────────────────
# dead-on-arrival taps
# ─────────────────────────────────────────────────────────────────────────────
def _tap(zone: Zone, bar: int = 40, level: float = 100.0) -> Event:
    return Event(kind=EV_TAP, bar=bar, symbol="TCS.NS", zid=zone.zid, tap_no=1,
                 level=level, price=99.0, ts=pd.Timestamp("2026-09-11"), zone=zone)


def _zone(**kw) -> Zone:
    base = dict(zid=0, born=30, origin=29, top=100.0, bot=98.0, entry=100.05,
                entry0=100.05, stop=97.0, atr0=1.0, taps=1, state=ST_TAPPED,
                departed=True, dead_bar=-1, dead_reason="")
    base.update(kw)
    return Zone(**base)


def test_tap_whose_own_bar_broke_the_stop_is_not_sent():
    """The level failed at the same moment it was touched.

    The alert block advertises a buy limit, a stop and three R-targets for a
    trade that is already invalid, and — with the default
    ``include_invalidations: false`` — the failure itself is never sent, so the
    user only ever sees the buy side.
    """
    z = _zone(state=ST_DEAD, dead_bar=40, dead_reason="close_below_stop")
    ok, why = passes_filters(_tap(z, bar=40), {"price": 96.0}, AlertConfig())
    assert not ok
    assert "failed on the same bar" in why
    assert "close_below_stop" in why


def test_exhausted_tap_is_also_dead_on_arrival():
    z = _zone(state=ST_DEAD, dead_bar=40, dead_reason="exhausted", taps=5)
    ok, why = passes_filters(_tap(z, bar=40, level=100.0), {"price": 99.0},
                             AlertConfig(events=["tap1", "tap"]))
    assert not ok and "exhausted" in why


def test_a_tap_that_died_on_a_later_bar_is_still_sent():
    """The signal was valid when it fired; a later failure must not erase it."""
    z = _zone(state=ST_DEAD, dead_bar=44, dead_reason="close_below_stop")
    ok, why = passes_filters(_tap(z, bar=40), {"price": 99.5}, AlertConfig())
    assert ok, why


def test_a_live_tap_is_still_sent():
    z = _zone(state=ST_TAPPED, dead_bar=-1)
    ok, why = passes_filters(_tap(z, bar=40), {"price": 99.5}, AlertConfig())
    assert ok, why


def test_dead_on_arrival_filter_can_be_switched_off():
    """Strict indicator parity: Pine's ``anyTap`` fires regardless."""
    z = _zone(state=ST_DEAD, dead_bar=40, dead_reason="close_below_stop")
    cfg = AlertConfig(skip_dead_on_arrival=False)
    ok, why = passes_filters(_tap(z, bar=40), {"price": 96.0}, cfg)
    assert ok, why


def test_dead_on_arrival_only_applies_to_taps():
    """An approach/confirm on a bar that later died is a different claim."""
    z = _zone(state=ST_DEAD, dead_bar=40, dead_reason="close_below_stop")
    ev = Event(kind=EV_APPROACH, bar=40, symbol="TCS.NS", zid=0, level=100.05,
               price=100.4, ts=pd.Timestamp("2026-09-11"), zone=z)
    ok, _why = passes_filters(ev, {"price": 100.4}, AlertConfig())
    assert ok


# ─────────────────────────────────────────────────────────────────────────────
# end to end: a dead-on-arrival bar produces no alert
# ─────────────────────────────────────────────────────────────────────────────
def test_scanner_reports_the_reason_in_the_cycle_tally(monkeypatch):
    """The cycle summary must say the level already failed, not just "filtered"."""
    from precision_tap.scanner import _skip_bucket
    assert _skip_bucket("level failed on the same bar (close_below_stop)") == "level already failed"


def test_rendered_alert_never_quotes_a_stop_above_the_price_it_advertises():
    """Sanity over the rendered block for a healthy tap."""
    z = _zone()
    ev = _tap(z, level=z.entry)
    ctx = {"price": 99.5, "change_pct": -0.4, "atr": 1.0, "rvol": 1.4,
           "exchange": "NSE", "timeframe": "1d", "entry": z.entry,
           "origin_date": "2026-09-01"}
    text = render_message(ev, ctx, AlertConfig(), Params.default(), parse_mode="plain")
    assert "TAP 1" in text
    assert f"{z.entry:.2f}" in text
    assert f"{z.stop:.2f}" in text
    assert not math.isnan(z.risk) and z.risk > 0


# ─────────────────────────────────────────────────────────────────────────────
# the defence-confirmed alert has its own message, not the tap block
# ─────────────────────────────────────────────────────────────────────────────
def _confirmed(zone: Zone, bar: int = 42, close: float = 134.10) -> Event:
    return Event(kind=EV_CONFIRMED, bar=bar, symbol="TCS.NS", zid=zone.zid, level=zone.top,
                 price=close, ts=pd.Timestamp("2026-09-22"), zone=zone,
                 detail={"rvol": 1.8, "clv": 0.81, "micro_bos": 0.42})


def _confirmed_zone(**kw) -> Zone:
    base = dict(zid=3, born=30, origin=29, top=131.84, bot=129.60, entry=132.90,
                entry0=132.32, stop=129.20, atr0=2.6, taps=1, tap_bars=[40],
                state=ST_CONFIRMED, departed=True, adaptive=True, confirm_bar=42)
    base.update(kw)
    return Zone(**base)


def _confirmed_ctx(z: Zone, **over):
    ctx = {"price": 134.10, "change_pct": 1.9, "atr": 2.6, "rvol": 0.9,   # newest-bar rvol ≠ defence bar's
           "exchange": "NSE", "timeframe": "1d", "entry": z.entry, "origin_date": "2026-09-02"}
    ctx.update(over)
    return ctx


def test_confirmed_alert_is_not_rendered_as_a_buy_order():
    """A defence confirmation is a verdict on the tap, not a new entry.

    Reusing the Tap block prints "Buy limit / Dist to lvl" for a level price
    has already left, which reads like a late Tap 1.  The confirmed message
    must carry none of the order vocabulary and name the level that *held*.
    """
    z = _confirmed_zone()
    text = render_message(_confirmed(z), _confirmed_ctx(z), AlertConfig(), Params.default(),
                          parse_mode="plain")
    head = text.splitlines()[0]
    assert "OB DEFENCE CONFIRMED" in head and head.startswith("🛡")
    assert "Buy limit" not in text
    assert "Dist to lvl" not in text
    assert "trigger" not in text
    assert "Defended" in text and "132.32" in text        # the level Tap 1 touched (entry0) …
    assert "Tap 1 held" in text
    assert "129.20" in text                                # … and the stop that still applies
    assert "not a fresh entry" in text


def test_confirmed_alert_describes_the_defence_bar():
    """Every gate of the indicator's ``defence`` rule is printed, with the bar's numbers.

    A ``confirmed`` alert the user cannot check against the chart is a claim,
    not a signal, so the message restates ``INDICATOR.txt``'s expression in the
    same order — ``barstate.isconfirmed``, the pending window, ``close > open``,
    ``clv >= confirmCLV``, ``rvol >= confirmRVOL``, ``close > top`` and
    ``microBOS`` — using the defence bar's own values from the event detail
    (RVOL here is 1.8, while the newest bar in ``ctx`` shows 0.9).
    """
    z = _confirmed_zone()
    p = Params.default()
    text = render_message(_confirmed(z), _confirmed_ctx(z), AlertConfig(), p, parse_mode="plain")
    assert "confirmed 2 bars after the tap" in text          # bar 42 − tap bar 40
    assert f"(window {p.confirm_bars})" in text
    assert "closed bar ✔" in text                            # barstate.isconfirmed
    assert "close > open ✔" in text                          # no `open` in this detail
    assert "CLV 0.81" in text and f"(≥ {p.confirm_clv:.2f})" in text
    assert "RVOL 1.8×" in text and "RVOL 0.9×" not in text    # detail wins over ctx
    assert f"(≥ {p.confirm_rvol:.1f})" in text
    assert f"close > OB top {z.top:.2f} ✔" in text
    assert f"micro-BOS close 134.10 > {p.confirm_bos_len}-bar high" in text
    assert "(by 0.42)" in text
    assert "state → confirmed" in text


def _engine_shaped_detail(**over):
    """The detail dict the engine attaches to ``EV_CONFIRMED`` (engine.py emit)."""
    d = {"rvol": 1.8, "clv": 0.81, "micro_bos": 0.42, "open": 131.60, "close": 134.10,
         "zone_top": 131.84, "bos_ref": 133.68, "bars_since_tap": 2, "confirm_window": 3,
         "closed_bar": True, "taps_at_event": 1, "tap_bar": 40, "state_at_event": ST_CONFIRMED,
         "entry_at_event": 132.32}
    d.update(over)
    return d


def test_confirmed_alert_quotes_the_bars_that_confirmed_not_a_later_retap():
    """The message must describe the defence bar, even if the block was tapped again.

    ``Event.zone`` is a live reference: a repeat tap *after* the defence bar
    increments ``zone.taps``, lifts ``zone.entry`` and — if the zone is
    adaptive — moves the pre-order.  Rendering from the live zone would then
    print "Tap 2 held" for a signal that was Tap 1, and quote a level that did
    not exist on the bar the alert is about.  The engine snapshots those inputs
    onto the event; this pins that they win.
    """
    z = _confirmed_zone(taps=2, tap_bars=[40, 45], entry=134.00, confirm_bar=42)
    ev = _confirmed(z)
    ev.detail = _engine_shaped_detail()                      # snapshot: Tap 1 at 132.32
    text = render_message(ev, _confirmed_ctx(z), AlertConfig(), Params.default(), parse_mode="plain")
    assert "Tap 1 held" in text and "not a fresh entry" in text
    assert "Defended     132.32" in text
    assert "Defended     134.00" not in text
    assert "taps 1/4" in text                                # snapshot, not z.taps == 2
    assert "state → confirmed" in text


def test_confirmed_alert_prints_the_gate_values_from_the_engine():
    """With a full engine detail the gates print as inequalities, not prose."""
    z = _confirmed_zone()
    ev = _confirmed(z)
    ev.detail = _engine_shaped_detail()
    text = render_message(ev, _confirmed_ctx(z), AlertConfig(), Params.default(), parse_mode="plain")
    assert "close 134.10 > open 131.60 ✔" in text
    assert "CLV 0.81 (≥ 0.65) ✔" in text
    assert "RVOL 1.8× (≥ 1.3) ✔" in text
    assert "close > OB top 131.84 ✔" in text
    assert "micro-BOS close 134.10 > 3-bar high 133.68 ✔ (by 0.42)" in text
    assert "closed bar ✔" in text


def test_confirmed_alert_reports_open_r_from_the_defended_level():
    z = _confirmed_zone()
    text = render_message(_confirmed(z), _confirmed_ctx(z), AlertConfig(), Params.default(),
                          parse_mode="plain")
    risk = z.entry0 - z.stop                                  # 3.12
    r_now = (134.10 - z.entry0) / risk                        # 0.57 R
    assert f"({r_now:.2f} R from Tap 1)" in text
    # targets are still measured from the defended level, so they line up with
    # the Tap 1 alert the user already has on screen
    assert f"R1 {z.entry0 + risk:.2f}" in text
    # a raised pre-order is mentioned as a *revisit* level, not as a buy limit
    assert "next pre-order 132.90" in text


def test_confirmed_after_a_repeat_tap_names_the_live_level():
    """Tap 2+ touch the (possibly raised) live entry, so that is what was defended."""
    z = _confirmed_zone(taps=2, tap_bars=[40, 45], confirm_bar=46)
    text = render_message(_confirmed(z, bar=46), _confirmed_ctx(z), AlertConfig(),
                          Params.default(), parse_mode="plain")
    assert "Tap 2 held" in text
    assert "Defended     132.90" in text
    assert "confirmed 1 bar after the tap" in text


def test_confirmed_alert_survives_every_parse_mode():
    z = _confirmed_zone()
    ev, ctx = _confirmed(z), _confirmed_ctx(z)
    html = render_message(ev, ctx, AlertConfig(), Params.default(), parse_mode="HTML")
    assert html.startswith("<b>🛡 OB DEFENCE CONFIRMED</b>")
    assert "P&amp;L" in html and "<" not in html.replace("<b>", "").replace("</b>", "") \
        .replace("<i>", "").replace("</i>", "")
    md = render_message(ev, ctx, AlertConfig(), Params.default(), parse_mode="MarkdownV2")
    assert "OB DEFENCE CONFIRMED" in md
    assert "\\." in md                                        # escaped for MarkdownV2


def test_tap_alert_is_unchanged_by_the_confirmed_layout():
    """The order block stays exactly as before for taps — only `confirmed` moved."""
    z = _confirmed_zone(state=ST_TAPPED)
    tap = Event(kind=EV_TAP, bar=40, symbol="TCS.NS", zid=z.zid, tap_no=1, level=z.entry0,
                price=131.7, ts=pd.Timestamp("2026-09-18"), zone=z)
    text = render_message(tap, _confirmed_ctx(z, price=131.71, change_pct=-0.14),
                          AlertConfig(), Params.default(), parse_mode="plain")
    assert "Buy limit    132.32   ← Tap 1 trigger" in text
    assert "Dist to lvl" in text
    assert "Defended" not in text


def test_engine_confirmed_event_renders_through_the_dispatcher():
    """End to end: a real engine `confirmed` reaches Telegram as its own message.

    The synthetic frame (seed 0) produces several defences; render the last one
    exactly as the dispatcher would and check it took the dedicated layout.
    """
    from precision_tap.alerts import AlertDispatcher
    from precision_tap.data import synthetic_frame

    df = synthetic_frame(seed=0)
    p = Params.default()
    res = run_engine(df, p, symbol="SYN.NS")
    conf = res.by(EV_CONFIRMED)
    assert conf, "fixture must produce a defence"
    ev = conf[-1]
    ctx = {"price": float(df["close"].iloc[ev.bar]), "atr": 1.0, "exchange": "NSE",
           "timeframe": "1d", "entry": ev.zone.entry}

    class _TG:
        dry_run = False
        configured = True
        cfg = type("C", (), {"parse_mode": "plain"})()
        sent = []

        def send_text(self, text, **kw):
            self.sent.append(text)
            return [type("R", (), {"ok": True, "error": "", "permanent": False})()]

    tg = _TG()
    cfg = AlertConfig(events=["tap1", "confirmed"], chart=False)
    d = AlertDispatcher(cfg, tg, store=None, params=p)
    out = d.dispatch([(ev, ctx)])
    assert out.sent == 1 and len(tg.sent) == 1
    msg = tg.sent[0]
    assert msg.startswith("🛡 OB DEFENCE CONFIRMED")
    assert "Defended" in msg and "held" in msg
    assert "Buy limit" not in msg
    assert f"{ev.zone.stop:.2f}" in msg


# ─────────────────────────────────────────────────────────────────────────────
# the delivery layer must not add its own conditions to the confirmed signal
# ─────────────────────────────────────────────────────────────────────────────
def test_the_shipped_config_keeps_the_confirmed_signal_on():
    """The defence confirmation is the point of the parity work — it ships enabled.

    ``INDICATOR.txt`` fires an alertcondition for every ``DEFENCE CONFIRMED``
    label.  A config that drops ``confirmed`` from ``alerts.events`` (or turns
    the closed-bar parity pass off) silently narrows that set, so the two edits
    that would do it are pinned here.
    """
    import yaml
    from pathlib import Path
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    alerts = raw.get("alerts", {})
    assert "confirmed" in alerts.get("events", []), \
        "config.yaml must keep the defence-confirmed alert enabled"
    assert alerts.get("match_indicator_100", True), \
        "config.yaml must keep the closed-bar parity pass on (confirmed is closed-bar only)"
    assert AlertConfig().events and "confirmed" in AlertConfig().events


def test_confirmed_passes_the_default_filters_and_the_reason_when_it_does_not():
    """A defence inherits the ordinary alert gates — no extra, hidden ones.

    The gates that *do* apply (allow-list, price band, liquidity, zone age) are
    the same ones every other event passes, and each returns a quotable reason
    so a cycle that skipped a defence can say why.
    """
    z = _confirmed_zone()
    ev = _confirmed(z)
    ctx = _confirmed_ctx(z, avg_dollar_volume=5e9)
    ok, why = passes_filters(ev, ctx, AlertConfig())
    assert ok, why
    # allow-list: disabling `confirmed` is the one filter that is *about* this signal
    cfg = AlertConfig(events=["tap1", "approach"])
    ok, why = passes_filters(ev, ctx, cfg)
    assert not ok and "confirmed not enabled" in why
    # the ordinary gates, unchanged for confirmed:
    assert passes_filters(ev, ctx, AlertConfig(min_price=200.0))[0] is False
    assert passes_filters(ev, ctx, AlertConfig(min_liquidity_dollar_volume=1e12))[0] is False
    old = _confirmed_zone(born=0)                      # 42 bars before the defence
    ok, why = passes_filters(_confirmed(old), ctx, AlertConfig(max_age_bars=30))
    assert not ok and "zone age 42 > max_age_bars 30" in why
    # ...and an unset age limit (the shipped default) never drops one
    assert passes_filters(_confirmed(old), ctx, AlertConfig(max_age_bars=0))[0] is True


def test_confirmed_dedupe_is_keyed_by_zone_and_bar_not_by_symbol():
    """Two blocks can confirm on the same session; they are two signals.

    ``once_per_symbol_per_day`` exists to stop repeat *taps* on one level
    spamming a chat.  A confirmation is a per-bar state change: a block can tap,
    confirm, be tapped again and confirm again — and on an intraday timeframe
    both defence bars carry the same date.  Keying those by ``(zone, day)``
    would drop the second one, i.e. send less than the indicator's
    ``DEFENCE CONFIRMED`` alertcondition fired.
    """
    from precision_tap.alerts import dedupe_key
    cfg = AlertConfig(once_per_symbol_per_day=True)
    a, b = _confirmed_zone(zid=1), _confirmed_zone(zid=2)
    ka, kb = dedupe_key(_confirmed(a), cfg), dedupe_key(_confirmed(b), cfg)
    assert ka != kb and ka.endswith("|1|2026-09-22T00:00:00") and kb.endswith("|2|2026-09-22T00:00:00")
    # the second defence of the *same* block, minutes later on the same day:
    # a different bar, so a different signal — the first alert does not silence it
    again = _confirmed(a, bar=43)
    again.ts = pd.Timestamp("2026-09-22 09:45")
    assert dedupe_key(again, cfg) != ka
    # ...while the identical event replayed by the next live cycle is deduped,
    # because its bar timestamp is unchanged
    replay = _confirmed(a)
    assert dedupe_key(replay, cfg) == ka
    # the flag only ever *narrows* taps/approaches; it must not enter this key
    cfg_off = AlertConfig(once_per_symbol_per_day=False)
    assert dedupe_key(_confirmed(a), cfg_off) == ka


def test_both_defences_of_one_block_reach_telegram():
    """End to end: two ``DEFENCE CONFIRMED`` labels, two messages.

    The fixture (``tests/test_pine_reference.py``) defends one block at bars 41
    and 44 of a 5-minute session — the indicator prints two labels there, and
    both must survive the ledger.  A ledger keyed by ``(zone, day)`` counts the
    second as "already alerted" and the user never hears about it.
    """
    from precision_tap.alerts import AlertDispatcher
    from precision_tap.state import StateStore

    from test_pine_reference import _double_defence_frame

    df = _double_defence_frame()
    p = Params.default()
    res = run_engine(df, p, symbol="RETAP.NS")
    conf = res.by(EV_CONFIRMED)
    assert [(e.bar, e.zid, e.ts.date().isoformat()) for e in conf] == \
        [(41, 0, "2021-03-01"), (44, 0, "2021-03-01")], "fixture drifted"
    items = [(ev, {"price": float(df["close"].iloc[ev.bar]), "atr": 1.0, "exchange": "NSE",
                   "timeframe": "5m"}) for ev in conf]

    class _TG:
        dry_run = False
        configured = True
        cfg = type("C", (), {"parse_mode": "plain"})()
        sent = []

        def send_text(self, text, **kw):
            self.sent.append(text)
            return [type("R", (), {"ok": True, "error": "", "permanent": False})()]

    with StateStore(":memory:") as store:
        tg = _TG()
        d = AlertDispatcher(AlertConfig(events=["confirmed"], chart=False), tg, store=store, params=p)
        first = d.dispatch(items)
        # a later cycle replays the same closed history: those two are old news
        second = d.dispatch(items)
    assert first.sent == 2 and len(tg.sent) == 2
    assert second.sent == 0 and second.skipped == 2
    assert all(m.startswith("🛡 OB DEFENCE CONFIRMED") for m in tg.sent)
    assert "Tap 2 held" in tg.sent[1] and "Tap 1 held" in tg.sent[0]
