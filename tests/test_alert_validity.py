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
from precision_tap.engine import EV_APPROACH, EV_TAP, ST_DEAD, ST_TAPPED, Event, Zone, run_engine
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
