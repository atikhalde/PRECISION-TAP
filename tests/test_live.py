"""The live-market path: feed freshness, the always-on scheduler, Telegram delivery.

Everything here runs offline — the market feed is a deterministic synthetic
generator whose last bar is today's session, and Telegram is a local HTTP
double.  These are the regressions that made the live scanner produce nothing
on a real market:

* a live scan being served a closed-bar (or last-night's) cached frame,
  flagged ``live=True`` anyway;
* scheduled scans inside the session running in closed-bar mode, so no
  intraday tap could ever be seen;
* the closed-bar parity pass never being populated, so ``confirmed`` alerts
  could not fire in live mode at all;
* an unconfigured Telegram client swallowing every alert into a retry queue
  that could never drain.
"""
from __future__ import annotations

import json
import threading
import types
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from precision_tap.data import Bars, DataSource, synthetic_frame
from precision_tap.engine import EV_CONFIRMED, EV_TAP, Event, run_engine
from precision_tap.live import LiveLoop, _parse_hhmm
from precision_tap.params import (AlertConfig, DataConfig, LiveConfig, Params, ScanConfig,
                                  TelegramConfig, TradeConfig)
from precision_tap.scanner import Scanner, SymbolScan, _is_stale_bar
from precision_tap.state import StateStore

TZ = "Asia/Kolkata"

#: Seeds of :func:`synthetic_frame` that are known to contain the setup a given
#: test needs (verified against the engine, so the fixtures are deterministic).
TAP_SEED = 0            # a Tap 1 on the *forming* bar
CLOSED_SEED = 1         # the Tap 1 sits on the last closed bar


# ─────────────────────────────────────────────────────────────────────────────
# market fixtures
# ─────────────────────────────────────────────────────────────────────────────
def _today():
    return pd.Timestamp.now(tz=TZ).normalize()


def _last_session():
    """:func:`_today` rolled onto the most recent trading weekday — the session
    the provider fixtures stamp their prints into (and what ``exchange_clock``
    pins "now" to), so this test is not hostage to the host's calendar."""
    d = _today()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _quiet_bars(df: pd.DataFrame, count: int = 2) -> pd.DataFrame:
    """Append unchanged bars, pushing the last real signal onto a closed bar."""
    c = float(df["close"].iloc[-1])
    v = float(df["volume"].tail(20).median())
    idx = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=count)
    add = pd.DataFrame([{"open": c, "high": c * 1.001, "low": c * 0.999,
                         "close": c, "volume": v}] * count, index=idx)
    from precision_tap.data import normalize_ohlcv
    return normalize_ohlcv(pd.concat([df, add]), daily=True)


def live_frame(seed: int = TAP_SEED, n: int = 520, *, days_back: int = 0,
               setup: str = "tap") -> pd.DataFrame:
    """Synthetic daily history whose newest bar is today's session.

    ``setup="tap"``     a Tap 1 lands on the forming bar (an intraday alert).
    ``setup="closed"``  the forming bar is inert; the signals sit on the last
                        closed bars, so only the parity pass can surface them.
    ``days_back``       shifts the series into the past (a feed that went dead).
    """
    if setup == "closed":
        # the final Tap 1 lands two bars back, so only the parity pass sees it
        df = _quiet_bars(synthetic_frame(n=n - 2, seed=seed, tap_on_last_bar=True))
    else:
        df = synthetic_frame(n=n, seed=seed, tap_on_last_bar=True)
    end = _today().tz_localize(None) - pd.Timedelta(days=days_back)
    return df.set_axis(df.index + (end - df.index[-1]))


def patch_source(monkeypatch, *, days_back: int = 0, live: bool = True,
                 setup: str = "tap"):
    """Point :class:`DataSource` at the synthetic live feed (no network)."""
    seed = CLOSED_SEED if setup == "closed" else TAP_SEED

    def _get(self, symbol, *, lookback_days=None, end=None, use_cache=True):
        df = live_frame(seed, days_back=days_back, setup=setup)
        return Bars(symbol=symbol, df=df,
                    meta={"currency": "INR", "fullExchangeName": "NSE",
                          "exchangeTimezoneName": TZ},
                    live=live, source="fake-live")

    monkeypatch.setattr(DataSource, "get", _get)
    monkeypatch.setattr(DataSource, "get_many",
                        lambda self, symbols, **kw: {s: self.get(s) for s in symbols})


def scan_cfg(symbols=("RELIANCE.NS", "TCS.NS", "INFY.NS"), **kw):
    return ScanConfig(
        params=Params.default(),
        data=DataConfig(provider="yfinance", universe=list(symbols), min_bars=90,
                        lookback_days=600, cache_dir="data/cache_test_live"),
        alert=AlertConfig(events=["tap1", "approach", "confirmed"], chart=False,
                          min_liquidity_dollar_volume=0, min_price=0, **kw.pop("alert", {})),
        trade=TradeConfig(),
        telegram=kw.pop("telegram", TelegramConfig(enabled=False)),
        live=LiveConfig(**kw.pop("live", {})),
        state_db=":memory:",
        out_dir="results_test_live",
    )


class _TGHandler(BaseHTTPRequestHandler):
    sent = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode("utf-8", "ignore")
        type(self).sent.append({"method": self.path.rsplit("/", 1)[-1], "raw": raw})
        body = json.dumps({"ok": True, "result": {"message_id": len(type(self).sent)}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def telegram_server():
    _TGHandler.sent = []
    srv = HTTPServer(("127.0.0.1", 0), _TGHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", _TGHandler.sent
    srv.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# scheduling inputs
# ─────────────────────────────────────────────────────────────────────────────
def test_malformed_schedule_times_are_ignored():
    """A typo must not become a job scheduled for 00:00."""
    assert _parse_hhmm("09:35") == (9, 35)
    assert _parse_hhmm(" 9:5 ") == (9, 5)
    for bad in ("", "oops", "9", "24:00", "12:60", None, "9:35:00"):
        assert _parse_hhmm(bad) is None, bad


# ─────────────────────────────────────────────────────────────────────────────
# feed freshness
# ─────────────────────────────────────────────────────────────────────────────
def test_cache_freshness_is_judged_by_session_not_ttl(tmp_path):
    """A frame a session behind is stale no matter how young the file is."""
    cfg = DataConfig(provider="yfinance", cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=10 ** 6, eod_cache_max_age_minutes=10 ** 6)
    src = DataSource(cfg, live=True)
    fresh, stale = live_frame(), live_frame(days_back=4)
    assert src._has_current_bar(fresh) is True
    assert src._has_current_bar(stale) is False
    assert src._cache_usable(fresh, None) is True
    assert src._cache_usable(stale, None) is False
    # a historical (backtest/verify) request is exempt — its data is frozen
    assert src._cache_usable(stale, "2024-05-01") is True
    # and so is an offline provider
    csv = DataSource(DataConfig(provider="csv", cache_dir=str(tmp_path / "c")), live=True)
    assert csv._cache_usable(stale, None) is True


def test_live_and_closed_bar_fetches_use_separate_cache_entries(tmp_path, monkeypatch,
                                                                exchange_clock):
    """A live scan must never be handed a frame cached by a closed-bar scan."""
    from test_providers import FakeTicker          # the fake yfinance module

    mod = types.ModuleType("yfinance")
    mod.Ticker = FakeTicker
    monkeypatch.setitem(__import__("sys").modules, "yfinance", mod)
    FakeTicker.calls = []
    cache = str(tmp_path / "cache")

    def make(live):
        return DataSource(DataConfig(provider="yfinance", cache_dir=cache,
                                     cache_max_age_minutes=10 ** 6,
                                     eod_cache_max_age_minutes=10 ** 6), live=live)

    eod = make(False).get("RELIANCE.NS")
    assert eod.ok and eod.source == "yfinance"
    n_after_eod = len(FakeTicker.calls)

    live = make(True).get("RELIANCE.NS")
    assert live.ok
    assert len(FakeTicker.calls) > n_after_eod, "live scan reused the closed-bar cache"
    assert live.source == "yfinance+intraday"
    assert live.live is True
    # the merged bar is stamped with the session FakeTicker built its prints from
    # (_last_session); under exchange_clock the provider agrees, so this holds on
    # a weekend — a raw _today() here only matched on weekday afternoons.
    assert pd.Timestamp(live.df.index[-1]).date() == _last_session().date()


def test_a_cache_hit_is_only_live_when_it_holds_todays_bar(tmp_path):
    """``Bars.live`` describes the *data*, not the caller's mode."""
    cfg = DataConfig(provider="yfinance", cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=10 ** 6, eod_cache_max_age_minutes=10 ** 6)
    src = DataSource(cfg, live=True)
    src.cache.store("X_1d_600d_live", live_frame())
    src.cache.store("Y_1d_600d_live", live_frame(days_back=3))
    assert src._has_current_bar(src.cache.load("X_1d_600d_live")) is True
    assert src._has_current_bar(src.cache.load("Y_1d_600d_live")) is False


# ─────────────────────────────────────────────────────────────────────────────
# stale-bar guard
# ─────────────────────────────────────────────────────────────────────────────
def test_stale_bar_detection_is_strict_intraday_and_lenient_eod():
    today = _today()
    assert _is_stale_bar(today, TZ, strict=True) is False
    assert _is_stale_bar(today, TZ, strict=False) is False
    # a feed stuck on yesterday must not fire a "price is tapping the OB now" alert
    assert _is_stale_bar(today - pd.Timedelta(days=1), TZ, strict=True) is True
    assert _is_stale_bar(today - pd.Timedelta(days=10), TZ, strict=True) is True
    # but a closed-bar scan may legitimately report the previous session
    # (Monday morning reporting Friday, a weekend run, a one-day holiday)
    assert _is_stale_bar(today - pd.Timedelta(days=3), TZ, strict=False) is False
    # a feed that has fallen several sessions behind is broken either way
    assert _is_stale_bar(today - pd.Timedelta(days=21), TZ, strict=False) is True


def test_live_scan_refuses_a_feed_that_stopped_updating(monkeypatch):
    patch_source(monkeypatch, days_back=6)
    cfg = scan_cfg(alert={"skip_stale_bars": True, "recent_bars": 3})
    with StateStore(":memory:") as store:
        rep = Scanner(cfg, store=store, dry_run=True).scan(live=True, progress=False)
        assert rep.usable == 0, rep.notes
        assert rep.alerts == []
        assert any("not today's session" in n for n in rep.notes)


def test_stale_guard_only_applies_to_live_capable_feeds(monkeypatch):
    """A csv/synthetic replay is frozen by definition — it cannot be "stale"."""
    patch_source(monkeypatch, days_back=6)
    cfg = scan_cfg(alert={"skip_stale_bars": True, "recent_bars": 3})
    cfg.data.provider = "csv"
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=True)
        st = sc._scan_frame("RELIANCE.NS", sc.source.get("RELIANCE.NS"), live=True)
        assert st.stale is False
        sc.close()


def test_historical_replay_is_exempt_from_the_stale_guard(monkeypatch):
    """``scan --end DATE`` / ``verify`` deliberately look at an old bar."""
    patch_source(monkeypatch, days_back=6)
    cfg = scan_cfg(alert={"skip_stale_bars": True, "recent_bars": 3})
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=True)
        bars = sc.source.get("RELIANCE.NS")
        assert sc._scan_frame("RELIANCE.NS", bars, live=True, historical=True).stale is False
        assert sc._scan_frame("RELIANCE.NS", bars, live=True, historical=False).stale is True
        sc.close()


# ─────────────────────────────────────────────────────────────────────────────
# closed-bar parity pass (alerts.match_indicator_100)
# ─────────────────────────────────────────────────────────────────────────────
def test_live_scan_populates_the_closed_bar_parity_pass(monkeypatch):
    """Live mode must also replay the frame with the last bar closed, so events
    the indicator has *already* printed — every ``confirmed``, which can never
    come from the forming bar — remain deliverable."""
    patch_source(monkeypatch)
    sc = Scanner(scan_cfg(), store=None, dry_run=True)
    bars = sc.source.get("RELIANCE.NS")
    st = sc._scan_frame("RELIANCE.NS", bars, live=True)
    p = sc._params_for("RELIANCE.NS")
    closed = run_engine(bars.df, p, symbol="RELIANCE.NS", intrabar_last=False, zone_cap=10 ** 6)
    want = {(e.kind, e.bar, e.zid) for e in closed.events if not e.intrabar}
    got = {(e.kind, e.bar, e.zid) for e in st.events_closed}
    assert got == want
    assert want, "fixture should contain closed-bar events"
    assert all(not e.intrabar for e in st.events_closed)
    sc.close()


def test_parity_pass_can_be_switched_off(monkeypatch):
    patch_source(monkeypatch)
    sc = Scanner(scan_cfg(alert={"match_indicator_100": False}), store=None, dry_run=True)
    st = sc._scan_frame("RELIANCE.NS", sc.source.get("RELIANCE.NS"), live=True)
    assert st.events_closed == []
    sc.close()


def _ev(kind, bar, intrabar, zid=1, tap_no=1):
    return Event(kind=kind, bar=bar, symbol="X.NS", zid=zid, tap_no=tap_no,
                 level=100.0, price=99.0, intrabar=intrabar)


def _craft_scan(n=520, *, live_events=(), closed_events=()):
    st = SymbolScan(symbol="X.NS", ok=True, bars=n, live=True)
    st.events = list(live_events)
    st.events_closed = list(closed_events)
    st.result = types.SimpleNamespace(index=pd.RangeIndex(n))
    return st


def test_recent_events_mixes_forming_bar_and_closed_bar_signals():
    """Live window = intraday events on the forming bar + the last closed bar(s)."""
    n = 520
    sc = Scanner(scan_cfg(), store=None, dry_run=True)
    sc.cfg.alert.recent_bars = 1
    st = _craft_scan(n,
                     live_events=[_ev(EV_TAP, n - 1, True), _ev(EV_CONFIRMED, n - 1, True)],
                     closed_events=[_ev(EV_TAP, n - 1, False),      # duplicate: must not double-fire
                                    _ev(EV_CONFIRMED, n - 2, False),
                                    _ev(EV_TAP, n - 5, False)])     # outside the window
    out = sc._recent_events(st, live=True)
    got = sorted((e.kind, e.bar, e.intrabar) for e in out)
    # `confirmed` cannot come from a forming bar, so the n-1 one is dropped
    assert got == [("confirmed", n - 2, False), ("tap", n - 1, True)]
    sc.close()


def test_recent_events_closed_bar_mode_ignores_the_parity_pass():
    n = 520
    sc = Scanner(scan_cfg(), store=None, dry_run=True)
    sc.cfg.alert.recent_bars = 2
    st = _craft_scan(n, live_events=[_ev(EV_TAP, n - 1, False)],
                     closed_events=[_ev(EV_TAP, n - 2, False)])
    assert [e.bar for e in sc._recent_events(st, live=False)] == [n - 1]
    sc.close()


def test_live_scan_delivers_a_closed_bar_signal(monkeypatch):
    """End to end: a signal the forming bar cannot produce still reaches the alert set."""
    patch_source(monkeypatch, setup="closed")
    cfg = scan_cfg(alert={"recent_bars": 3})
    with StateStore(":memory:") as store:
        rep = Scanner(cfg, store=store, dry_run=True).scan(live=True, progress=False)
        rows = {r.symbol: r for r in rep.rows}
        from_closed = [(e.symbol, e.kind, e.bar) for _, e in rep.alerts if not e.intrabar]
        assert from_closed, f"no closed-bar signal surfaced: {rep.summary_line}"
        for sym, _kind, bar in from_closed:
            assert bar < len(rows[sym].df) - 1, "a parity event must sit on a closed bar"


# ─────────────────────────────────────────────────────────────────────────────
# Telegram delivery
# ─────────────────────────────────────────────────────────────────────────────
def test_live_scan_sends_alerts_to_telegram(monkeypatch, telegram_server):
    api, sent = telegram_server
    patch_source(monkeypatch)
    cfg = scan_cfg(telegram=TelegramConfig(enabled=True, bot_token="123:abc",
                                           chat_ids=["42"], api_base=api,
                                           min_seconds_between_messages=0.0, timeout_sec=5))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store)
        rep = sc.scan(live=True, progress=False)
        sc.close()
        assert rep.alerts, rep.summary_line
        assert rep.dispatch.sent == len(rep.alerts)
        assert len(sent) >= len(rep.alerts)
        assert sent[0]["method"] == "sendMessage"
        assert "chat_id=42" in sent[0]["raw"]
        assert "parse_mode=HTML" in sent[0]["raw"]
        assert all(r["sent"] for r in store.recent_alerts(limit=20))


def test_unconfigured_telegram_logs_alerts_instead_of_burying_them(monkeypatch):
    """No token/chat id must not mean "silently queued forever"."""
    patch_source(monkeypatch)
    cfg = scan_cfg(telegram=TelegramConfig(enabled=True, bot_token="", chat_ids=[]))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=False)   # the `run` path: no --no-send
        assert sc.telegram is None
        rep = sc.scan(live=True, progress=False)
        sc.close()
        assert rep.alerts, rep.summary_line
        assert rep.dispatch.sent == 0
        assert rep.dispatch.skipped == len(rep.alerts)
        # rendered and recorded, but NOT left pending — there is nothing to retry with
        assert store.pending(limit=50) == []
        assert len(store.recent_alerts(limit=20)) == len(rep.alerts)
        assert sc.dispatcher.retry_pending() == 0


def test_dedupe_stops_a_second_live_cycle_resending(monkeypatch, telegram_server):
    api, sent = telegram_server
    patch_source(monkeypatch)
    cfg = scan_cfg(telegram=TelegramConfig(enabled=True, bot_token="123:abc",
                                           chat_ids=["42"], api_base=api,
                                           min_seconds_between_messages=0.0))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store)
        first = sc.scan(live=True, progress=False)
        n = len(sent)
        second = sc.scan(live=True, progress=False)
        sc.close()
    assert first.dispatch.sent > 0
    # nothing new to report, so the second cycle must not even reach dispatch
    assert second.alerts == [] and (second.dispatch is None or second.dispatch.sent == 0)
    assert len(sent) == n, "the same session's alerts were sent twice"


# ─────────────────────────────────────────────────────────────────────────────
# the always-on loop
# ─────────────────────────────────────────────────────────────────────────────
class FakeClock:
    """Virtual monotonic + wall clock, so a whole trading day runs instantly."""

    def __init__(self, start: datetime):
        self.now = start
        self.mono = 1000.0

    def monotonic(self):
        return self.mono

    def sleep(self, secs):
        secs = max(0.0, float(secs))
        self.now += timedelta(seconds=secs)
        self.mono += secs


class FakeScanner:
    """Records every cycle the scheduler triggers: (wall time, live mode)."""

    def __init__(self, cfg, clock=None):
        self.cfg = cfg
        self.telegram = None
        self.dispatcher = type("D", (), {"deliverable": False,
                                         "retry_pending": lambda *a, **k: 0})()
        self.calls = []
        self.clock = clock

    def universe(self):
        return ["RELIANCE.NS"]

    def scan(self, *, live, progress=False, **kw):
        self.calls.append((self.clock.now if self.clock else None, live))
        from precision_tap.scanner import ScanReport
        return ScanReport(mode="live" if live else "eod", notes=["x"])


@pytest.fixture
def loop_clock(monkeypatch):
    """Install a virtual clock over :mod:`precision_tap.live`."""
    import precision_tap.live as L

    def install(start: datetime):
        clock = FakeClock(start)
        monkeypatch.setattr(L, "time",
                            type("T", (), {"monotonic": clock.monotonic, "sleep": clock.sleep})())
        monkeypatch.setattr(L, "_local_now", lambda tzname: clock.now)
        real_open = L._market_open
        monkeypatch.setattr(L, "_market_open",
                            lambda tzname=None, now=None, session=None:
                            real_open(tzname, now or clock.now, session))
        return clock

    return install


def _run_until(loop_clock, cfg, start, hours):
    """Drive ``LiveLoop.run()`` until the virtual clock passes the deadline."""
    import precision_tap.live as L
    clock = loop_clock(datetime.fromisoformat(f"2026-09-11T{start}:00"))
    sc = FakeScanner(cfg, clock)
    stop_at = clock.now + timedelta(hours=hours)
    real_sleep = clock.sleep

    def sleep_guard(secs):
        real_sleep(secs)
        if clock.now >= stop_at:
            raise KeyboardInterrupt
    L.time.sleep = sleep_guard
    try:
        L.LiveLoop(cfg, sc, heartbeat=True).run()
    except KeyboardInterrupt:
        pass
    return sc, clock


def test_loop_polls_intraday_and_runs_mid_session_scans_live(loop_clock):
    """A scan scheduled inside the session must run in live mode.

    This is the bug that cost three of the five daily scans: the mode was chosen
    from the trigger tag, so 09:35 / 11:30 / 13:30 always ran closed-bar and
    could never see an intraday tap.
    """
    cfg = scan_cfg(live={"intraday_poll_minutes": 10,
                         "scan_times": ["09:35", "11:30", "13:30", "15:35", "16:10"],
                         "run_on_startup": False})
    sc, _ = _run_until(loop_clock, cfg, "09:00", 12)
    live_cycles = [t for t, live in sc.calls if live]
    assert len(live_cycles) > 10, f"intraday polling did not happen: {sc.calls}"
    inside_session = [t for t, live in sc.calls if t.hour * 60 + t.minute < 15 * 60 + 30]
    assert inside_session and all(
        live for t, live in sc.calls if t.hour * 60 + t.minute < 15 * 60 + 30), sc.calls


def test_loop_runs_post_close_scans_in_closed_bar_mode(loop_clock):
    cfg = scan_cfg(live={"intraday_poll_minutes": 10,
                         "scan_times": ["15:35", "16:10"], "run_on_startup": False})
    sc, _ = _run_until(loop_clock, cfg, "15:00", 5)
    after_close = [(t, live) for t, live in sc.calls if t.hour * 60 + t.minute >= 15 * 60 + 30]
    assert after_close and all(not live for _, live in after_close), sc.calls


def test_run_once_is_live_during_the_session(loop_clock):
    """``run --once`` is the cron entry point; it must be able to alert intraday."""
    import precision_tap.live as L
    cfg = scan_cfg()
    clock = loop_clock(datetime.fromisoformat("2026-09-11T11:00:00"))
    sc = FakeScanner(cfg, clock)
    L.LiveLoop(cfg, sc)._scan("once")
    assert sc.calls == [(clock.now, True)]


def test_loop_ignores_malformed_schedule_entries(loop_clock):
    cfg = scan_cfg(live={"scan_times": ["oops", "25:99", "10:00"], "run_on_startup": False,
                         "intraday_poll_minutes": 0, "heartbeat_daily_time": ""})
    sc, _ = _run_until(loop_clock, cfg, "09:00", 4)
    assert len(sc.calls) == 1, f"expected only the 10:00 scan, got {sc.calls}"


def test_loop_recovers_from_a_failing_cycle(loop_clock):
    """One bad cycle must not kill the loop (provider blips are normal)."""
    import precision_tap.live as L
    cfg = scan_cfg(live={"scan_times": ["10:00", "11:00"], "run_on_startup": True,
                         "intraday_poll_minutes": 0, "heartbeat_daily_time": ""})
    clock = loop_clock(datetime.fromisoformat("2026-09-11T09:00:00"))

    class Boom(FakeScanner):
        n = 0

        def scan(self, *, live, progress=False, **kw):
            Boom.n += 1
            if Boom.n == 1:
                raise RuntimeError("provider exploded")
            return super().scan(live=live, progress=progress, **kw)

    sc = Boom(cfg, clock)
    stop_at = clock.now + timedelta(hours=5)

    def sleep_guard(secs):
        clock.sleep(secs)
        if clock.now >= stop_at:
            raise KeyboardInterrupt
    L.time.sleep = sleep_guard
    try:
        L.LiveLoop(cfg, sc).run()
    except KeyboardInterrupt:
        pass
    assert Boom.n >= 3, "the loop died on the first failure"


# ─────────────────────────────────────────────────────────────────────────────
# alert content
# ─────────────────────────────────────────────────────────────────────────────
def test_tap_alert_prints_the_level_that_was_touched():
    """``raise_after_first_tap`` lifts ``zone.entry`` *on the tap bar*.

    By the time the alert is rendered the zone's live pre-order is already the
    raised level, so printing ``zone.entry`` as "Buy limit ← Tap 1 trigger"
    quotes a price the tap never touched.  The Pine label makes the same
    distinction: "TAP n" at the touched price, then "Next <raised>".
    """
    from precision_tap.alerts import render_message
    from precision_tap.engine import EV_TAP, Event, Zone
    from precision_tap.params import AlertConfig, Params

    z = Zone(zid=0, born=10, origin=9, top=100.0, bot=98.0, entry=100.60, entry0=100.05,
             stop=97.0, atr0=1.0, taps=1, state=1, departed=True, adaptive=True)
    ev = Event(kind=EV_TAP, bar=20, symbol="TCS.NS", zid=0, tap_no=1, level=100.05,
               price=100.00, zone=z)
    ctx = {"price": 100.20, "change_pct": -0.14, "atr": 1.0, "rvol": 1.4,
           "exchange": "NSE", "timeframe": "1d", "entry": z.entry, "origin_date": "2026-09-01"}
    text = render_message(ev, ctx, AlertConfig(), Params.default(), parse_mode="plain")
    assert "100.05" in text, text          # the level that was actually tapped
    assert "100.60" in text, text          # the next (raised) pre-order
    assert "next pre-order" in text, text


def test_non_tap_alerts_quote_the_live_pre_order():
    """Approach / confirmed are about the level that is live *now*."""
    from precision_tap.alerts import render_message
    from precision_tap.engine import EV_APPROACH, Event, Zone
    from precision_tap.params import AlertConfig, Params

    z = Zone(zid=1, born=10, origin=9, top=100.0, bot=98.0, entry=100.60, entry0=100.05,
             stop=97.0, atr0=1.0, taps=1, state=1, departed=True, adaptive=True)
    ev = Event(kind=EV_APPROACH, bar=21, symbol="TCS.NS", zid=1, level=100.60,
               price=100.90, zone=z)
    ctx = {"price": 100.90, "change_pct": 0.2, "atr": 1.0, "rvol": 1.1,
           "exchange": "NSE", "timeframe": "1d", "entry": z.entry}
    text = render_message(ev, ctx, AlertConfig(), Params.default(), parse_mode="plain")
    assert "100.60" in text, text


def test_a_quiet_cycle_says_why_nothing_was_sent(monkeypatch):
    """"Why did nothing fire?" is the first question after an empty cycle, so
    the report must carry the filter tally instead of a bare `alerts matched: 0`."""
    patch_source(monkeypatch)
    cfg = scan_cfg(alert={"recent_bars": 3})
    cfg.alert.min_liquidity_dollar_volume = 9e18        # nothing can pass
    sc = Scanner(cfg, store=None, dry_run=True)
    rep = sc.scan(live=True, progress=False)
    sc.close()
    assert rep.alerts == []
    assert any(n.startswith("nothing to send") and "illiquid" in n for n in rep.notes), rep.notes


def test_exchange_label_maps_provider_codes():
    from precision_tap.scanner import exchange_label
    # Yahoo reports the NSE as "NSI"; the alert should read NSE
    assert exchange_label("NSI") == "NSE"
    assert exchange_label("nse") == "NSE"
    assert exchange_label("BOM") == "BSE"
    assert exchange_label("NASDAQ") == "NASDAQ"      # unknown codes pass through
    assert exchange_label(None) == ""
