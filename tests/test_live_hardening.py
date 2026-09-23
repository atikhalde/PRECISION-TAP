"""Hardening regressions: the live pipeline must survive the real market.

Covers the fixes for "green run, silent Telegram":

* yfinance transient failures are retried + throttled (like the chart-API path
  always was), and the daily fetch uses a bounded start/end window instead of
  ``period="max"``;
* a dead primary provider fails over to ``data.fallback_provider`` per symbol;
* ``scan`` / ``run --once`` exit non-zero when nothing was usable (feed dead)
  or when signals were found but not delivered (Telegram broken) — a broken
  pipeline can no longer look like a quiet market;
* ``telegram-test --validate-only`` proves token + chat id via getMe/getChat
  without sending anything;
* quote buttons link the suffixed Yahoo symbol; long captions fall back to
  text instead of being silently truncated; the scheduler wakes for the
  heartbeat; a fully-skipped cycle names the cause.
"""
from __future__ import annotations

import json
import sys
import threading
import types
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from precision_tap.alerts import build_buttons, render_message
from precision_tap.data import Bars, DataSource
from precision_tap.engine import EV_TAP, Event, Zone
from precision_tap.live import LiveLoop
from precision_tap.params import (AlertConfig, DataConfig, LiveConfig, Params, ScanConfig,
                                  TelegramConfig)
from precision_tap.scanner import Scanner
from precision_tap.state import StateStore
from precision_tap.telegram import TelegramClient, TelegramPermanentError

from test_live import patch_source, scan_cfg  # noqa: E402  (offline fixtures)
from test_providers import FakeTicker, _daily  # noqa: E402  (fake yfinance module)


@pytest.fixture
def fake_yf(monkeypatch, tmp_path):
    mod = types.ModuleType("yfinance")
    mod.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    FakeTicker.calls = []
    return mod


def _src(tmp_path, **kw):
    d = dict(provider="yfinance", symbol_suffix=".NS", min_bars=50, lookback_days=400,
             cache_dir=str(tmp_path / "cache"), cache_max_age_minutes=0,
             eod_cache_max_age_minutes=0, rate_limit_per_sec=0, interval="1d")
    d.update(kw)
    return DataSource(DataConfig(**d), live=False)


# ─────────────────────────────────────────────────────────────────────────────
# yfinance retry + throttling + bounded window
# ─────────────────────────────────────────────────────────────────────────────
def test_yfinance_daily_uses_bounded_window_not_max(fake_yf, tmp_path):
    src = _src(tmp_path, lookback_days=900)
    bars = src.get("RELIANCE")
    assert bars.ok, bars.error
    first = FakeTicker.calls[0]
    assert "max" not in str(first.get("period", "")), first
    assert first.get("start") and first.get("end"), first


def test_yfinance_transient_failure_is_retried(fake_yf, tmp_path, monkeypatch):
    src = _src(tmp_path, retry_max=3, retry_backoff=0.01)
    tk = FakeTicker("RELIANCE.NS")
    calls = {"n": 0}
    real_history = FakeTicker.history

    def flaky(self, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("curl: connection reset")
        return real_history(self, **kw)

    monkeypatch.setattr(FakeTicker, "history", flaky)
    out = src._yfinance_history(tk, {"period": "1d", "interval": "1d"}, what="test")
    assert out is not None and len(out) > 0
    assert calls["n"] == 3


def test_yfinance_gives_up_after_retry_max(fake_yf, tmp_path, monkeypatch):
    src = _src(tmp_path, retry_max=2, retry_backoff=0.01)
    tk = FakeTicker("RELIANCE.NS")

    def dead(self, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(FakeTicker, "history", dead)
    with pytest.raises(RuntimeError, match="after 2 tries"):
        src._yfinance_history(tk, {"period": "1d", "interval": "1d"}, what="test")


def test_yfinance_signature_errors_are_not_retried(fake_yf, tmp_path, monkeypatch):
    src = _src(tmp_path, retry_max=3, retry_backoff=0.01)
    tk = FakeTicker("RELIANCE.NS")
    calls = {"n": 0}

    def bad_sig(self, **kw):
        calls["n"] += 1
        raise TypeError("history() got an unexpected keyword argument 'frobnicate'")

    monkeypatch.setattr(FakeTicker, "history", bad_sig)
    with pytest.raises(TypeError):
        src._yfinance_history(tk, {"period": "1d", "interval": "1d"}, what="test")
    assert calls["n"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# provider failover
# ─────────────────────────────────────────────────────────────────────────────
def test_dead_primary_fails_over_to_fallback(tmp_path, monkeypatch):
    cfg = DataConfig(provider="yfinance", fallback_provider="yahoo", symbol_suffix=".NS",
                     min_bars=50, lookback_days=400, cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=0, eod_cache_max_age_minutes=0)
    src = DataSource(cfg, live=False)

    def primary_down(self, symbol, days, end):
        raise RuntimeError("yfinance exploded")

    def fallback_ok(self, symbol, days, end):
        return Bars(symbol=symbol, df=_daily(), source="yahoo")

    monkeypatch.setattr(DataSource, "_yfinance", primary_down)
    monkeypatch.setattr(DataSource, "_yahoo", fallback_ok)
    bars = src.get("RELIANCE.NS")
    assert bars.ok, bars.error
    assert bars.source == "yahoo+failover"


def test_failover_can_be_disabled(tmp_path, monkeypatch):
    cfg = DataConfig(provider="yfinance", fallback_provider="", symbol_suffix=".NS",
                     min_bars=50, lookback_days=400, cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=0, eod_cache_max_age_minutes=0)
    src = DataSource(cfg, live=False)

    def primary_down(self, symbol, days, end):
        raise RuntimeError("yfinance exploded")

    monkeypatch.setattr(DataSource, "_yfinance", primary_down)
    bars = src.get("RELIANCE.NS")
    assert not bars.ok and "yfinance exploded" in bars.error


def test_both_providers_down_reports_both_errors(tmp_path, monkeypatch):
    cfg = DataConfig(provider="yfinance", fallback_provider="yahoo", symbol_suffix=".NS",
                     min_bars=50, lookback_days=400, cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=0, eod_cache_max_age_minutes=0)
    src = DataSource(cfg, live=False)
    monkeypatch.setattr(DataSource, "_yfinance",
                        lambda self, s, d, e: (_ for _ in ()).throw(RuntimeError("down-1")))
    monkeypatch.setattr(DataSource, "_yahoo",
                        lambda self, s, d, e: (_ for _ in ()).throw(RuntimeError("down-2")))
    bars = src.get("RELIANCE.NS")
    assert not bars.ok
    assert "down-1" in bars.error and "down-2" in bars.error


def test_both_providers_down_names_each_provider_once(tmp_path, monkeypatch):
    """One label per provider — the failure message is read by a human.

    The primary error is already built as ``<provider>: <what it said>``, so the
    failover composition must not prefix it a second time; and the providers'
    own raised texts must not open with their own name either.  What reached the
    chat before: ``yfinance: yfinance: yfinance returned no rows; yahoo: yahoo
    request failed after 3 tries: …``.
    """
    cfg = DataConfig(provider="yfinance", fallback_provider="yahoo", symbol_suffix=".NS",
                     min_bars=50, lookback_days=400, cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=0, eod_cache_max_age_minutes=0)
    src = DataSource(cfg, live=False)
    monkeypatch.setattr(DataSource, "_yfinance",
                        lambda self, s, d, e: (_ for _ in ()).throw(RuntimeError("down-1")))
    monkeypatch.setattr(DataSource, "_yahoo",
                        lambda self, s, d, e: (_ for _ in ()).throw(RuntimeError("down-2")))
    err = src.get("RELIANCE.NS").error
    assert err == "yfinance: down-1; yahoo: down-2", err
    assert err.count("yfinance") == 1 and err.count("yahoo") == 1
    # the raised texts do not name their own provider (the caller labels them)
    assert "request failed after" in _http_json_error_text(src)
    assert not _http_json_error_text(src).startswith("yahoo")
    assert not _yfinance_history_error_text().startswith("yfinance")


def _http_json_error_text(src) -> str:
    from precision_tap.data import DataSource
    try:
        src._http_json("http://127.0.0.1:1/nope", {})
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("expected a RuntimeError")


def _yfinance_history_error_text() -> str:
    from precision_tap.data import DataSource

    class _Tk:
        symbol = "X.NS"

        def history(self, **kw):
            raise RuntimeError("boom")

    cfg = DataConfig(provider="yfinance", symbol_suffix=".NS", retry_max=1,
                     cache_dir="/tmp/pt-mut-test-cache")
    src = DataSource(cfg, live=False)
    try:
        src._yfinance_history(_Tk(), {"period": "1d"}, what="daily X.NS")
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("expected a RuntimeError")


# ─────────────────────────────────────────────────────────────────────────────
# Telegram validation without noise
# ─────────────────────────────────────────────────────────────────────────────
class _ChatHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n).decode("utf-8", "ignore")
        method = self.path.rsplit("/", 1)[-1]
        type(self).requests.append((method, raw))
        if method == "getMe":
            body = {"ok": True, "result": {"username": "tapbot", "first_name": "Tap"}}
            code = 200
        elif method == "getChat":
            if "chat_id=999" in raw.replace("%20", " "):
                body = {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}
                code = 400
            else:
                body = {"ok": True, "result": {"id": 42, "type": "private", "first_name": "Me"}}
                code = 200
        elif method in ("sendMessage", "sendPhoto"):
            body = {"ok": True, "result": {"message_id": 7}}
            code = 200
        else:
            body = {"ok": False, "error_code": 404, "description": "not found"}
            code = 404
        blob = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *a):
        pass


@pytest.fixture
def chat_server():
    _ChatHandler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _ChatHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _tg(base, chats=("42",)):
    return TelegramClient(TelegramConfig(enabled=True, bot_token="123:abc",
                                         chat_ids=list(chats), api_base=base,
                                         min_seconds_between_messages=0.0, timeout_sec=5))


def test_validate_chats_passes_without_sending(chat_server):
    tg = _tg(chat_server)
    found = tg.validate_chats()
    assert found["bot"]["username"] == "tapbot"
    assert found["chats"]["42"]["type"] == "private"
    methods = [m for m, _ in _ChatHandler.requests]
    assert methods == ["getMe", "getChat"]
    assert "sendMessage" not in methods and "sendPhoto" not in methods


def test_validate_chats_fails_on_unknown_chat(chat_server):
    tg = _tg(chat_server).cfg
    tg.chat_ids = ["999"]
    with pytest.raises(TelegramPermanentError, match="chat not found"):
        TelegramClient(tg).validate_chats()


def test_long_caption_falls_back_to_text(chat_server, tmp_path):
    tg = _tg(chat_server)
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG-not-really")
    res = tg.send_photo("x" * 2000, str(png))
    assert all(r.ok for r in res)
    methods = [m for m, _ in _ChatHandler.requests]
    assert "sendPhoto" not in methods
    assert "sendMessage" in methods


def test_short_caption_still_sends_photo(chat_server, tmp_path):
    tg = _tg(chat_server)
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG-not-really")
    res = tg.send_photo("short caption", str(png))
    assert all(r.ok for r in res)
    assert [m for m, _ in _ChatHandler.requests] == ["sendPhoto"]


def test_quote_button_keeps_the_exchange_suffix():
    z = Zone(zid=0, born=10, origin=9, top=100.0, bot=98.0, entry=100.05,
             entry0=100.05, stop=97.0, atr0=1.0)
    ev = Event(kind=EV_TAP, bar=20, symbol="TCS.NS", zid=0, tap_no=1,
               level=100.05, price=100.0, zone=z)
    btns = build_buttons(ev, {"exchange": "NSE"}, AlertConfig())
    urls = [b["url"] for row in btns["inline_keyboard"] for b in row]
    assert "https://finance.yahoo.com/quote/TCS.NS" in urls, urls
    assert any("symbol=NSE:TCS" in u for u in urls), urls


def test_render_message_accepts_uppercase_plain():
    z = Zone(zid=0, born=10, origin=9, top=100.0, bot=98.0, entry=100.05,
             entry0=100.05, stop=97.0, atr0=1.0)
    ev = Event(kind=EV_TAP, bar=20, symbol="TCS.NS", zid=0, tap_no=1,
               level=100.05, price=100.0, zone=z)
    ctx = {"price": 100.2, "exchange": "NSE", "timeframe": "1d"}
    # TelegramConfig upper-cases parse_mode, so "plain" arrives as "PLAIN"
    text = render_message(ev, ctx, AlertConfig(), Params.default(), parse_mode="PLAIN")
    assert "<b>" not in text and "PRECISION OB" in text


# ─────────────────────────────────────────────────────────────────────────────
# loud exit codes: a broken pipeline must not look like a quiet market
# ─────────────────────────────────────────────────────────────────────────────
def test_scan_exits_3_when_nothing_was_usable(monkeypatch, tmp_path, capsys):
    import precision_tap.data as D
    from precision_tap.cli import cmd_scan

    monkeypatch.setattr(D.DataSource, "get_many",
                        lambda self, symbols, **kw: {
                            s: Bars(symbol=s, df=pd.DataFrame(), error="boom") for s in symbols})

    class A:
        config = None
        set = ["data.provider=yfinance", "data.universe=RELIANCE.NS,TCS.NS",
               "data.min_bars=90", f"data.cache_dir={tmp_path/'cache'}",
               f"out_dir={tmp_path/'out'}", "state_db=:memory:", "alerts.chart=false",
               "telegram.enabled=false"]
        env_file = None
        symbols = None
        days = None
        provider = None
        limit = None
        events = None
        recent_bars = None
        no_charts = True
        min_dollar_volume = None
        trigger = target_r = stop_mode = trail = time_stop = risk = capital = max_positions = None
        live = False
        eod = True
        end = None
        no_send = True
        report_only = False
        retried = False
        messages = False
        top = 5
        workers = 1

    assert cmd_scan(A()) == 3
    assert "0/2 symbols usable" in capsys.readouterr().err


def test_scan_exits_4_when_alerts_never_reach_telegram(monkeypatch, tmp_path, capsys):
    import precision_tap.data as D
    from precision_tap.cli import cmd_scan

    patch_source(monkeypatch)  # the feed works and signals exist

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            body = json.dumps({"ok": False, "error_code": 400,
                               "description": "Bad Request: chat not found"}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        api = f"http://127.0.0.1:{srv.server_address[1]}"

        class A:
            config = None
            set = ["data.provider=yfinance", "data.universe=RELIANCE.NS",
                   "data.min_bars=90", f"data.cache_dir={tmp_path/'cache'}",
                   f"out_dir={tmp_path/'out'}", "state_db=:memory:", "alerts.chart=false",
                   "alerts.min_liquidity_dollar_volume=0", "alerts.min_price=0",
                   "telegram.bot_token=123:abc", "telegram.chat_ids=999",
                   f"telegram.api_base={api}", "telegram.min_seconds_between_messages=0"]
            env_file = None
            symbols = None
            days = None
            provider = None
            limit = None
            events = None
            recent_bars = None
            no_charts = True
            min_dollar_volume = None
            trigger = target_r = stop_mode = trail = time_stop = risk = capital = max_positions = None
            live = True
            eod = False
            end = None
            no_send = False
            report_only = False
            retried = False
            messages = False
            top = 5
            workers = 1

        assert cmd_scan(A()) == 4
        assert "NOT delivered" in capsys.readouterr().err
    finally:
        srv.shutdown()


def test_scan_stays_green_when_the_feed_has_no_fresh_bar(monkeypatch, tmp_path, capsys):
    """Holiday-shaped outcome (fetches work, nothing is fresh) must NOT fail.

    Failing it would paint every 15-min slot of every NSE holiday red and spam
    a failure ping each time — the cycle note + step summary already say why.
    """
    import precision_tap.data as D
    from precision_tap.cli import cmd_scan

    frame = _daily()

    monkeypatch.setattr(D.DataSource, "get_many",
                        lambda self, symbols, **kw: {
                            s: Bars(symbol=s, df=frame, live=False, source="fake")
                            for s in symbols})

    class A:
        config = None
        set = ["data.provider=yfinance", "data.universe=RELIANCE.NS,TCS.NS",
               "data.min_bars=90", f"data.cache_dir={tmp_path/'cache'}",
               f"out_dir={tmp_path/'out'}", "state_db=:memory:", "alerts.chart=false",
               "telegram.enabled=false"]
        env_file = None
        symbols = None
        days = None
        provider = None
        limit = None
        events = None
        recent_bars = None
        no_charts = True
        min_dollar_volume = None
        trigger = target_r = stop_mode = trail = time_stop = risk = capital = max_positions = None
        live = True
        eod = False
        end = None
        no_send = True
        report_only = False
        retried = False
        messages = False
        top = 5
        workers = 1

    assert cmd_scan(A()) == 0
    assert "no usable symbols" in capsys.readouterr().out


def test_telegram_test_validate_only_passes(chat_server, tmp_path, capsys, monkeypatch):
    from precision_tap.cli import cmd_telegram

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    class A:
        config = None
        set = [f"telegram.api_base={chat_server}", "telegram.bot_token=123:abc",
               "telegram.chat_ids=42", "telegram.min_seconds_between_messages=0",
               f"out_dir={tmp_path/'out'}", "state_db=:memory:"]
        env_file = None
        discover = False
        validate_only = True
        timeout = 5.0
        text = None
        symbols = None
        days = None
        provider = None
        limit = None
        events = None
        recent_bars = None
        no_charts = True
        min_dollar_volume = None
        trigger = target_r = stop_mode = trail = time_stop = risk = capital = max_positions = None

    assert cmd_telegram(A()) == 0
    out = capsys.readouterr().out
    assert "VALIDATION PASSED" in out
    assert [m for m, _ in _ChatHandler.requests] == ["getMe", "getChat"]


# ─────────────────────────────────────────────────────────────────────────────
# scheduler + scanner diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def test_sleep_schedule_wakes_for_the_heartbeat():
    cfg = scan_cfg(live={"scan_times": ["23:59"], "intraday_poll_minutes": 0,
                         "heartbeat_daily_time": "12:00", "run_on_startup": False})
    loop = LiveLoop(cfg, scanner=None, heartbeat=True)
    now = datetime(2026, 9, 11, 11, 50)          # 10 min before the heartbeat
    assert loop._sleep_seconds(now) == pytest.approx(600.0, abs=5.0)


def test_fully_skipped_live_cycle_names_the_cause(monkeypatch, tmp_path):
    frame = _daily()

    def fake_get_many(self, symbols, **kwargs):
        return {s: Bars(symbol=s, df=frame, live=False, source="fake") for s in symbols}

    monkeypatch.setattr("precision_tap.scanner.DataSource.get_many", fake_get_many)
    cfg = ScanConfig(data=DataConfig(provider="yfinance", universe=["RELIANCE.NS"],
                                    min_bars=90, cache_dir=str(tmp_path / "cache")),
                     alert=AlertConfig(chart=False, min_liquidity_dollar_volume=0, min_price=0),
                     telegram=TelegramConfig(enabled=False),
                     state_db=":memory:", out_dir=str(tmp_path / "results"))
    sc = Scanner(cfg, dry_run=True)
    try:
        rep = sc.scan(live=True, progress=False)
    finally:
        sc.close()
    assert rep.usable == 0
    assert any(n.startswith("no usable symbols") for n in rep.notes), rep.notes


def test_yfinance_transient_typeerror_is_retried(fake_yf, tmp_path, monkeypatch):
    """A TypeError from inside yfinance is a hiccup, not a signature problem.

    ``Ticker.history`` walks ``_get_ticker_tz`` → ``self.info`` first, and when
    that metadata request comes back empty yfinance raises
    ``TypeError: argument of type 'NoneType' is not iterable``.  Treating every
    TypeError as "this build rejects the kwarg" skipped the retry *and* the
    backoff for every symbol in the universe, so one flappy minute zeroed a whole
    cycle.  Only the ``unexpected keyword argument`` shape is permanent (and the
    ``_history`` shim heals even that).
    """
    src = _src(tmp_path, retry_max=3, retry_backoff=0.01)
    tk = FakeTicker("RELIANCE.NS")
    calls = {"n": 0}
    real_history = FakeTicker.history

    def no_info_yet(self, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TypeError("argument of type 'NoneType' is not iterable")
        return real_history(self, **kw)

    monkeypatch.setattr(FakeTicker, "history", no_info_yet)
    out = src._yfinance_history(tk, {"period": "1d", "interval": "1d"}, what="test")
    assert out is not None and len(out) > 0
    assert calls["n"] == 3, calls


def test_whole_universe_survives_a_flaky_yfinance_metadata_path(fake_yf, tmp_path, monkeypatch):
    """…and the cycle still scans, instead of reporting every symbol as broken."""
    from precision_tap.data import Bars
    import precision_tap.data as D

    src = _src(tmp_path, retry_max=3, retry_backoff=0.01, min_bars=50)
    state = {"n": 0}
    real = FakeTicker.history

    def flaky(self, **kw):
        state["n"] += 1
        if state["n"] <= 2:                       # fails twice, then recovers
            raise TypeError("argument of type 'NoneType' is not iterable")
        return real(self, **kw)

    monkeypatch.setattr(FakeTicker, "history", flaky)
    bars = src.get("RELIANCE.NS", use_cache=False)
    assert bars.ok, bars.error
    assert state["n"] >= 3
