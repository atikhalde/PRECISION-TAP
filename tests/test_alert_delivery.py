"""Delivery regressions — the "scanner ran, Telegram was silent" family.

Each test here reproduces a way the scanner can look perfectly healthy in its
own log while nothing ever reaches the chat:

* a cycle that could not deliver (``--no-send``, log-only, no token) writing the
  de-duplication key anyway, so the *next* cycle — the one with a working bot —
  is told "already alerted" and stays quiet for the rest of the session;
* a Telegram rejection that cannot heal (401 bad token, 400 chat not found,
  403 bot blocked) being parked in the retry queue and quietly re-tried forever;
* a cycle that found signals but delivered none logging the same line as a
  healthy one;
* a quiet cycle giving no reason at all, so "no signals" and "broken pipeline"
  are indistinguishable.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest

from precision_tap.data import synthetic_frame
from precision_tap.live import LiveLoop
from precision_tap.params import TelegramConfig
from precision_tap.scanner import Scanner
from precision_tap.state import StateStore
from precision_tap.telegram import TelegramClient, TelegramPermanentError

from test_live import patch_source, scan_cfg


@pytest.fixture
def ok_telegram():
    """A Bot API double that accepts everything."""
    sent = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            sent.append(self.rfile.read(n).decode("utf-8", "ignore"))
            body = json.dumps({"ok": True, "result": {"message_id": len(sent)}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", sent
    srv.shutdown()


@pytest.fixture
def rejected_telegram():
    """A Bot API double that answers 401 — the classic revoked/mistyped token."""
    calls = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            calls["n"] += 1
            body = json.dumps({"ok": False, "error_code": 401,
                               "description": "Unauthorized"}).encode()
            self.send_response(401)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", calls
    srv.shutdown()


def _tg(api, **kw):
    return TelegramConfig(enabled=True, bot_token="123:abc", chat_ids=["42"], api_base=api,
                          min_seconds_between_messages=0.0, timeout_sec=5, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# the de-duplication ledger
# ─────────────────────────────────────────────────────────────────────────────
def test_a_cycle_that_could_not_deliver_does_not_consume_the_alert(monkeypatch, ok_telegram):
    """``scan --no-send`` then ``run`` must still deliver.

    This is the documented onboarding order, and it used to be fatal: the dry
    run recorded the alert key, ``seen()`` counted it, and every later cycle
    dropped the same signal as "already alerted" — for the whole session.
    """
    api, sent = ok_telegram
    patch_source(monkeypatch)
    with StateStore(":memory:") as store:
        dry = Scanner(scan_cfg(), store=store, dry_run=True)          # `scan --no-send`
        first = dry.scan(live=True, progress=False)
        dry.close()
        assert first.alerts, first.summary_line
        assert first.dispatch.sent == 0 and len(sent) == 0

        real = Scanner(scan_cfg(telegram=_tg(api)), store=store)      # `run`
        second = real.scan(live=True, progress=False)
        real.close()

    assert second.alerts, f"the dry run swallowed the signal: {second.notes}"
    assert second.dispatch.sent == len(second.alerts)
    assert len(sent) == len(second.alerts)


def test_given_up_rows_are_revived_but_delivered_rows_stay_deduped():
    with StateStore(":memory:") as store:
        assert store.record_alert("K1", symbol="S", event="tap1") is True
        store.give_up(["K1"])                       # no transport / permanent 4xx
        assert store.seen("K1") is False, "a never-delivered alert must not count as seen"
        assert store.record_alert("K1", symbol="S", event="tap1") is True, \
            "the retry with a working transport must be allowed"

        assert store.record_alert("K2", symbol="S", event="tap1") is True
        store.mark("K2", sent=True)
        assert store.seen("K2") is True
        assert store.record_alert("K2", symbol="S", event="tap1") is False, \
            "a delivered alert must never be re-sent"

        assert store.record_alert("K3", symbol="S", event="tap1") is True
        store.mark("K3", sent=False, error="boom")  # queued for retry
        assert store.seen("K3") is True
        assert store.record_alert("K3", symbol="S", event="tap1") is False
        assert {r.key for r in store.pending(limit=10)} == {"K1", "K3"}, \
            "the revived row and the queued row both still owe a delivery"


def test_a_third_cycle_does_not_resend_what_was_delivered(monkeypatch, ok_telegram):
    api, sent = ok_telegram
    patch_source(monkeypatch)
    cfg = scan_cfg(telegram=_tg(api))
    with StateStore(":memory:") as store:
        dry = Scanner(scan_cfg(), store=store, dry_run=True)
        dry.scan(live=True, progress=False)
        dry.close()
        for _ in range(2):
            sc = Scanner(cfg, store=store)
            sc.scan(live=True, progress=False)
            sc.close()
    assert len(sent) == 3, f"expected exactly one delivery per symbol, got {len(sent)}"


# ─────────────────────────────────────────────────────────────────────────────
# permanent rejections
# ─────────────────────────────────────────────────────────────────────────────
def test_a_permanent_rejection_is_reported_instead_of_retried(monkeypatch, rejected_telegram):
    api, calls = rejected_telegram
    patch_source(monkeypatch)
    cfg = scan_cfg(telegram=_tg(api, max_retries=3))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store)
        rep = sc.scan(live=True, progress=False)
        retried = sc.dispatcher.retry_pending()
        rows = store.recent_alerts(limit=10)
        sc.close()

    assert rep.alerts, rep.summary_line
    assert rep.dispatch.sent == 0 and rep.dispatch.failed == len(rep.alerts)
    assert any(n.startswith("DELIVERY PROBLEM") for n in rep.notes), rep.notes
    assert any("401" in e for e in rep.dispatch.errors)
    # a 401 is not retried: one attempt per alert, and the retry pass stays idle
    assert calls["n"] == len(rep.alerts)
    assert retried == 0
    assert all(r["sent"] == StateStore.GIVEN_UP for r in rows)


def test_permanent_errors_are_not_retried_by_the_client(rejected_telegram):
    api, calls = rejected_telegram
    tg = TelegramClient(_tg(api, max_retries=4))
    with pytest.raises(TelegramPermanentError):
        tg._post("sendMessage", {"chat_id": "42", "text": "hi"})
    assert calls["n"] == 1, "a 401 must not be retried with backoff"


# ─────────────────────────────────────────────────────────────────────────────
# a quiet cycle must explain itself
# ─────────────────────────────────────────────────────────────────────────────
def test_a_quiet_cycle_explains_why_nothing_was_sent(monkeypatch):
    """Signals that the filters rejected used to be tallied only when *some*
    other alert still went out."""
    # setup="closed" parks the Tap 1 two bars back, so the window holds nothing
    # but repeat taps, which `alerts.events` does not include
    patch_source(monkeypatch, setup="closed")
    cfg = scan_cfg(alert={"recent_bars": 1})
    with StateStore(":memory:") as store:
        rep = Scanner(cfg, store=store, dry_run=True).scan(live=True, progress=False)
    assert rep.alerts == []
    assert rep.events_in_window > 0 and rep.filtered == rep.events_in_window
    assert any(n.startswith("nothing to send — repeat tap") for n in rep.notes), rep.notes


def test_a_cycle_with_no_signal_at_all_says_so(monkeypatch):
    """The other quiet case: the indicator simply printed nothing.

    Before, this cycle produced no note whatsoever, so "a quiet session" and
    "the pipeline is dead" looked identical in the log.
    """
    import precision_tap.data as D
    from precision_tap.data import Bars, normalize_ohlcv

    df = synthetic_frame(n=400, seed=3)                      # no engineered tap at the end
    end = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
    df = df.set_axis(df.index + (end - df.index[-1]))

    monkeypatch.setattr(D.DataSource, "get_many",
                        lambda self, symbols, **kw: {
                            s: Bars(symbol=s, df=df, live=True,
                                    meta={"exchangeTimezoneName": "Asia/Kolkata"},
                                    source="fake") for s in symbols})
    with StateStore(":memory:") as store:
        rep = Scanner(scan_cfg(), store=store, dry_run=True).scan(live=True, progress=False)
    assert rep.usable > 0
    assert rep.events_in_window == 0
    assert any(n.startswith("nothing to send — no indicator signal") for n in rep.notes), rep.notes


def test_a_cycle_with_no_transport_says_it_did_not_send(monkeypatch):
    patch_source(monkeypatch)
    with StateStore(":memory:") as store:
        sc = Scanner(scan_cfg(), store=store, dry_run=True)
        rep = sc.scan(live=True, progress=False)
        sc.close()
    assert rep.alerts
    assert any(n.startswith("NOT SENT") for n in rep.notes), rep.notes


def test_the_live_loop_logs_the_delivery_outcome(monkeypatch, ok_telegram, caplog):
    """`run` used to log only the summary line — the same text for a cycle that
    delivered three taps and one that delivered none."""
    import logging

    api, _sent = ok_telegram
    patch_source(monkeypatch)
    cfg = scan_cfg(telegram=_tg(api))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store)
        loop = LiveLoop(cfg, sc, store=store, heartbeat=False)
        with caplog.at_level(logging.INFO, logger="precision_tap.live"):
            loop._scan("test")
        sc.close()
    assert any("delivery sent=3" in r.getMessage() for r in caplog.records), \
        [r.getMessage() for r in caplog.records]


# ─────────────────────────────────────────────────────────────────────────────
# `livecheck` — the one-command answer to "is it actually working?"
# ─────────────────────────────────────────────────────────────────────────────
def test_livecheck_passes_when_transport_feed_and_scanner_all_work(monkeypatch, ok_telegram,
                                                                  tmp_path, capsys):
    import precision_tap.data as D
    from precision_tap.cli import cmd_livecheck
    from precision_tap.data import Bars

    api, sent = ok_telegram
    patch_source(monkeypatch)

    class A:                                     # the argparse namespace cmd_livecheck reads
        config = None
        set = [f"telegram.api_base={api}", "telegram.bot_token=123:abc", "telegram.chat_ids=42",
               "telegram.min_seconds_between_messages=0", "data.provider=yfinance",
               "data.universe=RELIANCE.NS,TCS.NS,INFY.NS", "data.min_bars=90",
               f"data.cache_dir={tmp_path/'cache'}", f"out_dir={tmp_path/'out'}",
               "state_db=:memory:", "alerts.chart=false",
               "alerts.min_liquidity_dollar_volume=0", "alerts.min_price=0"]
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
        probe = 3
        no_message = False
        no_cache = True

    assert cmd_livecheck(A()) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out
    assert "sendMessage     ok" in out
    assert "delivered — check the chat" in out
    assert any("TAP 1" in s or "chat_id=42" in s for s in sent), "no alert reached the mock bot"


def test_livecheck_fails_loudly_without_credentials(monkeypatch, tmp_path, capsys):
    from precision_tap.cli import cmd_livecheck

    patch_source(monkeypatch)

    class A:
        config = None
        set = ["telegram.bot_token=", "telegram.chat_ids=", "data.provider=yfinance",
               "data.universe=RELIANCE.NS", "data.min_bars=90",
               f"data.cache_dir={tmp_path/'cache'}", f"out_dir={tmp_path/'out'}",
               "state_db=:memory:", "alerts.chart=false"]
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
        probe = 3
        no_message = False
        no_cache = True

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert cmd_livecheck(A()) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out
