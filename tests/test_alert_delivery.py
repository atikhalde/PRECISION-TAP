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


# ─────────────────────────────────────────────────────────────────────────────
# a cycle that found alerts with nothing to send them with must not be green
# ─────────────────────────────────────────────────────────────────────────────
def _scan_args(tmp_path, extra=(), **over):
    """The argparse namespace `cmd_scan` reads, wired to an offline config."""

    class A:
        config = None
        set = [f"--SET"]
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
        no_send = False
        report_only = False
        retried = False
        messages = False
        top = 5
        workers = 1

    a = A()
    a.set = ["data.provider=yfinance", "data.universe=RELIANCE.NS,TCS.NS,INFY.NS",
             "data.min_bars=90", f"data.cache_dir={tmp_path / 'cache'}",
             f"out_dir={tmp_path / 'out'}", "state_db=:memory:", "alerts.chart=false",
             # the synthetic feed has toy prices/volumes, so the production floors
             # would filter it — that is not what these tests are about
             "alerts.min_liquidity_dollar_volume=0", "alerts.min_price=0",
             "telegram.enabled=false", *extra]
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_scan_without_a_transport_exits_nonzero(tmp_path, monkeypatch, capsys):
    """The classic silent deployment: cron/systemd without the secrets in its env.

    Every alert is found, rendered and logged as log-only, exit 0 — so the cron
    log looks fine and the chat stays empty forever.  ``--no-send`` is an explicit
    dry run and must stay green; this is not.
    """
    from precision_tap.cli import cmd_scan

    patch_source(monkeypatch)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    rc = cmd_scan(_scan_args(tmp_path))
    err = capsys.readouterr().err
    assert rc == 4, (rc, err)
    assert "nothing could send them" in err, err
    assert "telegram.enabled" in err or "TELEGRAM_BOT_TOKEN" in err, err


def test_explicit_dry_runs_stay_green(tmp_path, monkeypatch, capsys):
    """`--no-send` and `--report-only` ask for silence, so silence is not a failure."""
    from precision_tap.cli import cmd_scan

    patch_source(monkeypatch)
    assert cmd_scan(_scan_args(tmp_path, no_send=True)) == 0
    assert cmd_scan(_scan_args(tmp_path, report_only=True)) == 0
    capsys.readouterr()


def test_no_transport_note_names_the_reason(monkeypatch):
    """"NOT SENT" has to say *why*, or the user is still guessing."""
    patch_source(monkeypatch)
    with StateStore(":memory:") as store:
        rep = Scanner(scan_cfg(), store=store, dry_run=True).scan(live=True, progress=False)
    assert rep.alerts, rep.summary_line
    note = [n for n in rep.notes if n.startswith("NOT SENT")]
    assert note, rep.notes
    assert "telegram.enabled is false" in note[0], note[0]

    tg_off = scan_cfg(telegram=TelegramConfig(enabled=True, bot_token="", chat_ids=[]))
    with StateStore(":memory:") as store:
        rep2 = Scanner(tg_off, store=store, dry_run=True).scan(live=True, progress=False)
    note2 = [n for n in rep2.notes if n.startswith("NOT SENT")]
    assert note2 and "TELEGRAM_BOT_TOKEN" in note2[0], rep2.notes


def test_a_queued_row_is_released_when_no_transport_can_drain_it():
    """A retry that can never run must not hold the day's de-duplication key.

    ``record_alert`` parks an alert as QUEUED (a delivery is owed) until it is
    marked delivered.  Without a transport the retry pass used to return early and
    leave the row there — and a queued row counts as "already alerted", so the
    same signal was lost for the rest of the day even after the bot came back.
    """
    from precision_tap.alerts import AlertDispatcher
    from precision_tap.params import AlertConfig, Params

    with StateStore(":memory:") as store:
        assert store.record_alert("K1", symbol="RELIANCE.NS", event="tap1") is True
        assert store.seen("K1") is True, "a delivery is owed on a fresh row"
        dead = AlertDispatcher(AlertConfig(), None, store, Params(), dry_run=True)
        assert dead.retry_pending() == 0
        assert store.seen("K1") is False, "no transport → free the key, do not hold it"


def test_a_queued_row_is_retried_when_the_transport_works(ok_telegram):
    from precision_tap.alerts import AlertDispatcher
    from precision_tap.params import AlertConfig, Params

    api, sent = ok_telegram
    with StateStore(":memory:") as store:
        store.record_alert("K1", symbol="RELIANCE.NS", event="tap1", message="🎯 tap")
        live = AlertDispatcher(AlertConfig(), TelegramClient(_tg(api)), store, Params())
        assert live.retry_pending() == 1
        assert store.seen("K1") is True, "it is delivered now, so it must count as seen"
        assert sent and "tap" in sent[0]


def test_env_file_is_read_from_the_documented_paths(tmp_path, monkeypatch):
    """`/etc/precision-tap.env` is where the deploy docs put the secrets.

    A cron entry has no shell environment, so a config whose token is
    ``${TELEGRAM_BOT_TOKEN}`` expanded to nothing, every alert was logged instead
    of sent, and the run stayed green.  Both ``PRECISION_TAP_ENV_FILE`` and the
    documented path now close that hole.
    """
    from precision_tap import config as C

    env = tmp_path / "precision-tap.env"
    env.write_text("TELEGRAM_BOT_TOKEN=42:abc\nTELEGRAM_CHAT_ID=-100999\n", encoding="utf-8")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    monkeypatch.setenv("PRECISION_TAP_ENV_FILE", str(env))
    cfg = C.load_config(None, env_path=None)
    assert cfg.telegram.bot_token == "42:abc" and cfg.telegram.chat_ids == ["-100999"], \
        "the override must feed ${TELEGRAM_BOT_TOKEN} expansion"

    monkeypatch.delenv("PRECISION_TAP_ENV_FILE")
    monkeypatch.setattr(C, "ENV_FILE_CANDIDATES", (tmp_path / "absent.env", env))
    cfg2 = C.load_config(None, env_path=None)
    assert cfg2.telegram.bot_token == "42:abc", "the candidate list must be walked in order"
    assert "/etc/precision-tap.env" in (".env", "config/.env", "/etc/precision-tap.env")


def test_missing_env_file_is_reported(tmp_path, monkeypatch, caplog):
    """A typo in the env path must be visible, not a silent drop to log-only."""
    import logging

    from precision_tap import config as C

    monkeypatch.setenv("PRECISION_TAP_ENV_FILE", str(tmp_path / "nope.env"))
    with caplog.at_level(logging.WARNING, logger="precision_tap.config"):
        C.load_dotenv()
    assert any("nope.env" in r.getMessage() for r in caplog.records), caplog.text


def test_env_file_does_not_clobber_the_real_environment(tmp_path, monkeypatch):
    """The new env-file lookup must keep ``load_dotenv``'s non-clobbering rule.

    Secrets in the process environment win over the file unless ``override`` is
    asked for — and reading ``PRECISION_TAP_ENV_FILE`` inside the loader is one
    shadowed name away from silently reversing that.
    """
    import os

    from precision_tap import config as C

    env = tmp_path / "precision-tap.env"
    env.write_text("TELEGRAM_BOT_TOKEN=from-file\n", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-shell")
    monkeypatch.setenv("PRECISION_TAP_ENV_FILE", str(env))
    assert C.load_dotenv() == []
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "from-shell"
    assert C.load_dotenv(override=True) == ["TELEGRAM_BOT_TOKEN"]
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "from-file"


# ─────────────────────────────────────────────────────────────────────────────
# a silent cycle must name the session it evaluated
#
# Two runs were dispatched on a Saturday, both went green, and the chat stayed
# empty.  Nothing was broken: `alerts.recent_bars: 1` means a cycle evaluates
# exactly one bar, and on a non-trading day that bar is the *previous* session —
# the same bar every cycle since the close has already evaluated.  But the cycle
# said only "no indicator signal … a quiet session", so the two runs looked like
# two independent attempts the scanner had silently failed.  These pin the
# cycle's self-description.
# ─────────────────────────────────────────────────────────────────────────────
TZ = "Asia/Kolkata"


def _quiet_cycle(monkeypatch, *, now, bar, live=None, **alert_kw):
    """One cycle with the exchange clock pinned to ``now`` and a feed whose
    newest bar is ``bar`` — no network, no Telegram, no engineered signal."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import precision_tap.data as D
    import precision_tap.scanner as SC
    from precision_tap.data import Bars

    pinned = now if now.tzinfo else datetime(*now.timetuple()[:6], tzinfo=ZoneInfo(TZ))

    def _pinned(tz: str = TZ):
        return pinned if tz == TZ else pinned.astimezone(ZoneInfo(tz))

    monkeypatch.setattr(D, "_now_tz", _pinned)
    monkeypatch.setattr(SC, "_session_now", _pinned)

    df = synthetic_frame(n=400, seed=3)                  # nothing engineered at the end
    df = df.set_axis(df.index + (pd.Timestamp(bar) - df.index[-1]))
    monkeypatch.setattr(
        D.DataSource, "get_many",
        lambda self, symbols, **kw: {
            s: Bars(symbol=s, df=df, live=False, source="fake",
                    meta={"currency": "INR", "fullExchangeName": "NSE",
                          "exchangeTimezoneName": TZ}) for s in symbols})
    # `scan_cfg` already zeroes the liquidity floor and disables charts
    cfg = scan_cfg(alert=dict(alert_kw))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=True)
        rep = sc.scan(live=live, progress=False)
        sc.close()
    assert rep.usable > 0, rep.notes                     # the cycle really did evaluate
    return rep


def test_a_weekend_cycle_names_the_bar_it_evaluated(monkeypatch):
    """Sat 14:18 IST, feed newest = Fri: green, silent, and *expected*."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 12, 14, 18, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-11"))
    assert rep.session_now == "2026-09-12"
    assert rep.bar_session == "2026-09-11"
    assert rep.expected_session == "2026-09-11"          # the feed is NOT behind
    assert rep.market_state == "closed" and rep.trading_day is False
    note = [n for n in rep.notes if n.startswith("MARKET CLOSED")]
    assert note, rep.notes
    assert "2026-09-11" in note[0] and "recent_bars" in note[0]
    assert "--recent-bars" in note[0]                    # …and how to review the week instead
    # the report the workflow summarises carries it too
    d = rep.to_dict()
    assert d["bar_session"] == "2026-09-11" and d["market_state"] == "closed"
    assert "bar=2026-09-11" in rep.summary_line


def test_a_settled_trading_day_does_not_claim_the_market_is_closed(monkeypatch):
    """Fri 20:00 IST with Friday's settled bar: a normal quiet EOD cycle."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 11, 20, 0, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-11"))
    assert rep.bar_session == rep.session_now == "2026-09-11"
    assert rep.trading_day is True
    assert not any(n.startswith(("MARKET CLOSED", "NO FRESH SESSION", "FEED BEHIND"))
                   for n in rep.notes), rep.notes


def test_a_closed_bar_cycle_mid_session_names_the_previous_session(monkeypatch):
    """A forced ``--eod`` at 10:00 reads yesterday — and must say so instead of
    implying today's session was scanned and found wanting."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 11, 10, 0, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-10"), live=False)
    assert rep.market_state == "open"
    note = [n for n in rep.notes if n.startswith("NO FRESH SESSION")]
    assert note, rep.notes
    assert "2026-09-10" in note[0]
    assert not any(n.startswith("MARKET CLOSED") for n in rep.notes), rep.notes


def test_a_feed_behind_the_session_it_should_have_is_named(monkeypatch):
    """Saturday, but the newest bar is Wednesday: that is a lagging feed (or a
    holiday), not a quiet market, and it must not be reported as one."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 12, 14, 18, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-09"))
    assert rep.expected_session == "2026-09-11"
    assert rep.bar_session < rep.expected_session
    note = [n for n in rep.notes if n.startswith("FEED BEHIND")]
    assert note, rep.notes
    assert "2026-09-09" in note[0] and "2026-09-11" in note[0]
    assert not any(n.startswith("MARKET CLOSED") for n in rep.notes), rep.notes


def test_the_quiet_note_names_the_bar_it_looked_at(monkeypatch):
    """`recent_bars: 1` makes "which bar?" the whole question — so the
    "nothing to send" note carries the date instead of just "a quiet session"."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 12, 14, 18, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-11"))
    assert rep.events_in_window == 0
    note = [n for n in rep.notes if n.startswith("nothing to send — no indicator signal")]
    assert note, rep.notes
    assert "newest 2026-09-11" in note[0]
    assert "quiet session" not in note[0]                # there was no session at all


def test_a_widened_window_says_how_many_bars_it_covered(monkeypatch):
    """The same note must stay honest when the window is widened."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 12, 14, 18, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-11"), recent_bars=3)
    note = [n for n in rep.notes if n.startswith("MARKET CLOSED")]
    assert note and "recent_bars=3" in note[0], rep.notes


def test_the_cli_says_which_session_a_silent_cycle_evaluated(monkeypatch, tmp_path, capsys):
    """The console is what a human reads first: `scan` on a Saturday must name
    the bar it evaluated instead of only printing "alerts matched: 0"."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import precision_tap.data as D
    import precision_tap.scanner as SC
    from precision_tap.cli import cmd_scan
    from precision_tap.data import Bars

    sat = datetime(2026, 9, 12, 14, 18, tzinfo=ZoneInfo(TZ))

    def _pinned(tz: str = TZ):
        return sat if tz == TZ else sat.astimezone(ZoneInfo(tz))

    monkeypatch.setattr(D, "_now_tz", _pinned)
    monkeypatch.setattr(SC, "_session_now", _pinned)

    df = synthetic_frame(n=400, seed=3)
    df = df.set_axis(df.index + (pd.Timestamp("2026-09-11") - df.index[-1]))
    monkeypatch.setattr(D.DataSource, "get_many",
                        lambda self, symbols, **kw: {
                            s: Bars(symbol=s, df=df, live=False, source="fake")
                            for s in symbols})

    class A:
        config = None
        set = ["data.provider=yfinance", "data.universe=RELIANCE.NS,TCS.NS",
               "data.min_bars=90", f"data.cache_dir={tmp_path / 'cache'}",
               f"out_dir={tmp_path / 'out'}", "state_db=:memory:", "alerts.chart=false",
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
        eod = False
        end = None
        no_send = True
        report_only = False
        retried = False
        messages = False
        top = 5
        workers = 1

    assert cmd_scan(A()) == 0                            # a closed market is not a failure
    out = capsys.readouterr().out
    assert "bar=2026-09-11" in out                       # the summary line carries the bar
    assert "session: evaluated 2026-09-11 · today is 2026-09-12" in out
    assert "not a trading day" in out
    assert "MARKET CLOSED" in out


def test_a_frozen_replay_does_not_comment_on_the_exchange_clock(monkeypatch, tmp_path):
    """``--provider csv`` / a historical ``--end`` is frozen by definition, so
    "today" means nothing to it: it must not be told the market is closed."""
    import precision_tap.data as D
    from precision_tap.data import Bars
    from precision_tap.params import DataConfig

    df = synthetic_frame(n=400, seed=3)
    monkeypatch.setattr(D.DataSource, "get_many",
                        lambda self, symbols, **kw: {
                            s: Bars(symbol=s, df=df, live=False, source="csv:x.csv")
                            for s in symbols})
    cfg = scan_cfg()
    cfg.data = DataConfig(provider="csv", universe=list(cfg.data.universe), min_bars=90,
                          lookback_days=600, cache_dir=str(tmp_path / "cache"))
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=True)
        rep = sc.scan(live=False, progress=False)
        sc.close()
    assert rep.usable > 0, rep.notes
    assert rep.market_state == "" and rep.expected_session == ""
    assert rep.bar_session                       # still reported, for the step summary
    assert not any(n.startswith(("MARKET CLOSED", "NO FRESH SESSION", "FEED BEHIND"))
                   for n in rep.notes), rep.notes


def test_a_weekday_holiday_reports_the_feed_as_behind(monkeypatch):
    """Mon 2026-09-14 is Ganesh Chaturthi — a weekday the exchange is shut.

    The scanner has no holiday calendar (``live.trading_days`` only knows
    weekends), so the honest report is "the feed is behind the session it should
    already have, and a holiday explains it" — not "market closed", and not
    silence.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    rep = _quiet_cycle(monkeypatch, now=datetime(2026, 9, 14, 20, 0, tzinfo=ZoneInfo(TZ)),
                       bar=pd.Timestamp("2026-09-11"))
    assert rep.trading_day is True
    assert rep.session_now == "2026-09-14"
    assert rep.expected_session == "2026-09-14"
    note = [n for n in rep.notes if n.startswith("FEED BEHIND")]
    assert note, rep.notes
    assert "holiday" in note[0] and "2026-09-11" in note[0]
    assert not any(n.startswith("MARKET CLOSED") for n in rep.notes), rep.notes


def test_doctor_flags_the_sample_credentials_as_unconfigured(tmp_path, capsys):
    """`init` + `doctor` must not report a repo ready to deliver when it is not.

    `init` copies `.env.example` (sample token included), so a fresh checkout has
    a *well-formed* token: the pre-fix doctor said "all checks passed" and the
    first sign of trouble was a 401 on the first alert, days later.
    """
    from precision_tap.cli import cmd_doctor
    from precision_tap.telegram import SAMPLE_BOT_TOKEN, SAMPLE_CHAT_ID

    class A:
        config = None
        set = [f"telegram.bot_token={SAMPLE_BOT_TOKEN}", f"telegram.chat_ids={SAMPLE_CHAT_ID}",
               f"state_db={tmp_path/'state.sqlite3'}"]
        env_file = None
        verbose = quiet = False
        net = False
        no_cache = True

    rc = cmd_doctor(A())
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "sample" in out and "some checks need attention" in out

    # ... and a real token clears it (the check is about the sample, not about
    # reaching the network — doctor without --net never sends anything)
    A.set = ["telegram.bot_token=999:realtoken", "telegram.chat_ids=12345",
             f"state_db={tmp_path/'state.sqlite3'}"]
    assert cmd_doctor(A()) == 0
