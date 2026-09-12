#!/usr/bin/env python3
"""Offline rehearsal of the *live* pipeline: fake Yahoo feed + mock Telegram.

CI (and this sandbox) has no route to Yahoo, but the live path is exactly the
code that has to work while the market is open, so this drills it end to end:

    fake Yahoo /v8/finance/chart  ->  DataSource (yahoo | yfinance, live=True)
                                  ->  intraday rebuild of today's forming bar
                                  ->  Scanner (live mode + closed-bar parity pass)
                                  ->  AlertDispatcher -> Telegram (mock API)

Nothing here touches the network.  Run it with:

    python tools/live_drill.py                      # market-open simulation
    python tools/live_drill.py --mode eod           # after the close
    python tools/live_drill.py --provider yfinance  # the yfinance code path
    python tools/live_drill.py --cycles 2           # + assert no re-send

It exits non-zero unless every alert it found reached Telegram, so it works as a
CI smoke test (see .github/workflows/ci.yml and tests/test_live_drill.py).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import threading
import zlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import precision_tap.data as pt_data                      # noqa: E402
import precision_tap.scanner as pt_scanner                 # noqa: E402
from precision_tap.config import load_config               # noqa: E402
from precision_tap.data import DataSource, synthetic_frame  # noqa: E402
from precision_tap.live import LiveLoop                    # noqa: E402
from precision_tap.scanner import Scanner                  # noqa: E402
from precision_tap.state import StateStore                 # noqa: E402
from precision_tap.telegram import TelegramClient          # noqa: E402

TZ = "Asia/Kolkata"
DEFAULT_CONFIG = ROOT / "config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# a deterministic synthetic NSE "market"
# ─────────────────────────────────────────────────────────────────────────────
def last_session() -> pd.Timestamp:
    """Today's date in exchange time, rolled off weekends."""
    day = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day


def market_frame(symbol: str, n: int = 520) -> pd.DataFrame:
    """Daily history for ``symbol`` whose newest bar is today's session.

    ``synthetic_frame(..., tap_on_last_bar=True)`` plants the displacement →
    OB → Tap-1 sequence on the final bars, so a live cycle has something real
    to find.  The seed is a CRC32 (``hash()`` is salted per process, which
    would make the drill non-deterministic).
    """
    seed = zlib.crc32(symbol.encode()) % 9973
    df = synthetic_frame(n=n, seed=seed, tap_on_last_bar=True, price0=100.0 + (seed % 40) * 10)
    return df.set_axis(df.index + (last_session() - df.index[-1]))


#: 5m prints in one session (09:15 → 15:30 at 5 minutes = 76 prints)
PRINTS = 76
#: how many of them the *daily* (delayed) series has caught up to (~12:45 IST)
PARTIAL_PRINTS = 40


def split_market(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the newest (still forming) bar into closed history + 5m prints.

    The path is engineered so the session low — the OB tap — prints *after*
    ``PARTIAL_PRINTS``: a cycle that skips the intraday rebuild is working from
    a bar that has not touched the level yet.
    """
    daily = df.iloc[:-1]
    row = df.iloc[-1]
    sess = pd.Timestamp(df.index[-1])
    idx = pd.date_range(sess + pd.Timedelta(hours=9, minutes=15), periods=PRINTS, freq="5min")
    up = np.linspace(row["open"], row["high"], 20)
    down = np.linspace(row["high"], row["low"], PRINTS - 20 - 25)
    back = np.linspace(row["low"], row["close"], 25)
    path = np.concatenate([up, down, back])
    o = np.concatenate([[row["open"]], path[:-1]])
    c = path
    intra = pd.DataFrame(
        {"open": o, "high": np.maximum(o, c) * 1.0004, "low": np.minimum(o, c) * 0.9996,
         "close": c, "volume": np.full(len(idx), row["volume"] / len(idx))}, index=idx)
    return daily, intra


def intraday_to_bar(intra: pd.DataFrame) -> pd.DataFrame:
    """Collapse 5m prints into one (partial) daily bar."""
    return pd.DataFrame(
        {"open": [intra["open"].iloc[0]], "high": [intra["high"].max()],
         "low": [intra["low"].min()], "close": [intra["close"].iloc[-1]],
         "volume": [intra["volume"].sum()]},
        index=pd.DatetimeIndex([intra.index[0].normalize()]))


class FakeYahoo:
    """Serves the synthetic market in Yahoo's ``v8/finance/chart`` envelope."""

    def __init__(self, symbols: List[str]):
        self.calls: List[Dict[str, Any]] = []
        self._frames = {s: market_frame(s) for s in symbols}

    @staticmethod
    def _envelope(df: pd.DataFrame, symbol: str, tzname: str = TZ) -> Dict[str, Any]:
        ts = [int(pd.Timestamp(t).tz_localize(tzname).timestamp()) for t in df.index]
        return {
            "chart": {
                "error": None,
                "result": [{
                    "meta": {"currency": "INR", "symbol": symbol, "exchangeName": "NSI",
                             "fullExchangeName": "NSE", "exchangeTimezoneName": tzname,
                             "regularMarketPrice": float(df["close"].iloc[-1])},
                    "timestamp": ts,
                    "indicators": {
                        "quote": [{"open": df["open"].tolist(), "high": df["high"].tolist(),
                                   "low": df["low"].tolist(), "close": df["close"].tolist(),
                                   "volume": df["volume"].tolist()}],
                        "adjclose": [{"adjclose": df["close"].tolist()}],
                    },
                }],
            }
        }

    def json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        symbol = urlparse(str(url)).path.rsplit("/", 1)[-1]
        self.calls.append({"symbol": symbol, **params})
        if symbol not in self._frames:
            self._frames[symbol] = market_frame(symbol)
        df = self._frames[symbol]
        interval = str(params.get("interval") or "1d")
        if interval == "1d":
            daily, intra = split_market(df)
            # the delayed daily series only carries today's bar *so far*
            return self._envelope(pd.concat([daily, intraday_to_bar(intra.iloc[:PARTIAL_PRINTS])]),
                                  symbol)
        if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
            _, intra = split_market(df)
            return self._envelope(intra, symbol)
        raise RuntimeError(f"unexpected interval {interval!r}")


# ─────────────────────────────────────────────────────────────────────────────
# provider doubles
# ─────────────────────────────────────────────────────────────────────────────
class FakeYFinance:
    """A stand-in ``yfinance`` module whose ``history()`` is as strict as 1.x.

    yfinance >= 1.0 forwards ``**kwargs`` to ``PriceHistory.history``, which no
    longer accepts ``progress`` / ``threads`` — so a call carrying them raises
    ``TypeError`` for every symbol.  This fake reproduces that exactly, which is
    the only way to catch the regression without Yahoo access.
    """

    def __init__(self, market: FakeYahoo):
        self.market = market

    def install(self) -> None:
        import types

        mod = types.ModuleType("yfinance")
        mod.__version__ = "1.7.0"
        outer = self

        class FastInfo(dict):
            pass

        class Ticker:
            def __init__(self, symbol):
                self.symbol = symbol
                self.fast_info = FastInfo(currency="INR", exchange="NSE")

            @property
            def tz(self):                       # yfinance >= 1.0 has no `Ticker.tz`
                raise AttributeError("tz")

            def history(self, period=None, interval="1d", start=None, end=None,
                        prepost=False, actions=True, auto_adjust=True, back_adjust=False,
                        repair=False, keepna=False, rounding=False, timeout=10,
                        raise_errors=False):
                if self.symbol not in outer.market._frames:
                    outer.market._frames[self.symbol] = market_frame(self.symbol)
                df = outer.market._frames[self.symbol]
                if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
                    _, intra = split_market(df)
                    return intra.rename(columns=str.capitalize)
                daily, intra = split_market(df)
                partial = intraday_to_bar(intra.iloc[:PARTIAL_PRINTS])
                return pd.concat([daily, partial]).tz_localize(TZ).rename(columns=str.capitalize)

        mod.Ticker = Ticker
        sys.modules["yfinance"] = mod


@contextlib.contextmanager
def patched(market: Optional[FakeYahoo], provider: str):
    """Point the providers at the fake market, and put everything back after.

    The market is fake but the calendar is not, so the exchange clock gets
    pinned here too: ``data._now_tz`` / ``scanner._session_now`` still ask the
    host clock whether a bar is from *today*, and on a weekend (or the hours
    after midnight IST) the host date has moved past ``last_session()`` — the
    live bar then looks stale and a correct scanner refuses to alert on it,
    which used to make this drill fail purely because of when CI ran.
    """
    saved_json = DataSource._http_json
    saved_yf = sys.modules.get("yfinance")
    now = (last_session() + pd.Timedelta(hours=11, minutes=30)).tz_localize(TZ).to_pydatetime()
    saved_now = (pt_data._now_tz, pt_scanner._session_now)
    pt_data._now_tz = lambda _tzname=TZ: now
    pt_scanner._session_now = lambda _tzname=TZ: now
    try:
        if market is not None:
            DataSource._http_json = lambda self, url, params: market.json(url, params)  # type: ignore[method-assign]
        if provider == "yfinance":
            FakeYFinance(market).install()
        yield
    finally:
        DataSource._http_json = saved_json                       # type: ignore[method-assign]
        pt_data._now_tz, pt_scanner._session_now = saved_now
        if saved_yf is not None:
            sys.modules["yfinance"] = saved_yf
        else:
            sys.modules.pop("yfinance", None)


# ─────────────────────────────────────────────────────────────────────────────
# mock Telegram (same contract as tools/mock_telegram_server.py, in-process)
# ─────────────────────────────────────────────────────────────────────────────
class MockTelegram:
    sent: List[Dict[str, Any]] = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n).decode("utf-8", "ignore")
            MockTelegram.sent.append({"method": self.path.rsplit("/", 1)[-1], "raw": raw})
            body = json.dumps({"ok": True, "result": {"message_id": len(MockTelegram.sent)}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    @classmethod
    def start(cls) -> str:
        cls.sent = []
        srv = HTTPServer(("127.0.0.1", 0), cls.H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{srv.server_address[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# the drill
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="offline rehearsal of the live pipeline")
    ap.add_argument("--mode", choices=("live", "eod"), default="live")
    ap.add_argument("--provider", default="yahoo", choices=("yahoo", "yfinance", "csv"))
    ap.add_argument("--symbols", default="RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cycles", type=int, default=1,
                    help="run N cycles against one store (>1 asserts de-duplication)")
    ap.add_argument("--out", default="results_drill")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra dotted config override, e.g. --set data.live_intraday_bar=false")
    a = ap.parse_args(argv)

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    api = MockTelegram.start()
    out = Path(a.out)
    cache = out / "cache"
    if cache.exists():                      # a fresh cache: the market is rebuilt each run
        for f in cache.glob("*.csv"):
            f.unlink()

    cfg = load_config(a.config, overrides=[
        f"data.provider={a.provider}",
        "data.universe=" + ",".join(symbols),
        "alerts.min_liquidity_dollar_volume=0",
        "alerts.min_price=0",
        "alerts.chart=false",
        "telegram.bot_token=123:localmock",
        "telegram.chat_ids=42",
        f"telegram.api_base={api}",
        "telegram.min_seconds_between_messages=0",
        f"data.cache_dir={cache}",
        "state_db=:memory:",
        f"out_dir={out}",
    ] + list(a.set))

    live = (a.mode == "live")
    print(f"\n=== drill: provider={a.provider} mode={a.mode} "
          f"market={'OPEN' if live else 'CLOSED'} cycles={a.cycles} ===")
    alerts = 0
    reports = []
    with patched(FakeYahoo(symbols), a.provider), StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, telegram=TelegramClient(cfg.telegram), dry_run=False)
        for cycle in range(1, max(1, a.cycles) + 1):
            rep = sc.scan(live=live, progress=False)
            reports.append(rep)
            print(f"cycle {cycle}: {rep.summary_line}")
            if rep.dispatch is not None:
                print(f"          delivery {rep.dispatch}")
            if cycle == 1:
                alerts = len(rep.alerts)
                if not a.quiet:
                    print(rep.console_table(10))
                    for note in rep.notes:
                        print("  note:", note)
            sc.dispatcher.retry_pending()
        # The scheduler entry point (`run --once`) picks its mode from the
        # exchange clock — it must follow the *market*, never the trigger, so a
        # mid-session cron cycle can see an intrabar tap.
        from precision_tap.scanner import _market_open
        real_open = _market_open(cfg.live.market_timezone,
                                 session=(cfg.live.session_open, cfg.live.session_close))
        loop = LiveLoop(cfg, sc, store=store)
        calls: List[bool] = []
        real_scan = sc.scan

        def spy(*, live=False, progress=False, **kw):
            calls.append(live)
            return real_scan(live=live, progress=progress, **kw)

        sc.scan = spy                                            # type: ignore[assignment]
        loop._scan("once")
        sc.scan = real_scan                                      # type: ignore[assignment]
        print(f"          scheduler scanned with live={calls} "
              f"(market {'OPEN' if real_open else 'CLOSED'} now)")
        sc.close()

    delivered = [m for m in MockTelegram.sent if m["method"] in ("sendMessage", "sendPhoto")]
    print(f"\ntelegram messages delivered: {len(delivered)}")
    if not a.quiet:
        for m in delivered[:3]:
            print("  ·", m["raw"][:150].replace("%0A", " "))
    problems = []
    if alerts == 0:
        problems.append("no alert was produced by the live cycle")
    if len(delivered) < alerts:
        problems.append(f"{alerts} alert(s) found but only {len(delivered)} reached Telegram")
    if len(delivered) > alerts and a.cycles > 1:
        problems.append(f"re-sent across cycles: {len(delivered)} messages for {alerts} alert(s)")
    if not calls or calls != [real_open]:
        problems.append(f"scheduler mode {calls} != market state (live={real_open})")
    print("\nRESULT:", "PASS" if not problems else "FAIL — " + "; ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
