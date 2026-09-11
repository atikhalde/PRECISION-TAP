"""The two ways the live feed used to die, plus the intraday rebuild contract.

Both regressions were invisible offline: the test-suite drives the scanner with
:func:`DataSource.get` monkeypatched, so nothing exercised the provider code
that actually runs against Yahoo.  Every failure mode looked identical from the
CLI — ``usable=0 errors=N`` and a quiet day with no alerts.

* ``yfinance >= 1.0`` rejects ``progress``/``threads`` on ``history()``, and
  ``Ticker.history`` still forwards ``**kwargs``, so the call raised
  ``TypeError`` for *every* symbol.
* the intraday rebuild of today's bar concatenated a **tz-naive** stamp onto a
  **tz-aware** daily index, which modern pandas refuses to sort — again killing
  every symbol, but only while the session was open.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from precision_tap.data import DataSource, merge_today_bar, synthetic_frame
from precision_tap.data import _now_tz
from precision_tap.params import DataConfig

TZ = "Asia/Kolkata"


def _last_session() -> pd.Timestamp:
    d = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _market(n: int = 400, seed: int = 11) -> pd.DataFrame:
    df = synthetic_frame(n=n, seed=seed)
    return df.set_axis(df.index + (_last_session() - df.index[-1]))


def _prints(df: pd.DataFrame, tz: str = TZ) -> pd.DataFrame:
    """Today's session as 5m prints (09:15 → 10:00 local)."""
    row = df.iloc[-1]
    sess = pd.Timestamp(df.index[-1])
    idx = pd.date_range(sess + pd.Timedelta(hours=9, minutes=15), periods=10, freq="5min", tz=tz)
    c = np.linspace(row["open"], row["close"], len(idx))
    o = np.concatenate([[row["open"]], c[:-1]])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001,
                         "low": np.minimum(o, c) * 0.999, "close": c,
                         "volume": np.full(len(idx), 1_000.0)}, index=idx)


def _envelope(df: pd.DataFrame, symbol: str, tzname: str = TZ) -> dict:
    """A Yahoo ``v8/finance/chart`` payload for ``df``."""
    ts = []
    for t in df.index:
        ts_i = pd.Timestamp(t)
        ts.append(int((ts_i.tz_convert(tzname) if ts_i.tz is not None else ts_i.tz_localize(tzname))
                      .timestamp()))
    return {"chart": {"error": None, "result": [{
        "meta": {"currency": "INR", "symbol": symbol, "fullExchangeName": "NSE",
                 "exchangeTimezoneName": tzname},
        "timestamp": ts,
        "indicators": {"quote": [{"open": df["open"].tolist(), "high": df["high"].tolist(),
                                  "low": df["low"].tolist(), "close": df["close"].tolist(),
                                  "volume": df["volume"].tolist()}],
                       "adjclose": [{"adjclose": (df["close"] * 0.5).tolist()}]}}]}}


# ─────────────────────────────────────────────────────────────────────────────
# the intraday rebuild
# ─────────────────────────────────────────────────────────────────────────────
def test_merge_appends_today_when_the_daily_series_has_not_caught_up():
    daily, intra = _market().iloc[:-1], _prints(_market())
    out = merge_today_bar(daily, intra, TZ)
    assert out is not None
    assert len(out) == len(daily) + 1
    assert pd.Timestamp(out.index[-1]).date() == _last_session().date()
    assert np.isclose(out["low"].iloc[-1], intra["low"].min())
    assert np.isclose(out["high"].iloc[-1], intra["high"].max())
    assert np.isclose(out["volume"].iloc[-1], intra["volume"].sum())


def test_merge_replaces_the_partial_bar_the_daily_series_already_carries():
    df = _market()
    daily, intra = df.iloc[:-1], _prints(df)
    partial = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
                            "volume": [1.0]}, index=[df.index[-1]])
    out = merge_today_bar(pd.concat([daily, partial]), intra, TZ)
    assert out is not None
    assert len(out) == len(daily) + 1, "today's partial bar must be replaced, not duplicated"
    assert out["close"].iloc[-1] > 1.0


@pytest.mark.parametrize("daily_tz,intra_tz", [("UTC", "UTC"), (TZ, "UTC"), ("UTC", TZ),
                                               (TZ, TZ), (None, TZ), (TZ, None)])
def test_merge_never_mixes_tz_aware_and_naive_indices(daily_tz, intra_tz):
    """pandas >= 2 refuses to sort a mixed index: the whole fetch used to die."""
    df = _market()
    daily = df.iloc[:-1].copy()
    intra = _prints(df, tz=intra_tz or TZ)
    if daily_tz:
        daily.index = daily.index.tz_localize(daily_tz)
    else:
        daily.index = pd.DatetimeIndex(daily.index)
    if intra_tz is None:
        intra.index = intra.index.tz_localize(None)
    out = merge_today_bar(daily, intra, TZ)
    assert out is not None and len(out) == len(daily) + 1
    assert pd.Timestamp(out.index[-1]).date() == _last_session().date()


def test_session_date_comes_from_the_exchange_clock_not_the_stamp():
    """A UTC-stamped feed: 00:00 IST is 18:30 UTC on the *previous* day.

    Reading ``.date`` off the raw index would look for the wrong session and
    leave yesterday's bar in place while appending a duplicate of today's.
    """
    df = _market()
    daily = df.iloc[:-1].copy()
    # Yahoo daily stamps are the session start in exchange time -> 18:30 UTC
    daily.index = (daily.index + pd.Timedelta(hours=9, minutes=30)).tz_localize("UTC")
    assert daily.index[-1].date() < _last_session().date()      # the trap
    intra = _prints(df, tz="UTC")
    out = merge_today_bar(daily, intra, TZ)
    assert out is not None
    assert len(out) == len(daily) + 1
    assert pd.Timestamp(out.index[-1]).tz_convert(TZ).date() == _last_session().date()


def test_merge_returns_none_when_the_prints_are_not_todays_session():
    """A stale intraday feed must not clobber a settled bar with a partial one."""
    daily = _market().iloc[:-1]
    today = _now_tz(TZ).date()
    stale = _prints(_market()).iloc[:2]
    stale.index = stale.index - pd.Timedelta(days=3)
    # with an expectation: refuse, so a settled bar is never clobbered
    assert merge_today_bar(daily, stale, TZ, session=today) is None
    assert merge_today_bar(daily, pd.DataFrame(), TZ, session=today) is None
    # ...and today's prints are accepted even though the daily lags behind
    assert merge_today_bar(daily, _prints(_market()), TZ, session=today) is not None
    # without one: the newest session present is rebuilt (caller's choice)
    assert merge_today_bar(daily, stale, TZ) is not None


# ─────────────────────────────────────────────────────────────────────────────
# the yahoo provider, live
# ─────────────────────────────────────────────────────────────────────────────
def test_yahoo_live_fetch_rebuilds_todays_bar(tmp_path, monkeypatch):
    """End to end through ``_yahoo``: tz-aware payload → live Bars, no crash."""
    df = _market()

    def fake_json(self, url, params):
        interval = str(params.get("interval") or "1d")
        if interval == "1d":
            return _envelope(pd.concat([df.iloc[:-1], df.iloc[-1:]]), "RELIANCE.NS")
        return _envelope(_prints(df), "RELIANCE.NS")

    monkeypatch.setattr(DataSource, "_http_json", fake_json)
    cfg = DataConfig(provider="yahoo", min_bars=50, lookback_days=400, interval="1d",
                     live_intraday_bar=True, intraday_interval="5m",
                     cache_dir=str(tmp_path / "cache"), cache_max_age_minutes=0)
    bars = DataSource(cfg, live=True).get("RELIANCE.NS", use_cache=False)
    assert bars.ok, bars.error
    assert bars.source == "yahoo+intraday"
    assert bars.live is True
    assert pd.Timestamp(bars.df.index[-1]).date() == _last_session().date()


def test_yahoo_honours_corporate_adjustments(tmp_path, monkeypatch):
    """The endpoint always ships adjclose, so the flag has to be applied here."""
    df = _market()

    def fake_json(self, url, params):
        return _envelope(df, "RELIANCE.NS")

    monkeypatch.setattr(DataSource, "_http_json", fake_json)
    out = {}
    for flag in (True, False):
        cfg = DataConfig(provider="yahoo", min_bars=50, lookback_days=400, interval="1d",
                         live_intraday_bar=False, corporate_adjustments=flag,
                         cache_dir=str(tmp_path / f"c{flag}"), cache_max_age_minutes=0)
        bars = DataSource(cfg, live=False).get("RELIANCE.NS", use_cache=False)
        assert bars.ok, bars.error
        out[flag] = float(bars.df["close"].iloc[-1])
    assert out[True] == pytest.approx(out[False] * 0.5)     # adjclose == close/2
    assert out[False] == pytest.approx(float(df["close"].iloc[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# the yfinance provider, live
# ─────────────────────────────────────────────────────────────────────────────
def test_yfinance_survives_a_build_that_dropped_progress_and_threads(tmp_path):
    """yfinance >= 1.0: ``PriceHistory.history`` rejects both kwargs.

    ``Ticker.history`` still forwards ``**kwargs``, so the TypeError surfaces
    per symbol — the scanner reported "no data" for the whole universe.
    """
    df = _market()

    class StrictTicker:
        """``history()`` with the 1.x signature — no ``**kwargs`` at the end."""

        def __init__(self, symbol):
            self.symbol = symbol
            self.fast_info = {"currency": "INR", "exchange": "NSE"}

        def history(self, period=None, interval="1d", start=None, end=None, prepost=False,
                    actions=True, auto_adjust=True, back_adjust=False, repair=False,
                    keepna=False, rounding=False, timeout=10, raise_errors=False):
            if interval in ("1m", "2m", "5m", "15m", "30m", "60m", "90m"):
                return _prints(df).rename(columns=str.capitalize)
            out = df.copy()
            out.index = out.index.tz_localize(TZ)
            return out.rename(columns=str.capitalize)

    mod = types.ModuleType("yfinance")
    mod.__version__ = "1.7.0"
    mod.Ticker = StrictTicker
    saved = sys.modules.get("yfinance")
    sys.modules["yfinance"] = mod
    try:
        cfg = DataConfig(provider="yfinance", min_bars=50, lookback_days=400, interval="1d",
                         live_intraday_bar=True, intraday_interval="5m",
                         cache_dir=str(tmp_path / "cache"), cache_max_age_minutes=0)
        bars = DataSource(cfg, live=True).get("RELIANCE.NS", use_cache=False)
        assert bars.ok, bars.error
        assert bars.source == "yfinance+intraday"
        assert bars.live is True
    finally:
        if saved is not None:
            sys.modules["yfinance"] = saved
        else:
            sys.modules.pop("yfinance", None)


def test_yfinance_uses_the_exchange_clock_even_without_ticker_tz(tmp_path):
    """yfinance >= 1.0 has no ``Ticker.tz``; the market clock comes from config.

    Without it the "is this bar from today's session?" test silently used the
    host clock, which disagrees with IST for hours at a time.
    """
    df = _market()

    class Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.fast_info = {"currency": "INR", "exchange": "NSE"}

        def history(self, **kw):
            out = df.copy()
            out.index = out.index.tz_localize(TZ)
            return out.rename(columns=str.capitalize)

    mod = types.ModuleType("yfinance")
    mod.__version__ = "1.7.0"
    mod.Ticker = Ticker
    saved = sys.modules.get("yfinance")
    sys.modules["yfinance"] = mod
    try:
        cfg = DataConfig(provider="yfinance", min_bars=50, lookback_days=400, interval="1d",
                         live_intraday_bar=False, cache_dir=str(tmp_path / "cache"),
                         cache_max_age_minutes=0)
        bars = DataSource(cfg, live=True).get("RELIANCE.NS", use_cache=False)
        assert bars.ok, bars.error
        assert bars.meta.get("exchangeTimezoneName") == TZ
        assert pd.Timestamp(bars.df.index[-1]).date() == _last_session().date()
    finally:
        if saved is not None:
            sys.modules["yfinance"] = saved
        else:
            sys.modules.pop("yfinance", None)
