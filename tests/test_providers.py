"""The yfinance path is exercised against a fake yfinance module (no network)."""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from precision_tap.data import DataSource, normalize_ohlcv, read_universe, synthetic_frame
from precision_tap.params import DataConfig


def _last_session():
    """Most recent weekday in the exchange timezone (today, rolled off weekends)."""
    d = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _daily(n=400, seed=5):
    """Synthetic history that ends on the most recent session — i.e. a live feed.

    A feed frozen in the past would not exercise any of the freshness logic the
    live scanner depends on.
    """
    df = synthetic_frame(n=n, seed=seed)
    shift = _last_session().tz_localize(None) - df.index[-1]
    return df.set_axis(df.index + shift)


class FakeTicker:
    calls = []

    def __init__(self, symbol):
        self.symbol = symbol
        self.tz = "Asia/Kolkata"

    @property
    def fast_info(self):
        return {"currency": "INR", "exchange": "NSE"}

    def history(self, **kw):
        FakeTicker.calls.append(kw)
        df = _daily()
        if kw.get("interval") in ("5m", "1m", "15m") or kw.get("period") == "1d":
            # today's forming bar, rebuilt from 5m prints (09:15 -> 15:30)
            idx = pd.date_range(_last_session() + pd.Timedelta(hours=9, minutes=15),
                                periods=76, freq="5min", tz="Asia/Kolkata")
            px = df["close"].iloc[-1]
            rng = np.random.default_rng(3)
            c = px * (1 + rng.normal(0, 0.002, len(idx))).cumprod()
            out = pd.DataFrame({"open": c * 0.999, "high": c * 1.004, "low": c * 0.996,
                                "close": c, "volume": 50_000}, index=idx)
            return out
        out = df.copy()
        out.index = out.index.tz_localize("Asia/Kolkata")
        return out


@pytest.fixture
def fake_yf(monkeypatch):
    mod = types.ModuleType("yfinance")
    mod.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    FakeTicker.calls = []
    return mod


def test_suffix_applied_and_frame_normalised(fake_yf, tmp_path):
    cfg = DataConfig(provider="yfinance", symbol_suffix=".NS", min_bars=50, lookback_days=400,
                     cache_dir=str(tmp_path / "cache"), cache_max_age_minutes=0, interval="1d")
    src = DataSource(cfg, live=False)
    assert src.fetch_symbol("RELIANCE") == "RELIANCE.NS"
    assert src.fetch_symbol("RELIANCE.NS") == "RELIANCE.NS"
    assert src.fetch_symbol("^NSEI") == "^NSEI"
    bars = src.get("RELIANCE")
    assert bars.ok, bars.error
    assert list(bars.df.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars.df["high"] >= bars.df["low"]).all()
    assert bars.df.index.is_monotonic_increasing
    assert FakeTicker.calls and FakeTicker.calls[0]["interval"] == "1d"
    assert FakeTicker.calls[0]["auto_adjust"] is True


def test_live_mode_merges_intraday_bar(fake_yf, tmp_path):
    cfg = DataConfig(provider="yfinance", symbol_suffix=".NS", min_bars=50, live_intraday_bar=True,
                     intraday_interval="5m", cache_dir=str(tmp_path / "cache"),
                     cache_max_age_minutes=0, lookback_days=400)
    src = DataSource(cfg, live=True)
    bars = src.get("TCS", use_cache=False)
    assert bars.ok, bars.error
    assert bars.source == "yfinance+intraday"
    assert bars.live is True
    last = bars.df.index[-1]
    # the merged bar is stamped with the intraday session date, midnight-normalised
    assert last.normalize().date() == last.date()
    assert any(k.get("period") == "1d" for k in FakeTicker.calls)
    assert any(k.get("interval") == "5m" for k in FakeTicker.calls)


def test_read_universe_suffixes_and_dedupes(tmp_path):
    f = tmp_path / "nse.txt"
    f.write_text("# comment\nRELIANCE\nTCS\nRELIANCE\n^NSEI\nINFY.NS\n")
    uni = read_universe(str(f), suffix=".NS")
    assert uni == ["RELIANCE.NS", "TCS.NS", "^NSEI", "INFY.NS"]


def test_csv_provider_roundtrip(tmp_path):
    from precision_tap.data import frame_to_csv, read_csv_frame
    df = _daily()
    frame_to_csv(df, tmp_path / "TATASTEEL.NS.csv")
    back = read_csv_frame(tmp_path / "TATASTEEL.NS.csv")
    assert len(back) == len(df)
    assert np.allclose(back["close"].to_numpy(), df["close"].to_numpy())
    cfg = DataConfig(provider="csv", history_dir=str(tmp_path), min_bars=50,
                     symbol_suffix=".NS", cache_dir=str(tmp_path / "c"),
                     cache_max_age_minutes=60)
    bars = DataSource(cfg).get("TATASTEEL")
    assert bars.ok and len(bars.df) == len(df)


def test_cache_ttl_zero_disables_cache(fake_yf, tmp_path):
    # live=False uses eod_cache_max_age_minutes, so zero both knobs
    cfg = DataConfig(provider="yfinance", symbol_suffix=".NS", min_bars=50, lookback_days=400,
                     cache_dir=str(tmp_path / "cache"), cache_max_age_minutes=0,
                     eod_cache_max_age_minutes=0, interval="1d")
    src = DataSource(cfg, live=False)
    src.get("TCS", use_cache=True)
    n = len(FakeTicker.calls)
    src.get("TCS", use_cache=True)
    assert len(FakeTicker.calls) == n * 2, "cache_max_age_minutes: 0 must mean no caching"
    assert list((tmp_path / "cache").glob("*.csv")) == []


def test_cache_hit_skips_fetch(fake_yf, tmp_path):
    cfg = DataConfig(provider="yfinance", symbol_suffix=".NS", min_bars=50, lookback_days=400,
                     cache_dir=str(tmp_path / "cache"), cache_max_age_minutes=10_000,
                     eod_cache_max_age_minutes=10_000, interval="1d")
    src = DataSource(cfg, live=False)
    src.get("INFY", use_cache=True)
    n = len(FakeTicker.calls)
    bars = src.get("INFY", use_cache=True)
    assert bars.source == "cache" and len(FakeTicker.calls) == n
