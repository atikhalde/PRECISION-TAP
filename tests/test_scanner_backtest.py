import math

import numpy as np
import pandas as pd
import pytest

from precision_tap.backtest import Backtester
from precision_tap.data import synthetic_frame, write_demo_dataset
from precision_tap.engine import EV_APPROACH, EV_CONFIRMED, EV_TAP, Params, run_engine
from precision_tap.params import AlertConfig, DataConfig, ScanConfig, TradeConfig
from precision_tap.scanner import Scanner, _market_open, _same_session
from precision_tap.selftest import GOLDEN, GOLDEN_LIVE, base_params, golden_frame
from precision_tap.state import StateStore


def cfg_for(symbols, **kw):
    d = dict(provider="csv", history_dir=kw.pop("history", "data/csv_test"), cache_dir="data/cache_test",
             universe=list(symbols), min_bars=90, lookback_days=600, symbol_suffix="")
    d.update(kw.pop("data", {}))
    return ScanConfig(params=Params.default(), data=DataConfig(**d),
                      alert=AlertConfig(events=["tap1", "approach", "confirmed"], chart=False,
                                        skip_stale_bars=False, min_liquidity_dollar_volume=0),
                      trade=TradeConfig(), state_db=":memory:", out_dir="results_test")


@pytest.fixture(scope="module")
def demo_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("csv")
    write_demo_dataset(d, symbols=("RELIANCE.NS", "TCS.NS", "INFY.NS"), n=600, tap_on_last_bar=2)
    return str(d)


def test_scanner_fires_on_last_bar(demo_data):
    cfg = cfg_for(("RELIANCE.NS", "TCS.NS", "INFY.NS"), history=demo_data)
    with StateStore(":memory:") as store:
        rep = Scanner(cfg, store=store, dry_run=True).scan(live=False, progress=False)
        assert rep.usable == 3
        assert len(rep.alerts) >= 1, rep.summary_line
        assert all(ev.kind == EV_TAP and ev.tap_no == 1 for _, ev in rep.alerts)
        # second scan of the same data must be de-duplicated
        rep2 = Scanner(cfg, store=store, dry_run=True).scan(live=False, progress=False)
        assert rep2.dispatch is None or getattr(rep2.dispatch, "sent", 0) == 0


def test_mintick_resolution_per_market(demo_data):
    cfg = cfg_for(("X",), history=demo_data)
    cfg.data.tick_sizes = {".NS": 0.05, ".BO": 0.05}
    cfg.params = Params.default().replace(mintick=0.01)
    sc = Scanner(cfg, store=None, dry_run=True)
    assert sc._params_for("RELIANCE.NS").mintick == 0.05
    assert sc._params_for("SOMETHING.BO").mintick == 0.05
    assert sc._params_for("RELIANCE").mintick == 0.05, "bare symbols are treated as NSE"
    assert sc._params_for("AAPL.OQ").mintick == 0.01, "foreign tickers never get an NSE tick"
    sc.close()


def test_india_and_daily_are_enforced():
    from precision_tap.params import DataConfig
    with pytest.raises(ValueError, match="daily"):
        DataConfig(interval="15m")
    with pytest.raises(ValueError, match="Indian markets"):
        DataConfig(market="NASDAQ")
    with pytest.raises(ValueError, match="symbol_suffix"):
        DataConfig(symbol_suffix=".L")
    from precision_tap.data import DataSource
    src = DataSource(DataConfig())
    assert src.fetch_symbol("RELIANCE") == "RELIANCE.NS"
    assert src.fetch_symbol("TCS.BO") == "TCS.BO"
    assert src.fetch_symbol("^NSEI") == "^NSEI"
    with pytest.raises(ValueError, match="not an NSE/BSE ticker"):
        src.fetch_symbol("AAPL.OQ")


def test_scanner_alerts_equal_closed_bar_indicator(demo_data):
    """The EOD scan must emit EXACTLY the indicator's confirmed signal set — no more,
    no less — for the newest bar. This is the 100%-match contract, end to end."""
    import precision_tap.data as D
    cfg = cfg_for(("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"), history=demo_data)
    cfg.alert.recent_bars = 3
    frames = {s: b.df for s, b in D.DataSource(cfg.data).get_many(cfg.data.universe,
                                                                progress=False).items() if b.ok}
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=True)
        rep = sc.scan(live=False, progress=False)
        for st in rep.rows:
            if not st.ok:
                continue
            eng = run_engine(st.df, sc._params_for(st.symbol), symbol=st.symbol,
                             intrabar_last=False, zone_cap=10 ** 6)
            n = len(st.df)
            want = sorted((e.kind, e.bar, e.zid) for e in eng.events if e.bar >= n - 3
                          and e.kind in (EV_TAP, EV_APPROACH, EV_CONFIRMED))
            got = sorted((e.kind, e.bar, e.zid) for e in st.events
                          if e.bar >= n - 3 and e.kind in (EV_TAP, EV_APPROACH, EV_CONFIRMED))
            assert got == want, (st.symbol, got, want)
            # and the alert window only ever contains events on the newest bar(s)
            assert all(e.bar >= n - cfg.alert.recent_bars for _, e in rep.alerts
                       if e.symbol == st.symbol)
        sc.close()


def test_session_and_staleness_helpers():
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-03 11:00").to_pydatetime()) is True
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-03 20:00").to_pydatetime()) is False
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-01 11:00").to_pydatetime()) is False  # Sat
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-03 16:00").to_pydatetime(),
                        session=("09:15", "17:00")) is True
    # tz-aware: a naive now() is read as the runner's UTC date, which differs from
    # the IST session date after 18:30Z — that made this assertion a clock flake.
    assert _same_session(pd.Timestamp.now(tz="Asia/Kolkata"), "Asia/Kolkata") is True
    assert _same_session(pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=1),
                         "Asia/Kolkata") is False


def test_backtest_matches_engine_events(demo_data):
    cfg = cfg_for(("RELIANCE.NS", "TCS.NS", "INFY.NS"), history=demo_data)
    import precision_tap.data as D
    frames = {s: b.df for s, b in D.DataSource(cfg.data).get_many(cfg.data.universe, progress=False).items()
              if b.ok}
    res = Backtester(cfg, progress=False).run(frames, collect_events=True)
    assert res.n_trades > 0
    for t in res.trades:
        assert t.r_multiple == t.r_multiple and t.exit_bar >= t.entry_bar
        assert math.isclose(t.risk_per_share, t.entry_price - t.initial_stop, rel_tol=1e-9)
        assert t.net_ret_pct <= t.ret_pct          # costs only ever hurt
    m = res.metrics
    assert 0.0 <= m["win_rate"] <= 100.0
    assert m["signals"] == res.n_trades
    assert res.equity is not None and len(res.equity) > 0


def test_fill_and_stop_assumptions(synthetic_daily):
    """Conservative fills: limit buy never worse than the level; same-bar worst path."""
    cfg = ScanConfig(params=Params.default(), trade=TradeConfig(target_r=1.5, same_bar_stop=True,
                                                                 commission_bps=0, slippage_bps=0,
                                                                 spread_bps=0))
    res = Backtester(cfg, progress=False).run({"SYN": synthetic_daily})
    eng = run_engine(synthetic_daily, Params.default(), symbol="SYN", zone_cap=10 ** 6)
    levels = {e.bar: e.level for e in eng.events if e.kind == EV_TAP and e.tap_no == 1}
    for t in res.trades:
        assert t.entry_price <= levels[t.entry_bar] + 1e-9
        if t.exit_reason in ("initial_stop", "stop", "trailing_stop"):
            assert t.exit_price <= t.initial_stop + 1e-9
