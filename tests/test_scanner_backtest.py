import math

import numpy as np
import pandas as pd
import pytest

from precision_tap.backtest import Backtester
from precision_tap.data import synthetic_frame, write_demo_dataset
from precision_tap.engine import EV_CONFIRMED, EV_TAP, Params, run_engine
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
    assert sc._params_for("AAPL").mintick == 0.01
    sc.close()


def test_session_and_staleness_helpers():
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-03 11:00").to_pydatetime()) is True
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-03 20:00").to_pydatetime()) is False
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-01 11:00").to_pydatetime()) is False  # Sat
    assert _market_open("Asia/Kolkata", pd.Timestamp("2024-06-03 16:00").to_pydatetime(),
                        session=("09:15", "17:00")) is True
    assert _same_session(pd.Timestamp.now(), "Asia/Kolkata") is True


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
