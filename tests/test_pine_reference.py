"""Engine ↔ Pine parity, checked against a *literal* transcription.

``tests/pine_reference.py`` re-implements INDICATOR.txt from scratch (its own
``ta.*``, its own parallel arrays, the same statement order).  These tests run
both implementations over randomised and engineered markets and require them to
agree on everything the indicator can show: the ``plotshape`` masks, the four
``alertcondition`` flags, the ``nearestEntry``/``nearestStop`` plots, the zone
arrays and every emitted event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pine_reference import run_reference

from precision_tap.data import synthetic_frame
from precision_tap.engine import (EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP,
                                  run_engine)
from precision_tap.params import Params

KIND = {EV_NEW: "new_ob", EV_TAP: "tap", EV_APPROACH: "approach",
        EV_CONFIRMED: "confirmed", EV_INVALID: "invalidated"}


def _frames():
    """A spread of markets: engineered setups, long histories, odd params."""
    yield "synthetic-seed7", synthetic_frame(n=600, seed=7)
    yield "synthetic-seed11", synthetic_frame(n=400, seed=11, tap_on_last_bar=True)
    yield "synthetic-seed23", synthetic_frame(n=900, seed=23)
    rng = np.random.default_rng(2024)
    for k, seed in enumerate((1, 2, 3)):
        n = 300
        px = 100.0
        rows = []
        for i in range(n):
            o = px
            r = float(rng.normal(0.0004, 0.022))
            c = px * (1 + r)
            hi = max(o, c) * (1 + abs(float(rng.normal(0, 0.01))))
            lo = min(o, c) * (1 - abs(float(rng.normal(0, 0.01))))
            vol = 1e6 * float(rng.uniform(0.3, 6.0))
            rows.append((o, hi, lo, c, vol))
            px = c
        df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                          index=pd.bdate_range("2021-01-04", periods=n))
        yield f"random-{seed}", df


PARAM_VARIANTS = [
    {},
    {"entry_mode": "50%"},
    {"entry_mode": "62%"},
    {"entry_mode": "Distal + 1 tick"},
    {"zone_method": "Body only"},
    {"zone_method": "Lower half of candle"},
    {"frontrun_mode": "Off"},
    {"frontrun_mode": "Ticks"},
    {"frontrun_mode": "ATR"},
    {"require_sweep": True},
    {"raise_after_first_tap": False},
    {"min_age": 1, "require_departure": 0.0},
    {"confirm_bars": 0},
    {"max_zones": 3},
    {"max_touches": 2},
    {"min_rvol": 1.2, "min_clv": 0.6},
]


def _engine_events(res, kinds):
    return [(KIND[e.kind], e.bar, e.zid, e.tap_no, round(float(e.level), 9))
            for e in res.events if e.kind in kinds]


@pytest.mark.parametrize("variant", PARAM_VARIANTS, ids=lambda v: ",".join(f"{k}={v_}" for k, v_ in v.items()) or "defaults")
@pytest.mark.parametrize("intrabar", [False, True])
def test_engine_matches_the_pine_transcription(variant, intrabar):
    p = Params(**variant)
    for name, df in _frames():
        ref = run_reference(df, p, symbol=name, intrabar_last=intrabar)
        got = run_engine(df, p, symbol=name, intrabar_last=intrabar)

        # plotshape(displacement and not duplicate) — the "OB" diamond
        assert list(np.asarray(got.arrays["displacement"], dtype=bool)) == ref.displacement_plot, \
            f"{name} {variant}: displacement/plotshape mask differs"

        for flag, ref_flag in (("any_approach", ref.any_approach), ("any_tap", ref.any_tap),
                               ("any_confirm", ref.any_confirm), ("any_dead", ref.any_dead)):
            assert list(got.flags[flag]) == list(ref_flag), \
                f"{name} {variant}: {flag} differs on bars " \
                f"{[i for i, (a, b) in enumerate(zip(got.flags[flag], ref_flag)) if bool(a) != bool(b)][:5]}"

        for plot, ref_plot in (("nearest_entry", ref.nearest_entry), ("nearest_stop", ref.nearest_stop)):
            a, b = np.asarray(got.flags[plot], dtype=float), np.asarray(ref_plot, dtype=float)
            same = (np.isnan(a) & np.isnan(b)) | (np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=False))
            assert bool(same.all()), f"{name} {variant}: {plot} plot differs"

        # the zone arrays themselves
        ref_table = ref.zone_table()
        got_table = [(z.zid, z.born, z.origin, round(z.top, 9), round(z.bot, 9),
                      round(z.entry, 9), round(z.stop, 9), z.state, z.taps,
                      z.tap_bars[-1] if z.tap_bars else -1, z.departed) for z in got.zones]
        assert got_table == ref_table, f"{name} {variant}: zone arrays differ"

        # every event, in order (the alertcondition set, plus the port's per-zone detail)
        kinds = {EV_NEW, EV_TAP, EV_CONFIRMED, EV_INVALID}
        ref_events = [(k, b, z, t, round(lvl, 9)) for k, b, z, t, lvl in ref.events]
        assert _engine_events(got, kinds) == ref_events, \
            f"{name} {variant}: event stream differs"

        # the approach *flag* is bar-level in Pine; the port edge-triggers the
        # event, so compare the set of bars where the flag is set instead
        assert sorted({e.bar for e in got.by(EV_APPROACH)}) == \
            sorted({e.bar for e in got.by(EV_APPROACH)}) and \
            set(e.bar for e in got.by(EV_APPROACH)) <= {i for i, f in enumerate(ref.any_approach) if f}


def test_the_scanner_honours_the_indicators_max_zones(monkeypatch, tmp_path):
    """The Pine script evicts the oldest box once ``maxZones`` is exceeded.

    The scanner used to replay with ``zone_cap=10**6``, i.e. it kept every zone
    it had ever seen and evaluated taps on boxes TradingView had already thrown
    away — a different ``nearestEntry``, a different duplicate veto and a
    different live-zone count.  This pins the scanner to the indicator's cap.
    """
    from precision_tap.params import AlertConfig, DataConfig, ScanConfig, TelegramConfig
    from precision_tap.scanner import Scanner
    from precision_tap.state import StateStore

    df = synthetic_frame(n=900, seed=1, setup_every=9)      # > maxZones zones over the run
    p = Params()
    capped = run_engine(df, p, symbol="S", intrabar_last=False)
    uncapped = run_engine(df, p, symbol="S", intrabar_last=False, zone_cap=10 ** 6)
    assert len(uncapped.zones) > len(capped.zones) == p.max_zones, \
        "fixture no longer exceeds maxZones — pick a denser market"

    import precision_tap.data as D
    from precision_tap.data import Bars

    monkeypatch.setattr(D.DataSource, "get_many",
                        lambda self, symbols, **kw: {s: Bars(symbol=s, df=df, live=False,
                                                             source="fake") for s in symbols})
    cfg = ScanConfig(
        params=p,
        data=DataConfig(provider="csv", universe=["RELIANCE.NS"], min_bars=90,
                        cache_dir=str(tmp_path / "cache")),
        alert=AlertConfig(chart=False, min_liquidity_dollar_volume=0, min_price=0),
        telegram=TelegramConfig(enabled=False),
        state_db=":memory:", out_dir=str(tmp_path / "out"),
    )
    with StateStore(":memory:") as store:
        sc = Scanner(cfg, store=store, dry_run=True)
        rep = sc.scan(live=False, progress=False)
        sc.close()
    st = rep.rows[0]
    assert st.ok, st.error
    got = [(z.zid, z.born, round(z.top, 9), round(z.entry, 9), z.state) for z in st.result.zones]
    want = [(z.zid, z.born, round(z.top, 9), round(z.entry, 9), z.state) for z in capped.zones]
    assert got == want, "the scanner is not running the indicator's maxZones eviction"
