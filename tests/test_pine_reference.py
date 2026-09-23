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


def _ob_block(off: float, attempt: str):
    """One ``displacement → departure → tap → defence attempt`` block.

    ``off`` shifts the whole block in price so consecutive blocks leave zones
    that do not overlap (the duplicate veto compares *entry* levels).  Every
    value is a plain number, so the block reads as a chart: an origin candle
    with a volume-confirmed displacement off it, a departure, a retrace into
    the block that taps it, and then one specific attempt at a defence.
    """
    bars = [
        (off + 0.00, off + 0.55, off - 0.50, off + 0.05, 1_000_000),   # calm
        (off + 0.05, off + 0.60, off - 0.45, off + 0.10, 1_000_000),   # calm
        (off + 0.10, off + 0.35, off - 0.50, off - 0.40, 1_000_000),   # ORIGIN (bearish)
        (off + 0.55, off + 2.10, off + 0.45, off + 2.00, 8_000_000),   # DISPLACEMENT
        (off + 2.00, off + 3.00, off + 1.95, off + 2.90, 1_200_000),   # departure
        (off + 2.90, off + 3.00, off + 2.60, off + 2.70, 1_000_000),
        (off + 2.70, off + 2.80, off + 2.40, off + 2.50, 1_000_000),
        (off + 2.50, off + 2.60, off + 2.10, off + 2.20, 1_000_000),
    ]
    if attempt.startswith("inside"):
        # the retrace stops *inside* the block, so the last three highs sit
        # below `top` and `close > top` becomes the deciding gate
        bars += [
            (off + 0.05, off + 0.08, off - 0.60, off - 0.40, 1_500_000),   # TAP
            (off - 0.40, off - 0.30, off - 0.70, off - 0.45, 1_000_000),
            (off - 0.50, off - 0.42, off - 0.75, off - 0.50, 1_000_000),
        ]
        attempts = {
            "inside_below_top": (off - 0.30, off + 0.09, off - 0.35, off + 0.085, 4_000_000),
            "inside_above_top": (off - 0.30, off + 0.40, off - 0.35, off + 0.35, 4_000_000),
        }
    else:
        bars += [
            (off + 0.80, off + 1.20, off - 0.60, off + 0.60, 1_500_000),   # TAP (deep sweep)
            (off + 0.60, off + 0.70, off + 0.45, off + 0.65, 1_000_000),   # quiet (low > entry)
            (off + 0.65, off + 0.75, off + 0.45, off + 0.70, 1_000_000),
        ]
        attempts = {
            "good":     (off + 0.60, off + 1.60, off + 0.55, off + 1.50, 4_000_000),
            "bearish":  (off + 1.55, off + 1.60, off + 0.55, off + 1.50, 4_000_000),   # close <= open
            "low_clv":  (off + 0.90, off + 2.20, off + 0.85, off + 1.30, 4_000_000),   # close near the low
            "low_rvol": (off + 0.60, off + 1.60, off + 0.55, off + 1.50, 1_100_000),   # no volume
            "no_bos":   (off + 0.60, off + 1.15, off + 0.55, off + 1.00, 4_000_000),   # under the 3-bar high
        }
    bars += [attempts[attempt],
             (off + 1.40, off + 1.50, off + 1.30, off + 1.45, 1_000_000),
             (off + 1.45, off + 1.55, off + 1.35, off + 1.50, 1_000_000),
             (off + 1.50, off + 1.60, off + 1.40, off + 1.55, 1_000_000)]
    return bars


def _stress_frame(kinds, *, level: float = 100.0, step: float = 6.0, warm: int = 30) -> pd.DataFrame:
    rows = [(level, level + 0.5, level - 0.5, level, 1_000_000)] * warm
    for i, kind in enumerate(kinds):
        rows += _ob_block(level + (i + 1) * step, kind)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=pd.bdate_range("2021-01-04", periods=len(rows)))


def _defence_stress_frames():
    """Markets whose taps are followed by *near-miss* defence bars.

    The confirmation gates are the ones a differential test can silently miss:
    a random walk rarely prints a bar that fails ``close > open``,
    ``clv >= confirmCLV`` or ``close > top`` while satisfying everything else —
    so those three gates could be deleted from the engine with every frame
    above still agreeing.  These frames put each attempt next to a control:
    ``good`` versus ``bearish``/``low_clv``/``low_rvol``/``no_bos``, and a rally
    that closes above the last three highs but inside the block versus one that
    closes just above its top.
    """
    yield "stress-defence-a", _stress_frame(
        ["good", "inside_above_top", "bearish", "good", "inside_below_top", "low_clv"])
    yield "stress-defence-b", _stress_frame(
        ["no_bos", "good", "low_rvol", "good", "good", "inside_above_top"], step=5.0)
    yield "stress-retap", _double_defence_frame()


def _retap_block(off: float, *, double: bool = True):
    """One block that is *defended twice* (Pine ``state 2`` is still tappable).

    ``INDICATOR.txt`` lets a confirmed block be tapped again — and the second
    tap can itself be confirmed within ``confirmBars``.  The bars below put the
    second defence's close above the three-bar high that includes the *first*
    defence bar, so only the ``microBOS`` window decides it.
    """
    bars = [
        (off + 0.00, off + 0.55, off - 0.50, off + 0.05, 1_000_000),   # calm
        (off + 0.05, off + 0.60, off - 0.45, off + 0.10, 1_000_000),   # calm
        (off + 0.10, off + 0.35, off - 0.50, off - 0.40, 1_000_000),   # ORIGIN (bearish)
        (off + 0.55, off + 2.10, off + 0.45, off + 2.00, 8_000_000),   # DISPLACEMENT
        (off + 2.00, off + 3.00, off + 1.95, off + 2.90, 1_200_000),   # departure
        (off + 2.90, off + 3.00, off + 2.60, off + 2.70, 1_000_000),
        (off + 2.70, off + 2.80, off + 2.40, off + 2.50, 1_000_000),
        (off + 2.50, off + 2.60, off + 2.10, off + 2.20, 1_000_000),
        (off + 0.80, off + 1.20, off - 0.60, off + 0.60, 1_500_000),   # TAP 1
        (off + 0.60, off + 0.70, off + 0.45, off + 0.65, 1_000_000),
        (off + 0.65, off + 0.75, off + 0.45, off + 0.70, 1_000_000),
        (off + 0.60, off + 1.60, off + 0.55, off + 1.50, 4_000_000),   # DEFENCE 1
    ]
    if double:
        bars += [
            (off + 0.25, off + 0.35, off - 0.20, off + 0.05, 1_500_000),  # TAP 2 (retap)
            (off + 0.30, off + 0.40, off + 0.32, off + 0.35, 1_000_000),  # no tap, CLV short
            (off + 0.35, off + 1.75, off + 0.30, off + 1.65, 4_000_000),  # DEFENCE 2
        ]
    bars += [(off + 1.70, off + 1.80, off + 1.60, off + 1.75, 1_000_000),
             (off + 1.75, off + 1.85, off + 1.65, off + 1.80, 1_000_000),
             (off + 1.80, off + 1.90, off + 1.70, off + 1.85, 1_000_000)]
    return bars


def _double_defence_frame() -> pd.DataFrame:
    """Two defences of one zone, on 5-minute bars inside a single session.

    Intraday on purpose: both ``DEFENCE CONFIRMED`` bars then carry the same
    *date*, which is the case an alert ledger keyed by ``(zone, day)`` would
    collapse into one message while the indicator printed two labels.
    """
    rows = [(100.0, 100.5, 99.5, 100.0, 1_000_000)] * 30
    rows += _retap_block(100.0)
    rows += _retap_block(120.0, double=False)          # control: defends once
    idx = pd.date_range("2021-03-01 09:15", periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"],
                        index=idx)


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
    yield from _defence_stress_frames()


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
    {"zone_method": "Body to low"},
    # confirmation gates, pushed to both ends of their range: a defence that
    # clears a 0.95 CLV / 2.5x RVOL bar is rare, a 0.5 CLV / 0.5x RVOL one with
    # a 5-bar breakout window is common — the frames above sit between them
    {"confirm_clv": 0.95, "confirm_rvol": 2.5},
    {"confirm_clv": 0.5, "confirm_rvol": 0.5, "confirm_bos_len": 5, "confirm_bars": 5},
    # easy displacement, so the near-miss markets actually produce zones/taps
    {"min_clv": 0.55, "min_rvol": 1.1, "min_range_atr": 0.5, "min_body_frac": 0.3,
     "structure_len": 4, "min_age": 1, "require_departure": 0.2},
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
        # event (one per zone per approach spell), so every approach event must
        # sit on a bar whose flag is set — the flags themselves were compared
        # against the transcription above
        assert set(e.bar for e in got.by(EV_APPROACH)) <= \
            {i for i, f in enumerate(ref.any_approach) if f}


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


def test_a_defended_block_can_be_defended_a_second_time():
    """Pine's ``state 2`` is still tappable, so one block can defend twice.

    The port must not treat a confirmation as the end of the zone's life: a
    repeat tap moves it back to ``state 1`` and the same ``defence`` expression
    can fire again within ``confirmBars``.  Two ``DEFENCE CONFIRMED`` labels in
    the indicator therefore mean two ``confirmed`` events here — and, at the
    alert layer, two messages (``tests/test_alert_validity.py``).
    """
    df = _double_defence_frame()
    p = Params()
    ref = run_reference(df, p, symbol="RETAP.NS", intrabar_last=False)
    got = run_engine(df, p, symbol="RETAP.NS", intrabar_last=False)
    conf = got.by(EV_CONFIRMED)
    assert [(e.zid, e.bar, e.tap_no) for e in conf] == [(0, 41, 0), (0, 44, 0)], \
        [(e.zid, e.bar) for e in conf]
    assert [e.bar for e in conf] == [b for k, b, *_ in ref.events if k == "confirmed"]
    events = [(e.kind, e.bar, e.zid, e.tap_no) for e in got.events if e.kind != EV_APPROACH]
    assert events == [(k, b, z, t) for k, b, z, t, _ in ref.events]
    # the second defence is the *second tap*, not the first one held
    assert [e.tap_no for e in got.by(EV_TAP)] == [1, 2]
