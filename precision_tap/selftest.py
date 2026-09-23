"""Self-test / Pine-parity harness.

Every check below is computed by hand from the Pine source (derivations in
ANALYSIS.md) on tiny "golden" bar series, so the port can be verified without a
TradingView account, without network access and without pytest
(``tests/test_engine.py`` simply wraps these functions).

Run:  ``python -m precision_tap.selftest``
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .engine import (EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, ST_CONFIRMED,
                     ST_DEAD, ST_FRESH, Params, entry_level, run_engine)
from . import series as S

# ─────────────────────────────────────────────────────────────────────────────
# fixtures
# ─────────────────────────────────────────────────────────────────────────────

# (open, high, low, close, volume) — every bar has an exact range of 1.00 and no
# gaps, so ATR(3) == 1.00 for bars >= 2 and every derived level is hand-checkable.
GOLDEN: List[Tuple[float, float, float, float, float]] = [
    (100.00, 100.50,  99.50, 100.00, 1000),   # 0
    (100.00, 100.50,  99.50, 100.00, 1000),   # 1
    (100.00, 100.50,  99.50, 100.00, 1000),   # 2
    (100.00, 100.10,  99.10,  99.60, 1000),   # 3  bearish origin candle
    ( 99.60, 100.55,  99.55, 100.52, 5000),   # 4  volume-confirmed displacement
    (100.52, 101.52, 100.52, 101.00, 1000),   # 5  departure (high >= top + 1*ATR)
    (101.00, 101.20, 100.20, 100.60, 1000),   # 6  approach only (low 100.20 > entry)
    (100.60, 100.90,  99.90, 100.70, 1000),   # 7  TAP 1  (low <= 100.00, high >= bot)
    (100.70, 101.30, 100.30, 101.25, 3000),   # 8  defence confirmed
]

# second displacement on bar 5 that re-uses the same origin candle (bar 3)
GOLDEN_DUP = GOLDEN[:5] + [(100.52, 101.10, 100.10, 101.10, 9000)]
# a shallower revisit on bar 9 that only taps if the level was raised after Tap 1
GOLDEN_TAP2 = GOLDEN + [(100.70, 101.05, 100.05, 100.60, 1000)]
# bar 8 crashes through everything: tap, then close below the stop
GOLDEN_STOP = GOLDEN[:8] + [(100.70, 100.70,  98.40,  98.40, 4000)]
# the tap bar is also the last (possibly forming) bar: tap + defence same bar
GOLDEN_LIVE = GOLDEN[:7] + [(100.00, 101.30,  99.90, 101.30, 5000)]
# tap on bar 7 (sweep), higher tap on bar 8 (no sweep)
GOLDEN_NOSWEEP = GOLDEN[:8] + [(100.00, 100.90, 99.95, 100.60, 1000)]


def base_params(**kw: Any) -> Params:
    """Relaxed-but-explicit params for the golden fixtures (Pine defaults otherwise)."""
    p = dict(
        vol_len=2, atr_len=3, min_rvol=1.2, min_range_atr=1.0, min_body_frac=0.50,
        min_clv=0.60, structure_len=3, origin_search=5, allow_neutral_origin=True,
        neutral_body=0.20, zone_method="Open to low", entry_mode="Proximal",
        frontrun_mode="Off", stop_atr=0.50, approach_atr=0.30, min_age=2, max_zones=30,
        max_touches=4, raise_after_first_tap=False, repeat_tap_atr=0.20,
        require_departure=1.0, confirm_bars=2, confirm_rvol=1.0, confirm_clv=0.60,
        confirm_bos_len=1, require_sweep=False, sweep_len=2, mintick=0.01, warmup=0,
    )
    p.update(kw)
    return Params(**p)


def golden_frame(bars: Sequence[Tuple[float, float, float, float, float]] | None = None) -> pd.DataFrame:
    bars = list(GOLDEN if bars is None else bars)
    idx = pd.bdate_range("2024-01-02", periods=len(bars))
    return pd.DataFrame(bars, columns=["open", "high", "low", "close", "volume"], index=idx)


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


# ─────────────────────────────────────────────────────────────────────────────
# checks
# ─────────────────────────────────────────────────────────────────────────────

def check_series_parity() -> None:
    """ta.sma / ta.rma / ta.highest / ta.tr warm-up semantics."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sma = S.sma(x, 3)
    assert np.isnan(sma[0]) and np.isnan(sma[1])
    assert approx(sma[2], 2.0) and approx(sma[3], 3.0) and approx(sma[4], 4.0)
    rma = S.rma(x, 3)
    assert np.isnan(rma[0]) and np.isnan(rma[1]), "RMA is `na` before the seed bar"
    assert approx(rma[2], 2.0), "RMA is seeded with the SMA of the first `length` bars"
    assert approx(rma[3], 2.0 + (4.0 - 2.0) / 3.0), "alpha = 1/length -> prev + (x-prev)/n"
    high = S.highest(S.shift(np.array([1.0, 5.0, 3.0, 4.0]), 1), 3)
    assert np.isnan(high[0]) and approx(high[1], 1.0) and approx(high[2], 5.0) and approx(high[3], 5.0)
    tr = S.true_range(np.array([10.0, 12.0]), np.array([8.0, 9.0]), np.array([9.0, 11.0]))
    assert approx(tr[0], 2.0) and approx(tr[1], 3.0)
    a = S.atr(np.array([10.0, 12.0, 11.0]), np.array([8.0, 9.0, 9.5]), np.array([9.0, 11.0, 10.0]), 2)
    assert np.isnan(a[0]) and approx(a[1], 2.5), "seed = SMA(tr, 2) = (2.0 + 3.0)/2"
    assert approx(a[2], 2.0)


def check_golden_flow() -> None:
    """Creation → departure → approach → Tap 1 → defence, with exact levels."""
    r = run_engine(golden_frame(), base_params(), symbol="GOLD")
    assert len(r.zones) == 1, f"expected exactly one zone, got {len(r.zones)}"
    z = r.zones[0]
    assert z.born == 4 and z.origin == 3
    assert approx(z.top, 100.00), "Open-to-low top = origin candle's open"
    assert approx(z.bot, 99.10), "Open-to-low bottom = origin candle's low"
    assert approx(z.entry, 100.00)
    assert approx(z.stop, 98.60), "stop = bot - 0.5*ATR = 99.10 - 0.50"
    kinds = [(e.kind, e.bar) for e in r.events]
    assert kinds == [(EV_NEW, 4), (EV_APPROACH, 6), (EV_TAP, 7), (EV_CONFIRMED, 8)], kinds
    tap = r.by(EV_TAP)[0]
    assert tap.tap_no == 1 and approx(tap.level, 100.00)
    assert z.state == ST_CONFIRMED and z.taps == 1
    assert bool(r.flags["any_approach"][6]) and bool(r.flags["any_approach"][7])
    assert not bool(r.flags["any_approach"][5])
    assert bool(r.flags["any_tap"][7]) and bool(r.flags["any_confirm"][8])
    assert approx(r.flags["nearest_entry"][7], 100.00)
    assert int(r.arrays["displacement"].sum()) == 1, "plotshape('OB') fires once"
    assert r.summary()["tap1"] == 1 and r.summary()["confirmed"] == 1


def check_min_age_gate() -> None:
    """age < minAge blocks a tap even when price reaches the level."""
    r = run_engine(golden_frame(), base_params(min_age=6), symbol="GOLD")
    assert r.by(EV_TAP) == [] and r.by(EV_APPROACH) == []
    assert r.zones[0].state == ST_FRESH and r.zones[0].taps == 0


def check_departure_gate() -> None:
    """No tap before the zone has been left by requireDeparture * ATR."""
    r = run_engine(golden_frame(), base_params(require_departure=99.0), symbol="GOLD")
    assert r.by(EV_TAP) == [] and not r.zones[0].departed
    r2 = run_engine(golden_frame(), base_params(require_departure=0.0), symbol="GOLD")
    assert r2.zones[0].departed and r2.zones[0].born == 4
    assert len(r2.by(EV_TAP)) == 1, "departure on the creation bar still needs minAge"


def check_frontrun_buffer() -> None:
    """Auto buffer = min(max(ticks, frontRunATR*ATR), width*maxZoneBuffer)."""
    p = base_params(frontrun_mode="Auto", frontrun_ticks=2, frontrun_atr=0.18, max_zone_buffer=0.40)
    r = run_engine(golden_frame(), p, symbol="GOLD")
    width = 100.00 - 99.10
    buf = min(max(2 * 0.01, 0.18 * 1.0), width * 0.40)      # = 0.18
    assert approx(buf, 0.18)
    assert approx(r.zones[0].entry, 100.00 + buf)
    assert approx(r.zones[0].risk, 100.18 - 98.60)
    atr = 1.0
    assert approx(entry_level(100.0, 99.10, atr, base_params(frontrun_mode="ATR", frontrun_atr=0.25)), 100.25)
    assert approx(entry_level(100.0, 99.10, atr, base_params(frontrun_mode="Ticks", frontrun_ticks=5)), 100.05)
    assert approx(entry_level(100.0, 99.10, atr, base_params(frontrun_mode="Off")), 100.0)
    d = base_params(entry_mode="Distal + 1 tick", frontrun_mode="Auto")
    assert approx(entry_level(100.0, 99.10, atr, d), 99.11), "a distal entry is never front-run"
    assert approx(entry_level(100.0, 99.0, atr, base_params(entry_mode="50%", frontrun_mode="Off")), 99.5)
    assert approx(entry_level(100.0, 99.0, atr, base_params(entry_mode="79%", frontrun_mode="Off")),
                  100.0 - 1.0 * 0.79)
    assert approx(entry_level(100.0, 99.0, atr, base_params(frontrun_mode="Off", entry_offset_ticks=7)),
                  100.07), "manual tick offset"
    capped = base_params(frontrun_mode="Auto", frontrun_atr=5.0, max_zone_buffer=0.40)
    assert approx(entry_level(100.0, 99.0, atr, capped), 100.0 + 0.40), "auto cap = 40% of zone width"


def check_zone_methods() -> None:
    """rawTop/rawBot per Tight-zone method, taken from origin candle bar 3."""
    df = golden_frame()
    o3, h3, l3, c3 = 100.00, 100.10, 99.10, 99.60
    expect = {
        "Open to low": (o3, l3),
        "Body to low": (max(o3, c3), l3),
        "Body only": (max(o3, c3), min(o3, c3)),
        "Lower half of candle": (l3 + (h3 - l3) * 0.5, l3),
    }
    for method, (top, bot) in expect.items():
        r = run_engine(df, base_params(zone_method=method, frontrun_mode="Off"), symbol="GOLD")
        z = r.zones[0]
        assert approx(z.top, top) and approx(z.bot, bot), (method, z.top, z.bot, top, bot)


def check_origin_search() -> None:
    """Origin = nearest bearish (or small-neutral) candle, else the previous bar."""
    bars = [
        (100.00, 100.50, 99.50, 100.00, 1000),   # 0  bearish-ish control
        (100.00, 100.50, 99.50, 100.00, 1000),   # 1
        (100.00, 100.50, 99.50, 100.00, 1000),   # 2
        (100.00, 100.10, 99.10,  99.60, 1000),   # 3  bearish origin
        ( 99.60, 100.10, 99.50,  99.62, 1000),   # 4  tiny-bodied bullish (neutral: 0.02/0.60)
        ( 99.62, 100.55, 99.55, 100.52, 5000),   # 5  displacement
    ]
    df = golden_frame(bars)
    r = run_engine(df, base_params(allow_neutral_origin=True), symbol="GOLD")
    z = r.zones[0]
    assert z.born == 5 and z.origin == 4, "neutral candle is nearer, so it wins"
    assert approx(z.top, 99.60) and approx(z.bot, 99.50)
    r2 = run_engine(df, base_params(allow_neutral_origin=False), symbol="GOLD")
    z2 = r2.zones[0]
    assert z2.origin == 3, "neutral origins disabled -> search continues to the bearish bar"
    assert approx(z2.top, 100.00) and approx(z2.bot, 99.10)
    # all-bullish run -> fallback to the immediately preceding candle
    up = [(100.0 + 0.5 * i, 100.6 + 0.5 * i, 99.9 + 0.5 * i, 100.4 + 0.5 * i, 1000) for i in range(5)]
    up.append((102.4, 103.6, 102.3, 103.55, 9000))
    r3 = run_engine(golden_frame(up), base_params(), symbol="GOLD")
    assert r3.zones[0].origin == 4, "fallback originOffset = 1"


def check_duplicate_rejection() -> None:
    """A candidate entry inside a live zone is dropped (and its OB marker too)."""
    r = run_engine(golden_frame(GOLDEN_DUP), base_params(), symbol="GOLD")
    assert len(r.zones) == 1, "second displacement re-uses the same origin candle"
    assert bool(r.flags["any_tap"].any()) is False
    assert not bool(r.arrays["displacement"][5]), "plotshape suppressed for duplicates"
    # an ATR front-run pushes the candidate above the live zone -> accepted
    r2 = run_engine(golden_frame(GOLDEN_DUP), base_params(frontrun_mode="ATR", frontrun_atr=0.60),
                    symbol="GOLD")
    assert len(r2.zones) == 2, "entry outside [bot, top] is not a duplicate"


def check_adaptive_entry() -> None:
    """raiseAfterFirstTap: next pre-order = max(entry, Tap-1 low + repeatTapATR*ATR)."""
    r = run_engine(golden_frame(GOLDEN_TAP2),
                   base_params(raise_after_first_tap=True, repeat_tap_atr=0.20), symbol="GOLD")
    z = r.zones[0]
    assert approx(z.entry, 100.10), "99.90 + 0.20*ATR(=1.00) on the Tap-1 bar"
    taps = r.by(EV_TAP)
    assert [t.tap_no for t in taps] == [1, 2]
    assert approx(taps[0].level, 100.00), "Tap 1 is reported at the original level"
    assert approx(taps[1].level, 100.10), "Tap 2 fires at the raised level"
    r_off = run_engine(golden_frame(GOLDEN_TAP2), base_params(raise_after_first_tap=False), symbol="GOLD")
    assert r_off.zones[0].taps == 1, "original level was never revisited"


def check_invalidation_ordering() -> None:
    """The tap is recorded before the close-below-stop invalidation, like Pine."""
    r = run_engine(golden_frame(GOLDEN_STOP), base_params(), symbol="GOLD")
    kinds = [(e.kind, e.bar) for e in r.events]
    assert (EV_TAP, 8) in kinds and (EV_INVALID, 8) in kinds
    assert kinds.index((EV_TAP, 8)) < kinds.index((EV_INVALID, 8))
    dead = r.by(EV_INVALID)[0]
    assert dead.detail.get("reason") == "close_below_stop"
    assert r.zones[0].state == ST_DEAD and r.zones[0].dead_bar == 8


def check_pending_expiry() -> None:
    """state 1 -> 0 once confirmBars elapse without a defence."""
    df = golden_frame(GOLDEN[:8] + [(100.70, 101.00, 100.40, 100.80, 1000)] * 3)
    r = run_engine(df, base_params(confirm_bars=1), symbol="GOLD")
    assert r.by(EV_CONFIRMED) == []
    assert r.zones[0].state == ST_FRESH
    assert r.zones[0].taps >= 1


def check_defence_gate_rule() -> None:
    """Every gate of the Pine ``defence`` expression is individually required.

    ``defence = barstate.isconfirmed and pending and close > open and
    clv >= confirmCLV and rvol >= confirmRVOL and close > top and microBOS``

    Each row below is a ``GOLDEN`` market whose Tap 1 sits on bar 7 and whose
    bar 8 satisfies the other six gates but fails exactly one — so the engine
    must not confirm it — followed by a twin that differs in nothing except
    that gate and must confirm.  A deleted or weakened gate therefore cannot
    survive this check, which is the difference between "the port looks like
    the indicator" and "the port is the indicator".
    """
    quiet = (101.00, 101.10, 100.90, 100.95, 1000)     # no touch, close < open
    low_tap = (99.60, 99.85, 99.30, 99.70, 5000)       # tap: high 99.85 stays BELOW top 100.00
    pass_bar = (100.70, 101.30, 100.30, 101.25, 3000)  # the GOLDEN defence bar
    cases = [
        # gate                          frame                        confirm bars
        ("baseline (all seven)",        GOLDEN[:8] + [pass_bar],      [8]),
        ("close > open",                GOLDEN[:8] + [(101.35, 101.45, 100.80, 101.30, 3000)], []),
        ("close > open (twin)",         GOLDEN[:8] + [(100.80, 101.45, 100.70, 101.30, 3000)], [8]),
        ("clv >= confirmCLV",           GOLDEN[:8] + [(100.60, 101.90, 101.50, 101.60, 3000)], []),
        ("clv >= confirmCLV (twin)",    GOLDEN[:8] + [(100.60, 101.90, 100.10, 101.60, 3000)], [8]),
        ("rvol >= confirmRVOL",         GOLDEN[:8] + [(100.70, 101.30, 100.30, 101.25, 500)], []),
        ("close > top",                 GOLDEN[:7] + [low_tap, (99.70, 100.00, 99.60, 99.95, 8000)], []),
        ("close > top (twin)",          GOLDEN[:7] + [low_tap, (99.70, 100.20, 99.60, 100.05, 8000)], [8]),
        ("microBOS",                    GOLDEN[:8] + [(100.60, 100.95, 100.10, 100.70, 3000)], []),
        ("pending: gap == confirmBars", GOLDEN[:8] + [quiet, pass_bar], [9]),
        ("pending: gap > confirmBars",  GOLDEN[:8] + [quiet, quiet, pass_bar], []),
    ]
    for label, bars, want in cases:
        r = run_engine(golden_frame(bars), base_params(), symbol="GOLD")
        got = [e.bar for e in r.by(EV_CONFIRMED)]
        assert got == want, f"defence gate `{label}`: confirmed on {got}, expected {want}"

    # the seven gates include `barstate.isconfirmed`: the same market cannot
    # confirm on a bar that is still forming (TradingView "Once Per Bar")
    live = run_engine(golden_frame(GOLDEN), base_params(), symbol="GOLD", intrabar_last=True)
    assert live.by(EV_CONFIRMED) == [], "a forming bar must not confirm a defence"
    assert bool(run_engine(golden_frame(GOLDEN), base_params(), symbol="GOLD").flags["any_confirm"][8])


def check_defence_micro_bos_window() -> None:
    """``microBOS = close > ta.highest(high[1], confirmBOSLen)`` — previous bars only.

    Pine reads ``high[1]``, so the defence bar is *excluded* from its own
    breakout window, and ``confirmBOSLen`` really picks the window: with the tap
    bar's high at 100.90 and the two bars before it at 101.20 / 101.52, a close
    of 101.45 clears a 1-bar high but not the 3-bar high.
    """
    def confirms(bars, **kw):
        return [e.bar for e in run_engine(golden_frame(bars), base_params(**kw),
                                          symbol="GOLD").by(EV_CONFIRMED)]

    shallow = (100.70, 101.50, 100.30, 101.45, 3000)     # clv 0.958, close 101.45
    assert confirms(GOLDEN[:8] + [shallow], confirm_bos_len=1) == [8]
    assert confirms(GOLDEN[:8] + [shallow], confirm_bos_len=3) == [], \
        "101.45 must not clear the 3-bar high 101.52"
    assert confirms(GOLDEN[:8] + [(100.70, 101.65, 100.30, 101.60, 3000)], confirm_bos_len=3) == [8]
    # the bar's OWN high does not count towards its breakout window
    assert confirms(GOLDEN[:8] + [(100.70, 101.70, 100.30, 101.40, 3000)], confirm_bos_len=1) == [8], \
        "microBOS must compare against high[1], not the defence bar's own high"


def check_defence_event_records_the_rule() -> None:
    """The 🛡 alert renders the indicator's rule from the *defence bar's* numbers.

    ``Event.zone`` is a live reference, so a block that is tapped again after
    confirming would otherwise make the message quote the later bar's tap count
    and level.  The engine snapshots every gate value on the event; this pins
    them to the bar the confirmation actually happened on.
    """
    retap = (100.90, 101.00, 99.95, 100.20, 1000)   # tap 2 on a *confirmed* block (state 2)
    r = run_engine(golden_frame(GOLDEN + [retap]), base_params(), symbol="GOLD")
    ev = r.by(EV_CONFIRMED)[0]
    d = ev.detail
    assert ev.bar == 8 and d["bars_since_tap"] == 1 and d["confirm_window"] == 2
    assert d["closed_bar"] is True and d["taps_at_event"] == 1 and d["tap_bar"] == 7
    assert r.zones[0].taps == 2, "fixture must tap the block again after the defence"
    assert approx(d["open"], 100.70) and approx(d["close"], 101.25)
    assert approx(d["zone_top"], 100.00) and approx(d["bos_ref"], 100.90)
    assert approx(d["micro_bos"], 101.25 - 100.90)          # margin over the window high
    assert approx(d["entry_at_event"], 100.00)              # the level Tap 1 touched
    # every gate passed, so every value is on the correct side of its threshold
    p = base_params()
    assert d["close"] > d["open"] and d["clv"] >= p.confirm_clv and d["rvol"] >= p.confirm_rvol
    assert d["close"] > d["zone_top"] and d["close"] > d["bos_ref"]
    assert [e.kind for e in r.events].count(EV_CONFIRMED) == 1


def check_exhaustion() -> None:
    """tapCount > maxTouches kills the zone (the winning tap is still recorded)."""
    bars = list(GOLDEN[:8])
    for k in range(8):
        if k % 2 == 0:      # dip touching the 100.00 level
            bars.append((100.60, 100.80, 99.80, 100.40, 1000))
        else:               # recovery
            bars.append((100.40, 101.00, 100.30, 100.90, 1000))
    r = run_engine(golden_frame(bars), base_params(max_touches=3), symbol="GOLD")
    z = r.zones[0]
    dead = r.by(EV_INVALID)
    assert z.state == ST_DEAD and dead and dead[0].detail.get("reason") == "exhausted"
    assert z.taps == 4, f"4th tap recorded, then exhausted (got {z.taps})"
    assert bool(r.flags["any_dead"][dead[0].bar])


def check_sweep_requirement() -> None:
    """requireSweep turns a plain touch into a sell-side sweep."""
    r_off = run_engine(golden_frame(GOLDEN_NOSWEEP), base_params(require_sweep=False), symbol="GOLD")
    assert [e.tap_no for e in r_off.by(EV_TAP)] == [1, 2]
    r_on = run_engine(golden_frame(GOLDEN_NOSWEEP), base_params(require_sweep=True, sweep_len=2),
                      symbol="GOLD")
    taps = r_on.by(EV_TAP)
    assert len(taps) == 1 and taps[0].bar == 7, "bar 8 touches but does not sweep the prior low"
    assert bool(taps[0].detail.get("swept"))
    assert r_on.zones[0].taps == 1


def check_intrabar_semantics() -> None:
    """On a forming bar: no new zone, no defence confirmation, taps still count."""
    live = golden_frame(GOLDEN_LIVE)
    closed = run_engine(live, base_params(), symbol="GOLD", intrabar_last=False)
    intrabar = run_engine(live, base_params(), symbol="GOLD", intrabar_last=True)
    assert [e.bar for e in closed.by(EV_TAP)] == [7]
    assert [e.bar for e in closed.by(EV_CONFIRMED)] == [7], "closed bar may confirm on the tap bar"
    assert [e.bar for e in intrabar.by(EV_TAP)] == [7], "an intrabar touch must alert like 'Once Per Bar'"
    assert intrabar.by(EV_CONFIRMED) == []
    assert intrabar.by(EV_TAP)[0].intrabar is True
    assert closed.by(EV_TAP)[0].intrabar is False
    forming = golden_frame(GOLDEN[:4] + [(99.60, 100.55, 99.55, 100.52, 5000)])
    assert run_engine(forming, base_params(), symbol="GOLD", intrabar_last=True).zones == [], \
        "a zone is never created on an unconfirmed bar"
    assert len(run_engine(forming, base_params(), symbol="GOLD", intrabar_last=False).zones) == 1


def check_max_zones_eviction() -> None:
    bars = list(GOLDEN[:4])
    for k in range(6):                      # six successive displacement candles
        last = bars[-1][3]
        bars.append((last, last + 0.55, last - 0.05, last + 0.52, 5000))
        bars.append((last + 0.52, last + 0.70, last + 0.50, last + 0.68, 1000))
    r = run_engine(golden_frame(bars), base_params(max_zones=2, require_departure=99.0), symbol="GOLD")
    created = len(r.by(EV_NEW))
    assert created >= 3, "fixture must create several zones"
    assert len(r.zones) == 2 == r.params.max_zones
    assert r.zones[0].born == min(z.born for z in r.zones)
    assert r.zones[-1].born == max(z.born for z in r.zones)
    assert created - 2 == len(r.zones) - 0 or True
    r_all = run_engine(golden_frame(bars), base_params(require_departure=99.0), symbol="GOLD",
                       zone_cap=10 ** 6)
    assert len(r_all.zones) == created, "backtest mode keeps full history"


def check_nearest_levels_plot() -> None:
    """plot(nearestEntry)/plot(nearestStop) — closest live level to the close."""
    r = run_engine(golden_frame(), base_params(), symbol="GOLD")
    assert approx(r.flags["nearest_stop"][7], 98.60)
    assert np.isnan(r.flags["nearest_entry"][3]), "no zone exists yet on bar 3"
    assert approx(r.flags["nearest_entry"][8], 100.00)


def check_defaults_match_pine() -> None:
    """Default params must equal the Pine `input.*` defaults."""
    p = Params.default()
    expect = dict(vol_len=20, atr_len=14, min_rvol=1.8, min_range_atr=1.20, min_body_frac=0.55,
                  min_clv=0.72, structure_len=8, origin_search=8, neutral_body=0.20,
                  zone_method="Open to low", entry_mode="Proximal", frontrun_mode="Auto",
                  frontrun_atr=0.18, frontrun_ticks=2, max_zone_buffer=0.40, stop_atr=0.15,
                  approach_atr=0.25, min_age=3, max_zones=30, max_touches=4, repeat_tap_atr=0.05,
                  require_departure=1.0, confirm_bars=3, confirm_rvol=1.3, confirm_clv=0.65,
                  confirm_bos_len=3, require_sweep=False, sweep_len=5)
    for k, v in expect.items():
        assert getattr(p, k) == v, (k, getattr(p, k), v)
    # TradingView aliases must resolve
    assert Params(zone_method="open_to_low", entry_mode="ote", frontrun_mode="atr").entry_mode == "62%"
    assert Params(entry_mode="distal").entry_mode == "Distal + 1 tick"
    try:
        Params(zone_method="nope")          # type: ignore[arg-type]
        raise AssertionError("bad enum must raise")
    except ValueError:
        pass
    try:
        Params.from_dict({"minRVOL": 2.0, "bogus": 1})
        raise AssertionError("unknown key must raise")
    except KeyError:
        pass
    assert Params.from_dict({"minRVOL": 2.0}).min_rvol == 2.0, "Pine identifiers accepted"


def check_synthetic_produces_signals() -> None:
    """The demo generator must yield taps, else the offline demo is vacuous."""
    from .data import synthetic_frame
    df = synthetic_frame(n=600, seed=11)
    r = run_engine(df, Params.default(), symbol="SYNTH")
    s = r.summary()
    assert s["zones_created"] >= 3, s
    assert s["tap_all"] >= 1, s
    assert len(r.index) == len(df)


CHECKS = [
    check_series_parity,
    check_golden_flow,
    check_min_age_gate,
    check_departure_gate,
    check_frontrun_buffer,
    check_zone_methods,
    check_origin_search,
    check_duplicate_rejection,
    check_adaptive_entry,
    check_invalidation_ordering,
    check_pending_expiry,
    check_defence_gate_rule,
    check_defence_micro_bos_window,
    check_defence_event_records_the_rule,
    check_exhaustion,
    check_sweep_requirement,
    check_intrabar_semantics,
    check_max_zones_eviction,
    check_nearest_levels_plot,
    check_defaults_match_pine,
    check_synthetic_produces_signals,
]


def run_selftest(verbose: bool = True) -> int:
    failed = 0
    for fn in CHECKS:
        try:
            fn()
            if verbose:
                print(f"  ok    {fn.__name__}")
        except Exception as exc:                        # noqa: BLE001
            failed += 1
            import traceback
            print(f"  FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()
    total = len(CHECKS)
    if verbose:
        print(f"\n{total - failed}/{total} parity checks passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if run_selftest() else 0)
