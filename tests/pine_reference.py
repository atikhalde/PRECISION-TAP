"""A deliberately independent transcription of ``INDICATOR.txt`` (Pine v6).

Nothing in here imports :mod:`precision_tap.engine` or :mod:`precision_tap.series`.
The rolling primitives are re-written as naive ``O(n·len)`` loops straight from
TradingView's documented ``ta.*`` semantics, and the state machine is a
line-by-line transliteration of the Pine source — parallel arrays, the same
statement order, the same ``na`` behaviour.

The point of the duplication is the diff: when the vectorised engine disagrees
with the indicator, this file disagrees with the engine, and the test in
``test_pine_reference.py`` prints exactly which bar and which field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NA = float("nan")


def _na(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ─────────────────────────────────────────────────────────────────────────────
# ta.* primitives (documented TradingView warm-up semantics)
# ─────────────────────────────────────────────────────────────────────────────

def ta_sma(x: Sequence[float], length: int) -> List[float]:
    """``ta.sma`` — na until ``length`` observations exist, then a plain mean."""
    out: List[float] = [NA] * len(x)
    for i in range(len(x)):
        if i + 1 < length:
            continue
        win = list(x[i - length + 1: i + 1])
        if any(_na(v) for v in win):
            continue
        out[i] = sum(win) / length
    return out


def ta_rma(x: Sequence[float], length: int) -> List[float]:
    """``ta.rma`` — Wilder recursion seeded with the SMA of the first ``length``."""
    out: List[float] = [NA] * len(x)
    if len(x) < length or length < 1:
        return out
    seed = list(x[:length])
    out[length - 1] = NA if any(_na(v) for v in seed) else sum(seed) / length
    alpha = 1.0 / length
    for i in range(length, len(x)):
        prev, v = out[i - 1], x[i]
        if _na(prev) or _na(v):
            out[i] = prev
            continue
        out[i] = alpha * v + (1.0 - alpha) * prev
    return out


def ta_true_range(h: Sequence[float], l: Sequence[float], c: Sequence[float]) -> List[float]:
    """``ta.tr`` — on bar 0 ``close[1]`` is na, so the range is ``high - low``."""
    out: List[float] = []
    for i in range(len(h)):
        if i == 0:
            out.append(h[i] - l[i])
        else:
            out.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return out


def ta_atr(h, l, c, length: int) -> List[float]:
    return ta_rma(ta_true_range(h, l, c), length)


def _extreme(x: Sequence[float], length: int, want_max: bool) -> List[float]:
    """``ta.highest`` / ``ta.lowest`` over *available* bars during warm-up."""
    out: List[float] = [NA] * len(x)
    for i in range(len(x)):
        win = [v for v in x[max(0, i - length + 1): i + 1] if not _na(v)]
        if not win:
            continue
        out[i] = max(win) if want_max else min(win)
    return out


def ta_highest(x: Sequence[float], length: int) -> List[float]:
    return _extreme(x, length, True)


def ta_lowest(x: Sequence[float], length: int) -> List[float]:
    return _extreme(x, length, False)


def shifted(x: Sequence[float], n: int) -> List[float]:
    """Pine ``x[n]`` — na where the history does not reach back that far."""
    out = [NA] * len(x)
    for i in range(n, len(x)):
        out[i] = x[i - n]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# outputs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RefZone:
    zid: int
    born: int
    origin: int
    top: float
    bot: float
    entry: float
    entry0: float
    stop: float
    state: int = 0
    taps: int = 0
    tap_bars: List[int] = field(default_factory=list)
    departed: bool = False
    adaptive: bool = False


@dataclass
class RefResult:
    symbol: str
    zones: List[RefZone]
    events: List[Tuple[str, int, int, int, float]]     # kind, bar, zid, tap_no, level
    displacement_plot: List[bool]                        # plotshape(displacement and not duplicate)
    any_approach: List[bool]
    any_tap: List[bool]
    any_confirm: List[bool]
    any_dead: List[bool]
    nearest_entry: List[float]
    nearest_stop: List[float]

    def zone_table(self) -> List[Tuple]:
        """Comparable snapshot of the zone arrays at the end of the run."""
        return [(z.zid, z.born, z.origin, round(z.top, 9), round(z.bot, 9),
                 round(z.entry, 9), round(z.stop, 9), z.state, z.taps,
                 z.tap_bars[-1] if z.tap_bars else -1, z.departed) for z in self.zones]


DEPTH = {"Proximal": 0.0, "50%": 0.50, "62%": 0.62, "70.5%": 0.705,
         "79%": 0.79, "Distal + 1 tick": 1.0}


def run_reference(df: pd.DataFrame, p, *, symbol: str = "",
                  intrabar_last: bool = False) -> RefResult:
    """Execute the Pine script bar by bar over ``df``."""
    o = [float(v) for v in df["open"].to_numpy()]
    h = [float(v) for v in df["high"].to_numpy()]
    l = [float(v) for v in df["low"].to_numpy()]
    c = [float(v) for v in df["close"].to_numpy()]
    v = [float(v) for v in df["volume"].to_numpy()]
    n = len(df)
    tick = float(p.mintick)

    # ── Detection group ────────────────────────────────────────────────────
    atr = ta_atr(h, l, c, p.atr_len)
    vol_ma = ta_sma(v, p.vol_len)
    rvol = [0.0 if _na(m) or m <= 0 else v[i] / m for i, m in enumerate(vol_ma)]
    rng = [max(h[i] - l[i], tick) for i in range(n)]
    body = [abs(c[i] - o[i]) for i in range(n)]
    clv = [NA if rng[i] == 0 else (c[i] - l[i]) / rng[i] for i in range(n)]
    prior_structure = ta_highest(shifted(h, 1), p.structure_len)
    prior_low = ta_lowest(shifted(l, 1), p.sweep_len)
    bos_ref = ta_highest(shifted(h, 1), p.confirm_bos_len)

    displacement_plot: List[bool] = [False] * n
    any_approach: List[bool] = [False] * n
    any_tap: List[bool] = [False] * n
    any_confirm: List[bool] = [False] * n
    any_dead: List[bool] = [False] * n
    nearest_entry: List[float] = [NA] * n
    nearest_stop: List[float] = [NA] * n

    # Pine's `var` arrays
    tops: List[float] = []
    bots: List[float] = []
    entries: List[float] = []
    stops: List[float] = []
    born: List[int] = []
    states: List[int] = []
    taps: List[int] = []
    tap_bars: List[int] = []
    departed: List[bool] = []
    zids: List[int] = []
    zobjs: List[RefZone] = []
    events: List[Tuple[str, int, int, int, float]] = []
    zid_seq = 0

    def front_buffer(width: float, atr_now: float) -> float:
        tick_buffer = p.frontrun_ticks * tick
        atr_buffer = p.frontrun_atr * (0.0 if _na(atr_now) else atr_now)
        if p.frontrun_mode == "ATR":
            return atr_buffer
        if p.frontrun_mode == "Ticks":
            return tick_buffer
        if p.frontrun_mode == "Off":
            return 0.0
        return min(max(tick_buffer, atr_buffer), width * p.max_zone_buffer)

    for i in range(n):
        confirmed = not (intrabar_last and i == n - 1)      # barstate.isconfirmed
        ai = atr[i] if not _na(atr[i]) else 0.0

        # ── displacement ───────────────────────────────────────────────────
        displacement = (
            confirmed
            and c[i] > o[i]
            and rvol[i] >= p.min_rvol
            and not _na(ai) and rng[i] >= ai * p.min_range_atr
            and body[i] / rng[i] >= p.min_body_frac
            and not _na(clv[i]) and clv[i] >= p.min_clv
            and not _na(prior_structure[i]) and c[i] > prior_structure[i]
        )

        # ── origin candle ──────────────────────────────────────────────────
        origin_offset: Optional[int] = None
        if displacement:
            for j in range(1, p.origin_search + 1):
                k = i - j
                if k < 0:
                    break
                crank = max(h[k] - l[k], tick)
                bearish = c[k] < o[k]
                neutral = p.allow_neutral_origin and (abs(c[k] - o[k]) / crank) <= p.neutral_body
                if origin_offset is None and (bearish or neutral):
                    origin_offset = j
            if origin_offset is None:
                origin_offset = 1
        raw_top = raw_bot = NA
        if displacement and origin_offset is not None:
            k = i - origin_offset
            if k >= 0:
                if p.zone_method == "Open to low":
                    raw_top, raw_bot = o[k], l[k]
                elif p.zone_method == "Body to low":
                    raw_top, raw_bot = max(o[k], c[k]), l[k]
                elif p.zone_method == "Body only":
                    raw_top, raw_bot = max(o[k], c[k]), min(o[k], c[k])
                else:
                    raw_top, raw_bot = l[k] + (h[k] - l[k]) * 0.50, l[k]

        # ── duplicate veto ─────────────────────────────────────────────────
        duplicate = False
        if displacement and not _na(raw_top) and raw_top > raw_bot and len(tops) > 0:
            width = raw_top - raw_bot
            cand = raw_top - width * DEPTH[p.entry_mode]
            cand = cand + front_buffer(width, ai) + p.entry_offset_ticks * tick
            if p.entry_mode == "Distal + 1 tick":
                cand = raw_bot + tick
            for zi in range(len(tops)):
                if states[zi] >= 0 and cand <= tops[zi] and cand >= bots[zi]:
                    duplicate = True

        # ── create the zone ────────────────────────────────────────────────
        if displacement and not duplicate and not _na(raw_top) and raw_top > raw_bot:
            width = raw_top - raw_bot
            entry = raw_top - width * DEPTH[p.entry_mode]
            entry = entry + front_buffer(width, ai) + p.entry_offset_ticks * tick
            if p.entry_mode == "Distal + 1 tick":
                entry = raw_bot + tick
            stop = raw_bot - ai * p.stop_atr
            tops.append(raw_top)
            bots.append(raw_bot)
            entries.append(entry)
            stops.append(stop)
            born.append(i)
            states.append(0)
            taps.append(0)
            tap_bars.append(-1)
            departed.append(False)
            zids.append(zid_seq)
            zobjs.append(RefZone(zid=zid_seq, born=i, origin=i - origin_offset, top=raw_top,
                                 bot=raw_bot, entry=entry, entry0=entry, stop=stop))
            events.append(("new_ob", i, zid_seq, 0, entry))
            zid_seq += 1

        displacement_plot[i] = bool(displacement and not duplicate)

        # ── maxZones eviction (Pine shifts exactly one per bar) ────────────
        if len(tops) > p.max_zones:
            for arr in (tops, bots, entries, stops, born, states, taps, tap_bars,
                        departed, zids, zobjs):
                arr.pop(0)

        # ── per-zone live evaluation ───────────────────────────────────────
        ne, ns = NA, NA
        for zi in range(len(tops)):
            state = states[zi]
            if state < 0:
                continue
            top, bot, entry, stop = tops[zi], bots[zi], entries[zi], stops[zi]
            age = i - born[zi]

            if not departed[zi] and h[i] >= top + ai * p.require_departure:
                departed[zi] = True

            if _na(ne) or abs(c[i] - entry) < abs(c[i] - ne):
                ne, ns = entry, stop

            approaching = (departed[zi] and age >= p.min_age and c[i] > entry
                           and l[i] <= entry + ai * p.approach_atr)
            if approaching:
                any_approach[i] = True

            touched = departed[zi] and age >= p.min_age and l[i] <= entry and h[i] >= bot
            swept = (not _na(prior_low[i])) and l[i] < prior_low[i]
            qualified = touched and (not p.require_sweep or swept)
            if qualified and (tap_bars[zi] < 0 or i > tap_bars[zi]):
                taps[zi] += 1
                tap_bars[zi] = i
                states[zi] = 1
                state = 1
                tap_level = entry
                if p.raise_after_first_tap and taps[zi] == 1:
                    entries[zi] = max(entry, l[i] + ai * p.repeat_tap_atr)
                    zobjs[zi].adaptive = entries[zi] > entry
                    entry = entries[zi]
                any_tap[i] = True
                events.append(("tap", i, zids[zi], taps[zi], tap_level))

            pending = state == 1 and tap_bars[zi] >= 0 and (i - tap_bars[zi]) <= p.confirm_bars
            micro_bos = (not _na(bos_ref[i])) and c[i] > bos_ref[i]
            defence = (confirmed and pending and c[i] > o[i] and not _na(clv[i])
                       and clv[i] >= p.confirm_clv and rvol[i] >= p.confirm_rvol
                       and c[i] > top and micro_bos)
            if defence:
                states[zi] = 2
                state = 2
                any_confirm[i] = True
                events.append(("confirmed", i, zids[zi], 0, top))

            if state == 1 and (i - tap_bars[zi]) > p.confirm_bars:
                states[zi] = 0
                state = 0

            if state >= 0 and (c[i] < stop or taps[zi] > p.max_touches):
                states[zi] = -1
                any_dead[i] = True
                events.append(("invalidated", i, zids[zi], 0, stop))

        nearest_entry[i], nearest_stop[i] = ne, ns

    # fold the surviving Pine arrays back into the zone objects so the caller
    # can compare them against engine.EngineResult.zones
    for zi, z in enumerate(zobjs):
        z.entry, z.state, z.taps = entries[zi], states[zi], taps[zi]
        z.tap_bars = [tap_bars[zi]] if tap_bars[zi] >= 0 else []
        z.departed = departed[zi]

    return RefResult(symbol=symbol, zones=zobjs, events=events,
                     displacement_plot=displacement_plot, any_approach=any_approach,
                     any_tap=any_tap, any_confirm=any_confirm, any_dead=any_dead,
                     nearest_entry=nearest_entry, nearest_stop=nearest_stop)


def zones_by_zid(zones: Sequence[RefZone], zid: int) -> RefZone:
    for z in zones:
        if z.zid == zid:
            return z
    raise KeyError(zid)
