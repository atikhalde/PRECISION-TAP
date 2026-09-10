"""Precision OB / Tap engine — a bar-exact port of the Pine v6 state machine.

Design notes (see ANALYSIS.md for the full derivation):

* All *series* maths (ATR, RVOL, CLV, rolling structure) is vectorised in
  :mod:`precision_tap.series`; all *stateful* logic (zone creation, duplicate
  rejection, departure, tap counting, adaptive entry, confirmation, invalidation)
  is a single sequential pass, because — exactly as in Pine — each bar reads and
  rewrites the zone arrays.
* Zone order is preserved (oldest first) so that ``max_zones`` eviction drops the
  same zone the indicator drops.
* ``intrabar_last=True`` reproduces TradingView's live bar: ``barstate.isconfirmed``
  is False there, so no zone is *created* and no *defence* is confirmed, while the
  approach/tap/invalidation checks still evaluate against the running high/low.
  This is what makes an intraday "price touched the OB" alert non-repainting in the
  same sense the Pine alert is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .params import ENTRY_DEPTH, Params
from . import series as S

# Zone state, identical to the Pine comment "0 fresh, 1 tapped/pending, 2 confirmed, -1 invalid"
ST_FRESH = 0
ST_TAPPED = 1
ST_CONFIRMED = 2
ST_DEAD = -1

STATE_NAME = {ST_FRESH: "fresh", ST_TAPPED: "tapped", ST_CONFIRMED: "confirmed", ST_DEAD: "dead"}

# Event kinds (the alertcondition list of the indicator)
EV_NEW = "new_ob"
EV_APPROACH = "approach"
EV_TAP = "tap"
EV_CONFIRMED = "confirmed"
EV_INVALID = "invalidated"

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Zone:
    """One precision order block, exactly as stored in the Pine arrays."""

    zid: int
    born: int                    # displacement bar index (`born` array)
    origin: int                  # origin candle bar index (born - originOffset)
    top: float                   # rawTop
    bot: float                   # rawBot
    entry: float                 # live pre-order level (mutated after Tap 1)
    entry0: float                # entry at creation
    stop: float
    atr0: float                  # ATR on the creation bar
    state: int = ST_FRESH
    taps: int = 0
    tap_bars: List[int] = field(default_factory=list)
    departed: bool = False
    adaptive: bool = False
    dead_bar: int = -1
    dead_reason: str = ""
    confirm_bar: int = -1
    approach_bars: List[int] = field(default_factory=list)
    approach_prev: bool = False          # for edge-triggered pre-alerts

    @property
    def width(self) -> float:
        return self.top - self.bot

    @property
    def risk(self) -> float:
        return self.entry - self.stop

    @property
    def state_name(self) -> str:
        return STATE_NAME[self.state]

    def snapshot(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state_name"] = self.state_name
        d["width"] = self.width
        d["risk"] = self.risk
        return d


@dataclass
class Event:
    """A discrete signal produced by the engine (maps 1:1 to alertcondition/plotshape)."""

    kind: str
    bar: int
    symbol: str
    zid: int
    tap_no: int = 0
    level: float = float("nan")          # the price the event is about (entry / stop / top)
    price: float = float("nan")          # market price at trigger (close for closed bars)
    ts: Optional[pd.Timestamp] = None
    intrabar: bool = False
    zone: Optional[Zone] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    # ── convenience ──────────────────────────────────────────────────────
    @property
    def is_tap1(self) -> bool:
        return self.kind == EV_TAP and self.tap_no == 1

    def key(self) -> str:
        day = self.ts.date().isoformat() if self.ts is not None else str(self.bar)
        return f"{self.symbol}|{self.kind}|{self.zid}|{day}"

    def to_dict(self) -> Dict[str, Any]:
        z = self.zone
        out = {
            "event": self.kind,
            "symbol": self.symbol,
            "bar": self.bar,
            "date": self.ts.date().isoformat() if self.ts is not None else "",
            "datetime": str(self.ts) if self.ts is not None else "",
            "zone_id": self.zid,
            "tap_no": self.tap_no,
            "level": self.level,
            "price": self.price,
            "intrabar": self.intrabar,
            "zone_top": z.top if z else float("nan"),
            "zone_bot": z.bot if z else float("nan"),
            "entry": z.entry if z else float("nan"),
            "entry_at_event": self.level if self.kind in (EV_TAP, EV_APPROACH) else (z.entry if z else float("nan")),
            "stop": z.stop if z else float("nan"),
            "zone_born": z.born if z else -1,
            "origin_bar": z.origin if z else -1,
            "age": (self.bar - z.born) if z else -1,
            "taps_total": z.taps if z else 0,
            "zone_state": z.state_name if z else "",
        }
        out.update(self.detail)
        return out

    def fmt(self, decimals: int = 4) -> str:
        f = lambda v: "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{decimals}f}"
        z = self.zone
        base = f"{self.kind.upper():<11} {self.symbol:<8} bar{self.bar:>6}"
        if self.kind == EV_NEW and z is not None:
            return (base + f" OB[{z.zid}] zone {f(z.top)}/{f(z.bot)} entry {f(z.entry)} stop {f(z.stop)}")
        if self.kind == EV_TAP and z is not None:
            return (base + f" TAP {self.tap_no} @ {f(self.level)} stop {f(z.stop)} "
                           f"(entry {f(self.level)} → risk {f(z.risk)})")
        if self.kind == EV_APPROACH and z is not None:
            return base + f" approaching {f(self.level)} (price {f(self.price)})"
        if self.kind == EV_CONFIRMED and z is not None:
            return base + f" DEFENCE above {f(z.top)}"
        if self.kind == EV_INVALID and z is not None:
            return base + f" DEAD ({self.detail.get('reason', '')})"
        return base


@dataclass
class EngineResult:
    """Everything one symbol's pass produced."""

    symbol: str
    params: Params
    zones: List[Zone]
    events: List[Event]
    arrays: Dict[str, np.ndarray]
    flags: Dict[str, np.ndarray]        # any_approach / any_tap / any_confirm / any_dead / nearest_*
    index: pd.Index
    skipped: str = ""

    # ── queries ──────────────────────────────────────────────────────────
    def by(self, *kinds: str) -> List[Event]:
        k = set(kinds)
        return [e for e in self.events if e.kind in k]

    @property
    def tap1(self) -> List[Event]:
        return [e for e in self.events if e.kind == EV_TAP and e.tap_no == 1]

    @property
    def live_zones(self) -> List[Zone]:
        return [z for z in self.zones if z.state >= 0]

    def last_event(self, *kinds: str) -> Optional[Event]:
        evs = self.by(*kinds) if kinds else self.events
        return evs[-1] if evs else None

    def summary(self) -> Dict[str, int]:
        return {
            "zones_created": len(self.zones),
            "zones_live": len(self.live_zones),
            "new_ob": len(self.by(EV_NEW)),
            "approach": len(self.by(EV_APPROACH)),
            "tap1": len(self.tap1),
            "tap_all": len(self.by(EV_TAP)),
            "confirmed": len(self.by(EV_CONFIRMED)),
            "invalidated": len(self.by(EV_INVALID)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Indicator maths shared by the engine and the plotter
# ─────────────────────────────────────────────────────────────────────────────

def compute_arrays(df: pd.DataFrame, p: Params) -> Dict[str, np.ndarray]:
    """Vectorised half of the Pine script: the ``Detection`` group."""
    o = S.as_float(df["open"].to_numpy())
    h = S.as_float(df["high"].to_numpy())
    l = S.as_float(df["low"].to_numpy())
    c = S.as_float(df["close"].to_numpy())
    v = S.as_float(df["volume"].to_numpy())
    tick = p.mintick

    atr = S.atr(h, l, c, p.atr_len)
    vol_ma = S.sma(v, p.vol_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        rvol = np.where(vol_ma > 0, v / vol_ma, 0.0)
    rng = np.maximum(h - l, tick)
    body = np.abs(c - o)
    clv = (c - l) / rng
    body_frac = body / rng
    up = c > o

    prior_structure = S.highest(S.shift(h, 1), p.structure_len)      # ta.highest(high[1], structureLen)
    prior_low = S.lowest(S.shift(l, 1), p.sweep_len)               # ta.lowest(low[1], sweepLen)
    bos_ref = S.highest(S.shift(h, 1), p.confirm_bos_len)          # ta.highest(high[1], confirmBOSLen)

    with np.errstate(invalid="ignore"):
        displacement = (
            up
            & (rvol >= p.min_rvol)
            & np.isfinite(atr)
            & (rng >= atr * p.min_range_atr)
            & (body_frac >= p.min_body_frac)
            & (clv >= p.min_clv)
            & np.isfinite(prior_structure)
            & (c > prior_structure)
        )
    displacement = np.asarray(displacement, dtype=bool)
    # Pine: ATR / volume SMA are still `na` during warm-up, so no signal can exist
    # before bar max(atr_len, vol_len) - 1.
    displacement[: max(p.atr_len, p.vol_len) - 1] &= False

    # origin-candle scan, per displacement bar (bearish first, small neutral allowed)
    origin_off = np.full(c.size, -1, dtype="int64")
    for i in np.flatnonzero(displacement):
        for j in range(1, p.origin_search + 1):
            k = i - j
            if k < 0:
                break
            crank = max(h[k] - l[k], tick)
            bearish = c[k] < o[k]
            neutral = p.allow_neutral_origin and (abs(c[k] - o[k]) / crank) <= p.neutral_body
            if bearish or neutral:
                origin_off[i] = j
                break
        if origin_off[i] < 0:          # fallback: immediately preceding candle
            origin_off[i] = 1 if i >= 1 else -1
        if origin_off[i] < 0:
            displacement[i] = False

    raw_top = np.full(c.size, np.nan)
    raw_bot = np.full(c.size, np.nan)
    ok = displacement & (origin_off > 0)
    for i in np.flatnonzero(ok):
        k = i - int(origin_off[i])
        o_, c_, h_, l_ = o[k], c[k], h[k], l[k]
        if p.zone_method == "Open to low":
            t, b = o_, l_
        elif p.zone_method == "Body to low":
            t, b = max(o_, c_), l_
        elif p.zone_method == "Body only":
            t, b = max(o_, c_), min(o_, c_)
        else:  # "Lower half of candle"
            t, b = l_ + (h_ - l_) * 0.50, l_
        raw_top[i], raw_bot[i] = t, b

    return {
        "open": o, "high": h, "low": l, "close": c, "volume": v,
        "atr": atr, "vol_ma": vol_ma, "rvol": rvol, "rng": rng, "body": body,
        "body_frac": body_frac, "clv": clv, "up": up,
        "prior_structure": prior_structure, "prior_low": prior_low, "bos_ref": bos_ref,
        "displacement": displacement, "origin_off": origin_off,
        "raw_top": raw_top, "raw_bot": raw_bot,
    }


def front_buffer(width: float, atr_now: float, p: Params) -> float:
    """``f_front_buffer`` — the deliberate near-miss / front-run offset."""
    tick_buffer = p.frontrun_ticks * p.mintick
    atr_buffer = p.frontrun_atr * (atr_now if math.isfinite(atr_now) else 0.0)
    if p.frontrun_mode == "ATR":
        return atr_buffer
    if p.frontrun_mode == "Ticks":
        return tick_buffer
    if p.frontrun_mode == "Off":
        return 0.0
    return min(max(tick_buffer, atr_buffer), width * p.max_zone_buffer)


def entry_level(top: float, bot: float, atr_now: float, p: Params) -> float:
    """``entry`` for a zone — proximal edge minus depth fraction plus front-run buffer."""
    if p.entry_mode == "Distal + 1 tick":
        return bot + p.mintick
    width = top - bot
    depth = ENTRY_DEPTH[p.entry_mode]
    base = top - width * depth
    return base + front_buffer(width, atr_now, p) + p.entry_offset_ticks * p.mintick


# ─────────────────────────────────────────────────────────────────────────────
# The sequential state machine
# ─────────────────────────────────────────────────────────────────────────────

def run_engine(
    df: pd.DataFrame,
    p: Optional[Params] = None,
    *,
    symbol: str = "",
    intrabar_last: bool = False,
    start_bar: int = 0,
    zone_cap: Optional[int] = None,
) -> EngineResult:
    """Replay the indicator over ``df`` and return zones + events.

    Parameters
    ----------
    df : DataFrame with ``open/high/low/close/volume`` (DatetimeIndex preferred).
    intrabar_last : treat the final row as the still-forming bar.
    start_bar : first bar index to evaluate (earlier bars only warm the state).
    zone_cap : override ``params.max_zones`` (used by the backtester to keep
        history for long runs without changing any signal).
    """
    p = p or Params.default()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    A = compute_arrays(df, p)
    o, h, l, c = A["open"], A["high"], A["low"], A["close"]
    atr, rvol, clv = A["atr"], A["rvol"], A["clv"]
    displacement, raw_top, raw_bot = A["displacement"], A["raw_top"], A["raw_bot"]
    prior_low, bos_ref = A["prior_low"], A["bos_ref"]
    origin_off = A["origin_off"]

    n = len(df)
    max_zones = int(zone_cap or p.max_zones)
    idx = df.index

    zones: List[Zone] = []
    events: List[Event] = []
    zid_seq = 0
    n_created = 0

    any_approach = np.zeros(n, dtype=bool)
    any_tap = np.zeros(n, dtype=bool)
    any_confirm = np.zeros(n, dtype=bool)
    any_dead = np.zeros(n, dtype=bool)
    nearest_entry = np.full(n, np.nan)
    nearest_stop = np.full(n, np.nan)

    def ts(i: int) -> Optional[pd.Timestamp]:
        try:
            t = idx[i]
            return t if isinstance(t, pd.Timestamp) else pd.Timestamp(t)
        except Exception:
            return None

    def emit(kind: str, i: int, z: Zone, tap_no: int = 0, level=float("nan"),
             price=float("nan"), live=False, **detail) -> Event:
        ev = Event(kind=kind, bar=i, symbol=symbol, zid=z.zid, tap_no=tap_no, level=level,
                   price=c[i] if math.isnan(price) else price, ts=ts(i), intrabar=live,
                   zone=z, detail=detail)
        events.append(ev)
        return ev

    for i in range(n):
        is_live = bool(intrabar_last and i == n - 1)     # barstate.isconfirmed == False
        confirmed = not is_live
        ai = atr[i]
        ai = ai if math.isfinite(ai) else 0.0

        # ── new zone (closed bars only — the non-repainting half) ─────────
        if displacement[i] and confirmed:
            top, bot = float(raw_top[i]), float(raw_bot[i])
            if math.isfinite(top) and math.isfinite(bot) and top > bot:
                cand = entry_level(top, bot, ai, p)
                duplicate = False
                for z in zones:                       # reject entry inside a live zone
                    if z.state >= 0 and cand <= z.top and cand >= z.bot:
                        duplicate = True
                        break
                if not duplicate:
                    z = Zone(
                        zid=zid_seq, born=i, origin=i - int(origin_off[i]),
                        top=top, bot=bot, entry=cand, entry0=cand,
                        stop=bot - ai * p.stop_atr, atr0=ai,
                    )
                    zid_seq += 1
                    n_created += 1
                    zones.append(z)
                    emit(EV_NEW, i, z, level=cand,
                         zone_width=top - bot, origin_offset=int(origin_off[i]),
                         rvol=float(rvol[i]), atr=ai)
                else:
                    A["displacement"][i] = False       # keep plotshape == creation parity
            else:
                A["displacement"][i] = False

        # ── max_zones eviction (Pine shifts the oldest) ───────────────────
        while len(zones) > max_zones:
            zones.pop(0)

        # ── per-zone live evaluation ─────────────────────────────────────
        ne, ns = math.nan, math.nan
        if zones:
            close_i = c[i]
            open_i = o[i]
            low_i, high_i = l[i], h[i]
            rvol_i = float(rvol[i])
            clv_i = float(clv[i])
            plow = float(prior_low[i]) if math.isfinite(prior_low[i]) else math.nan
            bosv = float(bos_ref[i]) if math.isfinite(bos_ref[i]) else math.nan
            for z in zones:
                if z.state < 0:
                    continue
                age = i - z.born
                # departure: price must have left the zone before it counts as a tap target
                if (not z.departed) and high_i >= z.top + ai * p.require_departure:
                    z.departed = True
                if math.isnan(ne) or abs(close_i - z.entry) < abs(close_i - ne):
                    ne, ns = z.entry, z.stop

                if z.departed and age >= p.min_age:
                    # pre-alert: price still above the level but inside the buffer
                    now_approach = bool(close_i > z.entry and low_i <= z.entry + ai * p.approach_atr)
                    if now_approach:
                        any_approach[i] = True           # Pine `alertcondition(approaching)`
                        z.approach_bars.append(i)
                        if not z.approach_prev:          # edge-triggered event (no spam)
                            emit(EV_APPROACH, i, z, level=z.entry, price=close_i, live=is_live,
                                 distance_atr=(close_i - z.entry) / ai if ai else math.nan)
                    z.approach_prev = now_approach

                    touched = low_i <= z.entry and high_i >= z.bot
                    if p.require_sweep:
                        swept = math.isfinite(plow) and low_i < plow
                        touched = touched and swept
                    if touched and (not z.tap_bars or z.tap_bars[-1] != i):
                        z.taps += 1
                        z.tap_bars.append(i)
                        z.state = ST_TAPPED
                        tap_level = z.entry
                        if p.raise_after_first_tap and z.taps == 1:
                            adaptive = max(z.entry, low_i + ai * p.repeat_tap_atr)
                            if adaptive > z.entry:
                                z.entry = adaptive
                                z.adaptive = True
                        any_tap[i] = True
                        emit(EV_TAP, i, z, tap_no=z.taps, level=tap_level, price=low_i,
                             live=is_live, swept=(math.isfinite(plow) and low_i < plow),
                             atr=ai, rvol=rvol_i, touched_bot=z.bot,
                             adaptive_entry=z.entry if z.adaptive else float("nan"))

                # defence confirmation (closed bars only)
                pending = z.state == ST_TAPPED and bool(z.tap_bars) and (i - z.tap_bars[-1]) <= p.confirm_bars
                if confirmed and pending and close_i > open_i and clv_i >= p.confirm_clv \
                        and rvol_i >= p.confirm_rvol and close_i > z.top \
                        and (math.isfinite(bosv) and close_i > bosv):
                    z.state = ST_CONFIRMED
                    z.confirm_bar = i
                    any_confirm[i] = True
                    emit(EV_CONFIRMED, i, z, level=z.top, price=close_i,
                         rvol=rvol_i, clv=clv_i, micro_bos=float(close_i - bosv))
                elif z.state == ST_TAPPED and z.tap_bars and (i - z.tap_bars[-1]) > p.confirm_bars:
                    z.state = ST_FRESH          # pending window expired -> back to fresh

                if close_i < z.stop or z.taps > p.max_touches:
                    z.state = ST_DEAD
                    z.dead_bar = i
                    z.dead_reason = "close_below_stop" if close_i < z.stop else "exhausted"
                    any_dead[i] = True
                    emit(EV_INVALID, i, z, level=z.stop, price=close_i, reason=z.dead_reason,
                         last_tap_bar=(z.tap_bars[-1] if z.tap_bars else -1))

        nearest_entry[i], nearest_stop[i] = ne, ns

    flags = {
        "any_approach": any_approach, "any_tap": any_tap, "any_confirm": any_confirm,
        "any_dead": any_dead, "nearest_entry": nearest_entry, "nearest_stop": nearest_stop,
    }
    return EngineResult(symbol=symbol, params=p, zones=zones, events=events, arrays=A,
                        flags=flags, index=idx)


def run_symbol(df: pd.DataFrame, p: Params, symbol: str = "", **kw) -> EngineResult:
    """Thin wrapper used by the scanner/backtest workers."""
    return run_engine(df, p, symbol=symbol, **kw)
