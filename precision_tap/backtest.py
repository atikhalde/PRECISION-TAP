"""Backtest engine for the Precision Tap signal set (daily bars).

Two phases, because they answer different questions.

**Phase A — per-symbol signal replay.** The exact engine state machine is replayed
over history (identical to a closed-bar TradingView alert, so results are
non-repainting). Every traded event becomes a *position* simulated against the real
daily OHLC with explicit, conservative fill assumptions:

* a Tap-1 limit order fills at ``min(bar_open, level)`` — a gap through the level
  fills you at the open (the better price for a buy);
* if a bar touches both stop and target, the **stop is filled first** (worst path);
* ``stop_mode = "close"`` mirrors the indicator's own invalidation rule
  (``close < stop``) instead of a wick fill;
* break-even / chandelier / structure trailing, time stop and structural exits are
  ours to configure, since the indicator defines an entry and a stop but no target.

**Phase B — portfolio accounting** (see :mod:`precision_tap.metrics`): trades are
replayed chronologically with compounding cash, ``risk_per_trade`` sizing and an
optional ``max_open_positions`` cap → equity curve + money metrics. Unconstrained
R-stats are reported alongside, because *that* is the honest read on the signal.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .engine import EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, EngineResult, Params, Zone, run_engine
from .params import ScanConfig, TradeConfig

log = logging.getLogger("precision_tap.backtest")

HORIZONS = (1, 2, 3, 5, 10, 20)


@dataclass
class Trade:
    symbol: str
    side: str = "long"
    trigger: str = "tap1"
    tap_no: int = 1
    zone_id: int = -1
    entry_bar: int = -1
    exit_bar: int = -1
    entry_date: Optional[pd.Timestamp] = None
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = float("nan")
    stop_price: float = float("nan")       # stop in force at the exit
    initial_stop: float = float("nan")
    target_price: float = float("nan")
    exit_price: float = float("nan")
    risk_per_share: float = float("nan")
    r_multiple: float = float("nan")
    ret_pct: float = float("nan")          # gross price return
    net_ret_pct: float = float("nan")      # after commission/slippage/spread
    bars_held: int = 0
    exit_reason: str = ""
    mae_r: float = float("nan")
    mfe_r: float = float("nan")
    atr_at_entry: float = float("nan")
    confirmed: bool = False                # zone printed a defence before the exit
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    shares: float = 0.0
    notional: float = 0.0
    skipped_by_cap: bool = False

    @property
    def won(self) -> bool:
        return bool(self.r_multiple > 0) if math.isfinite(self.r_multiple) else False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entry_date"] = str(self.entry_date.date()) if self.entry_date is not None else ""
        d["exit_date"] = str(self.exit_date.date()) if self.exit_date is not None else ""
        d["won"] = self.won
        return d


@dataclass
class BacktestResult:
    config: Optional[ScanConfig] = None
    trades: List[Trade] = field(default_factory=list)
    equity: Optional[pd.Series] = None
    r_equity: Optional[pd.Series] = None
    per_symbol: Optional[pd.DataFrame] = None
    per_year: Optional[pd.DataFrame] = None
    exit_table: Optional[pd.DataFrame] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    signal_study: Optional[pd.DataFrame] = None      # stats table (forward returns, MFE/MAE)
    signal_events: Optional[pd.DataFrame] = None     # one row per Tap 1
    events: List[Dict[str, Any]] = field(default_factory=list)
    tap_events: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)
    run_at: str = ""
    universe: int = 0
    bars: int = 0
    span: str = ""

    @property
    def n_trades(self) -> int:
        return len(self.trades)


class Backtester:
    def __init__(self, cfg: ScanConfig, *, progress: bool = True):
        self.cfg = cfg
        self.p: Params = cfg.params
        self.tc: TradeConfig = cfg.trade
        self.progress = progress

    # ── public ───────────────────────────────────────────────────────────
    def run(self, frames: Dict[str, pd.DataFrame], *, start: Optional[str] = None,
            end: Optional[str] = None, collect_events: bool = False) -> BacktestResult:
        """``frames``: {symbol: OHLCV DataFrame} on the daily timeframe."""
        from .metrics import attach_portfolio, metrics_for
        res = BacktestResult(config=self.cfg, run_at=datetime.now().isoformat(timespec="seconds"),
                             universe=len(frames))
        t0 = time.time()
        done = 0
        for sym, df in sorted(frames.items()):
            done += 1
            if df is None or len(df) < max(self.p.warmup, 40):
                continue
            try:
                trades, events, taps = self._run_symbol(sym, df, start, end)
            except Exception as exc:                    # one bad symbol must not kill the run
                log.debug("backtest failed for %s: %s", sym, exc)
                res.errors.append(f"{sym}: {exc}")
                continue
            res.trades.extend(trades)
            res.tap_events[sym] = taps
            if collect_events:
                res.events.extend(events)
            res.bars += len(df)
            if self.progress and (done % 50 == 0 or done == len(frames)):
                log.info("backtest %d/%d symbols · %d trades", done, len(frames), len(res.trades))
        res.trades.sort(key=lambda t: ((t.entry_date or pd.Timestamp.min), t.symbol))
        attach_portfolio(res, self.tc)
        res.metrics = metrics_for(res, self.cfg)
        stats, raw = signal_study(res.tap_events, frames, self.p, start=start, end=end)
        res.signal_study, res.signal_events = stats, raw
        all_dates = [d for t in res.trades for d in (t.entry_date, t.exit_date) if d is not None]
        if all_dates:
            res.span = f"{min(all_dates).date()} → {max(all_dates).date()}"
        log.info("backtest done: %d trades over %s in %.1fs", res.n_trades, res.span or "n/a",
                 time.time() - t0)
        return res

    # ── phase A ──────────────────────────────────────────────────────────
    def _run_symbol(self, symbol: str, df: pd.DataFrame, start: Optional[str],
                    end: Optional[str]) -> Tuple[List[Trade], List[Dict[str, Any]], List[Tuple[int, float]]]:
        view = _slice(df, start, end)
        if len(view) < max(self.p.warmup, 40):
            return [], [], []
        eng: EngineResult = run_engine(view, self.p, symbol=symbol, zone_cap=10 ** 6)
        arr = view[["open", "high", "low", "close"]].to_numpy(float)
        o, h, l, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        atr = eng.arrays["atr"]
        idx = view.index
        zones = {z.zid: z for z in eng.zones}
        trades: List[Trade] = []
        events: List[Dict[str, Any]] = []
        taps: List[Tuple[int, float]] = []
        last_exit_bar = -1
        for ev in eng.events:
            if ev.kind == EV_TAP and ev.tap_no == 1:
                taps.append((ev.bar, float(ev.level)))
            if ev.kind not in (EV_TAP, EV_CONFIRMED, EV_INVALID):
                continue
            if ev.kind == EV_INVALID:
                events.append({"date": _d(ev.ts), "symbol": symbol, "event": "invalidated",
                               "zone": ev.zid, "price": ev.price,
                               "reason": ev.detail.get("reason", "")})
                continue
            name = ("tap1" if (ev.kind == EV_TAP and ev.tap_no == 1)
                    else "confirmed" if ev.kind == EV_CONFIRMED else "tap")
            if not self._trades(name, ev):
                continue
            z = zones.get(ev.zid)
            if z is None or ev.bar < self.p.warmup:
                continue
            if not self.tc.allow_rollover and ev.bar <= last_exit_bar:
                events.append({"date": _d(ev.ts), "symbol": symbol, "event": name, "zone": ev.zid,
                               "skipped": "overlapping_position"})
                continue
            atr_i = float(atr[ev.bar]) if math.isfinite(atr[ev.bar]) else 0.0
            if self.tc.max_entry_distance_atr and atr_i and (c[ev.bar] - ev.level) / atr_i > self.tc.max_entry_distance_atr:
                events.append({"date": _d(ev.ts), "symbol": symbol, "event": name, "zone": ev.zid,
                               "skipped": "too_far_from_level"})
                continue
            tr, exit_bar = self._simulate(symbol, o, h, l, c, atr, idx, ev, z, name)
            if tr is not None:
                tr.confirmed = z.confirm_bar >= 0 and ev.bar <= z.confirm_bar <= tr.exit_bar
                trades.append(tr)
                last_exit_bar = exit_bar
                events.append({"date": _d(tr.entry_date), "symbol": symbol, "event": name,
                               "zone": ev.zid, "level": tr.entry_price, "price": c[ev.bar],
                               "r": tr.r_multiple, "exit": tr.exit_reason, "tap_no": ev.tap_no,
                               "bars": tr.bars_held})
        return trades, events, taps

    def _trades(self, name: str, ev) -> bool:
        trig = self.tc.entry_trigger
        if trig == "tap1":
            return name == "tap1"
        if trig == "confirmed":
            return name == "confirmed"
        if trig == "tap1_or_confirmed":
            return name in ("tap1", "confirmed")
        if trig == "tap":
            return name in ("tap", "tap1") and self.tc.min_tap <= ev.tap_no <= self.tc.max_tap
        return False

    def _simulate(self, symbol, o, h, l, c, atr, idx, ev, z: Zone, name: str):
        tc = self.tc
        n = len(o)
        bar = int(ev.bar)
        level = float(ev.level) if math.isfinite(ev.level) else float(z.entry)
        fill = float(c[bar]) if name == "confirmed" else float(min(o[bar], level))
        if not (fill > 0):
            return None, bar
        stop = initial_stop = float(z.stop)
        risk = fill - stop
        if not math.isfinite(risk) or risk <= 0:
            return None, bar
        target = fill + risk * tc.target_r if tc.target_r and tc.target_r > 0 else math.inf
        costs = (tc.commission_bps + tc.slippage_bps + tc.spread_bps) / 10_000.0
        entry_eff = fill * (1.0 + costs)
        mae_r, mfe_r = 0.0, 0.0
        hi_since = fill
        be_done = False
        i = bar if (tc.same_bar_stop and name != "confirmed") else bar + 1
        exit_bar = exit_price = None
        reason = ""
        while i < n:
            mae_r = min(mae_r, (l[i] - entry_eff) / risk)
            mfe_r = max(mfe_r, (h[i] - entry_eff) / risk)
            stop_hit = (l[i] <= stop) if tc.stop_mode == "touch" else (c[i] < stop)
            if stop_hit:                                   # worst path: stop always wins
                exit_bar = i
                exit_price = min(o[i], stop) if tc.stop_mode == "touch" else c[i]
                reason = "trailing_stop" if stop > initial_stop * (1.0 + 1e-9) else "initial_stop"
                break
            if math.isfinite(target) and h[i] >= target:
                exit_bar, exit_price, reason = i, max(o[i], target), "target"
                break
            hi_since = max(hi_since, h[i])
            if tc.trail_mode == "breakeven" and not be_done and h[i] >= fill + 0.5 * risk:
                stop, be_done = max(stop, entry_eff), True
            elif tc.trail_mode == "chandelier":
                a = float(atr[i]) if math.isfinite(atr[i]) else risk
                stop = max(stop, hi_since - tc.chandelier_atr * a)
            elif tc.trail_mode == "structure":
                k = max(1, int(tc.structure_lookback))
                stop = max(stop, float(np.min(l[max(0, i - k):i + 1])))
            if tc.exit_on_reversal and c[i] < z.bot:
                exit_bar, exit_price, reason = i, c[i], "zone_flip"
                break
            if tc.time_stop_bars and (i - bar) >= int(tc.time_stop_bars):
                exit_bar, exit_price, reason = i, c[i], "time_stop"
                break
            if z.dead_bar == i:
                exit_bar, exit_price, reason = i, c[i], "zone_invalidated"
                break
            i += 1
        if exit_bar is None:
            if n - 1 <= bar:
                return None, bar
            exit_bar, exit_price, reason = n - 1, c[n - 1], "end_of_data"
        exit_eff = float(exit_price) * (1.0 - costs)
        tr = Trade(
            symbol=symbol, trigger=name, tap_no=int(getattr(ev, "tap_no", 0) or 0), zone_id=z.zid,
            entry_bar=bar, exit_bar=int(exit_bar), entry_date=idx[bar], exit_date=idx[exit_bar],
            entry_price=fill, stop_price=float(stop), initial_stop=initial_stop,
            target_price=(target if math.isfinite(target) else float("nan")),
            exit_price=float(exit_price), risk_per_share=risk,
            r_multiple=float((exit_eff - entry_eff) / risk),
            ret_pct=float((exit_price - fill) / fill * 100.0),
            net_ret_pct=float((exit_eff - entry_eff) / entry_eff * 100.0),
            bars_held=int(exit_bar - bar), exit_reason=reason,
            mae_r=float(mae_r), mfe_r=float(mfe_r),
            atr_at_entry=float(atr[bar]) if math.isfinite(atr[bar]) else float("nan"),
        )
        return tr, int(exit_bar)


# ─────────────────────────────────────────────────────────────────────────────
# Signal study — forward returns after a Tap 1 (independent of exit rules)
# ─────────────────────────────────────────────────────────────────────────────

def signal_study(tap_events: Dict[str, List[Tuple[int, float]]], frames: Dict[str, pd.DataFrame],
                 params: Params, *, start: Optional[str] = None, end: Optional[str] = None,
                 horizons: Sequence[int] = HORIZONS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(stats_table, raw_events)`` for Tap-1 forward-return analysis."""
    rows: List[Dict[str, Any]] = []
    for sym, taps in (tap_events or {}).items():
        df = frames.get(sym)
        if df is None or not taps:
            continue
        view = _slice(df, start, end)
        c = view["close"].to_numpy(float)
        h = view["high"].to_numpy(float)
        lo = view["low"].to_numpy(float)
        for bar, level in taps:
            if level <= 0 or bar >= len(c):
                continue
            row: Dict[str, Any] = {"symbol": sym, "date": str(view.index[bar].date()),
                                   "entry": float(level)}
            for k in horizons:
                j = bar + int(k)
                row[f"fwd_{k}"] = float((c[j] / level - 1.0) * 100.0) if j < len(c) else float("nan")
            w = slice(bar + 1, min(len(c), bar + 11))
            if h[w].size:
                row["mfe_10"] = float((h[w].max() / level - 1.0) * 100.0)
                row["mae_10"] = float((lo[w].min() / level - 1.0) * 100.0)
            else:
                row["mfe_10"] = row["mae_10"] = float("nan")
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out, out
    # baseline: every bar's forward return in the same window, for an edge read
    base_rows = []
    for sym, df in frames.items():
        view = _slice(df, start, end)
        c = view["close"].to_numpy(float)
        if len(c) < max(horizons) + 5:
            continue
        for k in horizons:
            base = (c[k:] / c[:-k] - 1.0) * 100.0
            if base.size:
                base_rows.append({"fwd_%d" % k: float(np.nanmean(base))})
    base = pd.DataFrame(base_rows).mean() if base_rows else pd.Series(dtype="float64")
    stats = {}
    for k in horizons:
        col = f"fwd_{k}"
        if col in out.columns:
            v = out[col].dropna()
            stats[col] = {"n": int(v.size), "mean": float(v.mean()) if v.size else float("nan"),
                          "median": float(v.median()) if v.size else float("nan"),
                          "win%": float((v > 0).mean() * 100) if v.size else float("nan"),
                          "baseline_mean": float(base.get(col, float("nan")))}
    if "mfe_10" in out.columns:
        for col in ("mfe_10", "mae_10"):
            v = out[col].dropna()
            stats[col] = {"n": int(v.size), "mean": float(v.mean()) if v.size else float("nan"),
                          "median": float(v.median()) if v.size else float("nan")}
    table = pd.DataFrame([{"event": k, **v} for k, v in stats.items()])
    return table, out


def _slice(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not start and not end:
        return df
    lo = pd.Timestamp(start) if start else df.index.min()
    hi = pd.Timestamp(end) if end else df.index.max()
    try:
        return df.loc[lo:hi]
    except Exception:
        return df


def _d(ts) -> str:
    try:
        return str(pd.Timestamp(ts).date())
    except Exception:
        return str(ts)


__all__ = ["Backtester", "BacktestResult", "Trade", "signal_study"]
