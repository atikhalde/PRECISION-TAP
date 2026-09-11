"""Performance maths: portfolio accounting, R-statistics, drawdown, Sharpe, tables."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .params import ScanConfig, TradeConfig


# ─────────────────────────────────────────────────────────────────────────────
# small stat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _finite(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype="float64") if len(list(x)) else np.array([], dtype="float64")
    return a[np.isfinite(a)]


def profit_factor(rs: np.ndarray) -> float:
    wins = float(rs[rs > 0].sum())
    losses = float(-rs[rs < 0].sum())
    if losses <= 0:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def max_drawdown(equity: pd.Series) -> Tuple[float, int, Optional[Any]]:
    """Return (max DD as a negative fraction, longest drawdown in observations, trough date)."""
    if equity is None or len(equity) < 2:
        return 0.0, 0, None
    values = equity.to_numpy(dtype="float64")
    peak = np.maximum.accumulate(values)
    dd = values / peak - 1.0
    i = int(np.argmin(dd))
    # longest stretch under water
    longest = cur = 0
    for v in dd:
        cur = cur + 1 if v < 0 else 0
        longest = max(longest, cur)
    return float(dd[i]), longest, (equity.index[i] if i < len(equity) else None)


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna().to_numpy(dtype="float64")
    if r.size < 3:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna().to_numpy(dtype="float64")
    if r.size < 3:
        return float("nan")
    downside = r[r < 0]
    if downside.size < 2:
        return float("nan")
    dsd = float(np.sqrt((downside ** 2).sum() / max(1, r.size)))
    return float(r.mean() / dsd * math.sqrt(periods_per_year)) if dsd > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — portfolio accounting
# ─────────────────────────────────────────────────────────────────────────────

def _naive(ts):
    """Strip tz info so a fixed business-day grid can be built safely."""
    try:
        t = pd.Timestamp(ts)
        return t.tz_localize(None) if t.tzinfo is not None else t
    except Exception:
        return ts


def attach_portfolio(res, tc: TradeConfig) -> None:
    """Fill in $-denominated sizing / P&L, the equity curves and the sub-tables."""
    trades = res.trades
    cash = float(tc.initial_capital)
    cost_frac = (tc.commission_bps + tc.slippage_bps + tc.spread_bps) / 10_000.0
    closed: List[Tuple[Any, float, float]] = []        # (exit_date, net_pnl, r)
    open_exits: List[Any] = []
    events: List[Tuple[Any, int]] = []
    for tr in trades:
        if tr.entry_date is not None:
            events.append((tr.entry_date, 1))
        if tr.exit_date is not None:
            events.append((tr.exit_date, -1))
    for tr in trades:
        while open_exits and open_exits[0] <= tr.entry_date:
            open_exits.pop(0)
        if tc.max_open_positions and len(open_exits) >= int(tc.max_open_positions):
            tr.skipped_by_cap = True
            continue
        risk_money = cash * (tc.risk_per_trade / 100.0)
        shares = risk_money / tr.risk_per_share if tr.risk_per_share > 0 else 0.0
        max_by_cash = (cash * 0.98) / tr.entry_price if tr.entry_price > 0 else 0.0
        capped = False
        if shares > max_by_cash > 0:
            shares, capped = max_by_cash, True
        if shares <= 0:
            tr.skipped_by_cap = True
            continue
        gross = shares * (tr.exit_price - tr.entry_price)
        fees = shares * (tr.entry_price + tr.exit_price) * cost_frac
        net = gross - fees
        tr.shares = float(shares)
        tr.notional = float(shares * tr.entry_price)
        tr.gross_pnl = float(gross)
        tr.net_pnl = float(net)
        tr.ret_pct = float(tr.ret_pct) 
        tr.exit_reason = tr.exit_reason + (" (cash-capped)" if capped else "")
        if tr.exit_date is not None:
            open_exits.append(tr.exit_date)
        closed.append((tr.exit_date, net, tr.r_multiple))

    # equity curve on the trading axis (realised-P&L steps; standard for daily bars)
    if closed:
        dates = [d for d, _, _ in closed if d is not None]
        pnl = pd.Series({d: 0.0 for d in dates})
        for d, net, _ in closed:
            if d is not None:
                pnl[d] = pnl.get(d, 0.0) + net
        pnl = pnl.sort_index()
        curve = tc.initial_capital + pnl.cumsum()
        # include the first trade's entry date as the start of the curve
        start = min([t.entry_date for t in trades if t.entry_date is not None] or dates)
        lo, hi = _naive(start), _naive(max(dates))
        curve.index = pd.DatetimeIndex([_naive(d) for d in curve.index])
        curve = curve.groupby(level=0).sum().sort_index()
        curve = curve.reindex(pd.date_range(start=lo, end=hi, freq="B")).ffill()
        curve = curve.fillna(tc.initial_capital)
        curve.iloc[0] = float(tc.initial_capital) if len(curve) else float(tc.initial_capital)
        res.equity = curve
    else:
        res.equity = pd.Series(dtype="float64")
    rs = [t.r_multiple for t in trades if t.exit_date is not None and math.isfinite(t.r_multiple)]
    if rs:
        r_idx = pd.DatetimeIndex([_naive(t.exit_date) for t in trades
                                  if t.exit_date is not None and math.isfinite(t.r_multiple)])
        r_ser = pd.Series(rs, index=r_idx).sort_index()
        res.r_equity = r_ser.cumsum()
    df = pd.DataFrame([t.to_dict() for t in trades])
    if not df.empty:
        res.per_symbol = (df.groupby("symbol")
                          .agg(trades=("symbol", "size"), wins=("won", "sum"),
                               total_r=("r_multiple", "sum"), avg_r=("r_multiple", "mean"),
                               net_pnl=("net_pnl", "sum"), avg_bars=("bars_held", "mean"))
                          .assign(win_rate=lambda d: d.wins / d.trades * 100.0)
                          .sort_values("total_r", ascending=False).reset_index())
        res.exit_table = (df.groupby("exit_reason")
                          .agg(trades=("symbol", "size"), total_r=("r_multiple", "sum"),
                               avg_r=("r_multiple", "mean"), win_rate=("won", "mean"),
                               net_pnl=("net_pnl", "sum"))
                          .assign(win_rate=lambda d: d.win_rate * 100.0)
                          .sort_values("trades", ascending=False).reset_index())
        dd = df.copy()
        dd["year"] = pd.to_datetime(dd["entry_date"]).dt.year
        res.per_year = (dd.groupby("year")
                        .agg(trades=("symbol", "size"), total_r=("r_multiple", "sum"),
                             avg_r=("r_multiple", "mean"), win_rate=("won", "mean"),
                             net_pnl=("net_pnl", "sum"))
                        .assign(win_rate=lambda d: d.win_rate * 100.0).reset_index())
    res.skipped = {"by_position_cap": int(sum(1 for t in trades if t.skipped_by_cap))}


# ─────────────────────────────────────────────────────────────────────────────
# Headline metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics_for(res, cfg: ScanConfig) -> Dict[str, Any]:
    tc = cfg.trade
    trades = [t for t in res.trades]
    closed = [t for t in trades if t.exit_bar >= 0]
    rs = _finite([t.r_multiple for t in closed])
    nets = _finite([t.net_pnl for t in closed])
    wins = int((rs > 0).sum()) if rs.size else 0
    losses = int((rs <= 0).sum()) if rs.size else 0
    m: Dict[str, Any] = {
        "universe": res.universe, "symbols_traded": len({t.symbol for t in closed}),
        "bars": res.bars, "span": res.span, "run_at": res.run_at,
        "signals": len(closed), "signals_per_year": 0.0,
        "win_rate": (wins / rs.size * 100.0) if rs.size else 0.0,
        "avg_r": float(rs.mean()) if rs.size else 0.0,
        "median_r": float(np.median(rs)) if rs.size else 0.0,
        "stdev_r": float(rs.std(ddof=1)) if rs.size > 1 else 0.0,
        "expectancy_r": float(rs.mean()) if rs.size else 0.0,
        "profit_factor": profit_factor(rs),
        "best_r": float(rs.max()) if rs.size else 0.0,
        "worst_r": float(rs.min()) if rs.size else 0.0,
        "avg_win_r": float(rs[rs > 0].mean()) if wins else 0.0,
        "avg_loss_r": float(rs[rs <= 0].mean()) if losses else 0.0,
        "payoff": (abs(float(rs[rs > 0].mean()) / float(rs[rs <= 0].mean()))
                   if wins and losses and rs[rs <= 0].mean() != 0 else float("nan")),
        "avg_mae_r": float(np.nanmean([t.mae_r for t in closed])) if closed else float("nan"),
        "avg_mfe_r": float(np.nanmean([t.mfe_r for t in closed])) if closed else float("nan"),
        "avg_bars_held": float(np.nanmean([t.bars_held for t in closed])) if closed else 0.0,
        "fees_total": float(np.nansum([abs(t.gross_pnl - t.net_pnl) for t in closed])) if closed else 0.0,
        "config": {"entry_trigger": tc.entry_trigger, "target_r": tc.target_r,
                   "stop_mode": tc.stop_mode, "trail_mode": tc.trail_mode,
                   "time_stop_bars": tc.time_stop_bars, "costs_bps_per_side":
                   tc.commission_bps + tc.slippage_bps + tc.spread_bps},
    }
    if res.equity is not None and len(res.equity) > 1:
        eq = res.equity.astype("float64")
        rets = eq.pct_change().dropna()
        years = max(1e-9, (eq.index[-1] - eq.index[0]).days / 365.25)
        dd, dd_len, dd_at = max_drawdown(eq)
        m.update({
            "initial_capital": float(cfg.trade.initial_capital),
            "final_equity": float(eq.iloc[-1]),
            "net_profit": float(eq.iloc[-1] - cfg.trade.initial_capital),
            "return_pct": float((eq.iloc[-1] / cfg.trade.initial_capital - 1.0) * 100.0),
            "cagr_pct": float(((eq.iloc[-1] / cfg.trade.initial_capital) ** (1.0 / years) - 1.0) * 100.0)
                        if eq.iloc[-1] > 0 else float("nan"),
            "max_drawdown_pct": dd * 100.0,
            "max_drawdown_days": int(dd_len),
            "max_drawdown_date": str(dd_at.date()) if dd_at is not None else "",
            "sharpe": sharpe(rets),
            "sortino": sortino(rets),
            "vol_pct_annual": float(rets.std(ddof=1) * math.sqrt(252) * 100.0) if len(rets) > 2 else float("nan"),
            "best_trade": float(eq.diff().max()), "worst_trade": float(eq.diff().min()),
            "signals_per_year": len(closed) / years,
        })
    if res.exit_table is not None and not res.exit_table.empty:
        m["exit_mix"] = {str(r.exit_reason): int(r.trades) for r in res.exit_table.itertuples()}
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in m.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Rendering helper (avoids the `tabulate` dependency)
# ─────────────────────────────────────────────────────────────────────────────

def md_table(df: Optional[pd.DataFrame], *, float_nd: int = 2, max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_no rows_"
    out = df.head(max_rows).copy()

    def fmt(v: Any) -> str:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{float_nd}f}" if abs(v) < 1e6 else f"{v:,.0f}"
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    cols = [str(c) for c in out.columns]
    rows = [[fmt(v) for v in rec] for rec in out.itertuples(index=False, name=None)]
    widths = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
              for i, c in enumerate(cols)]
    head = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |" for r in rows]
    extra = f"\n_{len(df) - len(rows)} more row(s) not shown_" if len(df) > len(rows) else ""
    return "\n".join([head, sep, *body]) + extra


def sweep_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = [c for c in df.columns if c not in ("config", "note")]
    return df.sort_values(by="final_equity", ascending=False).reset_index(drop=True) if "final_equity" in df.columns else df[order]


__all__ = ["attach_portfolio", "metrics_for", "md_table", "max_drawdown", "sharpe", "sortino",
           "profit_factor", "sweep_table"]
