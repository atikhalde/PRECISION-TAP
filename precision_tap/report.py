"""Report writers: markdown summary, CSV exports, equity chart, Telegram digest."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .alerts import RULE_LINE
from .metrics import md_table

log = logging.getLogger("precision_tap.report")

_KPI_ORDER = [
    ("signals", "trades"), ("win_rate", "win %"), ("avg_r", "avg R"), ("median_r", "median R"),
    ("profit_factor", "profit factor"), ("expectancy_r", "expectancy R"),
    ("payoff", "payoff"), ("best_r", "best R"), ("worst_r", "worst R"),
    ("avg_bars_held", "avg bars held"), ("avg_mae_r", "avg MAE R"), ("avg_mfe_r", "avg MFE R"),
]
_MONEY_ORDER = [
    ("initial_capital", "capital"), ("final_equity", "final equity"),
    ("return_pct", "return %"), ("cagr_pct", "CAGR %"), ("max_drawdown_pct", "max DD %"),
    ("sharpe", "Sharpe"), ("sortino", "Sortino"), ("vol_pct_annual", "vol % ann."),
    ("signals_per_year", "signals/yr"), ("fees_total", "fees $"),
]


@dataclass
class ReportPaths:
    dir: str = ""
    summary: str = ""
    trades: str = ""
    events: str = ""
    study: str = ""
    study_events: str = ""
    equity_png: str = ""
    metrics: str = ""
    per_symbol: str = ""

    def as_list(self) -> List[str]:
        return [v for v in [self.summary, self.trades, self.events, self.study, self.study_events,
                           self.equity_png, self.metrics, self.per_symbol] if v]


def _fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "—"
    if abs(f) >= 1e6:
        return f"{f:,.0f}"
    return f"{f:,.{nd}f}"


def summary_message(res, cfg) -> str:
    """Compact Telegram-sized digest of a backtest run."""
    m = res.metrics or {}
    lines = ["📊 PRECISION TAP · BACKTEST",
             f"{res.span} · {res.universe} symbols · {m.get('signals', 0)} trades",
             RULE_LINE]
    for key, label in _KPI_ORDER[:6]:
        v = m.get(key)
        if key == "win_rate":
            lines.append(f"{label:<14}{_fmt(v, 1)}%   ({m.get('signals', 0)} trades)")
        else:
            lines.append(f"{label:<14}{_fmt(v)}")
    lines.append(RULE_LINE)
    for key, label in _MONEY_ORDER[:6]:
        lines.append(f"{label:<14}{_fmt(m.get(key), 1)}" + ("%" if key.endswith("_pct") else ""))
    if m.get("exit_mix"):
        mix = " · ".join(f"{k} {v}" for k, v in sorted(m["exit_mix"].items(), key=lambda kv: -kv[1]))
        lines.append(f"exits: {mix}")
    lines.append(RULE_LINE)
    lines.append("synthetic demo data" if getattr(cfg.data, "provider", "") == "synthetic"
                 else f"{cfg.trade.entry_trigger} · TP {cfg.trade.target_r}R · "
                      f"{cfg.trade.stop_mode} stop · trail {cfg.trade.trail_mode} · "
                      f"time {cfg.trade.time_stop_bars}")
    return "\n".join(lines)


def render_markdown(res, cfg) -> str:
    m = res.metrics or {}
    out: List[str] = [f"# Precision Tap — backtest report", ""]
    out.append(f"_run {res.run_at} · universe {res.universe} symbols · {res.bars:,} bars · span {res.span}_")
    out.append("")
    out.append("## Signal statistics (R-space, all Tap signals)")
    out.append("")
    out.append("| metric | value | metric | value |")
    out.append("|---|---|---|---|")
    rows = [(lbl, m.get(k)) for k, lbl in _KPI_ORDER]
    for i in range(0, len(rows), 2):
        a = rows[i]
        b = rows[i + 1] if i + 1 < len(rows) else ("", None)
        unit = "%" if a[0].endswith("%") else ""
        unitb = "%" if str(b[0]).endswith("%") else ""
        out.append(f"| {a[0]} | {_fmt(a[1])}{unit} | {b[0]} | {_fmt(b[1])}{unitb} |")
    out.append("")
    out.append("## Portfolio (money-space, risk-sized)")
    out.append("")
    out.append("| metric | value | metric | value |")
    out.append("|---|---|---|---|")
    rows = [(lbl, m.get(k)) for k, lbl in _MONEY_ORDER]
    for i in range(0, len(rows), 2):
        a = rows[i]
        b = rows[i + 1] if i + 1 < len(rows) else ("", None)
        unit = "%" if a[0].endswith("%") else ""
        unitb = "%" if str(b[0]).endswith("%") else ""
        out.append(f"| {a[0]} | {_fmt(a[1])}{unit} | {b[0]} | {_fmt(b[1])}{unitb} |")
    out.append("")
    if res.exit_table is not None:
        out.append("## Exit mix")
        out.append("")
        out.append(md_table(res.exit_table))
        out.append("")
    if res.per_year is not None and len(res.per_year):
        out.append("## By year")
        out.append("")
        out.append(md_table(res.per_year))
        out.append("")
    if res.per_symbol is not None:
        out.append("## By symbol (top) / (bottom)")
        out.append("")
        out.append(md_table(res.per_symbol.head(15), max_rows=15))
        out.append("")
    if res.signal_study is not None and len(res.signal_study):
        out.append("## Tap-1 forward-return study (no position, no exits)")
        out.append("")
        out.append(md_table(res.signal_study, float_nd=3))
        out.append("")
    closed = [t for t in res.trades if t.exit_bar >= 0]
    if closed:
        best = max(closed, key=lambda t: (t.r_multiple if math.isfinite(t.r_multiple) else -9e9))
        worst = min(closed, key=lambda t: (t.r_multiple if math.isfinite(t.r_multiple) else 9e9))
        out.append("## Best / worst")
        out.append("")
        out.append(f"- 🏆 `{best.symbol}` {best.entry_date.date()} → {best.exit_date.date()} "
                   f"**{_fmt(best.r_multiple)}R** ({_fmt(best.net_ret_pct)}% net, {best.exit_reason})")
        out.append(f"- 💀 `{worst.symbol}` {worst.entry_date.date()} → {worst.exit_date.date()} "
                   f"**{_fmt(worst.r_multiple)}R** ({_fmt(worst.net_ret_pct)}% net, {worst.exit_reason})")
        out.append("")
    p = cfg.params
    out.append("## Settings used")
    out.append("")
    out.append("```yaml")
    out.append("# indicator (Pine parity)")
    for k in ("min_rvol", "min_range_atr", "min_body_frac", "min_clv", "structure_len",
              "origin_search", "zone_method", "entry_mode", "frontrun_mode", "frontrun_atr",
              "stop_atr", "approach_atr", "min_age", "max_touches", "require_departure",
              "raise_after_first_tap", "confirm_bars", "confirm_rvol", "confirm_clv",
              "require_sweep", "atr_len", "vol_len"):
        out.append(f"{k}: {getattr(p, k)!r}".replace("'", ""))
    out.append("# execution")
    for k, v in vars(cfg.trade).items():
        out.append(f"{k}: {v!r}".replace("'", ""))
    out.append("```")
    out.append("")
    out.append("_The indicator defines entries and a structural stop only — targets, trailing and "
               "costs above are the backtest's own assumptions. Not investment advice._")
    return "\n".join(out)


def render_equity_png(res, path: Path) -> Optional[str]:
    if res.equity is None or len(res.equity) < 3:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        log.info("matplotlib unavailable (%s) — skipping equity chart", exc)
        return None
    eq = res.equity.astype("float64")
    peak = eq.cummax()
    dd = (eq / peak - 1.0) * 100.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.2, 5.0), dpi=110, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0e1117")
    for a in (ax, ax2):
        a.set_facecolor("#0e1117")
        a.tick_params(colors="#c9d1d9", labelsize=7)
        for sp in a.spines.values():
            sp.set_color("#30363d")
        a.grid(True, color="#21262d", lw=0.5)
    ax.plot(eq.index, eq.values, color="#26a69a", lw=1.4, label="equity")
    ax.axhline(eq.iloc[0], color="#8b949e", lw=0.8, ls="--", label="start")
    if res.r_equity is not None and len(res.r_equity) > 2:
        ax2b = ax.twinx()
        ax2b.plot(res.r_equity.index, res.r_equity.values, color="#58a6ff", lw=0.9, alpha=0.75,
                  label="cumulative R")
        ax2b.tick_params(colors="#58a6ff", labelsize=6)
        ax2b.set_ylabel("cum R", color="#58a6ff", fontsize=7)
    ax.set_title(f"Precision Tap equity · {res.metrics.get('signals', 0)} trades · "
                 f"{res.span}", color="#e6edf3", fontsize=10, loc="left")
    ax.legend(fontsize=7, facecolor="#161b22", labelcolor="#c9d1d9", edgecolor="#30363d")
    ax2.fill_between(dd.index, dd.values, 0, color="#ef5350", alpha=0.55)
    ax2.set_ylabel("drawdown %", color="#8b949e", fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def write_backtest_report(res, cfg, out_dir: str | Path, *, tag: str = "",
                         charts: bool = True) -> ReportPaths:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = ReportPaths(dir=str(out))
    try:
        trades = pd.DataFrame([t.to_dict() for t in res.trades])
        if not trades.empty:
            cols = [c for c in ["symbol", "trigger", "tap_no", "zone_id", "entry_date", "entry_price",
                                "initial_stop", "target_price", "exit_date", "exit_price", "risk_per_share",
                                "r_multiple", "ret_pct", "net_ret_pct", "bars_held", "exit_reason",
                                "mae_r", "mfe_r", "atr_at_entry", "confirmed", "shares", "notional",
                                "gross_pnl", "net_pnl", "skipped_by_cap", "entry_bar", "exit_bar",
                                "stop_price"] if c in trades.columns]
            trades = trades[cols + [c for c in trades.columns if c not in cols]]
            rp.trades = str(out / f"trades_{stamp}.csv")
            trades.to_csv(rp.trades, index=False)
        if res.events:
            rp.events = str(out / f"events_{stamp}.csv")
            pd.DataFrame(res.events).to_csv(rp.events, index=False)
        if res.signal_study is not None and len(res.signal_study):
            rp.study = str(out / f"signal_study_{stamp}.csv")
            res.signal_study.to_csv(rp.study, index=False)
        if res.signal_events is not None and len(res.signal_events):
            rp.study_events = str(out / f"signal_events_{stamp}.csv")
            res.signal_events.to_csv(rp.study_events, index=False)
        for name, frame in (("per_symbol", res.per_symbol), ("per_year", res.per_year),
                            ("exit_mix", res.exit_table)):
            if frame is not None and len(frame):
                f = out / f"{name}_{stamp}.csv"
                frame.to_csv(f, index=False)
                if name == "per_symbol":
                    rp.per_symbol = str(f)
        rp.summary = str(out / f"summary_{stamp}.md")
        Path(rp.summary).write_text(render_markdown(res, cfg), encoding="utf-8")
        rp.metrics = str(out / f"metrics_{stamp}.json")
        Path(rp.metrics).write_text(json.dumps(res.metrics, indent=2, default=str), encoding="utf-8")
        if charts:
            png = render_equity_png(res, out / f"equity_{stamp}.png")
            if png:
                rp.equity_png = png
        latest = out / "latest_summary.md"
        latest.write_text(render_markdown(res, cfg), encoding="utf-8")
    except Exception as exc:
        log.error("report write failed: %s", exc)
    return rp
