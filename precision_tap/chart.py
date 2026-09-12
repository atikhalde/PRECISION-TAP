"""Chart rendering for Telegram alerts (matplotlib, no mplfinance dependency).

Draws the candlesticks, every live precision OB, the tapped zone highlighted, its
entry/stop lines and the Tap marker. Everything is optional: a rendering failure
must never cost an alert, so callers wrap this in try/except.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("precision_tap.chart")

C_UP = "#26a69a"
C_DOWN = "#ef5350"
C_ZONE = "#1469dc"
C_TAP = "#ff9800"
C_CONFIRM = "#00b469"
C_DEAD = "#787878"
C_STOP = "#e53935"


def _candles(ax, x: np.ndarray, o, h, l, c, wick_w=0.6, body_w=0.62) -> None:
    up = c >= o
    for mask, color in ((up, C_UP), (~up, C_DOWN)):
        if not mask.any():
            continue
        xs = x[mask]
        ax.vlines(xs, l[mask], h[mask], color=color, linewidth=0.9, zorder=2)
        body_low = np.minimum(o[mask], c[mask])
        body_h = np.maximum(np.abs(c[mask] - o[mask]), (h[mask] - l[mask]) * 0.001)
        ax.bar(xs, body_h, bottom=body_low, width=body_w, color=color,
               edgecolor=color, linewidth=0.4, zorder=3)


def render_chart(df: pd.DataFrame, zones, event=None, *, out_path: str | Path,
                 symbol: str = "", bars: int = 120, timeframe: str = "1d",
                 title_extra: str = "", dpi: int = 105, height: float = 4.2,
                 width: float = 8.6) -> Optional[str]:
    """Render one PNG; returns the path (or ``None`` when matplotlib is absent)."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:                                  # pragma: no cover
        log.info("matplotlib unavailable (%s) — skipping chart", exc)
        return None

    # The overlay is drawn on a truncated window, so every bar index has to be
    # rebased on the first bar that is actually plotted.  Measuring the offset
    # *after* the truncation (``len(df) - n``) always yields 0, which puts a zone
    # born at bar 596 at x=596 of a 120-bar window: negative width, nothing
    # visible, and the tap marker silently skipped — every alert then ships a
    # chart of bare candles while still looking like a success.
    window = max(30, int(bars))
    start_bar = max(0, len(df) - window)
    df = df.iloc[start_bar:]
    n = len(df)
    x = np.arange(n)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                                  gridspec_kw={"height_ratios": [3.1, 0.8]}, dpi=dpi)
    fig.patch.set_facecolor("#0e1117")
    for a in (ax, axv):
        a.set_facecolor("#0e1117")
        a.tick_params(colors="#c9d1d9", labelsize=7)
        for s in a.spines.values():
            s.set_color("#30363d")
        a.grid(True, color="#21262d", linewidth=0.5, alpha=0.7)

    _candles(ax, x, o, h, l, c)
    axv.bar(x, v, width=0.62, color=np.where(c >= o, C_UP, C_DOWN), alpha=0.75)

    live = [z for z in (zones or []) if getattr(z, "state", -1) >= 0]
    tapped = getattr(event, "zid", None)
    for z in live[-12:]:
        left = (z.born - start_bar) - 0.35
        right = n - 0.5
        color = {0: C_ZONE, 1: C_TAP, 2: C_CONFIRM}.get(int(z.state), C_ZONE)
        if z.zid == tapped:
            color = C_TAP
        ax.add_patch(Rectangle((max(left, -0.5), z.bot), (n - 0.5) - max(left, -0.5),
                               max(z.top - z.bot, 1e-9), facecolor=color, alpha=0.17,
                               edgecolor=color, linewidth=1.0, zorder=1))
        ax.axhline(z.entry, color=color, linewidth=1.1, linestyle="-", alpha=0.9)
        ax.axhline(z.stop, color=C_STOP, linewidth=0.8, linestyle="--", alpha=0.7)

    if event is not None and getattr(event, "bar", None) is not None:
        xi = event.bar - start_bar
        if 0 <= xi < n:
            ax.scatter([xi], [event.level if np.isfinite(event.level) else l[xi]],
                       s=52, facecolor=C_TAP, edgecolor="white", linewidth=0.8, zorder=6)
            ax.annotate(f"TAP {event.tap_no}" if event.kind == "tap" else event.kind.upper(),
                        (xi, (event.level if np.isfinite(event.level) else l[xi])),
                        xytext=(6, -14), textcoords="offset points", color="#ffd8a8",
                        fontsize=8, fontweight="bold", zorder=7)

    tick_at = list(range(0, n, max(1, n // 8)))
    labels = []
    for i in tick_at:
        ts = df.index[min(i, n - 1)]
        labels.append(pd.Timestamp(ts).strftime("%d %b" if timeframe.endswith("d") else "%d %b %H:%M"))
    ax.set_xticks(tick_at)
    ax.set_xticklabels(labels, color="#8b949e")
    axv.set_xticks(tick_at)
    axv.set_xticklabels([])
    axv.set_ylim(0, (max(v.max(), 1.0) if len(v) else 1.0) * 1.25)
    axv.set_ylabel("vol", color="#8b949e", fontsize=7)

    last = c[-1]
    ax.set_xlim(-0.8, n + 0.8)
    pad = (h.max() - l.min()) * 0.12 if n else 1.0
    ax.set_ylim(l.min() - pad, h.max() + pad)
    ax.set_title(f"{symbol} · {timeframe.upper()} · {last:,.2f}"
                 + (f"   {title_extra}" if title_extra else ""),
                 color="#e6edf3", fontsize=10, loc="left", pad=8)
    ax.margins(x=0)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return str(out)
