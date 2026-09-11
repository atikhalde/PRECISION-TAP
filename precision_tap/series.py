"""Pine-compatible rolling primitives.

TradingView's ``ta.*`` functions have specific warm-up semantics that a naive
pandas port gets wrong (and a wrong ATR seed silently changes every signal).
These helpers reproduce Pine exactly:

* ``ta.sma(x, n)``   -> ``na`` until ``n`` observations exist, then plain mean.
* ``ta.rma(x, n)``   -> ``na`` until bar ``n-1``, seeded with ``sma``, then the
  Wilder recursion ``alpha*x + (1-alpha)*prev`` with ``alpha = 1/n``.
* ``ta.highest/lowest(x, n)`` -> over *available* bars during warm-up (Pine
  does not return ``na`` there), ``na`` only where every value in the window is ``na``.
* ``x[1]`` shifting -> ``na`` at bar 0.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def as_float(x) -> np.ndarray:
    a = np.asarray(x, dtype="float64")
    return a


def shift(x: np.ndarray, n: int, fill: float = np.nan) -> np.ndarray:
    """Pine ``x[n]``: value n bars ago; missing history -> ``fill`` (na)."""
    x = as_float(x)
    out = np.full(x.shape, fill, dtype="float64")
    if n <= 0:
        return x.copy()
    if n < x.size:
        out[n:] = x[: x.size - n]
    return out


def sma(x: np.ndarray, length: int) -> np.ndarray:
    """``ta.sma``: NaN while fewer than ``length`` observations exist."""
    x = as_float(x)
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    if length < 1 or n < length:
        return out
    csum = np.cumsum(np.where(np.isnan(x), 0.0, x))
    # NaN inside the window makes the Pine mean NaN -> mirror that.
    cnan = np.cumsum(np.isnan(x).astype("int64"))
    win_sum = csum[length - 1 :] - np.concatenate(([0.0], csum[: n - length]))
    win_nan = cnan[length - 1 :] - np.concatenate(([0], cnan[: n - length]))
    out[length - 1 :] = np.where(win_nan > 0, np.nan, win_sum / length)
    return out


def rma(x: np.ndarray, length: int) -> np.ndarray:
    """``ta.rma`` (Wilder EMA) with the Pine SMA seed."""
    x = as_float(x)
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    if length < 1 or n < length:
        return out
    alpha = 1.0 / length
    seed = np.nanmean(x[:length]) if not np.isnan(x[:length]).any() else np.nan
    prev = seed
    out[length - 1] = prev
    for i in range(length, n):
        v = x[i]
        if np.isnan(v):
            continue
        prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _roll_extreme(x: np.ndarray, length: int, mode: int) -> np.ndarray:
    """Monotonic-deque rolling max (mode=1) / min (mode=-1) with Pine warm-up.

    ``na`` values are skipped (a short history is *not* an error in Pine's
    ``highest``/``lowest`` — it simply uses what exists), but a window that is
    entirely ``na`` yields ``na``.
    """
    x = as_float(x)
    n = x.size
    out = np.full(n, np.nan, dtype="float64")
    if n == 0:
        return out
    length = max(1, int(length))
    from collections import deque

    dq: deque = deque()  # indices, values in decreasing (max) / increasing (min) order
    for i in range(n):
        v = x[i]
        if not np.isnan(v):
            while dq:
                last = x[dq[-1]]
                if (mode == 1 and last <= v) or (mode == -1 and last >= v):
                    dq.pop()
                else:
                    break
            dq.append(i)
        while dq and dq[0] <= i - length:
            dq.popleft()
        if dq:
            out[i] = x[dq[0]]
    return out


def highest(x: np.ndarray, length: int) -> np.ndarray:
    return _roll_extreme(x, length, 1)


def lowest(x: np.ndarray, length: int) -> np.ndarray:
    return _roll_extreme(x, length, -1)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """``ta.tr``: max(h-l, |h-prev_c|, |l-prev_c|); first bar -> h-l (Pine uses
    ``nz`` semantics where the previous close is na)."""
    high, low, close = as_float(high), as_float(low), as_float(close)
    pc = shift(close, 1, fill=close[0] if close.size else np.nan)
    return np.maximum.reduce([high - low, np.abs(high - pc), np.abs(low - pc)])


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """``ta.atr(length)`` == ``ta.rma(ta.tr, length)``."""
    return rma(true_range(high, low, close), length)
