#!/usr/bin/env python3
"""Signal audit — does the scanner emit alerts that are *true* and *actionable*?

The rest of the offline tooling proves the pipeline runs: `selftest` pins Pine
parity bar-for-bar, `demo` walks the whole product, `tools/live_drill.py` drives
the live path through a fake feed and a fake Bot API.  None of them ask whether
the message a human receives is correct, because they all run on
`data.synthetic_frame` — a generator that *engineers* a displacement → OB → Tap-1
sequence every 41 bars, so it always has something to find.

This tool generates market data that mimics real NSE daily statistics instead
(lognormal returns with volatility clustering, autocorrelated volume that spikes
on big-move days, no planted setups) and then asks three questions:

  1. HOW OFTEN — gate pass-through, and the alert rate a universe should expect.
     Answers "is a quiet chat a quiet market or a broken scanner?"
  2. IS IT VALID — every alert is graded against independently re-derived rules:
     did the bar really touch the level, was the zone alive, is the advertised
     stop already breached, did price run away from the entry.
  3. DOES THE FORMING BAR LEAK — a live cycle replays the last *closed* bar for
     parity; if that pass is fed today's incomplete bar, the zone state printed
     on yesterday's alert (taps, entry, dead) describes a bar that had not
     happened yet.

    python tools/signal_audit.py
    python tools/signal_audit.py --symbols 60 --bars 750
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from precision_tap import alerts as A  # noqa: E402
from precision_tap.data import normalize_ohlcv  # noqa: E402
from precision_tap.engine import EV_INVALID, EV_TAP, run_engine  # noqa: E402
from precision_tap.params import AlertConfig, Params  # noqa: E402
from precision_tap import series as S  # noqa: E402

#: the shipped defaults (config.yaml → alerts)
ALERT_CFG = AlertConfig(events=["new_ob", "tap1", "approach", "confirmed"],
                        recent_bars=1, min_liquidity_dollar_volume=250_000_000,
                        min_price=20)
#: universe/nse.txt, for scaling the per-symbol rate to a per-day alert count
UNIVERSE_SIZE = 127


def realistic_frame(n: int = 750, seed: int = 1, price0: float = 400.0,
                    vol0: float = 0.016, shares: float = 4_000_000,
                    start: str = "2022-01-03") -> pd.DataFrame:
    """Daily OHLCV shaped like a liquid NSE name — with no planted setups."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)

    sigma = np.full(n, vol0)                                  # volatility clustering
    for i in range(1, n):
        sigma[i] = np.sqrt(0.86 * sigma[i - 1] ** 2 + 0.10 * vol0 ** 2
                           + 0.04 * (rng.standard_normal() * sigma[i - 1]) ** 2)
    ret = rng.standard_normal(n) * sigma + 0.0003

    lv = np.full(n, np.log(shares))                           # autocorrelated volume
    for i in range(1, n):
        lv[i] = 0.75 * lv[i - 1] + 0.25 * np.log(shares) + rng.normal(0, 0.35)
    volume = np.exp(lv) * (1.0 + 2.2 * np.abs(ret) / vol0)    # spikes on big moves

    close = price0 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = price0
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.004, n - 1))

    span = np.abs(rng.standard_normal(n)) * sigma * close * 0.9 + 0.002 * close
    hi_body, lo_body = np.maximum(open_, close), np.minimum(open_, close)
    up = rng.random(n) < 0.5 + 1.4 * np.tanh(ret / vol0)      # wick asymmetry
    high = hi_body + span * np.where(up, rng.uniform(0.25, 0.9, n), rng.uniform(0.05, 0.5, n))
    low = lo_body - span * np.where(up, rng.uniform(0.05, 0.5, n), rng.uniform(0.25, 0.9, n))

    df = pd.DataFrame({"open": open_, "high": np.maximum(high, hi_body),
                       "low": np.minimum(low, lo_body), "close": close,
                       "volume": volume}, index=idx)
    return normalize_ohlcv(df, daily=True)


def gate_report(df: pd.DataFrame, p: Params) -> dict:
    """Bars clearing each displacement gate alone, and all of them together."""
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    o, c, v = df["open"].to_numpy(float), df["close"].to_numpy(float), df["volume"].to_numpy(float)
    atr = S.atr(h, l, c, p.atr_len)
    vol_ma = S.sma(v, p.vol_len)
    rvol = np.where(vol_ma > 0, v / vol_ma, 0.0)
    rng_ = np.maximum(h - l, p.mintick)
    prior = S.highest(S.shift(h, 1), p.structure_len)
    sl = slice(p.warmup, len(df))
    g = {"bullish": (c > o)[sl], f"rvol>={p.min_rvol}": (rvol >= p.min_rvol)[sl],
         f"range>={p.min_range_atr}ATR": (rng_ >= atr * p.min_range_atr)[sl],
         f"body>={p.min_body_frac}": (np.abs(c - o) / rng_ >= p.min_body_frac)[sl],
         f"clv>={p.min_clv}": (((c - l) / rng_) >= p.min_clv)[sl],
         f"bos({p.structure_len})": (c > prior)[sl]}
    joint = np.ones(len(df) - p.warmup, dtype=bool)
    for m in g.values():
        joint &= np.nan_to_num(m, nan=False).astype(bool)
    return {"per_gate": {k: int(np.nan_to_num(v_, nan=False).sum()) for k, v_ in g.items()},
            "joint": int(joint.sum()), "bars": len(df) - p.warmup}


def grade(df: pd.DataFrame, p: Params, ev, all_events) -> list:
    """Independent re-derivation of whether this alert is true and actionable."""
    z, i = ev.zone, ev.bar
    row = df.iloc[i]
    close_i = float(row["close"])
    atr_i = ev.detail.get("atr")
    out = []
    if ev.kind != EV_TAP:
        return out
    if not (float(row["low"]) <= ev.level + 1e-9):
        out.append("TOUCH-NOT-MET")                    # the bar never reached the level
    died = [e for e in all_events if e.kind == EV_INVALID and e.bar == i and e.zid == ev.zid]
    if died:
        out.append("DEAD-ON-ARRIVAL:" + str(died[0].detail.get("reason")))
    if close_i < z.stop:
        out.append("CLOSED-BELOW-STOP")                # advertised stop already breached
    if atr_i and np.isfinite(atr_i) and atr_i > 0 and (ev.level - close_i) / atr_i > 1.0:
        out.append(f"RAN-{(ev.level - close_i) / atr_i:.1f}ATR-PAST-ENTRY")
    if ev.tap_no > p.max_touches:
        out.append("EXHAUSTED")
    return out


def audit(n_sym: int, n_bars: int) -> int:
    p = Params.default()
    tot = Counter(); kinds = Counter(); verdicts = Counter(); by_kind = Counter()
    gates = Counter(); leak_checked = leak_bad = 0; examples = {}

    for s in range(n_sym):
        px0 = float(np.random.default_rng(s).uniform(60, 2500))
        shares = float(np.random.default_rng(s + 500).uniform(5e5, 8e6))
        df = realistic_frame(n=n_bars, seed=7000 + s * 13, price0=px0, shares=shares)
        g = gate_report(df, p)
        gates.update(g["per_gate"]); tot["bars"] += g["bars"]; tot["joint"] += g["joint"]

        res = run_engine(df, p, symbol=f"SYM{s:02d}.NS")
        smry = res.summary()
        for k in ("zones_created", "new_ob", "approach", "tap1", "tap_all",
                  "confirmed", "invalidated"):
            tot[k] += smry[k]

        # ── 2. validity of what the scanner would actually send ──
        n = len(df)
        atr_last = float(res.arrays["atr"][n - 1])
        vol = df["volume"].tail(21).to_numpy(float); px = df["close"].tail(21).to_numpy(float)
        ctx = {"price": float(df["close"].iloc[-1]), "atr": atr_last, "change_pct": 0.0,
               "rvol": 1.0, "exchange": "NSE", "timeframe": "1d",
               "avg_dollar_volume": float(np.median(vol[1:] * px[1:])) if len(vol) >= 5 else 0.0}
        for ev in res.events:                     # a daily scanner sees each event on its own bar
            ok, _why = A.passes_filters(ev, ctx, ALERT_CFG)
            if not ok:
                continue
            tot["sent"] += 1
            probs = grade(df, p, ev, res.events)
            name = A.event_name(ev)
            if probs:
                tag = probs[0].split(":")[0]
                verdicts[tag] += 1; by_kind[(name, tag)] += 1
                examples.setdefault(tag, (f"SYM{s:02d}.NS", str(df.index[ev.bar])[:10], name,
                                          ev.level, float(ev.zone.stop),
                                          float(df["close"].iloc[ev.bar]), probs))
            else:
                verdicts["OK"] += 1; by_kind[(name, "OK")] += 1

        # ── 3. does the forming bar leak into closed-bar zone state? ──
        trunc = run_engine(df.iloc[:-1], p, symbol=f"S{s}", intrabar_last=False)
        full = run_engine(df, p, symbol=f"S{s}", intrabar_last=False)
        ref = {(e.kind, e.bar, e.zid): e for e in trunc.events if e.bar == n - 2}
        got = {(e.kind, e.bar, e.zid): e for e in full.events if e.bar == n - 2}
        for k in set(ref) & set(got):
            leak_checked += 1
            za, zb = got[k].zone, ref[k].zone
            if (za.taps, za.state, za.entry) != (zb.taps, zb.state, zb.entry):
                leak_bad += 1

    # ── report ──
    years = tot["bars"] / n_sym / 250.0
    print(f"realistic NSE-like market · {n_sym} symbols × {n_bars} bars "
          f"({tot['bars']} bars, {years:.1f} symbol-years)\n")
    print("1. HOW OFTEN")
    print(f"   displacement gates   : " + "  ".join(f"{k}={v}" for k, v in gates.items()))
    print(f"   all gates together   : {tot['joint']}  ({tot['joint'] / tot['bars'] * 100:.2f}% of bars)")
    print(f"   zones created        : {tot['zones_created']}   new_ob {tot['new_ob']}")
    print(f"   taps                 : {tot['tap1']} first / {tot['tap_all']} total")
    print(f"   approach / confirmed / invalidated : {tot['approach']} / {tot['confirmed']} / {tot['invalidated']}")
    print(f"   tap1 per symbol-year : {tot['tap1'] / n_sym / years:.2f}")
    per_day = tot["sent"] / tot["bars"] * UNIVERSE_SIZE
    print(f"   => alerts the scanner would SEND per day on a {UNIVERSE_SIZE}-name universe: {per_day:.2f}")
    print(f"      (a quiet chat on a day like that is a broken pipeline, not a quiet market)")

    print("\n2. IS IT VALID")
    sent = max(1, tot["sent"])
    for k, v in verdicts.most_common():
        print(f"   {v:>6}  {k:<34} {v / sent * 100:5.1f}%")
    ok_by_kind = Counter({k: v for (n, t), v in by_kind.items() if t == "OK" for k in [n]})
    print("   per kind sent/valid : " + ", ".join(
        f"{n} {ok_by_kind.get(n, 0)}/{sum(v for (nn, _t), v in by_kind.items() if nn == n)}"
        for n in sorted({n for n, _t in by_kind})))
    for tag, ex in examples.items():
        print(f"   e.g. {tag}: {ex[0]} {ex[1]} {ex[2]} level={ex[3]:.2f} "
              f"stop={ex[4]:.2f} close={ex[5]:.2f}")

    print("\n3. DOES THE FORMING BAR LEAK INTO CLOSED-BAR STATE")
    print(f"   closed-bar events compared          : {leak_checked}")
    print(f"   would be corrupted by today's bar   : {leak_bad}")
    print("   A non-zero count is the POINT, not a failure: it is the difference")
    print("   between replaying the whole frame (last bar treated as closed) and")
    print("   replaying df.iloc[:-1]. The scanner uses the latter, so the alerts it")
    print("   sends never carry it. Pinned by tests/test_alert_validity.py and")
    print("   tests/test_live.py::test_parity_event_zone_state_is_not_polluted_by_the_forming_bar.")

    bad = sum(v for k, v in verdicts.items() if k != "OK")
    print(f"\nRESULT: {verdicts.get('OK', 0)}/{sent} alerts valid "
          f"({verdicts.get('OK', 0) / sent * 100:.1f}%), {bad} graded problem(s)")
    return 1 if bad > sent * 0.02 else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", type=int, default=40)
    ap.add_argument("--bars", type=int, default=750)
    a = ap.parse_args()
    return audit(a.symbols, a.bars)


if __name__ == "__main__":
    sys.exit(main())
