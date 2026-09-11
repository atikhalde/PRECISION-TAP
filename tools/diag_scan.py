#!/usr/bin/env python3
"""TEMPORARY diagnostic for the silent live scanner. Removed once fixed.

Runs the real pipeline against the real feed and writes one markdown report that
shows, stage by stage, why a cycle did or did not put a message on Telegram:

  universe -> provider frame -> last-bar engine events -> alert window ->
  filters -> dedupe ledger -> dispatch/transport
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from precision_tap.alerts import dedupe_key, event_name, passes_filters
from precision_tap.config import load_config, setup_logging
from precision_tap.data import DataSource, session_date
from precision_tap.engine import EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, run_engine
from precision_tap.params import TelegramConfig
from precision_tap.scanner import Scanner, _is_stale_bar, _market_open
from precision_tap.state import StateStore
from precision_tap.telegram import TelegramClient

OUT = []


def say(line: str = "") -> None:
    OUT.append(line)
    print(line, flush=True)


def head(t):
    say("\n\n## " + t)


def main() -> int:
    cfg = load_config("config.yaml")
    setup_logging("DEBUG" if "-v" in sys.argv else "INFO")
    ist = pd.Timestamp.now(tz="Asia/Kolkata")
    head("0. clock / mode")
    say(f"utc={datetime.now(timezone.utc):%Y-%m-%d %H:%M} ist={ist:%Y-%m-%d %H:%M %a} "
        f"market_open={_market_open(cfg.live.market_timezone, session=(cfg.live.session_open, cfg.live.session_close))} "
        f"provider={cfg.data.provider} universe_file={cfg.data.universe_file}")
    uni = Scanner(cfg, dry_run=True).universe()
    say(f"universe={len(uni)} sample={uni[:6]}")

    head("1. provider frames (real fetch, no cache)")
    for tag, live in (("eod", False), ("live", True)):
        src = DataSource(cfg.data, live=live)
        for sym in uni[:6]:
            t0 = time.time()
            try:
                b = src.get(sym, use_cache=False)
            except Exception as exc:
                say(f"- {sym} [{tag}] EXC {type(exc).__name__}: {exc}")
                continue
            if not b.ok:
                say(f"- {sym} [{tag}] FAIL {b.error[:160]}")
                continue
            last = b.df.index[-1]
            say(f"- {sym} [{tag}] bars={len(b.df)} src={b.source} live={b.live} "
                f"last={last!r} sess={session_date(last, cfg.live.market_timezone)} "
                f"stale={_is_stale_bar(last, cfg.live.market_timezone, strict=live)} "
                f"close={b.df['close'].iloc[-1]:,.2f} {time.time()-t0:.1f}s")
        src.close()

    head("2. engine events on the newest 6 bars (closed-bar replay)")
    src = DataSource(cfg.data, live=False)
    probe = uni[:25]
    frames = src.get_many(probe, workers=6, progress=False)
    src.close()
    tot = {}
    for sym in probe:
        b = frames.get(sym)
        if b is None or not b.ok:
            say(f"- {sym}: no data ({(b.error if b else '')!s:.80})")
            continue
        df = b.df
        res = run_engine(df, cfg.params, symbol=sym, intrabar_last=False)
        n = len(df)
        recent = [e for e in res.events if e.bar >= n - 6]
        for e in recent:
            tot[event_name(e)] = tot.get(event_name(e), 0) + 1
        disp = [str(pd.Timestamp(df.index[i]).date()) for i in range(n)
                 if res.arrays["displacement"][i] and i >= n - 60]
        say(f"- {sym} bars={n} last={df.index[-1].date()} total_events={len(res.events)} "
            f"last6={[(event_name(e), e.bar - n + 1, e.tap_no) for e in recent]} "
            f"disp(60)={disp[-3:]}")
        ctx = {"price": float(df["close"].iloc[-1]), "avg_dollar_volume": float(
            pd.Series((df["volume"].tail(21) * df["close"].tail(21)).to_numpy()).median()),
            "atr": 1.0, "exchange": "NSE"}
        for e in recent:
            ok, why = passes_filters(e, ctx, cfg.alert)
            if not ok:
                say(f"    filtered: {event_name(e)} -> {why}")
    say(f"\ntotals(last6 bars over {len(probe)} symbols): {tot}")

    head("3. full cycle with a widened window (recent_bars=5, log-only)")
    cfg2 = load_config("config.yaml", overrides=["alerts.recent_bars=5"])
    cfg2.data.universe = uni
    cfg2.alert.chart = False
    cfg2.state_db = ":memory:"
    cfg2.out_dir = "results_diag"
    with StateStore(":memory:") as store:
        sc = Scanner(cfg2, store=store, dry_run=True)
        rep = sc.scan(live=False, progress=False)
        sc.close()
    say(f"mode={rep.mode} universe={rep.universe} usable={rep.usable} errors={rep.errors} "
        f"skipped={rep.skipped_symbols} events_total={rep.events_total} "
        f"in_window={rep.events_in_window} filtered={rep.filtered} alerts={len(rep.alerts)}")
    say("notes: " + " | ".join(rep.notes))
    for st, ev in rep.alerts[:12]:
        say(f"  {ev.symbol} {event_name(ev)} bar={ev.bar} ts={ev.ts} level={ev.level:.2f}")

    head("4. transport (validate + one real test message)")
    tg = TelegramClient(TelegramConfig(enabled=cfg.telegram.enabled, bot_token=cfg.telegram.bot_token,
                                       chat_ids=cfg.telegram.chat_ids, api_base=cfg.telegram.api_base,
                                       parse_mode=cfg.telegram.parse_mode,
                                       min_seconds_between_messages=0.0, proxy=cfg.telegram.proxy))
    say(f"configured={tg.configured} dry_run={tg.dry_run} chats={tg.cfg.chat_ids}")
    if tg.configured and not tg.dry_run:
        try:
            me = tg.get_me()
            say(f"getMe ok: @{me.get('username')}")
            res = tg.send_text("🔧 Precision Tap diagnostic: transport probe (no action needed).")
            say(f"send_text -> {[(r.ok, r.error) for r in res]}")
        except Exception as exc:
            say(f"transport EXCEPTION {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    rc = main()
    with open("diag-report.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    sys.exit(rc)
