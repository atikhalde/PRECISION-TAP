"""Command line interface: `python -m precision_tap <command>` (or `precision-tap`).

  init            write config.yaml / .env / universe file
  doctor          environment + config diagnostics (run this first)
  verify SYMBOL   print every zone/event the engine sees, for TradingView cross-check
  scan            one scan cycle over the universe -> Telegram alerts
  run             always-on loop (intraday polling + scheduled EOD scans)
  backtest        replay the same engine over history, report R + money metrics
  sweep           parameter grid search over the backtest
  zones           watchlist table: nearest live OB level per symbol
  alerts          recent alerts from the state db
  telegram-test   ping the bot (with --discover to find your chat id)
  livecheck       end-to-end live check: config -> Telegram -> feed -> one real cycle
  export-data     cache history to CSV for offline backtesting
  demo            end-to-end offline demo on synthetic NSE-style data
  selftest        Pine-parity checks (no network needed)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .alerts import event_name, render_message
from .config import load_config, setup_logging, config_to_dict
from .data import DataSource, frame_to_csv, read_universe, synthetic_frame, write_demo_dataset
from .engine import EV_APPROACH, EV_CONFIRMED, EV_INVALID, EV_NEW, EV_TAP, Params, run_engine
from .params import AlertConfig, DataConfig, TelegramConfig
from .report import summary_message, write_backtest_report
from .scanner import Scanner, _market_open
from .state import StateStore
from .telegram import TelegramClient

log = logging.getLogger("precision_tap.cli")

BANNER = "Precision Tap — NSE/BSE daily scanner & backtester (Pine port)"


# ─────────────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cfg(args) -> Any:
    cfg = load_config(args.config, overrides=args.set or (), env_path=args.env_file)
    if getattr(args, "symbols", None):
        cfg.data.universe = [s for s in args.symbols if s]
    if getattr(args, "days", None):
        cfg.data.lookback_days = int(args.days)
    if getattr(args, "provider", None):
        cfg.data.provider = args.provider
    if getattr(args, "limit", None):
        cfg.data.universe = (cfg.data.universe or read_universe(cfg.data.universe_file))[: args.limit]
    for flag, attr in (("trigger", "entry_trigger"), ("target_r", "target_r"),
                        ("stop_mode", "stop_mode"), ("trail", "trail_mode"),
                        ("time_stop", "time_stop_bars"), ("risk", "risk_per_trade"),
                        ("capital", "initial_capital"), ("max_positions", "max_open_positions")):
        val = getattr(args, flag, None)
        if val is not None:
            cur = getattr(cfg.trade, attr)
            setattr(cfg.trade, attr, type(cur)(val) if not isinstance(cur, bool) else bool(val))
    if getattr(args, "events", None):
        cfg.alert.events = [e.strip() for e in args.events.split(",") if e.strip()]
    if getattr(args, "recent_bars", None) is not None:
        cfg.alert.recent_bars = int(args.recent_bars)
    if getattr(args, "no_charts", False):
        cfg.alert.chart = False
    if getattr(args, "min_dollar_volume", None):
        cfg.alert.min_liquidity_dollar_volume = float(args.min_dollar_volume) * 1e7  # in crore
    return cfg


def _store(cfg, *, readonly: bool = False) -> StateStore:
    return StateStore(cfg.state_db)


def _scanner(cfg, *, dry_run: bool, store: Optional[StateStore] = None) -> Scanner:
    tg = None
    if not dry_run and cfg.telegram.enabled and cfg.telegram.bot_token and cfg.telegram.chat_ids:
        tg = TelegramClient(cfg.telegram)
    elif not dry_run and cfg.telegram.enabled:
        log.warning("telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) — "
                    "alerts will be printed and queued only")
    return Scanner(cfg, store=store, telegram=tg, dry_run=dry_run)


def _dt(index, i: int) -> str:
    """Bar index -> 'YYYY-MM-DD' (safe for tz-aware and naive indices)."""
    try:
        return str(pd.Timestamp(index[i]).date())
    except Exception:
        return str(i)


def _fmt(v: Any, nd: int = 2) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "—" if not math.isfinite(f) else f"{f:,.{nd}f}"


# ─────────────────────────────────────────────────────────────────────────────
# commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_init(args) -> int:
    root = Path(".")
    made: List[str] = []
    here = Path(__file__).resolve().parent.parent
    for name, src in (("config.yaml", here / "config.example.yaml"),
                      (".env", here / ".env.example")):
        dst = root / name
        if dst.exists():
            print(f"  exists  {name}")
            continue
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            made.append(name)
    uni = root / "universe" / "nse.txt"
    if not uni.exists():
        (here / "universe" / "nse.txt")
        uni.parent.mkdir(parents=True, exist_ok=True)
        if (here / "universe" / "nse.txt").exists():
            uni.write_text((here / "universe" / "nse.txt").read_text(encoding="utf-8"), encoding="utf-8")
            made.append(str(uni))
    for d in ("data/csv", "data/cache", "results", "logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    print("created: " + (", ".join(made) if made else "nothing (already present)"))
    print("\nNext: put your bot token + chat id in .env, then run\n"
          "  python -m precision_tap doctor\n"
          "  python -m precision_tap telegram-test\n"
          "  python -m precision_tap scan --no-send\n"
          "  python -m precision_tap run")
    return 0


def cmd_doctor(args) -> int:
    cfg = _cfg(args)
    ok = True
    print(BANNER + "\n")
    print(f"python            {sys.version.split()[0]}")
    for mod in ("pandas", "numpy", "requests", "yaml", "matplotlib", "yfinance"):
        try:
            m = __import__(mod)
            print(f"{mod:<16} {getattr(m, '__version__', 'ok')}")
        except Exception as exc:
            print(f"{mod:<16} MISSING ({exc})")
            ok = mod in ("pandas", "numpy", "yfinance") and False
    print(f"\nuniverse file     {cfg.data.universe_file}")
    uni = cfg.data.universe or read_universe(cfg.data.universe_file,
                                             suffix=cfg.data.symbol_suffix)
    print(f"symbols           {len(uni)}" + (f"  e.g. {', '.join(uni[:5])}" if uni else "  ← EDIT THIS"))
    if not uni:
        ok = False
    print(f"provider          {cfg.data.provider}  (interval {cfg.data.interval}, "
          f"suffix {cfg.data.symbol_suffix!r})")
    print(f"indicator         RVOL≥{cfg.params.min_rvol} range≥{cfg.params.min_range_atr}ATR "
          f"body≥{cfg.params.min_body_frac} CLV≥{cfg.params.min_clv} BOS={cfg.params.structure_len} "
          f"mintick={cfg.params.mintick}")
    print(f"alerts            {cfg.alert.events} recent_bars={cfg.alert.recent_bars} "
          f"chart={cfg.alert.chart} parity={cfg.alert.match_indicator_100}")
    print(f"session           {cfg.live.market_timezone} {cfg.live.session_open}-"
          f"{cfg.live.session_close} · poll {cfg.live.intraday_poll_minutes}min · "
          f"scans {cfg.live.scan_times}")
    print(f"market now        {'OPEN' if _market_open(cfg.live.market_timezone, session=(cfg.live.session_open, cfg.live.session_close)) else 'CLOSED'}")
    tok = cfg.telegram.bot_token
    shown = "set (" + tok.split(":")[0] + ":…)" if tok else "MISSING"
    tg_line = ("telegram          " + ("enabled" if cfg.telegram.enabled else "disabled")
               + " · token " + shown + " · " + str(len(cfg.telegram.chat_ids)) + " chat id(s)")
    print(tg_line)
    if cfg.telegram.enabled and not (tok and cfg.telegram.chat_ids):
        print("                  → set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env")
        print("                  → until then every alert is logged, never delivered")
        ok = False
    try:
        with StateStore(cfg.state_db) as st:
            st.start_run("doctor", 0) and None
        print(f"state db          writable ({cfg.state_db})")
    except Exception as exc:
        print(f"state db          FAIL {exc}")
        ok = False
    if getattr(args, "net", False):
        print("\nprobing data provider …")
        src = DataSource(cfg.data, live=_market_open(cfg.live.market_timezone,
                                                     session=(cfg.live.session_open, cfg.live.session_close)))
        probe = (uni[:3] if uni else ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        for sym in probe:
            t0 = time.time()
            b = src.get(sym, use_cache=not args.no_cache)
            if b.ok:
                print(f"  {sym:<16} {len(b.df):>5} bars  last {b.df.index[-1].date()}  "
                      f"close {b.df['close'].iloc[-1]:,.2f}  live={b.live}  "
                      f"({time.time() - t0:.1f}s)")
            else:
                print(f"  {sym:<16} ERROR {b.error[:120]}")
                ok = False
    print("\n" + ("all checks passed" if ok else "some checks need attention"))
    return 0 if ok else 1


def cmd_verify(args) -> int:
    """Dump the engine's full signal set for one symbol — the TradingView cross-check."""
    cfg = _cfg(args)
    src = DataSource(cfg.data, live=False)
    bars = src.get(args.symbol, end=args.end, lookback_days=cfg.data.lookback_days)
    if not bars.ok:
        print(f"no data for {args.symbol}: {bars.error}")
        return 2
    p = cfg.params
    df = bars.df
    if args.start:
        df = df.loc[pd.Timestamp(args.start):]
    res = run_engine(df, p, symbol=args.symbol, intrabar_last=False)
    print(f"{args.symbol} · {cfg.data.interval} · {len(df)} bars · {df.index[0].date()} → "
          f"{df.index[-1].date()} · mintick {p.mintick} · provider {bars.source}")
    print(f"\nzones ({len(res.zones)} created, {len(res.live_zones)} still live)")
    for z in res.zones:
        if args.bars_only and z.born < len(df) - args.bars_only:
            continue
        tap_dates = ", ".join(_dt(df.index, b) for b in z.tap_bars[:6])
        print(f"  #{z.zid:<3} {_dt(df.index, z.born)} origin {_dt(df.index, z.origin)} "
              f"[{z.origin - z.born}] zone {z.top:,.2f}/{z.bot:,.2f} entry {z.entry:,.2f} "
              f"(was {z.entry0:,.2f}) stop {z.stop:,.2f} state {z.state_name:<9} "
              f"taps {z.taps} at [{tap_dates}]"
              + ("  ← raiseAfterFirstTap" if z.adaptive else ""))
    print(f"\nevents ({len(res.events)})")
    for e in res.events:
        if args.events and e.kind not in args.events.split(","):
            continue
        d = str(pd.Timestamp(e.ts).date()) if e.ts is not None else str(e.bar)
        extra = (f" tap{e.tap_no}" if e.kind == EV_TAP else
                 f" reason={e.detail.get('reason')}" if e.kind == EV_INVALID else "")
        lvl = f"{e.level:,.2f}" if math.isfinite(e.level) else "—"
        print(f"  {d:<12} {e.kind:<11} z#{e.zid:<3} level {lvl:>10} close {e.price:,.2f}{extra}")
    n = len(df)
    print("\nclosed-bar flags (plotshape parity)")
    n_show = 24
    for label, mask in (("displacement → OB created", res.arrays["displacement"]),
                        ("anyTap    (plotshape TAP)", res.flags["any_tap"]),
                        ("anyApproach (pre-alert)", res.flags["any_approach"]),
                        ("anyConfirm  (DEFENCE OK)", res.flags["any_confirm"]),
                        ("anyDead     (invalidated)", res.flags["any_dead"])):
        days = [_dt(df.index, i) for i in range(n) if mask[i]]
        print(f"  {label:<24}: {', '.join(days[-n_show:]) or '—'}")
    return 0


def cmd_scan(args) -> int:
    cfg = _cfg(args)
    # same contract as `run`: with no working transport, log the alerts instead
    # of parking them in a retry queue that can never drain
    dry = bool(args.no_send) or not (cfg.telegram.enabled and cfg.telegram.bot_token
                                     and cfg.telegram.chat_ids)
    with _store(cfg) as store:
        sc = _scanner(cfg, dry_run=dry, store=store)
        if args.retried:
            n = sc.dispatcher.retry_pending()
            if n:
                print(f"re-sent {n} queued alert(s)")
        live = None
        if args.live:
            live = True
        elif args.eod:
            live = False
        rep = sc.scan(live=live, end=args.end, send=not args.report_only,
                      workers=args.workers)
        print(f"\n{BANNER}")
        print(f"mode={rep.mode} started={rep.started_at}  {rep.summary_line}")
        print()
        print(rep.console_table(args.top))
        print(f"\nalerts matched: {len(rep.alerts)}")
        for st, ev in rep.alerts:
            print("  " + ev.fmt())
        if rep.alerts and (args.messages or dry):
            print("\n--- rendered messages ---")
            for st, ev in rep.alerts:
                ctx = sc._context(st, ev)
                print(render_message(ev, ctx, cfg.alert, cfg.params, parse_mode="plain"))
                print("─" * 30)
        if rep.dispatch is not None and not isinstance(rep.dispatch, dict):
            print(f"\ndelivery: {rep.dispatch}")
        for note in rep.notes:
            print("  note:", note)
        sc.close()
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    setup_logging(cfg.log_level, cfg.live.log_file if not args.no_logfile else None)
    # Same contract as `scan`: with no working transport, log the alerts instead
    # of parking them in a retry queue that can never drain.
    dry = bool(args.no_send) or not (cfg.telegram.enabled and cfg.telegram.bot_token
                                     and cfg.telegram.chat_ids)
    with _store(cfg) as store:
        sc = _scanner(cfg, dry_run=dry, store=store)
        from .live import LiveLoop
        loop = LiveLoop(cfg, sc, store=store, heartbeat=not args.no_heartbeat,
                        max_cycles=args.max_cycles,
                        stop_after=(time.monotonic() + args.duration) if args.duration else None)
        try:
            if args.once:
                loop._scan("once")
                print("single cycle finished")
                return 0
            return loop.run()
        finally:
            sc.close()


def cmd_backtest(args) -> int:
    cfg = _cfg(args)
    setup_logging(cfg.log_level)
    uni = cfg.data.universe or read_universe(cfg.data.universe_file, suffix=cfg.data.symbol_suffix)
    if args.limit:
        uni = uni[: args.limit]
    print(f"loading {len(uni)} symbols via {cfg.data.provider} "
          f"({cfg.data.interval}, {args.days or cfg.data.lookback_days}d) …")
    src = DataSource(cfg.data, live=False)
    frames = {s: b.df for s, b in src.get_many(uni, lookback_days=args.days or cfg.data.lookback_days,
                                               workers=args.workers, progress=True).items() if b.ok}
    if not frames:
        print("no data — check `doctor --net`, or run the offline `demo`")
        return 2
    from .backtest import Backtester
    bt = Backtester(cfg, progress=not args.quiet)
    res = bt.run(frames, start=args.start, end=args.end, collect_events=not args.no_events)
    m = res.metrics or {}
    print(f"\n{'=' * 74}\nPrecision Tap backtest · {res.span} · {len(frames)} symbols · "
          f"{m.get('signals', 0)} trades\n{'=' * 74}")
    pairs = [("win rate %", "win_rate"), ("avg R", "avg_r"), ("median R", "median_r"),
             ("profit factor", "profit_factor"), ("expectancy R", "expectancy_r"),
             ("payoff", "payoff"), ("best R", "best_r"), ("worst R", "worst_r"),
             ("avg bars held", "avg_bars_held"), ("avg MAE R", "avg_mae_r"),
             ("avg MFE R", "avg_mfe_r"), ("final equity", "final_equity"),
             ("return %", "return_pct"), ("CAGR %", "cagr_pct"), ("max DD %", "max_drawdown_pct"),
             ("Sharpe", "sharpe"), ("Sortino", "sortino"), ("signals/yr", "signals_per_year"),
             ("fees paid", "fees_total")]
    for i in range(0, len(pairs), 2):
        a = pairs[i]
        b = pairs[i + 1] if i + 1 < len(pairs) else ("", None)
        sa = f"{a[0]:<14}{_fmt(m.get(a[1]))}"
        sb = f"{b[0]:<14}{_fmt(m.get(b[1]))}" if b[0] else ""
        print(f"  {sa:<34}{sb}")
    if res.exit_table is not None and len(res.exit_table):
        print("\nexits")
        print(res.exit_table.to_string(index=False))
    if res.signal_study is not None and len(res.signal_study):
        print("\nTap-1 forward returns (no exits — pure signal study)")
        print(res.signal_study.to_string(index=False))
    if res.per_symbol is not None and len(res.per_symbol):
        print(f"\ntop symbols\n{res.per_symbol.head(10).to_string(index=False)}")
    if res.errors:
        print(f"\n{len(res.errors)} symbol error(s): " + "; ".join(res.errors[:5]))
    if not args.no_report:
        paths = write_backtest_report(res, cfg, args.out or cfg.out_dir, tag=args.tag or "",
                                     charts=not args.no_charts)
        print("\nreport: " + "\n        ".join(paths.as_list()))
    if args.send_summary:
        tg = TelegramClient(cfg.telegram)
        text = summary_message(res, cfg)
        print("\nsending summary to telegram …")
        for r in tg.send_text(text):
            print("  ", "ok" if r.ok else f"FAILED {r.error}")
    return 0


def cmd_sweep(args) -> int:
    cfg = _cfg(args)
    grids: List[Tuple[str, List[Any]]] = []
    for spec in args.grid or []:
        if "=" not in spec:
            print("--grid expects key=v1,v2,v3")
            return 2
        key, _, vals = spec.partition("=")
        parsed: List[Any] = []
        for raw in vals.split(","):
            raw = raw.strip()
            try:
                parsed.append(int(raw) if raw.lstrip("-").isdigit() else float(raw))
            except ValueError:
                parsed.append(raw.strip("'\""))
        grids.append((key.strip(), parsed))
    if not grids:
        print("nothing to sweep; pass e.g. --grid backtest.target_r=1,2,3 "
              "--grid indicator.min_rvol=1.8,2.2")
        return 2
    uni = cfg.data.universe or read_universe(cfg.data.universe_file, suffix=cfg.data.symbol_suffix)
    if args.limit:
        uni = uni[: args.limit]
    src = DataSource(cfg.data, live=False)
    frames = {s: b.df for s, b in src.get_many(uni, lookback_days=args.days or cfg.data.lookback_days,
                                               workers=args.workers).items() if b.ok}
    combos = list(itertools.product(*(vals for _, vals in grids)))
    rows: List[Dict[str, Any]] = []
    for ci, combo in enumerate(combos, 1):
        ccfg = load_config(args.config, overrides=args.set or ())
        for (key, _), val in zip(grids, combo):
            top, _, leaf = key.partition(".")
            setattr({"indicator": ccfg.params, "params": ccfg.params, "backtest": ccfg.trade,
                     "trade": ccfg.trade, "alerts": ccfg.alert, "data": ccfg.data}.get(top, ccfg.params),
                    leaf, val)
        from .backtest import Backtester
        res = Backtester(ccfg, progress=False).run(frames, start=args.start, end=args.end)
        m = res.metrics or {}
        row = {key: val for (key, _), val in zip(grids, combo)}
        row.update({"trades": m.get("signals", 0), "win%": round(m.get("win_rate", 0), 1),
                    "avg_R": round(m.get("avg_r", 0), 3), "PF": round(m.get("profit_factor", 0) or 0, 2),
                    "ret%": round(m.get("return_pct", 0) or 0, 1),
                    "maxDD%": round(m.get("max_drawdown_pct", 0) or 0, 1),
                    "final_equity": round(m.get("final_equity", 0) or 0, 0),
                    "sharpe": round(m.get("sharpe", 0) or 0, 2)})
        rows.append(row)
        print(f"[{ci}/{len(combos)}] " + " ".join(f"{k}={v}" for k, v in row.items()))
    df = pd.DataFrame(rows).sort_values("final_equity", ascending=False)
    print("\n" + df.to_string(index=False))
    out = Path(cfg.out_dir) / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n{out}")
    return 0


def cmd_zones(args) -> int:
    cfg = _cfg(args)
    with _store(cfg) as store:
        sc = _scanner(cfg, dry_run=True, store=store)
        rep = sc.scan(symbols=([args.symbol] if args.symbol else None),
                      live=(True if args.live else None), send=False, progress=False)
        print(f"{'symbol':<15}{'last':>12}{'chg%':>7}{'entry':>12}{'stop':>12}{'ΔATR':>7}"
              f"{'age':>5}{'taps':>6}  state")
        rows = [r for r in rep.rows if r.ok and r.nearest_zone is not None]
        rows.sort(key=lambda r: abs(r.dist_atr) if math.isfinite(r.dist_atr) else 9e9)
        for r in rows[: args.top]:
            print(f"{r.symbol:<15}{r.price:>12,.2f}{r.change_pct:>7.2f}{r.nearest_entry:>12,.2f}"
                  f"{r.nearest_stop:>12,.2f}{r.dist_atr:>7.2f}{r.age:>5}{r.taps:>6}  {r.zone_state}")
        print(f"\n{len(rows)} symbols with a live precision OB (of {rep.usable} usable)")
        if not rows:
            print("no live zones — run `verify <SYMBOL>` to inspect the raw signal set")
        sc.close()
    return 0


def cmd_alerts(args) -> int:
    cfg = _cfg(args)
    with _store(cfg) as store:
        rows = store.recent_alerts(limit=args.limit, symbol=args.symbol)
        if not rows:
            print("no alerts recorded yet")
            return 0
        def _state(row) -> str:
            # sent: 1 delivered · 0 queued for retry · 2 given up (log-only / no transport)
            return {1: "sent", 0: "queued", 2: "log-only"}.get(int(row["sent"] or 0), "?")

        print(f"{'created':<20}{'symbol':<16}{'event':<12}{'level':>10}{'price':>10}  state")
        for r in rows:
            print(f"{str(r['created_at'])[:19]:<20}{r['symbol']:<16}{r['event']:<12}"
                  f"{_fmt(r['level']):>10}{_fmt(r['price']):>10}  {_state(r)}")
        pend = store.pending(limit=50)
        if pend:
            print(f"\n{len(pend)} queued for retry")
        last = store.last_run()
        if last:
            print(f"last run: {last.get('started_at')} mode={last.get('mode')} "
                  f"alerts={last.get('alerts')} errors={last.get('errors')}")
    return 0


def cmd_telegram(args) -> int:
    cfg = _cfg(args)
    tg = TelegramClient(cfg.telegram)
    if args.discover:
        if not cfg.telegram.bot_token:
            print("TELEGRAM_BOT_TOKEN is required to discover chats")
            return 2
        print("Send any message to your bot (e.g. /start), then wait …")
        chats = tg.discover_chats(timeout_sec=args.timeout)
        if not chats:
            print("no chats found — is the token right, and did you message the bot?")
            return 1
        for ch in chats:
            print(f"  chat_id={ch['chat_id']}  type={ch['type']}  title={ch['title']}")
        print(f"\nadd to .env:  TELEGRAM_CHAT_ID={chats[0]['chat_id']}")
        return 0
    me = tg.get_me() if cfg.telegram.bot_token else {}
    print(f"bot: @{me.get('username', '?')} ({me.get('first_name', 'no getMe — token missing?')})")
    print("chats: " + (", ".join(str(c) for c in cfg.telegram.chat_ids) or "NONE"))
    if not cfg.telegram.chat_ids:
        print("→ run `telegram-test --discover` after messaging the bot")
        return 2
    text = args.text or ("✅ <b>Precision Tap</b> is wired up.\n"
                         f"universe {len(cfg.data.universe or read_universe(cfg.data.universe_file, suffix=cfg.data.symbol_suffix))} symbols · "
                         f"{cfg.data.provider} · {cfg.data.interval} · {cfg.live.market_timezone}")
    res = tg.send_text(text)
    for r in res:
        print(("  ok   msg_id=" + str(r.message_id)) if r.ok else ("  FAIL " + r.error))
    return 0 if all(r.ok for r in res) else 1


def cmd_livecheck(args) -> int:
    """End-to-end live-market check: config → Telegram → market feed → one real cycle.

    "The scanner runs but Telegram is silent" has exactly four possible causes,
    and they are in four different places.  This walks all four in order and
    names the first one that fails, against the *real* market and the *real*
    bot — nothing here is mocked.
    """
    cfg = _cfg(args)
    probe = ([s for s in args.symbols if s] if getattr(args, "symbols", None)
             else (cfg.data.universe or read_universe(cfg.data.universe_file,
                                                      suffix=cfg.data.symbol_suffix))[:args.probe])
    open_now = _market_open(cfg.live.market_timezone,
                            session=(cfg.live.session_open, cfg.live.session_close))
    print(BANNER + "\n")
    failures: List[str] = []

    def stage(title: str) -> None:
        print(f"\n── {title} " + "─" * max(0, 58 - len(title)))

    # ── 1. configuration ──────────────────────────────────────────────────
    stage("1 · configuration")
    print(f"provider          {cfg.data.provider} · {cfg.data.interval} · suffix {cfg.data.symbol_suffix}")
    print(f"universe          {len(probe)} probe symbol(s): {', '.join(probe[:6])}"
          + (" …" if len(probe) > 6 else ""))
    print(f"alert events      {cfg.alert.events} · recent_bars={cfg.alert.recent_bars} · "
          f"liquidity floor {_fmt(cfg.alert.min_liquidity_dollar_volume / 1e7, 0)}cr · "
          f"min_price {_fmt(cfg.alert.min_price)}")
    print(f"market            {cfg.live.market_timezone} {cfg.live.session_open}–"
          f"{cfg.live.session_close} → {'OPEN' if open_now else 'CLOSED'}")
    if not cfg.telegram.enabled:
        failures.append("telegram.enabled is false — nothing will ever be sent")
        print("telegram          DISABLED by config")
    elif not (cfg.telegram.bot_token and cfg.telegram.chat_ids):
        failures.append("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing (.env)")
        print("telegram          token/chat MISSING → alerts are logged, never delivered")
    else:
        print(f"telegram          token {cfg.telegram.bot_token.split(':')[0]}:… · "
              f"{len(cfg.telegram.chat_ids)} chat id(s)")

    # ── 2. Telegram round trip ────────────────────────────────────────────
    stage("2 · Telegram round trip (real Bot API)")
    tg_ok = False
    if cfg.telegram.enabled and cfg.telegram.bot_token and cfg.telegram.chat_ids:
        tg = TelegramClient(cfg.telegram)
        try:
            me = tg.get_me()
            print(f"  getMe           ok — @{me.get('username')} ({me.get('first_name')})")
        except Exception as exc:
            failures.append(f"getMe failed — bad token, or no route to api.telegram.org: {exc}")
            print(f"  getMe           FAIL {exc}")
        if not args.no_message:
            try:
                res = tg.send_text("🔎 <b>Precision Tap livecheck</b> — transport test. "
                                   "If you can read this, delivery works.")
                for r in res:
                    if r.ok:
                        print(f"  sendMessage     ok — msg_id={r.message_id} chat {r.chat_id}")
                    else:
                        print(f"  sendMessage     FAIL chat {r.chat_id}: {r.error}")
                        failures.append(f"Telegram rejected the message: {r.error}")
                tg_ok = bool(res) and all(r.ok for r in res)
            except Exception as exc:
                failures.append(f"Telegram send failed: {exc}")
                print(f"  sendMessage     FAIL {exc}")
        else:
            print("  sendMessage     skipped (--no-message)")
    else:
        print("  skipped — no bot token / chat id")

    # ── 3. market feed ────────────────────────────────────────────────────
    stage("3 · market feed (real provider)")
    src = DataSource(cfg.data, live=open_now)
    feed_live = 0
    for sym in probe[:6]:
        t0 = time.time()
        b = src.get(sym, use_cache=not args.no_cache)
        if not b.ok:
            failures.append(f"{sym}: no data — {b.error}")
            print(f"  {sym:<16} FAIL {str(b.error)[:110]}")
            continue
        print(f"  {sym:<16} {len(b.df):>5} bars · last {b.df.index[-1].date()} · "
              f"close {b.df['close'].iloc[-1]:,.2f} · live={b.live} · {b.source} "
              f"({time.time() - t0:.1f}s)")
        feed_live += int(bool(b.live))
    src.close()
    if open_now and feed_live == 0:
        failures.append("the market is OPEN but no symbol returned today's forming bar — "
                        "a live cycle would skip every symbol")
        print("  ⚠ no live (forming) bar while the session is open")
    elif not open_now:
        print("  market closed — closed-bar mode; a forming bar is not expected")

    # ── 4. one real scan cycle ────────────────────────────────────────────
    stage("4 · one real scan cycle → Telegram")
    with _store(cfg) as store:
        sc = _scanner(cfg, dry_run=not tg_ok and not args.no_message, store=store)
        rep = sc.scan(symbols=probe, live=None, send=True, progress=False)
        print(f"  {rep.summary_line}")
        print(f"  events in window {rep.events_in_window} · filtered {rep.filtered} · "
              f"skipped symbols {rep.skipped_symbols}")
        for st, ev in rep.alerts[:10]:
            print("   • " + ev.fmt())
        if rep.dispatch is not None and not isinstance(rep.dispatch, dict):
            print(f"  delivery: {rep.dispatch}")
            if getattr(rep.dispatch, "undelivered", 0):
                failures.append(f"{rep.dispatch.undelivered} alert(s) found but not delivered — "
                                f"see the errors above")
        for note in rep.notes:
            if note.startswith(("DELIVERY PROBLEM", "NOT SENT", "nothing to send", "also filtered")):
                print(f"  note: {note}")
        if rep.alerts and tg_ok:
            print(f"  ✅ {len(rep.alerts)} alert(s) delivered — check the chat")
        elif not rep.alerts and not failures:
            print("  ℹ no signal in this window — that is a quiet market, not a broken scanner. "
                  "Widen it with --recent-bars 3 --events tap1,tap,approach,confirmed")
        sc.close()

    print("\n" + ("=" * 62))
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  ✗ " + f)
        return 1
    print("RESULT: PASS — transport, feed and scanner all healthy")
    return 0


def cmd_export(args) -> int:
    cfg = _cfg(args)
    uni = cfg.data.universe or read_universe(cfg.data.universe_file, suffix=cfg.data.symbol_suffix)
    if args.limit:
        uni = uni[: args.limit]
    out = Path(args.out or cfg.data.history_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = DataSource(cfg.data, live=False)
    frames = src.get_many(uni, lookback_days=args.days or 2000, workers=args.workers, progress=True)
    written = 0
    for sym, b in frames.items():
        if not b.ok:
            print(f"  {sym}: {b.error[:80]}")
            continue
        frame_to_csv(b.df, out / f"{sym}.csv")
        written += 1
    print(f"wrote {written}/{len(uni)} CSVs to {out} — set data.provider: csv to use them offline")
    return 0


def cmd_demo(args) -> int:
    """Fully offline end-to-end demo: synthetic NSE-style data → scan → Telegram-shaped
    messages → backtest report. Nothing here touches the network."""
    cfg = _cfg(args)
    cfg.data.provider = "csv"
    cfg.data.history_dir = "data/csv_demo"
    cfg.data.cache_dir = "data/cache_demo"
    cfg.alert.chart = not args.no_charts
    cfg.alert.min_liquidity_dollar_volume = 0.0   # synthetic volumes are toy numbers
    cfg.alert.min_price = 0.0
    cfg.state_db = "data/state_demo.sqlite3"
    syms = ("RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS", "SBIN.NS")
    write_demo_dataset(cfg.data.history_dir, symbols=syms, n=700, tap_on_last_bar=4)
    cfg.data.universe = list(syms)
    print("synthetic NSE-style history written to " + cfg.data.history_dir + "\n")
    with StateStore(":memory:") as store:
        sc = _scanner(cfg, dry_run=True, store=store)
        rep = sc.scan(live=False, send=True, progress=False)
        print(f"scan: {rep.summary_line}\n")
        print(rep.console_table(10) + "\n")
        print(f"alerts: {len(rep.alerts)}")
        for st, ev in rep.alerts[:4]:
            ctx = sc._context(st, ev)
            print("─" * 34)
            print(render_message(ev, ctx, cfg.alert, cfg.params, parse_mode="plain"))
        print("─" * 34)
        from .backtest import Backtester
        frames = {s: b.df for s, b in sc.source.get_many(list(syms), progress=False).items() if b.ok}
        res = Backtester(cfg, progress=False).run(frames)
        paths = write_backtest_report(res, cfg, "results_demo", tag="demo")
        m = res.metrics or {}
        print(f"\nbacktest: {m.get('signals', 0)} trades · win {m.get('win_rate', 0):.1f}% · "
              f"avg {m.get('avg_r', 0):.2f}R · PF {m.get('profit_factor') or 0:.2f} · "
              f"ret {m.get('return_pct') or 0:.1f}% · maxDD {m.get('max_drawdown_pct') or 0:.1f}%")
        print("report: " + ", ".join(paths.as_list()))
        print("\n" + summary_message(res, cfg))
        sc.close()
    return 0


def cmd_selftest(args) -> int:
    from .selftest import run_selftest
    print("Pine-parity checks (hand-computed golden fixtures):")
    return 1 if run_selftest(verbose=not args.quiet) else 0


# ─────────────────────────────────────────────────────────────────────────────
# parser
# ─────────────────────────────────────────────────────────────────────────────

def _common(sp) -> None:
    sp.add_argument("-c", "--config", default=None, help="YAML config (default ./config.yaml)")
    sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="dotted override, e.g. --set indicator.min_rvol=2.2")
    sp.add_argument("--env-file", default=None, help="dotenv path (default ./.env)")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.add_argument("-q", "--quiet", action="store_true")


def _data_opts(sp) -> None:
    sp.add_argument("-S", "--symbols", nargs="*", default=None, help="override the universe")
    sp.add_argument("--provider", default=None, help="yfinance | yahoo | csv | synthetic")
    sp.add_argument("--days", type=int, default=None, help="history window in days")
    sp.add_argument("--limit", type=int, default=0, help="truncate the universe")
    sp.add_argument("--workers", type=int, default=6, help="download threads")
    sp.add_argument("--min-dollar-volume", type=float, default=None,
                    help="filter: 20d median turnover in ₹ crore")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="precision-tap", description=BANNER,
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("init", help="write config/.env/universe skeletons")
    _common(p); p.set_defaults(fn=cmd_init)

    p = sub.add_parser("doctor", help="environment + config diagnostics")
    _common(p); p.add_argument("--net", action="store_true", help="also probe the data provider")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("verify", help="print every zone/event for one symbol (TV cross-check)")
    _common(p); p.add_argument("symbol"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--events", default=None, help="comma list filter, e.g. tap,confirmed")
    p.add_argument("--bars-only", type=int, default=0, help="only zones born in the last N bars")
    p.add_argument("--no-charts", action="store_true")
    _data_opts(p); p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("scan", help="one scan cycle -> alerts")
    _common(p); _data_opts(p)
    p.add_argument("--live", action="store_true", help="force intrabar (forming-bar) mode")
    p.add_argument("--eod", action="store_true", help="force closed-bar mode (indicator-exact)")
    p.add_argument("--end", default=None, help="historical replay up to this date")
    p.add_argument("--recent-bars", dest="recent_bars", type=int, default=None,
                   help="alert on events within the last N bars (default 1)")
    p.add_argument("--no-send", action="store_true", help="print instead of sending")
    p.add_argument("--report-only", action="store_true", help="compute + store, send nothing")
    p.add_argument("--retried", action="store_true", help="flush the retry queue first")
    p.add_argument("--messages", action="store_true", help="also print rendered messages")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--events", default=None, help="override alerts.events")
    p.add_argument("--no-charts", action="store_true")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("run", help="always-on loop for the live market")
    _common(p); _data_opts(p)
    p.add_argument("--once", action="store_true", help="single cycle then exit (cron-friendly)")
    p.add_argument("--no-send", action="store_true")
    p.add_argument("--no-heartbeat", action="store_true")
    p.add_argument("--no-logfile", action="store_true")
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--duration", type=float, default=0, help="stop after N seconds")
    p.add_argument("--events", default=None)
    p.add_argument("--recent-bars", dest="recent_bars", type=int, default=None)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("backtest", help="historical replay of Tap-1 signals")
    _common(p); _data_opts(p)
    p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--trigger", choices=["tap1", "tap", "confirmed", "tap1_or_confirmed"])
    p.add_argument("--target-r", dest="target_r", type=float)
    p.add_argument("--stop-mode", dest="stop_mode", choices=["touch", "close"])
    p.add_argument("--trail", dest="trail", choices=["none", "breakeven", "chandelier", "structure"])
    p.add_argument("--time-stop", dest="time_stop", type=int)
    p.add_argument("--risk", type=float, help="risk per trade, %% of equity")
    p.add_argument("--capital", type=float)
    p.add_argument("--max-positions", dest="max_positions", type=int)
    p.add_argument("--out", default=None, help="report directory")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--no-charts", action="store_true")
    p.add_argument("--no-events", action="store_true", help="skip events.csv")
    p.add_argument("--send-summary", action="store_true", help="post the summary to Telegram")
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("sweep", help="parameter grid search")
    _common(p); _data_opts(p)
    p.add_argument("--grid", action="append", help="dotted.key=v1,v2 (repeatable)")
    p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--no-charts", action="store_true")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("zones", help="nearest live OB per symbol")
    _common(p); _data_opts(p)
    p.add_argument("--symbol"); p.add_argument("--live", action="store_true")
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(fn=cmd_zones)

    p = sub.add_parser("alerts", help="recent alerts from the state db")
    _common(p); p.add_argument("--limit", type=int, default=40); p.add_argument("--symbol")
    p.set_defaults(fn=cmd_alerts)

    p = sub.add_parser("telegram-test", help="ping the bot")
    _common(p); p.add_argument("--discover", action="store_true")
    p.add_argument("--timeout", type=float, default=25.0)
    p.add_argument("--text", default=None)
    p.set_defaults(fn=cmd_telegram)

    p = sub.add_parser("livecheck",
                       help="end-to-end live check: config → Telegram → feed → one real cycle")
    _common(p); _data_opts(p)
    p.add_argument("--probe", type=int, default=6, help="how many universe symbols to probe")
    p.add_argument("--no-message", action="store_true", help="do not send the transport test ping")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_livecheck)

    p = sub.add_parser("export-data", help="cache history to CSV for offline runs")
    _common(p); _data_opts(p)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("demo", help="offline end-to-end demo (synthetic data)")
    _common(p); p.add_argument("--no-charts", action="store_true")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("selftest", help="Pine-parity checks")
    _common(p); p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    cfg_level = "INFO"
    try:
        probe = load_config(getattr(args, "config", None), overrides=getattr(args, "set", ()) or ())
        cfg_level = ("DEBUG" if getattr(args, "verbose", False)
                     else "WARNING" if getattr(args, "quiet", False) else probe.log_level)
        setup_logging(cfg_level, probe.live.log_file if args.cmd in ("run",) else None)
    except Exception as exc:
        if getattr(args, "verbose", False):
            raise
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if args.cmd not in ("run",):
        setup_logging(cfg_level)
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:                                # top-level: show a usable error
        if os.environ.get("PRECISION_TAP_DEBUG") or getattr(args, "verbose", False):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("(re-run with -v for a traceback)", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
