# PRECISION-TAP — NSE/BSE daily scanner + backtest engine (Telegram alerts)

A Python port of the TradingView Pine v6 indicator in [`INDICATOR.txt`](INDICATOR.txt)
— *"Institutional OB — Precision Tap & Pre-Order"* — plus a **live market scanner** and a
**backtest engine** built on the exact same engine code.

Scan NSE equities on the **daily** timeframe and get a **Telegram alert the moment price taps a
precision order block (Tap 1)**, with every rule of the indicator enforced: volume-confirmed
displacement, range/ATR expansion, body fraction, close-location value, structure break,
origin-candle selection, zone geometry, front-run buffer, minimum zone age, departure gate,
touch condition, optional sweep and duplicate rejection.

```
🎯 PRECISION OB · TAP 1
TCS.NS · NSE · 1d · 2024-09-06
──────────────────────────────
📌 Price        131.71  (-0.14%)
🟢 Buy limit    132.32   ← Tap 1 trigger
🔴 Stop         129.20   (-2.36%)
⬜ OB zone       131.84 → 129.60
🎯 Targets      R1 135.43 · R2 138.55 · R3 141.67
📏 Dist to lvl  -0.61 (-0.20 R)
──────────────────────────────
zone age 3 bars · taps 1/4 · departure ✔ · RVOL 0.7× · ATR 2.60 (2.0%)
origin candle 2024-09-02 · Open to low
defence: close > 131.84 within 3 bars, RVOL ≥ 1.3×, CLV ≥ 0.65
──────────────────────────────
[ 📈 Chart ]  [ 🔎 Quote ]        ← inline buttons (+ a candlestick chart with the OB drawn on it)
```

* **Logic parity:** the port is bar-exact, verified by 18 hand-computed fixtures
  (`python -m precision_tap.selftest`) — see [ANALYSIS.md](ANALYSIS.md) for the rule-by-rule derivation.
* **Live:** polls during the NSE session (intraday touch = TradingView "Once Per Bar") and runs
  scheduled scans at/after the close (closed-bar = "Once Per Bar Close").
* **No repaint surprises:** zones are only *created* on closed bars, exactly like the indicator.
* **Telegram:** HTML or MarkdownV2, inline buttons, chart PNG, per-chat throttling, 429 backoff,
  and a SQLite retry queue so an outage delays an alert instead of losing it.
* **Backtest:** same engine replayed over history → R-multiples, forward-return signal study,
  ₹-compounding equity, per-symbol/year/exit breakdown, markdown + CSV + PNG reports, grid sweep.

---

## 1. Install

```bash
git clone <this repo> && cd PRECISION-TAP
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # pandas, numpy, requests, PyYAML, yfinance (+ matplotlib, lxml)
cp config.example.yaml config.yaml       # or:  python -m precision_tap init
```

Python ≥ 3.9. `matplotlib` is only needed for chart attachments / the equity PNG;
`lxml` only for `universe_file: nifty500`-style scraping.

## 2. Telegram (2 minutes)

1. [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.
2. Message your bot anything (`/start`) so a chat exists.
3. ```bash
   echo 'TELEGRAM_BOT_TOKEN=123456:AA...' >> .env
   python -m precision_tap telegram-test --discover   # prints TELEGRAM_CHAT_ID
   echo 'TELEGRAM_CHAT_ID=111888' >> .env
   python -m precision_tap telegram-test               # → "ok msg_id=…"
   ```
   For a group/channel, add the bot and use its `-100…` id. No network? Test the whole delivery
   path locally: `python tools/mock_telegram_server.py 8099` + `--set telegram.api_base=http://127.0.0.1:8099`.

## 3. Sanity checks before going live

```bash
python -m precision_tap doctor --net      # deps, universe, tz/session, token+chat, live data probe
python -m precision_tap selftest          # 18 Pine-parity checks (offline, deterministic)
python -m precision_tap demo              # full offline pipeline on synthetic NSE-style data
python -m precision_tap scan --no-send    # one real cycle, prints instead of sending
python -m precision_tap verify RELIANCE.NS   # every zone + every event, for TradingView diffing
```

`verify` is the parity workbench: it prints each zone's `born / origin / top / bot / entry / stop /
state / taps` and the closed-bar `plotshape` date lists (`displacement→OB created`, `anyTap`,
`anyApproach`, `anyConfirm`, `anyDead`) so you can line them up against the chart in minutes.
Add `--end 2025-09-05 --start 2024-01-01 --bars-only 60` to zoom into a specific session.

## 4. Commands

| Command | What it does |
|---|---|
| `scan` | one cycle over the universe → alerts. `--live` / `--eod`, `--end YYYY-MM-DD`, `--recent-bars N`, `--no-send`, `--report-only`, `--messages`, `--top N` |
| `run` | **always-on loop**: intraday polling + scheduled scans + daily heartbeat. `--once` for a single cron cycle, `--duration SEC`, `--max-cycles N` |
| `backtest` | replay Tap-1 (or `confirmed`/`tap`) signals over history. `--start/--end`, `--trigger`, `--target-r`, `--stop-mode touch\|close`, `--trail none\|breakeven\|chandelier\|structure`, `--time-stop`, `--risk`, `--capital`, `--max-positions`, `--send-summary` |
| `sweep` | grid search: `--grid backtest.target_r=1.5,2.5,3.5 --grid indicator.min_rvol=1.5,1.8` |
| `zones` | watchlist table: nearest live OB level per symbol, sorted by distance in ATRs |
| `verify SYM` | full signal dump for one symbol (parity debugging) |
| `alerts` | recent alerts + retry queue from the state DB |
| `telegram-test` | ping / `--discover` the chat id |
| `export-data` | cache history to `data/csv/*.csv` for offline backtests (`--provider csv` to use it) |
| `demo` / `selftest` / `doctor` / `init` | offline demo / parity checks / diagnostics / scaffolding |

Common flags on every command: `-c config.yaml`, `--set section.key=value` (repeatable, dotted —
e.g. `--set indicator.min_rvol=2.2 --set alerts.events=tap1 --set data.provider=csv`),
`-S RELIANCE TCS INFY` (ad-hoc universe; `.NS` added automatically), `--provider`, `--days`,
`--limit`, `--workers`, `-v/-q`.

## 5. Configuration (`config.yaml`)

Every key under `indicator:` is the TradingView input with the same name and the same default — so a
saved TradingView preset transplants 1:1. Full semantics, including the *why*, in [ANALYSIS.md](ANALYSIS.md).

```yaml
data:
  provider: yfinance           # live source
  symbol_suffix: ".NS"         # ".BO" for BSE
  universe_file: universe/nse.txt   # or nifty50 | nifty100 | nifty200 | nifty500 | sensex30
  lookback_days: 900
  live_intraday_bar: true      # rebuild today's bar from 5m ticks (needed for touch alerts)
  corporate_adjustments: true  # back-adjusted OHLC, like TradingView's default
  tick_sizes: {".NS": 0.05, ".BO": 0.05}    # syminfo.mintick → front-run ticks

indicator:
  min_rvol: 1.8        min_range_atr: 1.20   min_body_frac: 0.55   min_clv: 0.72
  structure_len: 8     origin_search: 8      neutral_body: 0.20
  zone_method: "Open to low"                 # | Body to low | Body only | Lower half of candle
  entry_mode: "Proximal"                     # | 50% | 62% | 70.5% | 79% | Distal + 1 tick
  frontrun_mode: "Auto"                      # buy level ABOVE the visible edge; | ATR | Ticks | Off
  stop_atr: 0.15       approach_atr: 0.25     min_age: 3       max_touches: 4
  raise_after_first_tap: true                 # lift the next pre-order above the defended low
  require_departure: 1.0                      # price must first leave the OB by 1 ATR
  confirm_bars: 3      confirm_rvol: 1.3      confirm_clv: 0.65   confirm_bos_len: 3
  require_sweep: false                        # true → fewer, later, higher-quality taps

alerts:
  events: [tap1, approach, confirmed]         # new_ob | tap1 | tap | approach | confirmed | invalidated
  recent_bars: 1         # only the newest bar(s)
  match_indicator_100: true    # live mode also replays the last CLOSED bar (nothing can be missed)
  skip_stale_bars: true        # never alert when the last bar isn't today's session (holidays)
  once_per_symbol_per_day: true
  min_liquidity_dollar_volume: 5000000000     # ~₹5,000 crore median 20d turnover
  min_price: 20
  chart: true
```

## 6. Running it on the live market

```bash
python -m precision_tap run                  # supervised loop (recommended)
```

The loop wakes every `live.intraday_poll_minutes` (10) during 09:15–15:30 IST, and runs full scans at
`live.scan_times` (`09:35`, `11:30`, `13:30`, `15:35`, `16:10` — the 15:35/16:10 pair catches the
closing print and the settled EOD bar). It survives restarts (the SQLite ledger remembers what was
already sent), retries queued alerts, and sends a 15:45 heartbeat so silence is never ambiguous.
Logs go to `logs/precision_tap.log`, plus `results/scan_<mode>_<stamp>.md` per cycle.

**systemd** (`deploy/systemd/precision-tap.service`), **Docker/compose** (`deploy/`) and a
**cron** variant (`--once`) are included — see `deploy/README.md`.

## 7. Backtesting

```bash
python -m precision_tap backtest --start 2021-01-01 --end 2025-09-01
python -m precision_tap backtest --provider csv --trigger confirmed --target-r 3 --stop-mode close
```

Reports land in `results/`: `summary_*.md` (R-stats, money metrics, exit mix, per-year, per-symbol,
best/worst, settings echo), `trades_*.csv` (every fill, stop, R, MAE/MFE, ₹ P&L),
`signal_study_*.csv` (Tap-1 forward returns vs. an all-bars baseline), `equity_*.png`.

Honesty notes: the indicator defines **no take-profit**, so `target_r`/trailing/time-stop are the
backtest's assumptions, not the indicator's. Bars are daily, so any bar touching both stop and target
is resolved **worst path** (stop first) — that biases results *down*, deliberately. Vendor data
differences vs. TradingView can move a single marginal `rvol ≥ 1.8` gate; nothing else is ambiguous.

## 8. Layout

```
precision_tap/
  engine.py     ← the Pine state machine (zones, taps, defence, invalidation) — the parity surface
  series.py     ← Pine-exact ta.atr/sma/rma/highest/lowest/tr primitives
  params.py     ← every input mirrored 1:1, plus trade/alert/telegram/live/data config
  selftest.py   ← 18 hand-computed parity fixtures (also run by pytest)
  data.py       ← yfinance/yahoo/csv/synthetic providers, NSE universe presets, cache, intraday bar
  scanner.py    ← live scan: universe → engine → filters → dedupe → dispatch
  alerts.py     ← filtering, Telegram message rendering, dispatch, retry queue
  telegram.py   ← Bot API client (throttle, 429 backoff, 4096-split, MarkdownV2 escape, discovery)
  backtest.py   ← signal replay + conservative fill model + forward-return signal study
  metrics.py    ← R/money metrics, portfolio accounting, equity curve maths
  report.py     ← markdown/CSV/PNG reports + the Telegram digest
  live.py       ← session-aware scheduler (polling, scan times, heartbeat, signals)
  state.py      ← SQLite: alert ledger, dedupe, retry queue, zone memory, run log
  cli.py        ← the commands above
tests/          ← pytest: parity wrappers, yfinance adapter (fake), Telegram (local mock), scanner/backtest
universe/nse.txt, config.example.yaml, tools/mock_telegram_server.py, deploy/
```

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `Telegram not configured` | `.env` needs `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`; check `python -m precision_tap telegram-test` |
| No alerts at all, ever | the gates are strict by design: try `--set indicator.min_rvol=1.4 --set indicator.min_clv=0.65`, and check `alerts.min_liquidity_dollar_volume`/`min_price`; `verify SYM` shows what the engine sees |
| Alerts on the wrong day / holidays | keep `alerts.skip_stale_bars: true`; NSE holidays produce no new bar |
| `no data` / rate limited | raise `data.retry_max`, lower `data.rate_limit_per_sec`, or `export-data` once and run `--provider csv` |
| One symbol differs from TradingView | compare `data.corporate_adjustments` (adjusted vs raw) and the exact `mintick`; run `verify SYM --end <date>` |
| Too many messages | `alerts.events: [tap1]`, `once_per_symbol_per_day: true`, `daily_limit: 10`, `alerts.min_liquidity_dollar_volume` up |
| Need it offline | `python -m precision_tap export-data`, then `--provider csv` everywhere |

## 10. Disclaimer

Research tooling — not investment advice, not a trading system, and no execution. The indicator is a
*level-finding* tool: the market does not owe you a defence at the OB, so size for the printed stop.
Daily-bar backtests cannot see intrabar sequence, so treat the numbers as an order-of-magnitude read,
not a forward return expectation.
