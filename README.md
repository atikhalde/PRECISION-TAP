# PRECISION-TAP — NSE/BSE daily scanner + backtest engine (Telegram alerts)

A Python port of the TradingView Pine v6 indicator in [`INDICATOR.txt`](INDICATOR.txt)
— *"Institutional OB — Precision Tap & Pre-Order"* — plus a **live market scanner** and a
**backtest engine** built on the exact same engine code.

> **Scope, enforced in code (not just config):** `data.market` must be **NSE or BSE**, `data.interval`
> must be **1d**, `symbol_suffix` must be **`.NS`/`.BO`** — anything else raises at startup. Live data
> comes from **yfinance**. Bare tickers are assumed NSE (`RELIANCE` → `RELIANCE.NS`); a foreign ticker
> (`AAPL.OQ`) is rejected rather than silently scanned. That keeps the alert set exactly comparable to
> the TradingView chart you are looking at.

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
defence: closed bar within 3 bars of the tap · close > open · close > 131.84
         CLV ≥ 0.65 · RVOL ≥ 1.3× · close > 3-bar high
──────────────────────────────
[ 📈 Chart ]  [ 🔎 Quote ]        ← inline buttons (+ a candlestick chart with the OB drawn on it)
```

When the tapped level then **holds** (a closed bar back above the OB with volume, CLV and a
micro break of structure — the indicator's `DEFENCE CONFIRMED` label), a **separate** message
follows in the same chat. It is a verdict on the tap, not a new order, so it has its own layout:

```
🛡 OB DEFENCE CONFIRMED
TCS.NS · NSE · 1d · 2024-09-10
──────────────────────────────
📌 Price        134.10  (+1.90%)
✅ Defended     132.32   ← Tap 1 held
⬜ OB zone       131.84 → 129.60
🔴 Stop         129.20   (-3.65% from close)
📈 vs OB top    +2.26   (+1.71% above 131.84)
🎯 Targets      R1 135.44 · R2 138.56 · R3 141.68
📏 Open P&L     1.78 (0.57 R from Tap 1)
──────────────────────────────
confirmed 2 bars after the tap (window 3) · closed bar ✔ · close 134.10 > open 131.60 ✔
CLV 0.81 (≥ 0.65) ✔ · RVOL 1.8× (≥ 1.3) ✔ · close > OB top 131.84 ✔
micro-BOS close 134.10 > 3-bar high 133.68 ✔ (by 0.42)
zone age 12 bars · taps 1/4 · state → confirmed
origin candle 2024-09-02 · Open to low
next pre-order 132.90 if price revisits the block
defence confirmed — manage the open trade; this is not a fresh entry
```

* **Logic parity:** the port is bar-exact, verified by 21 hand-computed fixtures
  (`python -m precision_tap.selftest`) plus a line-by-line differential against an independent
  transcription of `INDICATOR.txt` (`tests/pine_reference.py`, run by `pytest -q`). The defence
  gates get their own markets: taps followed by *near-miss* defence bars (bearish close, low CLV,
  no volume, close under the OB top, close under the 3-bar high) and a block that is **defended
  twice**, so deleting a gate shows up as a difference rather than as silence. See
  [ANALYSIS.md](ANALYSIS.md) for the rule-by-rule derivation.
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
python -m precision_tap selftest          # 21 Pine-parity checks (offline, deterministic)
python -m precision_tap demo              # full offline pipeline on synthetic NSE-style data
python tools/live_drill.py                # the LIVE path offline: fake Yahoo feed → mock Telegram
python tools/signal_audit.py              # are the alerts *valid*? realistic (not engineered) market
python tools/mutation_probe.py            # can the suite *see* a broken defence rule? (deletes each gate)
python -m precision_tap scan --no-send    # one real cycle, prints instead of sending
python -m precision_tap livecheck         # config → real Telegram → real feed → one real cycle
python -m precision_tap verify RELIANCE.NS   # every zone + every event, for TradingView diffing
```

`tools/signal_audit.py` is the answer to *"are the alerts it sends actually any good?"*. The other
offline tools all run on `data.synthetic_frame`, which plants a displacement → OB → Tap-1 sequence
every 41 bars, so they can never tell you what a real market does. The audit generates data shaped
like real NSE dailies instead (volatility clustering, autocorrelated volume, no planted setups) and
reports the gate pass-through, the alert rate a 127-name universe should expect per day — so "quiet
chat" can be compared against a number — and a validity grade for every alert the scanner would
actually send: did the bar touch the level, was the zone alive, is the advertised stop already
breached, did price run away from the entry.

`livecheck` is the answer to *"is it actually working on the live market?"*. It talks to the real
Bot API and the real data provider, sends one test message, and prints a per-stage PASS/FAIL —
so "no alert arrived" is never a guessing game again. A dry `scan --no-send` first is safe: those
alerts are recorded as *not delivered*, so the next cycle with a working bot still sends them.

`tools/live_drill.py` is the one that catches "it works on my machine but not on the market":
it drives the real provider code (Yahoo chart API *and* yfinance, both faked) through the
intraday rebuild of today's forming bar, the scanner, dispatch and the Bot API, and exits
non-zero unless every alert it found was delivered. Two cycles by default, so it also
asserts the de-dupe ledger stops the second one re-sending. It needs no network at all —
which is why it runs in CI on every push.

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
| `livecheck` | **end-to-end live check** — config → real Bot API round trip → real feed → one real cycle, and names the first thing that fails |
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

**Scan mode follows the market, not the trigger.** Any cycle that lands inside 09:15–15:30 runs in
*live* mode: it rebuilds today's forming bar from `data.intraday_interval` ticks, so an intrabar tap
is caught the way TradingView's "Once Per Bar" alert is. Everything after the close runs in *eod*
mode on settled bars. Both modes also replay the frame with the last bar closed
(`alerts.match_indicator_100`), which is the only way a `confirmed` (OB defence) alert can be
produced — the indicator only confirms on a closed bar.

Live and closed-bar frames are cached under **separate keys**, and a cached frame is only reused if
its newest bar is the session the feed should already have — so a mid-session poll can never be
served last night's EOD frame. Keep `data.eod_cache_max_age_minutes` shorter than the gap between
your last two post-close `scan_times`, or the "settled bar" scan just re-reads the pre-close fetch.

**systemd** (`deploy/systemd/precision-tap.service`), **Docker/compose** (`deploy/`) and a
**cron** variant (`--once`) are included — see `deploy/README.md`.

### Hosted alternative: GitHub Actions

If you would rather not run a server, [`.github/workflows/live-scan.yml`](.github/workflows/live-scan.yml)
runs the same scanner on GitHub's runners against the real market and posts to the same chat.
Add two repository secrets and it is live:

| Secret / variable | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` *(secret)* | from [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` *(secret)* | `python -m precision_tap telegram-test --discover` |
| `SCAN_LIMIT` *(variable, optional)* | cap the universe to the first N symbols while you tune |
| `SCAN_SYMBOLS` *(variable, optional)* | ad-hoc universe, e.g. `RELIANCE,TCS,INFY` |
| `SCAN_EVENTS` *(variable, optional)* | e.g. `tap1` to cut the volume |
| `SCAN_MIN_DV` *(variable, optional)* | 20d median turnover floor in ₹ crore |
| `SCAN_RECENT_BARS` *(variable, optional)* | alert window in bars — config default `1`; `3`–`5` reviews the week after a weekend or holiday |

**It does *not* rely on cron for the cadence.** Cron is used only to *kick* the job; the job then
owns the session itself with the same always-on loop as `run` — intraday polling every
`live.intraday_poll_minutes` plus the `live.scan_times` closed-bar prints, each cycle picking
live-vs-closed-bar from the exchange clock. You can also start one by hand from the **Actions** tab
(one cycle by default, or the whole session with `loop=true`), with a forced `live`/`eod` mode, a
dry-run toggle, and `recent_bars` to widen the alert window — the way to ask *"what did I miss this
week?"* after a weekend or a holiday, since a default one-bar cycle can only re-read the last
completed session. The ledger still dedupes whatever an earlier cycle delivered, so a wider window
re-reports, it does not re-send.

Why the change matters: this workflow used to declare 29 cron slots a day, one every 15 minutes
across the session. Measured on its first trading day, **15 slots were due, 5 fired, every one of
them 1–4 hours late, and not one landed inside the NSE session** — so the intraday tap stream, the
entire point of the product, never executed, while all 5 runs reported success. GitHub's `schedule`
trigger is explicitly best-effort: it delays under load, drops runs when load stays high, and
deprioritises low-traffic repositories. Fifteen precise slots is fifteen independent chances to be
skipped; one kick that covers hours needs only one.

Five staggered kicks (09:05 / 11:25 / 13:05 / 14:55 / 15:25 / 16:15 IST, at off-the-hour minutes
because the top of the hour is GitHub's documented high-load moment) each run until 17:20 IST or
the 5h30m job cap, handing over to one another through the `concurrency` queue. A kick that starts
after the window has closed degrades to a single catch-up cycle, so redundancy costs about a minute
each and the ledger makes them idempotent. Coverage degrades instead of collapsing:

| kicks that reach a runner | session covered | post-close prints | digest |
|---|---|---|---|
| all six | 100% | ✔ | ✔ |
| any one of 11:25 / 13:05 / 14:55 / 15:25 / 16:15 | partial | ✔ | ✔ |
| only 09:05 | 85% (09:05→14:35) | ✘ | ✘ |
| none | — | ✘ | ✘ |

Three things to know before you rely on it:

* **The dedupe ledger is carried between runs with `actions/cache`.** Runners are ephemeral, and
  `data/state.sqlite3` is the only thing that stops the same Tap 1 being re-sent by the next cycle.
  A cache miss is survivable but re-delivers that day's alerts once — the run says so in a
  `::warning::` annotation rather than failing silently.
* **GitHub's cron can still drop every kick.** Five independent chances is far better than fifteen
  fragile ones, but it is not a guarantee, and the `only 09:05` row above is a real residual gap.
  For alerts you must not miss, run the systemd/Docker loop on a host you own — this workflow is
  the zero-maintenance option, not the guaranteed one.
* **A session loop holds one runner for hours.** Free on public repos; on a private repo that is a
  large slice of the 2000 min/month allowance — lower `JOB_CAP_MINUTES` or trim the universe with
  `SCAN_LIMIT`.

A failing run posts to Telegram, and the first job to finish after 16:40 IST posts an end-of-day
digest (idempotent through the ledger) so a quiet market is visibly different from a dead pipeline.
On a **non-trading day** any job posts it rather than waiting for 16:40 — there is no later slot to
protect, and a manual weekend dispatch is exactly when the chat is silent *and* unexplained. The
digest names the session it evaluated and counts that session's alerts by `bar_date`, so a Saturday
run answers "did Friday produce anything, and was it delivered?" instead of "no signals fired today".


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
  selftest.py   ← 21 hand-computed parity fixtures (also run by pytest)
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
tests/          ← pytest: parity wrappers, yfinance adapter (fake), Telegram (local mock),
                  scanner/backtest, and the **live path** (feed freshness, scheduler, delivery)
  pine_reference.py    ← a second, literal transcription of INDICATOR.txt (independent of
                         engine.py/series.py) — the diff *is* the parity proof
  test_pine_reference.py ← fuzzes engine ↔ transcription over 16 parameter sets × 5 markets
  test_alert_delivery.py ← "ran fine, chat silent" regressions (ledger, 4xx, quiet-cycle notes,
                         and the session a silent cycle actually evaluated)
  test_live_scan_workflow.py ← the hosted workflow: silent-green regressions (a step reading its
                         own outputs, a digest gated out on weekends, embedded-python syntax)
universe/nse.txt, config.example.yaml, deploy/
tools/live_drill.py        ← offline rehearsal of the LIVE path (fake Yahoo feed + mock Bot API)
tools/mutation_probe.py    ← deletes each `defence` gate in turn to prove the tests notice
tools/mock_telegram_server.py   ← local stand-in for api.telegram.org
.github/workflows/   ← ci.yml (offline parity + tests) and live-scan.yml (real market → Telegram)
```

## 9. Troubleshooting

Run `python -m precision_tap livecheck` first — it walks config → Telegram → feed → one real
cycle against the live market and prints `RESULT: PASS` or names the first thing that failed.

| Symptom | Fix |
|---|---|
| `Telegram not configured` | `.env` needs `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`; check `python -m precision_tap telegram-test` |
| Token set, chat silent, runs are green | `getMe` passing only proves the *token* — validate the chat too: `python -m precision_tap telegram-test --validate-only` (no message sent). A wrong chat id / un-started bot fails here instead of eating alerts |
| Two green runs, no alert — and it is a weekend or holiday | **Expected, and now labelled.** `alerts.recent_bars: 1` means a cycle evaluates exactly **one bar**, and on a non-trading day that bar is the *previous* session — so two runs are not two chances, they are the same bar read twice. The cycle names the bar it looked at: `MARKET CLOSED` (Sat/Sun, or a day outside `live.trading_days`), `NO FRESH SESSION` (a trading day, but the EOD print has not settled yet, or you forced `--eod` mid-session), `FEED BEHIND` (the newest bar is older than the session the feed should already have — an NSE holiday, or a lagging provider). The GitHub step summary prints **session evaluated** on the run page, and a non-trading day is annotated with a `::warning::` so it is visible without opening the log. On a weekend the digest is posted by *any* job (there is no later slot to protect) and reports that session's alerts. To scan more than one bar: `alerts.recent_bars: 2-3`, or the `recent_bars` input on a dispatched run (`SCAN_RECENT_BARS` to make it the default) |
| `SCAN FAILED … exit 2` | the universe resolved to **0 symbols** — a preset (`nifty500`, `allnse`…) whose every download source failed, or an empty/commented file. nseindia archives reject datacenter IPs (GitHub runners included); the presets fall back to Wikipedia tables automatically, but if all sources fail, point `data.universe_file` at a local file (`universe/nse.txt`) |
| `SCAN FAILED … exit 3` | the feed failed outright (`usable=0`, mostly errors) — provider outage or misconfiguration. A holiday-shaped cycle (fetches ok, no fresh bar) stays green by design |
| `SCAN FAILED … exit 4` | signals were found but **not** delivered. Two shapes: `DELIVERY PROBLEM` — Telegram rejected the send (usually a 400/403 chat problem); or `NOT SENT … log-only=N` — there was no transport to send with (`telegram.enabled: false`, or the token/chat never reached the process). Fix it and the alerts are re-offered next cycle — a queued row is never held against you by a transport that cannot drain it |
| `no usable symbols — …` (green) | the feed has no fresh bar — NSE holiday or a lagging vendor. Check the per-symbol notes; the next cycle recovers on its own |
| `FEED PROBLEM — …` | majority of symbols failed to fetch; the scan exits 3 so schedulers show red. The yfinance→yahoo per-symbol failover (`data.fallback_provider`) already tried |
| Ran fine, chat is silent, `sent=0 … skipped=N` | you looked at a dry cycle first (`scan --no-send`, or a run with no token). Those alerts are *logged, not delivered*, and the ledger marks them given-up rather than sent — the next cycle with a working bot still sends them. If you are on an older build, `DELETE FROM alerts WHERE sent=2;` in `data/state.sqlite3` clears them |
| `DELIVERY PROBLEM — … 401 Unauthorized` / `400 chat not found` / `403 bot can't initiate…` | permanent Telegram rejection: the token is wrong, the chat id is wrong, or you never pressed **Start** on the bot (in a group, add it and make it an admin). It is logged at ERROR and *not* retried; fix the credential and the alert is re-offered on the next cycle |
| `cycle … → delivery sent=0` in the log | the always-on loop now logs the delivery outcome separately from the signal count — a cycle that found three taps and delivered none no longer looks healthy |
| No alerts at all, ever | first separate *quiet* from *broken*: the cycle summary says which — `nothing to send … no indicator signal` is a quiet market; `usable=0` / `UNIVERSE EMPTY` / `DELIVERY PROBLEM` is a broken pipeline. The summary line and that note both carry `bar=<session>` / `newest <session>`: if it is not today's date, read the `MARKET CLOSED` / `NO FRESH SESSION` / `FEED BEHIND` note above it first — the cycle evaluated an older bar and there was nothing new to find. Then remember the default window is deliberately one bar (`alerts.recent_bars: 1`) and Taps need a zone that is ≥3 bars old, was left by ≥1 ATR and is revisited *on the newest bar* — on a quiet session even 500 names can legitimately produce zero taps. Widen honestly: `alerts.recent_bars: 2-3`, add `new_ob`/`approach` to `alerts.events`, lower `alerts.min_liquidity_dollar_volume` (₹500cr muted the entire mid/small-cap market — the default is now ₹25cr), and scan the broad market (`universe_file: nifty500` or `allnse`) instead of a 127-name starter file. `verify SYM` shows what the engine sees |
| GitHub Actions runs are green but no alerts ever arrive, and the run times look random | `schedule` is **best-effort** — this repo measured 15 due slots, 5 fired, all 1–4 h late, none inside the NSE session, every run reporting success. The workflow no longer asks cron for a cadence: cron only *kicks* the job (09:05 / 11:25 / 13:05 / 14:55 / 15:25 / 16:15 IST) and the job runs the always-on loop until 17:20 IST or the 5h30m cap. Check `gh run list --workflow=live-scan.yml` for the `schedule` rows against those times; if a whole day is missing, GitHub dropped every kick — for a guarantee run systemd/Docker (`deploy/README.md`) |
| Alert says `TAP 1` but the zone is already broken | that was a *dead-on-arrival* tap: the same bar touched the entry **and** closed below the stop, so the level failed at the moment it was tapped, and with `include_invalidations: false` the failure was never sent — only the buy side was. `alerts.skip_dead_on_arrival: true` (default) drops them and the cycle tally says `level already failed N`. Set it `false` for strict Pine `anyTap` parity |
| An alert about yesterday's bar shows today's price / tap count | the live closed-bar parity pass used to replay the whole frame with today's *forming* bar treated as closed, and `Event.zone` is a live reference — so yesterday's alert was rendered with today's tap count, today's raised pre-order, even `state=dead`. The pass now replays `df.iloc[:-1]` and the alert context is taken from the event's own bar (`tests/test_alert_validity.py`) |
| Want the full NSE market scanned | `data.universe_file: nifty500` (default) covers ~92% of NSE market cap; `allnse` pulls every listed equity (~2 000+ symbols — EOD cycle ≈ 8 min at the default 5 req/s, intraday ≈ double; keep `live.intraday_poll_minutes` ≥ 15 and the workflow timeout at 45 min). nseindia downloads fail from datacenter IPs — the presets fall back to Wikipedia; from a local machine they hit the official CSVs |
| `alerts matched: 0` but zones are found | read the `nothing to send — …` note at the end of the cycle: it tallies *why* every event was dropped (`illiquid 12 · event type disabled 6`). A quiet market plus a ₹500 crore turnover floor plus `tap1`-only often filters everything; `--min-dollar-volume 100` or `--events tap1,approach,confirmed,tap` widens it |
| `scan --no-send` shows alerts but `run` sends nothing | `.env` is missing/unreadable, so `run` falls back to log-only. It says so once at startup, on every cycle, and now **exits 4** — from `scan` and `run --once` immediately, and from a `run --duration` session loop once at the end, since one process covers the whole session and the per-cycle code never reaches the scheduler. `doctor` prints the token/chat status. Under cron/systemd the process environment is empty — put the secrets in `/etc/precision-tap.env` (also honoured via `PRECISION_TAP_ENV_FILE`) so the scanner can read them without a shell profile |
| Alert chart shows no order block / no `TAP` marker | the overlay is drawn on a truncated window (`alerts.chart_bars` bars) and had to be rebased onto it — a zone born at bar 596 of a 600-bar frame landed off-canvas. Boxes, entry/stop lines and the marker are placed relative to the first *drawn* bar now (`tests/test_alert_chart.py`) |
| Cycle runs every 10 min but the notes say `skipped` | the feed's newest bar is not a recent session (`skip_stale_bars`). Check `data.provider`, the yfinance version, and `doctor --net` |
| Alerts repeat yesterday's session | the intraday rebuild failed, so today's bar is missing; the stale guard then suppresses it. Check `data.live_intraday_bar` / `data.intraday_interval` |
| Alerts on the wrong day / holidays | keep `alerts.skip_stale_bars: true`; NSE holidays produce no new bar |
| `no data` / rate limited | raise `data.retry_max`, lower `data.rate_limit_per_sec`, or `export-data` once and run `--provider csv` |
| `usable=0 errors=N` on every symbol | the provider call itself is failing, not the gates. Run `scan -S RELIANCE -v` and read the `fetch failed for …` debug lines: a yfinance upgrade (`progress`/`threads` removed in 1.0) or a tz-aware/naive index clash in the intraday rebuild both look exactly like this. `python tools/live_drill.py` reproduces both offline |
| Alerts fire but the buy limit is not the price that was touched | expected with `raise_after_first_tap: true` — the zone's pre-order is lifted *on* the tap bar, so the message prints the tapped level as the trigger and the raised level as the next pre-order |
| One symbol differs from TradingView | compare `data.corporate_adjustments` (adjusted vs raw) and the exact `mintick`; run `verify SYM --end <date>` |
| Too many messages | `alerts.events: [tap1]`, `once_per_symbol_per_day: true`, `daily_limit: 10`, `alerts.min_liquidity_dollar_volume` up |
| Need it offline | `python -m precision_tap export-data`, then `--provider csv` everywhere |

## 10. Disclaimer

Research tooling — not investment advice, not a trading system, and no execution. The indicator is a
*level-finding* tool: the market does not owe you a defence at the OB, so size for the printed stop.
Daily-bar backtests cannot see intrabar sequence, so treat the numbers as an order-of-magnitude read,
not a forward return expectation.
