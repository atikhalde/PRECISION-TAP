# `INDICATOR.txt` — deep analysis, and how the Python port guarantees parity

Source: `INDICATOR.txt` — *Institutional OB — Precision Tap & Pre-Order* (`indicator("OB Precision Tap")`, Pine **v6**,
`overlay=true`, `max_boxes_count=150`, `max_lines_count=300`, `max_labels_count=300`).

This document is the specification the Python implementation (`precision_tap/`) was written against. Every formula below
is taken from the Pine source, and every branch is covered by a hand-computed fixture in `precision_tap/selftest.py`
(18 checks, runnable offline with `python -m precision_tap.selftest`).

---

## 1. What the indicator actually does

It is **not** a generic order-block detector. It is a *pre-order execution tool* with one job: find the exact price at
which institutions left a **volume-confirmed bullish displacement**, anchor a **tight** bearish-origin zone underneath
that displacement, place a buy level **just above** the visible zone edge (because liquid markets front-run visible
levels), and then alert on the first time price returns to tap that level (**Tap 1**).

Five stages, in order:

```
 ┌──────────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌───────────────────┐   ┌──────────────┐
 │ 1 DISPLACEMENT        │→ │ 2 ORIGIN CANDLE  │→ │ 3 ZONE + LEVELS   │→ │ 4 LIVE TAP (alert)  │→ │ 5 DEFENCE /   │
 │ vol + range + body +  │   │ nearest bearish/ │   │ top,bot,entry,stop│   │ departure, minAge,  │   │ INVALIDATION  │
 │ CLV + structure break │   │ neutral, 8 back  │   │ + front-run buffer│   │ low ≤ entry         │   │ confirm/kill  │
 └──────────────────────┘   └─────────────────┘   └──────────────────┘   └───────────────────┘   └──────────────┘
        closed bar               closed bar            closed bar           ANY bar (intrabar!)      closed bar
     (non-repainting)         (non-repainting)      (non-repainting)      ("Once Per Bar" alert)   (non-repainting)
```

That asymmetry — *creation* needs `barstate.isconfirmed`, the *tap* does not — is the single most important thing to
understand, and it is exactly why the port has two scan modes (§7).

---

## 2. Stage 1 — volume-confirmed displacement

```pine
atr            = ta.atr(atrLen)                      // 14, Wilder RMA of TR
volMA          = ta.sma(volume, volLen)              // 20
rvol           = volMA > 0 ? volume / volMA : 0.0
rng            = math.max(high - low, syminfo.mintick)
body           = math.abs(close - open)
clv            = (close - low) / rng                 // close-location value
priorStructure = ta.highest(high[1], structureLen)   // 8-bar high EXCLUDING the current bar

displacement = barstate.isconfirmed and close > open and rvol >= minRVOL
               and rng >= atr * minRangeATR and body / rng >= minBodyFrac
               and clv >= minCLV and close > priorStructure
```

All seven conditions are **AND**ed; there is no OR branch and no "or" default. Meaning, per gate:

| Gate | Default | What it enforces | Why it matters |
|---|---|---|---|
| `close > open` | — | bullish bar | the OB is a *demand* origin, so displacement must be up |
| `rvol ≥ 1.8` | 1.8 | ≥1.8× the 20-bar mean volume | participation, not drift |
| `rng ≥ 1.20×ATR` | 1.20 | expansion vs. the recent volatility regime | relative, so it works on a 0.8% ATR IT stock and a 4% ATR small cap alike |
| `body/rng ≥ 0.55` | 0.55 | 55%+ of the bar is body | a *displacement*, not a fighting/rejected bar |
| `clv ≥ 0.72` | 0.72 | close in the top 28% of the range | closed near the high → orders left on the table |
| `close > highest(high[1],8)` | 8 | structure break to the upside | it must actually break something |
| `barstate.isconfirmed` | — | closed bar only | **non-repainting** |

> ⚠️ `minRVOL` is measured against a **20-bar SMA**, not a 20-day *volume profile*; on NSE, a single Diwali/ expiry day
> or a block-deal session can lift the SMA for weeks and silently suppress signals. The port keeps this behaviour
> exactly — it is a property of the indicator, not a bug in the port.

---

## 3. Stage 2 — origin-candle selection

```pine
int originOffset = na
if displacement
    for j = 1 to originSearch                      // 8
        candleRange = math.max(high[j] - low[j], syminfo.mintick)
        bearish = close[j] < open[j]
        neutral = allowNeutral and math.abs(close[j] - open[j]) / candleRange <= neutralBody   // 0.20
        if na(originOffset) and (bearish or neutral)
            originOffset := j
if displacement and na(originOffset)               // fallback
    originOffset := 1
```

Key properties, all reproduced in `compute_arrays()`:

* The loop starts at **j = 1**, so the *nearest* qualifying candle wins (first match breaks out).
* The displacement bar itself is never the origin (j starts at 1).
* "Neutral" means a small body — `|c-o|/range ≤ 0.20` — and is only accepted if `allowNeutralOrigin`.
* If no bearish/neutral candle exists in 8 bars (a pure melt-up), it falls back to **j = 1** — so a zone can be drawn
  from a *bullish* origin. That is intentional in the source; the fallback is what makes the tool usable on strong trends.
* If `originOffset` cannot be resolved (bar 0 of the history) no zone is created.

---

## 4. Stage 3 — zone construction and the pre-order level

```pine
rawTop/rawBot  = f(zoneMethod, origin candle)        // 4 methods, see table
zoneWidth      = rawTop - rawBot
baseEntry      = rawTop - zoneWidth * f_depth(entryMode)
entry          = baseEntry + f_front_buffer(zoneWidth, atr) + entryOffsetTicks * syminfo.mintick
entry          := entryMode == "Distal + 1 tick" ? rawBot + syminfo.mintick : entry
stop           = rawBot - atr * stopATR              // atr here = ATR of the CREATION bar
```

### 4.1 Zone geometry (`zoneMethod`)

| Method | `rawTop` | `rawBot` |
|---|---|---|
| **Open to low** (default) | `open[origin]` | `low[origin]` |
| Body to low | `max(open,close)` | `low[origin]` |
| Body only | `max(open,close)` | `min(open,close)` |
| Lower half of candle | `low + (high-low)*0.50` | `low[origin]` |

Note that for a **bearish** origin candle `open > close`, so "Open to low" *includes* the body and the lower wick — the
comment in the source calls it "the recommended institutional mitigation range".

### 4.2 Depth (`entryMode`) — how far into the zone you are willing to bid

`f_depth`: Proximal 0.0 · 50% 0.50 · 62% 0.62 · 70.5% 0.705 · 79% 0.79 · Distal+1tick → 1.0 (special-cased).

### 4.3 The front-run buffer — the most distinctive idea in the script

```pine
f_front_buffer(width, atr) =>
    tickBuffer = frontRunTicks * syminfo.mintick          // 2 ticks
    atrBuffer  = frontRunATR  * atr                       // 0.18 ATR
    switch frontRunMode
        "ATR"   => atrBuffer
        "Ticks" => tickBuffer
        "Off"   => 0.0
        =>  math.min(math.max(tickBuffer, atrBuffer), width * maxZoneBuffer)   // "Auto"
```

**Auto = the larger of (2 ticks, 0.18 ATR), capped at 40% of the zone width.** Rationale, from the source comments:
*"liquid orders often front-run the visible zone"* — so the executable buy level is deliberately placed **above** the
OB edge, and the cap keeps it inside a sane distance for very tight zones. Consequence worth internalising: **the alert
fires before price reaches the drawn box**, and `entry` can even sit above `rawTop`.

Worked NSE example (₹, `mintick 0.05`, ATR 12.40):

```
origin candle (bearish):  open 1,420.00   high 1,424.00   low 1,408.25   close 1,411.00
zone (Open to low):       top = 1,420.00  bot = 1,408.25  width = 11.75
depth  (Proximal):        baseEntry = 1,420.00 - 11.75*0.00 = 1,420.00
buffer (Auto):            min( max(2*0.05, 0.18*12.40)=2.232 , 11.75*0.40=4.70 ) = 2.23
entry (pre-order BUY):    1,422.23
stop  (distal - 0.15ATR): 1,408.25 - 1.86 = 1,406.39
1R (risk)                 1,422.23 - 1,406.39 = 15.84  (1.11% of price)
R1/R2/R3                  1,438.07 / 1,453.91 / 1,469.75
```

`verify RELIANCE.NS` prints exactly these numbers so you can diff them against the TradingView label.

---

## 5. Stage 4 — the state machine (this is where "Tap 1" is defined)

State per zone (Pine comment: *"0 fresh, 1 tapped/pending, 2 confirmed, -1 invalid/exhausted"*):

```
                 ┌──────────────── departure gate ────────────────┐
   [created] ───▶│ state 0 (fresh)  age = bar_index - born        │
                 └────────────────────────────────────────────────┘
                        │ high ≥ top + requireDeparture(1.0)×ATR   → departed := true (never resets)
                        ▼
   age ≥ minAge(3) AND departed?
        ├─ close > entry AND low ≤ entry + approachATR(0.25)×ATR ……… 👀 approach  (pre-alert)
        └─ low ≤ entry AND high ≥ bot ────────────────────────────► 🎯 TAP
                                                        │ (requireSweep ⇒ also low < lowest(low[1],5))
                                                        │ one tap per zone per bar (tapBar guard)
                                                        ▼
                                    state 1 (tapped/pending), taps += 1
                                    if raiseAfterFirstTap and taps == 1:
                                        entry := max(entry, low + repeatTapATR(0.05)×ATR)   ← adaptive level
                        ┌───────────────────────────────┴────────────────────────────┐
   within confirmBars(3) bars: close>open & clv≥0.65 & rvol≥1.3 &                │
        close > top & close > highest(high[1],3)  ─────────────────────────────► 🛡 state 2 (DEFENCE CONFIRMED)
   otherwise (bar_index - tapBar > confirmBars)  ──────────────────────────────► state 0 again (re-armable)
                                                                                        │
   close < stop  OR  taps > maxTouches(4)  ────────────────────────────────────► ❌ state -1 (dead, frozen)
```

Ordering *inside a single bar* is load-bearing and reproduced line-for-line in `run_engine()`:

1. `displacement` → duplicate check → push zone;
2. `while len(zones) > max_zones: zones.pop(0)` (Pine `array.shift`) — **eviction can drop a live zone**;
3. per live zone, in creation order: **departure → nearest-level bookkeeping → approach → tap (+adaptive entry) →
   defence → pending-expiry (1 → 0) → invalidation**.

So a bar that wicks into the level *and* closes below the stop produces **both** a `TAP` and an `invalidated` event,
tap first. The port does the same (`check_invalidation_ordering`), and the backtester's alert is still delivered — the
`zone_invalidated` exit reason is a signal-quality marker, not a suppression.

---

## 6. Stage 5 — duplicate rejection, output objects, alerts

```pine
bool duplicate = false
if displacement and rawTop > rawBot and array.size(boxes) > 0
    for i = 0 to size-1
        if array.get(states,i) >= 0 and candidateEntry <= tops[i] and candidateEntry >= bots[i]
            duplicate := true
if displacement and not duplicate and rawTop > rawBot
    ... create box / entry line / stop line / label, push to all arrays
```

* Only **live** zones (state ≥ 0) veto a new one; dead ones do not.
* The veto test is on the **entry level**, not on box overlap → two overlapping OB boxes are perfectly allowed as long as
  their entries differ. Conversely, with `frontRunMode="Off"`, a second displacement that resolves to the same origin
  candle computes the same entry → always vetoed. (`check_duplicate_rejection` asserts both directions.)
* `plotshape(displacement and not duplicate, …)` and `alertcondition(displacement and not duplicate, …)` mean a vetoed
  candidate must produce **no marker and no "new OB" alert** — the port therefore clears the `displacement` flag for
  duplicates after suppression, which is also what makes `verify`'s date list match TradingView exactly.

Outputs: boxes (state-coloured: blue fresh / orange tapped / green confirmed / grey dead), an entry line, a dashed stop
line, labels (`PRECISION OB`, `TAP n`, `DEFENCE CONFIRMED`), 4 `plotshape`s, 3 data-window `plot`s
(`nearestEntry`, `nearestStop`, `rvol`), and 5 `alertcondition`s.

---

## 7. Non-repainting, and the two alert modes

The header comment states the contract: *"Closed-bar zone creation is non-repainting. A live touch alert can fire
intrabar."* That maps onto two distinct scanner modes:

| Mode | `run_engine(..., intrabar_last=…)` | Matches TradingView alert set to | Behaviour on the newest bar |
|---|---|---|---|
| `scan --eod` / `run` after 15:30 | `False` | **"Once Per Bar Close"** | the bar is complete: zones, taps, defences, invalidations all evaluated |
| `scan --live` / `run` during 09:15–15:30 | `True` | **"Once Per Bar"** | zones are **not created** and defences are **not confirmed** on the forming bar; approach/tap/invalidation *are* evaluated on the running high/low |

`alerts.match_indicator_100: true` (default) makes live mode do **both**: it evaluates the forming bar *and* replays the
last completed bar in closed-bar mode, then unions the two event sets (deduped by `(kind, bar, zone)`). A delayed or
partially-updated intraday feed therefore cannot make the scanner miss a signal the indicator printed at the close.

**What "100% match" means here, precisely:** *identical logic on identical bars.* Verified two ways:

1. **18 hand-computed fixtures** (`selftest.py`) covering ATR seeding, warm-up `na` semantics, every zone method,
   origin search + fallback, entry math for all four front-run modes, all six entry modes, `minAge`,
   `requireDeparture`, one-tap-per-bar, adaptive entry, pending expiry, exhaustion ordering, sweep gating, duplicate
   veto, eviction, intrabar semantics and nearest-level bookkeeping.
2. **A second, independent implementation.** `tests/pine_reference.py` is a fresh transliteration of
   `INDICATOR.txt` — its own naive `ta.sma/rma/atr/highest/lowest`, its own parallel zone arrays, the same statement
   order, written from the Pine source and sharing no code with `engine.py` or `series.py`.
   `tests/test_pine_reference.py` runs both over five markets × 16 parameter sets × {closed bar, forming bar} and
   requires agreement on the `plotshape` mask, all four `alertcondition` flags, the `nearestEntry`/`nearestStop`
   plots, the surviving zone arrays and the full event stream. A transcription mistake in either direction shows up
   as a diff naming the bar and the field.

It does **not** mean identical *data* — see §10.

---

## 8. Quirks in the source that are faithfully preserved (know these before trusting the alerts)

1. **`minAge` counts from the displacement bar, not the origin bar** (`born = bar_index` at creation). The zone's own
   candles are 1 bar "younger" than they look on the chart.
2. **`departed` is monotonic.** Once a zone has been left by 1×ATR it stays eligible forever; there is no "zone went
   stale, needs a fresh departure" reset.
3. **Zones never expire by time.** Only `close < stop`, `taps > maxTouches`, or `max_zones` eviction can kill a box.
   A 200-bar-old untouched OB is still "live" and can still alert. Use `alerts.max_age_bars` to impose your own TTL
   (the port adds this filter *without* touching the signal set).
4. **Confirmed zones (state 2) can be tapped again** — `touched` only requires `state >= 0`. So "TAP 3" on a green box
   is normal.
5. **After Tap 1 the entry moves UP** (`max(entry, low + 0.05×ATR)`), which makes later taps *easier*, not harder —
   intentional per the source comment ("later tests commonly turn just above Tap 1"), but it means repeat-tap counts can
   climb fast in chop, which interacts with (6).
6. **The 5th tap is recorded and then the zone is marked exhausted in the same bar.**
7. **ATR is read at different times for different purposes:** the stop uses the *creation* bar's ATR, while
   `requireDeparture`, `approachATR` and `repeatTapATR` use the *current* bar's ATR. Not a bug — but it means a
   volatility spike changes the approach/tap gates for old zones while their stop stays frozen.
8. **`Auto` can place the entry above `rawTop`** (buffer > 0 while depth = 0). If you want the alert strictly inside the
   box, use `frontrun_mode: "Off"`, or `entry_mode: "50%"`.
9. **`anyApproach`/`anyTap`/`anyConfirm`/`anyDead` are bar-level booleans** in Pine, so a TradingView alert cannot tell
   you *which* zone fired. The port emits per-zone events (with `zone_id`, top/bot/entry/stop) instead.
10. **Approach and tap can both be true on the same bar**; the port edge-triggers the *pre-alert event* (one per zone
    per approach spell) while keeping the per-bar flag identical for parity.
11. **`nearestEntry`/`nearestStop` include zones that are not yet departed** — display-only, never used for alerting.
12. **Distal + 1 tick bypasses both the buffer and the tick offset** (`entry = rawBot + mintick`).

---

## 9. Pine → Python mapping (where to look)

| Pine | Python (`precision_tap/`) |
|---|---|
| `ta.atr(14)` | `series.atr` → `series.rma(true_range(…))`, SMA-seeded like Pine |
| `ta.sma(volume, 20)`, `rvol` | `series.sma` + vectorised division in `engine.compute_arrays` |
| `ta.highest(high[1], n)` / `ta.lowest(low[1], n)` | `series.highest(series.shift(…,1), n)` (monotonic deque, available-bars warm-up) |
| `rng`, `body`, `clv`, `body/rng` | `compute_arrays` (all `np.maximum(…, mintick)` guarded) |
| `displacement` | `A["displacement"]` (bool array; cleared on duplicate, as `plotshape` implies) |
| origin search + `rawTop/rawBot` | `compute_arrays` (`origin_off`, `raw_top`, `raw_bot`) |
| `f_depth`, `f_front_buffer`, `entry`, `stop` | `engine.entry_level`, `engine.front_buffer` |
| `boxes/tops/bots/entries/stops/born/states/taps/tapBars/departed` arrays | `engine.Zone` dataclass fields (`top,bot,entry,entry0,stop,born,state,taps,tap_bars,departed`) |
| `if array.size(boxes) > maxZones` shift | `while len(zones) > max_zones: zones.pop(0)` |
| `approaching` / `touched` / `qualifiedTap` / `defence` / `invalid` | the per-zone loop in `engine.run_engine`, same order, emitting `Event`s |
| `plotshape`/`alertcondition` flags | `EngineResult.flags` (`any_approach/any_tap/any_confirm/any_dead`) + `events` |
| `plot(nearestEntry)`, `plot(nearestStop)` | `flags["nearest_entry"]/["nearest_stop"]` |
| `syminfo.mintick` | `Params.mintick`, per-symbol via `data.tick_sizes` (NSE/BSE = **0.05**) |
| `barstate.isconfirmed` | `run_engine(..., intrabar_last=bool)` |
| label text "PRECISION OB / Entry / Stop" | `alerts.render_lines` → the Telegram message body |

---

## 10. NSE/BSE specifics, and the one thing that cannot be made identical

This build is **locked to Indian equities on the daily timeframe** (`DataConfig.__post_init__` raises
otherwise: `market ∈ {NSE, BSE}`, `interval == "1d"`, `symbol_suffix ∈ {".NS", ".BO"}`). That is a
correctness decision, not a convenience one: the indicator's gates are all *relative to the bar's
statistics* — RVOL against the 20-bar volume mean, `range ≥ 1.2·ATR`, `body/range`, CLV, an 8-bar
structure break, `minAge = 3` bars, `maxTouches = 4` — and every one of those constants was tuned for
daily candles. On a 15-minute frame the same numbers describe a completely different market structure,
so "the same indicator" would quietly mean something else. Refusing the input is the honest behaviour;
the parity claim (§7) only holds on the timeframe the script was written for.

* **Suffix & tick** — `data.symbol_suffix: .NS` (BSE `.BO`) and `tick_sizes: {".NS": 0.05, ".BO": 0.05}`; TradingView's
  `syminfo.mintick` for these equities is 0.05, which feeds `Ticks`/`Auto` buffers and `Distal + 1 tick`.
* **Session** — `Asia/Kolkata`, 09:15–15:30, Mon–Fri. Pre-open (09:00–09:15) prints are deliberately excluded: the
  scanner treats the 09:15 continuous-session bar as the day's bar.
* **Holidays / stale feeds** — `alerts.skip_stale_bars: true` refuses to alert if the newest bar is not today's session,
  so a NSE holiday or a lagging vendor cannot produce a phantom "touch".
* **Corporate actions** — `corporate_adjustments: true` → yfinance `auto_adjust=True` (back-adjusted OHLC). TradingView
  also back-adjusts by default, so this is the closest match; for a *live* price alert you may prefer raw prices so the
  level matches your terminal — set `corporate_adjustments: false` and re-run `verify` on the split date to see the
  difference.
* **What genuinely cannot be identical:** the *bars themselves*. Yahoo/NSE-feed OHLC can differ from TradingView's NSE
  feed in (a) the exact 15:30 close print, (b) volume units (Yahoo's NSE volume is shares; TV's is shares too, but its
  aggregation of the 09:00–09:15 pre-open block can differ), and (c) which day's bar appears first after a
  midnight-maintenance window. A single tick of difference in `volume` moves `rvol`; `rvol ≥ 1.8` is a hard gate, so a
  signal can legitimately appear on one feed and not the other. Everything else is deterministic. `verify SYMBOL`
  exists precisely so you can diff dates/levels against a chart in minutes.

---

## 11. What the backtester adds (the indicator defines no exits)

The script gives you `entry` and `stop` — that's it. The backtest engine therefore makes its assumptions explicit
(`precision_tap/backtest.py`, all configurable):

* Fill = `min(bar_open, level)` for a Tap limit buy (gap-through fills better); `close` for a `confirmed` entry.
* **Worst path** on any bar that touches both stop and target → stop wins; `same_bar_stop` decides whether the entry
  bar can be stopped out at all.
* `stop_mode: touch` (low ≤ stop) is the realistic choice; `stop_mode: close` reproduces the indicator's own
  `close < stop` invalidation instead — run both and you have a bound on how much of the result is intrabar luck.
* Optional `target_r`, `breakeven` / `chandelier` / `structure` trailing, `time_stop_bars`, `exit_on_reversal`.
* Costs per side in bps (commission + slippage + spread); ₹ risk-fractional sizing and an optional
  `max_open_positions` for the money-space view; R-space stats are always reported unconstrained.
* A **signal study** (`fwd_1…fwd_20`, `mfe_10`, `mae_10`, plus an all-bars baseline mean) that tells you whether the
  Tap-1 *event* has edge before any exit rules are layered on. That is the number to look at first — if the signal
  study is flat, exit optimisation is curve-fitting.
