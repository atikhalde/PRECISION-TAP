# Deployment

Pick one. The first three run the *same* command (`python -m precision_tap run`) or its `--once`
variant on a host you own; the fourth runs it on GitHub's runners.

## systemd (VPS, recommended)

```bash
sudo useradd -r -m -d /opt/precision-tap precision
sudo -u precision git clone <this repo> /opt/precision-tap
cd /opt/precision-tap
sudo -u precision python3 -m venv .venv
sudo -u precision .venv/bin/pip install -r requirements.txt
sudo -u precision cp config.example.yaml config.yaml      # then edit: universe, gates, alerts
sudo tee /etc/precision-tap.env >/dev/null <<'ENV'
TELEGRAM_BOT_TOKEN=123456:AA...
TELEGRAM_CHAT_ID=111888
ENV
sudo chmod 600 /etc/precision-tap.env
sudo -u precision .venv/bin/python -m precision_tap doctor --net    # must pass
sudo -u precision .venv/bin/python -m precision_tap telegram-test   # message must arrive
sudo cp deploy/systemd/precision-tap.service /etc/systemd/system/
# set User=precision in the unit if %i is not expanded for your distro
sudo systemctl daemon-reload && sudo systemctl enable --now precision-tap
journalctl -u precision-tap -f
```

The unit sends `SIGINT` on stop, which the loop traps → it finishes the current scan cycle, so no
half-sent alerts and no duplicate Telegram pings after a restart (the ledger dedupes anyway).

## Docker

```bash
cd deploy
TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=... docker compose up -d --build
docker compose logs -f
```
`config.yaml` is mounted read-only; `data/` (cache + `state.sqlite3`) and `results/` persist on the host.
Backtest from the same image: `docker compose run --rm precision-tap python -m precision_tap backtest --start 2021-01-01`.

## cron

`crontab deploy/cron.example` after editing paths/user. Use `scan` (not `run`) — `--once` semantics are
the default for a single invocation. Prefer the `15:35`/`16:10` IST jobs if you only want the
indicator-exact closed-bar alert set; add the `*/15` intraday jobs for live touch alerts.

Cron gives the job no environment, so `deploy/cron.example` sets `PRECISION_TAP_ENV_FILE=/etc/precision-tap.env`
and the scanner loads the secrets from that file itself. If it is missing, the run does not fail quietly any
more: `scan` exits **4** with `alerts found but nothing could send them`. Check with
`sudo -u precision env -i /opt/precision-tap/.venv/bin/python -m precision_tap telegram-test` (that is exactly
the empty environment cron runs in) and `grep -c 'alert (log-only)' logs/cron.log`.

## GitHub Actions (no server)

`.github/workflows/live-scan.yml` runs the scanner on GitHub's runners against the live market and
posts to Telegram. Nothing to install, patch or monitor beyond the repo.

```
Settings → Secrets and variables → Actions
  New repository secret   TELEGRAM_BOT_TOKEN
  New repository secret   TELEGRAM_CHAT_ID
```

Then enable it (scheduled workflows are off until the file is on the default branch, and GitHub
disables any schedule after 60 days of repository inactivity):

```bash
gh workflow view live-scan.yml                       # is it enabled?
gh workflow run live-scan.yml -f dry_run=true        # first real run, one cycle, nothing sent
gh workflow run live-scan.yml -f mode=eod -f limit=25
gh workflow run live-scan.yml -f loop=true           # cover the rest of the session now
gh run list --workflow=live-scan.yml
```

Cadence: cron only *kicks* the job (09:05 / 11:25 / 13:05 / 14:55 / 15:25 / 16:15 IST); the job then
runs `precision_tap run --duration <budget>` and owns the cadence itself — intraday polling every
`live.intraday_poll_minutes` plus the `live.scan_times` closed-bar prints, until `SESSION_WINDOW_END_IST`
(17:20) or `JOB_CAP_MINUTES` (330, kept under GitHub's hard 6-hour job limit). Each cycle decides
intraday-vs-closed-bar from the exchange clock, so the UTC runner needs no timezone help. A kick that
starts after the window has closed degrades to one `scan` cycle, so queued redundant kicks cost ~1 min.

**State.** Runners are ephemeral, and `data/state.sqlite3` is the entire dedupe mechanism — without
it every cycle re-sends the same Tap 1. The workflow carries it forward with `actions/cache`
(restore newest → scan → WAL checkpoint → save under the run id). A cache miss does not fail the
run; it emits a `::warning::` saying today's alerts may repeat once. Only the ledger is cached, never
`data/cache`, so a cycle can't be handed yesterday's frames.

**Honest limits — read this before trusting it.** `schedule` is best-effort, and the earlier
"cron every 15 minutes" version of this workflow is a measured failure, not a theoretical one: on its
first trading day 15 slots were due, 5 fired, all 1–4 hours late, and **none landed inside the NSE
session**, so the intraday stream never ran while every run reported success. GitHub documents the
delay; the dropping and the deprioritisation of low-traffic repos it does not. The kick-and-loop
design above needs only *one* kick to reach a runner, and any one of the five from 11:25 IST onwards
covers the close and the digest on its own. The residual gap: if the 09:05 kick is the *only*
survivor, its cap ends the job at 14:35 IST and that day's post-close prints are missed. Consecutive
jobs are serialised by a `concurrency` group with `cancel-in-progress: false`, so a queued kick waits
rather than cancelling the one in flight. If a tap absolutely must reach you inside the session, run
systemd or Docker on a host you own — that is a guarantee, this is five good chances.

**Minutes.** One runner held for the session: up to ~5.5 h per weekday, plus ~1 min per redundant
kick. Free on public repos; on a private repo that is most of the 2000 min/month allowance — lower
`JOB_CAP_MINUTES`, drop kicks, or use `SCAN_LIMIT` to trim the universe.



* **Timezone**: NSE 09:15–15:30 IST, Mon–Fri. Keep the host on `Asia/Kolkata` (or set `TZ=`) so the
  scheduler and the stale-bar guard agree with the exchange.
* **Rate limits**: one Yahoo/yfinance request per symbol per cycle. 130 symbols ≈ 130 requests every
  10 min by default — tune `data.rate_limit_per_sec` / `live.max_workers` if you scale to Nifty 500.
* **Disk**: the CSV cache and SQLite ledger stay small (a few MB per 100 symbols/yr). `results/`
  grows one scan report per cycle — `logs/` + `results/` are safe to rotate/cron-clean.
* **Upgrades**: `precision_tap selftest` (18 parity checks) must stay green after any edit to
  `engine.py`/`series.py` — that is the parity contract with the Pine source.
* **Silence ≠ no signals**: leave `live.heartbeat_daily_time` set, or you cannot tell "flat market"
  from "dead process".
