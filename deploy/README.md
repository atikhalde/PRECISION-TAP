# Deployment

Pick one. All three run the *same* command (`python -m precision_tap run`) or its `--once` variant.

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

## Operational notes

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
