

## 0. clock / mode
utc=2026-09-11 15:17 ist=2026-09-11 20:47 Fri market_open=False provider=yfinance universe_file=universe/nse.txt
universe=1 sample=['RELIANCE']


## 1. provider frames (real fetch, no cache)
- RELIANCE [eod] FAIL yfinance: yfinance: argument of type 'NoneType' is not iterable; yahoo: yahoo request failed after 3 tries: HTTPSConnectionPool(host='query1.finance.yahoo.com',
- RELIANCE [live] FAIL yfinance: yfinance: yfinance returned no rows; yahoo: yahoo request failed after 3 tries: HTTPSConnectionPool(host='query1.finance.yahoo.com', port=443): Max re


## 2. engine events on the newest 6 bars (closed-bar replay)
- RELIANCE: no data (yfinance: yfinance: yfinance returned no rows; yahoo: yahoo request failed after)

totals(last6 bars over 1 symbols): {}


## 3. full cycle with a widened window (recent_bars=5, log-only)
mode=eod universe=1 usable=0 errors=1 skipped=0 events_total=0 in_window=0 filtered=0 alerts=0
notes: RELIANCE: data fetch failed — yfinance: yfinance: yfinance returned no rows; yahoo: yahoo request failed after 3 tries: HTTPSConnectionPool(host='query1.finance.yahoo.com', port=443): Max retries exceeded with url: /v8/finance/chart/RELIANCE.NS?interval=1d&range=900d&includePrePost=false&events=div%2Csplit (Caused by SSLError(SSLZeroReturnError(6, 'TLS/SSL connection has been closed (EOF) (_ssl.c:992)'))) | FEED PROBLEM — 1/1 symbols failed to fetch (see the per-symbol errors above); nothing was evaluated | nothing to send — no indicator signal in the last 5 bar(s) of 0 usable symbol(s) (0 with a live zone); gates not met or a quiet session | 8.6s | results_diag/scan_eod_20260911_151832.md


## 4. transport (validate + one real test message)
configured=False dry_run=False chats=[]
