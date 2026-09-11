"""Market data: fetch, normalise, cache.

Providers
---------
``yahoo``     Raw ``query1.finance.yahoo.com/v8/finance/chart`` endpoint via
              ``requests``. No API key, no extra dependency; robust to yfinance
              churn. Also used for the intraday rebuild of today's forming bar.
``yfinance``  Same data through the ``yfinance`` package (optional dependency),
              convenient if you already use it / want its corporate actions.
``csv``       Local OHLCV files (one per symbol, or ``history_dir/all.csv``).
``synthetic`` Deterministic random-walk generator with injected displacement
              bars — used by the test-suite and the offline demo.

Everything is normalised to a ``DataFrame[open, high, low, close, volume]`` with a
tz-aware-or-naive ``DatetimeIndex`` sorted ascending, exactly what the engine wants.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .params import DataConfig

log = logging.getLogger("precision_tap.data")

OHLCV = ["open", "high", "low", "close", "volume"]
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

#: NSE/BSE only (enforced in :class:`~precision_tap.params.DataConfig`), so the
#: exchange clock is always the Indian one.
MARKET_TZ = {"NSE": "Asia/Kolkata", "BSE": "Asia/Kolkata"}
DEFAULT_MARKET_TZ = "Asia/Kolkata"

#: The EOD daily bar only settles a while after the 15:30 close — before this
#: local time a closed-bar scan legitimately still shows the *previous* session.
EOD_SETTLE_MIN = 16 * 60 + 10

#: Providers that can return today's (possibly forming) bar.
LIVE_PROVIDERS = {"yfinance", "yahoo", "yahoo_chart", "chart"}


def _market_tz(name: str = "NSE") -> str:
    return MARKET_TZ.get(str(name or "").upper(), DEFAULT_MARKET_TZ)


def _now_tz(tzname: str) -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tzname))
    except Exception:
        return datetime.now().astimezone()


def _expected_last_bar_date(tzname: str = DEFAULT_MARKET_TZ, *, live: bool = False) -> date_cls:
    """Newest session date the feed should already be able to return.

    ``live``      -> today's session (the forming bar a touch alert needs).
    not ``live``  -> the last *completed* session; before the EOD bar settles
                     (``EOD_SETTLE_MIN``) that is the previous weekday.
    """
    now = _now_tz(tzname)
    d = now.date()
    if not live and (now.hour * 60 + now.minute) < EOD_SETTLE_MIN:
        d -= timedelta(days=1)
    while d.weekday() >= 5:                        # Sat/Sun roll back to Friday
        d -= timedelta(days=1)
    return d


def _bar_is_today(ts, meta: dict | None = None) -> bool:
    """Is this bar stamped with today's session date? (exchange timezone aware)"""
    try:
        tz = (meta or {}).get("exchangeTimezoneName")
        if tz:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo(tz))
        else:
            now = datetime.now()
        bar = pd.Timestamp(ts)
        if bar.tzinfo is not None and tz:
            try:
                bar = bar.tz_convert(tz)
            except Exception:
                pass
        if bar.date() == now.date():
            return True
        # tolerate a UTC-stamped feed (NSE session is 03:45-10:15 UTC)
        return bar.date() == datetime.now(timezone.utc).date()
    except Exception:
        return False


def normalize_ohlcv(df: pd.DataFrame, *, daily: bool = False) -> pd.DataFrame:
    """Lower-case, coerce numeric, drop incomplete rows, sort, de-duplicate."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV)
    out = df.copy()
    cols = {str(c).lower().replace(" ", "_"): c for c in out.columns}
    rename = {}
    for want, src in {
        "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"),
        "close": ("close", "c", "adj_close", "adjclose"), "volume": ("volume", "vol", "v"),
    }.items():
        for cand in src:
            if cand in cols:
                rename[cols[cand]] = want
                break
    out = out.rename(columns=rename)
    if "close" not in out.columns and "adj_close" in cols:
        out["close"] = out[cols["adj_close"]]
    for col in OHLCV:
        if col not in out.columns:
            if col == "volume":
                out["volume"] = 0.0
            else:
                raise ValueError(f"missing column {col!r} (have {list(df.columns)})")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[OHLCV]
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    # keep high >= low and clamp single-print bars to a tiny range
    out["high"] = np.maximum.reduce([out["high"].to_numpy(), out["open"].to_numpy(), out["close"].to_numpy()])
    out["low"] = np.minimum.reduce([out["low"].to_numpy(), out["open"].to_numpy(), out["close"].to_numpy()])
    if daily:
        out.index = out.index.normalize()
    out["volume"] = out["volume"].fillna(0.0)
    return out


def apply_adjclose(df: pd.DataFrame, adjclose: Optional[pd.Series]) -> pd.DataFrame:
    """Scale raw OHLC by adjclose/close so splits/dividends do not fakedisplace zones."""
    if adjclose is None or len(df) == 0:
        return df
    a = pd.to_numeric(pd.Series(adjclose).reindex(df.index), errors="coerce")
    ratio = (a / df["close"]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    ratio = ratio.fillna(1.0)
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * ratio
    return out


def parse_chart_payload(payload: dict, *, daily: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Parse a Yahoo ``v8/finance/chart`` response into a normalised frame."""
    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"yahoo chart error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError("empty yahoo chart result")
    res = results[0]
    meta = res.get("meta") or {}
    ts = res.get("timestamp") or []
    quote = ((res.get("indicators") or {}).get("quote") or [{}])[0] or {}
    df = pd.DataFrame(
        {
            "open": quote.get("open"), "high": quote.get("high"),
            "low": quote.get("low"), "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(pd.Index(ts), unit="s", utc=True),
    )
    tzname = meta.get("exchangeTimezoneName")
    if tzname:
        try:
            df.index = df.index.tz_convert(tzname)
        except Exception:
            pass
    adj = ((res.get("indicators") or {}).get("adjclose") or [{}])
    adjclose = pd.Series(adj[0].get("adjclose"), index=df.index) if adj and adj[0].get("adjclose") else None
    df = apply_adjclose(df, adjclose)
    df = normalize_ohlcv(df, daily=daily)
    return df, meta


def frame_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.to_csv(path)


def read_csv_frame(path: Path, *, daily: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(path)
    datecol = None
    for cand in ("date", "datetime", "timestamp", "time", "day"):
        for col in raw.columns:
            if str(col).strip().lower() == cand:
                datecol = col
                break
        if datecol is not None:
            break
    if datecol is None:
        datecol = raw.columns[0]
    raw[datecol] = pd.to_datetime(raw[datecol], errors="coerce", utc=None)
    raw = raw.set_index(datecol)
    cols = {c.lower(): c for c in raw.columns}
    ren = {}
    for target in OHLCV:
        for cand in (target, target.capitalize(), {"open": "o", "high": "h", "low": "l",
                                                   "close": "c", "volume": "v"}[target]):
            if cand in cols:
                ren[cols[cand]] = target
                break
    raw = raw.rename(columns=ren)
    keep = [c for c in OHLCV if c in raw.columns]
    return normalize_ohlcv(raw[keep], daily=daily)


# ─────────────────────────────────────────────────────────────────────────────
# Providers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bars:
    """Container returned by every provider."""

    symbol: str
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
    live: bool = False            # last bar is today's forming bar
    source: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return len(self.df) > 0 and not self.error


class Cache:
    """Tiny CSV-on-disk cache; TTL depends on whether the market is open."""

    def __init__(self, cfg: DataConfig, *, live: bool = False):
        self.ttl_min = float(cfg.cache_max_age_minutes if live else cfg.eod_cache_max_age_minutes)
        self.enabled = self.ttl_min > 0            # 0/negative disables the cache entirely
        self.dir = Path(cfg.cache_dir)
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.enabled = False

    def _path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
        return self.dir / f"{safe}.csv"

    def load(self, key: str) -> Optional[pd.DataFrame]:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        if self.ttl_min > 0:
            age_min = (time.time() - p.stat().st_mtime) / 60.0
            if age_min > self.ttl_min:
                return None
        try:
            df = read_csv_frame(p)
            return df if len(df) else None
        except Exception as exc:            # corrupt cache must never kill a scan
            log.warning("cache read failed for %s (%s)", p, exc)
            return None

    def store(self, key: str, df: pd.DataFrame) -> None:
        if not self.enabled:
            return
        try:
            frame_to_csv(df, self._path(key))
        except Exception as exc:
            log.warning("cache write failed for %s (%s)", key, exc)


class RateLimiter:
    def __init__(self, per_sec: float):
        self.min_gap = (1.0 / per_sec) if per_sec and per_sec > 0 else 0.0
        self._next = 0.0

    def wait(self) -> None:
        if self.min_gap <= 0:
            return
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self.min_gap


class DataSource:
    """Provider dispatch + caching + symbol universe."""

    def __init__(self, cfg: DataConfig, *, live: bool = False, proxy: str = ""):
        self.cfg = cfg
        self.live = live
        self.cache = Cache(cfg, live=live)
        self.limiter = RateLimiter(cfg.rate_limit_per_sec)
        self.proxy = proxy or ""
        self._session = None
        if cfg.provider in {"yahoo"}:
            try:
                import requests  # noqa: F401
                self._session = requests.Session()
                self._session.headers.update({"User-Agent": UA, "Accept": "application/json"})
                if self.proxy:
                    self._session.proxies.update({"http": self.proxy, "https": self.proxy})
            except Exception as exc:
                log.warning("requests unavailable (%s) — falling back to csv/synthetic", exc)
                self.cfg.provider = "csv"

    # ── public API ───────────────────────────────────────────────────────
    #: suffixes that are never Indian — rejected so a stray US ticker cannot quietly
    #: pollute the universe (and so `verify` never compares against the wrong feed)
    _FOREIGN = (".SS", ".SZ", ".L", ".DE", ".HK", ".TO", ".NYSE", ".MC", ".PA", ".SW",
                ".SI", ".JK", ".KL", ".SG", ".AX", ".VI", ".HE", ".ST", ".T", ".OL", ".DB")

    def fetch_symbol(self, symbol: str) -> str:
        """Normalise to a yfinance Indian ticker: ``RELIANCE`` -> ``RELIANCE.NS``.

        Indices (``^NSEI``, ``^BSESN``) pass through; non-Indian suffixes raise.
        """
        s = str(symbol).strip().upper()
        suf = (self.cfg.symbol_suffix or ".NS").strip().upper()
        if s.startswith("^"):
            return s
        if "." in s:
            _, _, tail = s.rpartition(".")
            if tail in {"NS", "BO"}:
                return s
            if tail in {f.lstrip(".") for f in self._FOREIGN} or tail not in {"NS", "BO"}:
                raise ValueError(f"{symbol!r} is not an NSE/BSE ticker "
                                 f"(expected a '.{suf.lstrip('.')}') suffix)")
            return s
        return s + suf

    def tick_for(self, symbol: str) -> float:
        """``syminfo.mintick`` by exchange suffix (NSE/BSE equities = 0.05)."""
        table = self.cfg.tick_sizes or {}
        s = str(symbol).upper()
        for suffix, val in table.items():
            if s.endswith(str(suffix).upper()):
                return float(val)
        return float(table.get(s[-3:] if s.startswith("^") else s[s.rfind("."):], 0.05))

    def get(self, symbol: str, *, lookback_days: Optional[int] = None,
            end: Optional[str] = None, use_cache: bool = True) -> Bars:
        cfg = self.cfg
        days = int(lookback_days or cfg.lookback_days)
        # live and closed-bar frames are cached under different keys: a live scan
        # must never be handed a frame fetched for (and valid for) a closed-bar
        # scan, and today's *forming* bar must never leak into the EOD cache.
        key = f"{symbol}_{cfg.interval}_{days}d_{end or ('live' if self.live else 'eod')}"
        if use_cache:
            cached = self.cache.load(key)
            if cached is not None and len(cached) >= cfg.min_bars and self._cache_usable(cached, end):
                return Bars(symbol=symbol, df=cached, source="cache",
                            live=bool(self.live and self._has_current_bar(cached)))
        try:
            symbol = self.fetch_symbol(symbol)
        except ValueError as exc:
            return Bars(symbol=symbol, df=pd.DataFrame(columns=OHLCV), error=str(exc))
        try:
            if cfg.provider in {"yahoo", "yahoo_chart"}:
                bars = self._yahoo(symbol, days, end)
            elif cfg.provider == "yfinance":
                bars = self._yfinance(symbol, days, end)
            elif cfg.provider in {"csv", "file", "local"}:
                bars = self._csv(symbol)
            elif cfg.provider in {"synthetic", "demo"}:
                bars = self._synthetic(symbol, days)
            else:
                raise ValueError(f"unknown data.provider {cfg.provider!r}")
        except Exception as exc:
            log.debug("fetch failed for %s: %s", symbol, exc)
            return Bars(symbol=symbol, df=pd.DataFrame(columns=OHLCV), error=str(exc))
        if bars.ok and use_cache and bars.source != "cache":
            self.cache.store(key, bars.df)
        return bars

    # ── cache freshness ──────────────────────────────────────────────────
    def _tracks_live_market(self, end: Optional[str]) -> bool:
        """True when this request is for *current* data (no historical ``end``)."""
        return end is None and self.cfg.provider in LIVE_PROVIDERS

    def _has_current_bar(self, df: pd.DataFrame) -> bool:
        """Does the newest row carry the session the feed should already have?"""
        if df is None or df.empty:
            return False
        try:
            return pd.Timestamp(df.index[-1]).date() >= self._expected_date()
        except Exception:
            return False

    def _expected_date(self) -> date_cls:
        return _expected_last_bar_date(_market_tz(self.cfg.market), live=bool(self.live))

    def _cache_usable(self, df: pd.DataFrame, end: Optional[str]) -> bool:
        """TTL is not enough: a cached frame can be a whole session behind.

        The on-disk cache is keyed by live/EOD scope, but the TTL is measured in
        minutes, and a session boundary is what actually makes a frame obsolete
        (last night's EOD frame is useless for this morning's touch check, and a
        frame fetched before the close is useless for the settled EOD scan).
        Offline providers (csv/synthetic) are exempt — their data is frozen by
        definition, so only the TTL applies.
        """
        if not self._tracks_live_market(end):
            return True
        return self._has_current_bar(df)

    def get_many(self, symbols: Sequence[str], *, lookback_days: Optional[int] = None,
                 end: Optional[str] = None, workers: int = 8,
                 progress: bool = True) -> Dict[str, Bars]:
        out: Dict[str, Bars] = {}
        symbols = [s for s in symbols if s]
        if not symbols:
            return out
        workers = max(1, min(int(workers or 1), 16))
        if workers == 1 or self.cfg.provider in {"csv", "synthetic"}:
            for i, s in enumerate(symbols, 1):
                out[s] = self.get(s, lookback_days=lookback_days, end=end)
                if progress and (i % 25 == 0 or i == len(symbols)):
                    log.info("data %d/%d", i, len(symbols))
            return out
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.get, s, lookback_days=lookback_days, end=end): s for s in symbols}
            done = 0
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    out[s] = fut.result()
                except Exception as exc:
                    out[s] = Bars(symbol=s, df=pd.DataFrame(columns=OHLCV), error=str(exc))
                done += 1
                if progress and (done % 25 == 0 or done == len(symbols)):
                    ok = sum(1 for b in out.values() if b.ok)
                    log.info("data %d/%d (usable %d)", done, len(symbols), ok)
        return out

    # ── yahoo chart API ──────────────────────────────────────────────────
    def _http_json(self, url: str, params: dict) -> dict:
        self.limiter.wait()
        last: Optional[Exception] = None
        for attempt in range(1, self.cfg.retry_max + 1):
            try:
                if self._session is not None:
                    r = self._session.get(url, params=params, timeout=25)
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"http {r.status_code}")
                    r.raise_for_status()
                    return r.json()
                import urllib.parse, urllib.request
                q = urllib.parse.urlencode(params)
                req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=25) as fh:
                    return json.loads(fh.read().decode("utf-8"))
            except Exception as exc:
                last = exc
                time.sleep(self.cfg.retry_backoff ** attempt)
        raise RuntimeError(f"yahoo request failed after {self.cfg.retry_max} tries: {last}")

    def _yahoo(self, symbol: str, days: int, end: Optional[str]) -> Bars:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": self.cfg.interval, "range": f"{max(days, 5)}d",
                  "includePrePost": "false", "events": "div,split"}
        if end:
            params.pop("range")
            end_dt = pd.Timestamp(end, tz="UTC")
            params["period1"] = int((end_dt - timedelta(days=days)).timestamp())
            params["period2"] = int(end_dt.timestamp())
        payload = self._http_json(url, params)
        daily = self.cfg.interval in ("1d", "d", "daily", "1wk")
        df, meta = parse_chart_payload(payload, daily=daily)
        source = "yahoo"
        if self.live and self.cfg.live_intraday_bar and not end:
            merged, live_flag = self._merge_live_bar(symbol, df)
            if merged is not None:
                df, live = merged, live_flag
                source = "yahoo+intraday"
            else:
                live = not df.empty and _bar_is_today(df.index[-1], meta)
        else:
            live = False
        df = self._trim(df, days)
        return Bars(symbol=symbol, df=df, meta={k: meta.get(k) for k in
                    ("currency", "exchangeName", "fullExchangeName", "symbol",
                     "exchangeTimezoneName", "regularMarketPrice") if meta.get(k) is not None},
                    live=live, source=source)

    def _merge_live_bar(self, symbol: str, daily: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], bool]:
        """Rebuild today's partial daily bar from intraday bars (touch detection needs
        the running high/low, which the delayed daily series may not have yet)."""
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": self.cfg.intraday_interval, "range": "1d", "includePrePost": "false"}
        try:
            payload = self._http_json(url, params)
            intra, meta = parse_chart_payload(payload, daily=False)
        except Exception as exc:
            log.debug("intraday fetch failed for %s: %s", symbol, exc)
            return None, False
        if intra.empty:
            return None, False
        tzname = meta.get("exchangeTimezoneName")
        try:
            today = pd.Timestamp.now(tz=tzname).date() if tzname else pd.Timestamp.now().date()
        except Exception:
            today = intra.index[-1].date()
        todays = intra[[d == today for d in intra.index.date]]
        if todays.empty:
            return None, False
        agg = pd.DataFrame({
            "open": todays["open"].iloc[0],
            "high": todays["high"].max(),
            "low": todays["low"].min(),
            "close": todays["close"].iloc[-1],
            "volume": todays["volume"].sum(),
        }, index=pd.DatetimeIndex([pd.Timestamp(datetime.combine(today, datetime.min.time()))]))
        base = daily[~pd.Index([d == today for d in daily.index.date])].copy()
        merged = pd.concat([base, agg])
        merged = merged.sort_index()
        return merged, True

    # ── yfinance ─────────────────────────────────────────────────────────
    def _yfinance(self, symbol: str, days: int, end: Optional[str]) -> Bars:
        import yfinance as yf
        kwargs: Dict[str, Any] = dict(interval=self.cfg.interval, auto_adjust=self.cfg.corporate_adjustments,
                                       progress=False, threads=False)
        if end:
            kwargs["end"] = str(pd.Timestamp(end).date())
            kwargs["start"] = str((pd.Timestamp(end) - timedelta(days=days)).date())
        else:
            kwargs["period"] = f"{max(days, 5)}d" if days < 730 else "max"
        tk = yf.Ticker(symbol)
        raw = tk.history(**kwargs)
        if raw is None or len(raw) == 0:
            raise RuntimeError("yfinance returned no rows")
        raw = raw.rename(columns=str.lower)
        daily = self.cfg.interval in ("1d", "d", "daily")
        df = normalize_ohlcv(raw, daily=daily)
        df = self._trim(df, days)
        meta: Dict[str, Any] = {"symbol": symbol}
        try:
            tzv = getattr(tk, "tz", None)
            if tzv:
                meta["exchangeTimezoneName"] = tzv
        except Exception:
            pass
        try:
            fi = getattr(tk, "fast_info", None)
            for key in ("currency", "exchange"):
                val = fi.get(key) if isinstance(fi, dict) else getattr(fi, key, None)
                if val:
                    meta["currency" if key == "currency" else "fullExchangeName"] = val
        except Exception:
            pass
        live = bool(self.live and not end and not df.empty
                    and _bar_is_today(df.index[-1], meta))
        src = "yfinance"
        if self.live and self.cfg.live_intraday_bar and not end and len(df):
            merged = self._yfinance_live_bar(tk, df)
            if merged is not None:
                df, src = merged, "yfinance+intraday"
                # only claim "live" when the rebuilt bar really is today's —
                # a merge that lands on an older session must not be trusted
                live = bool(not df.empty and _bar_is_today(df.index[-1], meta))
        return Bars(symbol=symbol, df=self._trim(df, days), meta=meta, live=live, source=src)

    def _yfinance_live_bar(self, tk, daily: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Rebuild today's forming daily bar from intraday ticks.

        A "price touched the OB" alert needs today's *running* high/low; the daily
        series can lag a few minutes during the NSE session.
        """
        try:
            intra = tk.history(period="1d", interval=self.cfg.intraday_interval,
                               auto_adjust=self.cfg.corporate_adjustments, progress=False,
                               threads=False)
        except Exception as exc:
            log.debug("intraday fetch failed: %s", exc)
            return None
        if intra is None or len(intra) == 0:
            return None
        intra = normalize_ohlcv(intra.rename(columns=str.lower), daily=False)
        if getattr(intra.index, "tz", None) is None and getattr(daily.index, "tz", None) is not None:
            try:
                intra.index = intra.index.tz_localize(daily.index.tz)
            except Exception:
                pass
        if intra.empty:
            return None
        # session date in *exchange* time, not the feed's stamp (a UTC-stamped
        # feed otherwise rolls the early IST bars into the previous day)
        tzname = _market_tz(self.cfg.market)
        try:
            local = intra.index.tz_convert(tzname) if getattr(intra.index, "tz", None) is not None \
                else intra.index.tz_localize(tzname)
        except Exception:
            local = intra.index
        today = local[-1].date()
        todays = intra[[d == today for d in local.date]]
        if todays.empty:
            return None
        stamp = pd.Timestamp(datetime.combine(today, datetime.min.time()))
        tz = getattr(daily.index, "tz", None)
        if tz is not None:
            try:
                stamp = stamp.tz_localize(tz)
            except Exception:
                pass
        bar = pd.DataFrame({
            "open": [todays["open"].iloc[0]], "high": [todays["high"].max()],
            "low": [todays["low"].min()], "close": [todays["close"].iloc[-1]],
            "volume": [todays["volume"].sum()],
        }, index=pd.DatetimeIndex([stamp]))
        base = daily[~pd.Index([d == today for d in daily.index.date])]
        out = pd.concat([base, bar]).sort_index()
        return normalize_ohlcv(out, daily=True)

    # ── local csv ────────────────────────────────────────────────────────
    def _csv(self, symbol: str) -> Bars:
        root = Path(self.cfg.history_dir)
        candidates = [root / f"{symbol}.csv", root / f"{symbol}.CSV", root / f"{symbol.upper()}.csv",
                      root / "all.csv", root / "data.csv"]
        for path in candidates:
            if path.exists():
                df = read_csv_frame(path)
                if path.name.lower() in {"all.csv", "data.csv"} and "symbol" in (c.lower() for c in df.columns):
                    scol = next(c for c in df.columns if str(c).lower() == "symbol")
                    sub = df[df[scol].astype(str).str.upper() == symbol.upper()]
                    df = normalize_ohlcv(sub.drop(columns=[scol]))
                days = self.cfg.lookback_days
                return Bars(symbol=symbol, df=self._trim(df, days), source=f"csv:{path.name}")
        raise FileNotFoundError(f"no csv for {symbol} under {root}")

    # ── synthetic ────────────────────────────────────────────────────────
    def _synthetic(self, symbol: str, days: int) -> Bars:
        seed = abs(hash(symbol)) % 10_000 if symbol else 0
        df = synthetic_frame(n=min(days, 1200), seed=seed, start="2022-01-03")
        return Bars(symbol=symbol, df=df, meta={"symbol": symbol, "currency": "USD",
                                               "fullExchangeName": "SYNTH"}, source="synthetic")

    def _trim(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df.empty or not days:
            return df
        cutoff = df.index[-1] - pd.Timedelta(days=int(days * 1.6))
        out = df[df.index >= cutoff]
        return out if len(out) >= self.cfg.min_bars else df


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic market generator (tests, demo, offline CI)
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_frame(n: int = 600, seed: int = 7, start: str = "2022-01-03",
                    price0: float = 120.0, vol: float = 0.010, setup_every: int = 41,
                    dip: float = 0.012, displacement: float = 0.036,
                    defend_prob: float = 0.55, crash_prob: float = 0.25,
                    tap_on_last_bar: bool = False) -> pd.DataFrame:
    """Deterministic random walk that *contains* the setups the indicator hunts.

    Every ``setup_every`` bars is an engineered sequence:

    1. a bearish origin candle,
    2. a volume-confirmed displacement bar (+``displacement``) that breaks the
       8-bar high and leaves a fresh precision OB at the origin candle,
    3. a 3-5 bar fade that pierces the OB entry (→ Tap 1), which then either
       defends (→ confirmation), breaks the stop (→ invalidation) or chops.

    Used for the offline demo, the test-suite and CI — no network needed.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n); v = np.empty(n)
    px = float(price0)
    bv = 1.5e6
    for i in range(n):
        r = float(rng.normal(0.0002, vol))
        o[i] = px
        c[i] = px * (1.0 + r)
        h[i] = max(o[i], c[i]) + abs(px * vol * float(rng.uniform(0.1, 0.9)))
        l[i] = min(o[i], c[i]) - abs(px * vol * float(rng.uniform(0.1, 0.9)))
        v[i] = bv * float(rng.uniform(0.55, 1.7))
        px = c[i]

    def set_bar(k: int, open_: float, close_: float, hi: float, lo: float, vol_: float) -> None:
        if k < 0 or k >= n or not (np.isfinite(open_) and np.isfinite(close_)):
            return
        o[k], c[k] = open_, close_
        h[k] = max(hi, open_, close_)
        l[k] = min(lo, open_, close_)
        v[k] = vol_

    step = max(9, int(setup_every))
    origins = (list(range(n - 5, 8, -step))[::-1] if tap_on_last_bar
               else list(range(6, n - 14, step)))
    for s0 in origins:
        P = float(c[s0 - 1])
        # 1) bearish origin candle -> the precision OB will be [low .. open] of this bar
        set_bar(s0, P, P * (1 - dip), P * (1 + 0.004), P * (1 - dip - 0.005),
                bv * float(rng.uniform(0.6, 1.2)))
        top = o[s0]
        # 2) displacement: big close, close of the origin bar as open, ~4x volume
        oc, cc = float(c[s0]), float(c[s0]) * (1 + displacement)
        set_bar(s0 + 1, oc, cc, cc * 1.006, oc * 0.998, bv * float(rng.uniform(3.4, 5.2)))
        # 3) fade into the entry level (a hair above the OB top) 3-4 bars later
        entry, target, bars = top * 1.003, top * 0.999, 4
        for k in range(1, bars + 1):
            i = s0 + 1 + k
            if i >= n:
                break
            prev = float(c[i - 1])
            tgt = prev + (target - prev) * min(1.0, 0.45 * k)
            set_bar(i, prev, tgt, prev * 1.005, min(prev, tgt) * 0.995,
                    bv * float(rng.uniform(0.5, 1.2)))
        tap_bar = s0 + 1 + bars - 1
        path = float(rng.random())
        if path < defend_prob:            # defence + follow-through (keeps the zone alive)
            for k in range(1, 12):
                i = tap_bar + k
                if i >= n:
                    break
                op = float(c[i - 1])
                tgt = op * (1 + (0.024 if k == 1 else 0.016 if k == 2 else 0.006))
                set_bar(i, op, tgt, tgt * 1.004, op * 0.998,
                        bv * (3.2 if k <= 2 else float(rng.uniform(0.7, 1.5))))
        elif path < defend_prob + crash_prob:     # breakdown below the distal stop
            for k in range(1, 6):
                i = tap_bar + k
                if i >= n:
                    break
                op = float(c[i - 1])
                tgt = op * (1 - 0.016 * k)
                set_bar(i, op, tgt, op * 1.002, tgt * 0.996,
                        bv * float(rng.uniform(1.0, 2.4)))
        else:                                     # chop around the level
            for k in range(1, 6):
                i = tap_bar + k
                if i >= n:
                    break
                op = float(c[i - 1])
                tgt = op * (1 + float(rng.normal(0.0, 0.006)))
                set_bar(i, op, tgt, max(op, tgt) * 1.004, min(op, tgt) * 0.996,
                        bv * float(rng.uniform(0.6, 1.4)))

    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    return normalize_ohlcv(df, daily=True)


# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────

NSE_INDICES = {
    "nifty50":     "https://en.wikipedia.org/wiki/Nifty_50",
    "nifty100":    "https://en.wikipedia.org/wiki/Nifty_100",
    "nifty200":    "https://en.wikipedia.org/wiki/Nifty_200",
    "nifty500":    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "niftynext50": "https://en.wikipedia.org/wiki/Nifty_Next_50",
    "sensex30":    "https://en.wikipedia.org/wiki/BSE_SENSEX",
    "banknifty":   "https://en.wikipedia.org/wiki/Nifty_Bank",
}


def read_universe(source: Optional[str] = None, *, text: str = "",
                  suffix: str = ".NS") -> List[str]:
    """A comma list, a file path (one ticker per line, ``#`` comments allowed), or an
    index preset (``nifty50`` | ``nifty100`` | ``nifty200`` | ``nifty500`` |
    ``sensex30`` | ``banknifty`` | ``url:https://...``). Indian tickers get the
    yfinance ``.NS`` suffix automatically (``suffix=".BO"`` for BSE)."""
    out: List[str] = []
    if text:
        out += [t for t in re.split(r"[,\s]+", text) if t]
    if source:
        s = str(source).strip()
        low = s.lower()
        if low in {"sp500", "s&p500"}:
            out += _scrape_sp500()
        elif low in NSE_INDICES:
            out += _http_symbols(NSE_INDICES[low], table=not NSE_INDICES[low].endswith(".csv"),
                                 csv=NSE_INDICES[low].endswith(".csv"))
        elif low.startswith("http"):
            out += _http_symbols(s)
        elif Path(s).exists():
            for line in Path(s).read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    out += [t for t in re.split(r"[,\s]+", line) if t]
        elif "," in s:
            out += [t.strip() for t in s.split(",") if t.strip()]
        elif s:
            out.append(s)
    seen, clean = set(), []
    for t in out:
        t = t.strip().strip('"').strip("'").upper()
        if not t or t.startswith("//"):
            continue
        if suffix and "." not in t and not t.startswith("^"):
            t += suffix
        if t not in seen:
            seen.add(t)
            clean.append(t)
    return clean


_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _scrape_sp500() -> List[str]:
    return _http_symbols(_SP500_URL, table=True)


def _http_symbols(url: str, *, table: bool = False, csv: bool = False) -> List[str]:
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=25)
        r.raise_for_status()
        html = r.text
    except Exception as exc:
        log.error("universe download failed (%s) — using empty list", exc)
        return []
    if csv:
        try:
            df = pd.read_csv(io.StringIO(html))
            col = next((c for c in df.columns
                        if str(c).strip().lower() in ("symbol", "series", "security identifier")),
                       df.columns[0])
            return [str(x).strip().upper() for x in df[col].tolist() if str(x).strip()]
        except Exception as exc:
            log.warning("universe csv parse failed: %s", exc)
            return []
    if table:
        try:
            tables = pd.read_html(io.StringIO(html))
            for t in tables:
                cols = [str(c) for c in t.columns]
                if any("Symbol" in c for c in cols):
                    sym = next(c for c in cols if "Symbol" in c)
                    vals = [str(x).strip().upper() for x in t[sym].tolist()]
                    return [v for v in vals if re.fullmatch(r"[A-Z0-9&_-]{1,16}", v)]
        except Exception as exc:
            log.warning("sp500 table parse failed: %s", exc)
        return []
    return re.findall(r"\b[A-Z]{1,5}(\.[A-Z]{1,2})?\b", html)[:2000]


def write_demo_dataset(out_dir: str | Path, symbols: Sequence[str] = ("RELIANCE.NS", "TCS.NS",
                       "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS", "SBIN.NS"), n: int = 700,
                       tap_on_last_bar: Optional[int] = None) -> List[Path]:
    """Create CSV history so the whole pipeline runs with zero network access."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    symbols = list(symbols)
    align = len(symbols) if tap_on_last_bar is None else int(tap_on_last_bar)
    for i, s in enumerate(symbols):
        df = synthetic_frame(n=n, seed=101 + i * 13, start="2022-01-03",
                             price0=60.0 + 40 * i, tap_on_last_bar=(i < align))
        p = out / f"{s}.csv"
        frame_to_csv(df, p)
        paths.append(p)
    return paths
