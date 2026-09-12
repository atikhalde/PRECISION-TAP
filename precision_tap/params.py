"""Indicator parameters — a 1:1 mirror of the TradingView Pine v6 inputs of
"Institutional OB — Precision Tap & Pre-Order" (INDICATOR.txt).

Every field name maps to one ``input.*`` line in the Pine source, and every
default is the Pine default, so a TradingView setting can be transplanted into
``config.yaml`` without translation. See ANALYSIS.md for the mapping table.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Option vocabularies (identical to the Pine `options=[...]` lists)
# ─────────────────────────────────────────────────────────────────────────────

ZONE_METHODS: List[str] = ["Open to low", "Body to low", "Body only", "Lower half of candle"]
ENTRY_MODES: List[str] = ["Proximal", "50%", "62%", "70.5%", "79%", "Distal + 1 tick"]
FRONTRUN_MODES: List[str] = ["Auto", "ATR", "Ticks", "Off"]

_ZONE_ALIASES = {
    "open to low": "Open to low",
    "open_to_low": "Open to low",
    "opentolow": "Open to low",
    "o2l": "Open to low",
    "body to low": "Body to low",
    "body_to_low": "Body to low",
    "bodytolow": "Body to low",
    "body only": "Body only",
    "body_only": "Body only",
    "bodyonly": "Body only",
    "lower half of candle": "Lower half of candle",
    "lower_half": "Lower half of candle",
    "lower half": "Lower half of candle",
    "half": "Lower half of candle",
}

_ENTRY_ALIASES = {
    "proximal": "Proximal",
    "prox": "Proximal",
    "top": "Proximal",
    "50%": "50%",
    "50": "50%",
    "0.50": "50%",
    "mid": "50%",
    "median": "50%",
    "62%": "62%",
    "62": "62%",
    "0.62": "62%",
    "ote": "62%",
    "70.5%": "70.5%",
    "70.5": "70.5%",
    "705": "70.5%",
    "79%": "79%",
    "79": "79%",
    "0.79": "79%",
    "distal + 1 tick": "Distal + 1 tick",
    "distal": "Distal + 1 tick",
    "distal+1": "Distal + 1 tick",
    "bot": "Distal + 1 tick",
}

_FRONTRUN_ALIASES = {
    "auto": "Auto",
    "atr": "ATR",
    "ticks": "Ticks",
    "tick": "Ticks",
    "off": "Off",
    "none": "Off",
    "false": "Off",
}

# entry_mode -> fraction of the zone width to subtract from the proximal edge
ENTRY_DEPTH = {
    "Proximal": 0.0,
    "50%": 0.50,
    "62%": 0.62,
    "70.5%": 0.705,
    "79%": 0.79,
    "Distal + 1 tick": 1.0,
}


def _norm_enum(value: Any, aliases: Mapping[str, str], allowed: Iterable[str], name: str) -> str:
    if isinstance(value, str) and value in set(allowed):
        return value
    key = str(value).strip().lower()
    if key in aliases:
        return aliases[key]
    raise ValueError(f"{name}={value!r} is not one of {sorted(set(allowed))}")


# ─────────────────────────────────────────────────────────────────────────────
# Params
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Params:
    """All indicator inputs. Field order mirrors the Pine source top-to-bottom."""

    # ── 1 — Volume-confirmed displacement ────────────────────────────────
    vol_len: int = 20                 # volLen       "Local volume average"
    atr_len: int = 14                 # atrLen       "ATR length"
    min_rvol: float = 1.8             # minRVOL      "Minimum displacement RVOL"
    min_range_atr: float = 1.20       # minRangeATR  "Minimum displacement range ÷ ATR"
    min_body_frac: float = 0.55       # minBodyFrac  "Minimum bullish body fraction"
    min_clv: float = 0.72             # minCLV       "Minimum close location"
    structure_len: int = 8            # structureLen "Structure-break lookback"
    origin_search: int = 8            # originSearch "Search back for origin candle"
    allow_neutral_origin: bool = True  # allowNeutral "Allow small neutral origin candle"
    neutral_body: float = 0.20        # neutralBody  "Neutral body maximum ÷ range"

    # ── 2 — Exact order-block zone ───────────────────────────────────────
    zone_method: str = "Open to low"  # zoneMethod   "Tight-zone method"
    entry_mode: str = "Proximal"      # entryMode    "Pre-order entry"
    frontrun_mode: str = "Auto"       # frontRunMode "Near-miss/front-run buffer"
    frontrun_atr: float = 0.18        # frontRunATR  "ATR buffer above OB"
    frontrun_ticks: int = 2           # frontRunTicks "Minimum tick buffer"
    max_zone_buffer: float = 0.40     # maxZoneBuffer "Auto buffer cap as zone fraction"
    entry_offset_ticks: int = 0       # entryOffsetTicks "Additional manual offset in ticks"
    stop_atr: float = 0.15            # stopATR      "Stop buffer below distal ÷ ATR"
    approach_atr: float = 0.25        # approachATR  "Pre-alert distance above entry ÷ ATR"
    min_age: int = 3                  # minAge       "Minimum bars before first tap"
    max_zones: int = 30               # maxZones     "Maximum zones"
    max_touches: int = 4              # maxTouches   "Maximum touches before exhaustion"
    raise_after_first_tap: bool = True  # raiseAfterFirstTap "Adapt entry after first defended tap"
    repeat_tap_atr: float = 0.05      # repeatTapATR "Repeat-entry buffer above Tap-1 low ÷ ATR"
    require_departure: float = 1.0    # requireDeparture "Required departure above proximal ÷ ATR"

    # ── 3 — Optional defence confirmation ────────────────────────────────
    confirm_bars: int = 3             # confirmBars  "Bars after tap allowed for confirmation"
    confirm_rvol: float = 1.3         # confirmRVOL  "Confirmation minimum RVOL"
    confirm_clv: float = 0.65         # confirmCLV   "Confirmation minimum close location"
    confirm_bos_len: int = 3          # confirmBOSLen "Confirmation micro-BOS lookback"
    require_sweep: bool = False       # requireSweep "Require sell-side sweep on tap"
    sweep_len: int = 5                # sweepLen     "Sweep lookback"

    # ── Symbol microstructure (Pine `syminfo.mintick`) ───────────────────
    mintick: float = 0.05             # tick size; NSE/BSE equities = 0.05 (US = 0.01)
    warmup: int = 60                  # bars of history needed before signals are trusted

    # ── normalisation ────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self.zone_method = _norm_enum(self.zone_method, _ZONE_ALIASES, ZONE_METHODS, "zone_method")
        self.entry_mode = _norm_enum(self.entry_mode, _ENTRY_ALIASES, ENTRY_MODES, "entry_mode")
        self.frontrun_mode = _norm_enum(
            self.frontrun_mode, _FRONTRUN_ALIASES, FRONTRUN_MODES, "frontrun_mode"
        )
        self.entry_depth = ENTRY_DEPTH[self.entry_mode]
        if self.mintick <= 0:
            raise ValueError("mintick must be > 0")
        self.max_zones = max(1, int(self.max_zones))
        self.max_touches = max(1, int(self.max_touches))
        self.min_age = max(1, int(self.min_age))
        self.sweep_len = max(2, int(self.sweep_len))
        self.origin_search = max(1, min(20, int(self.origin_search)))

    # ── constructors ─────────────────────────────────────────────────────
    @classmethod
    def default(cls) -> "Params":
        return cls()

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "Params":
        """Build from a mapping, ignoring unknown keys but reporting them."""
        data = dict(data or {})
        # accept the exact Pine identifiers too, so a settings export works verbatim
        pine_alias = {
            "volLen": "vol_len", "atrLen": "atr_len", "minRVOL": "min_rvol",
            "minRangeATR": "min_range_atr", "minBodyFrac": "min_body_frac",
            "minCLV": "min_clv", "structureLen": "structure_len",
            "originSearch": "origin_search", "allowNeutral": "allow_neutral_origin",
            "neutralBody": "neutral_body", "zoneMethod": "zone_method",
            "entryMode": "entry_mode", "frontRunMode": "frontrun_mode",
            "frontRunATR": "frontrun_atr", "frontRunTicks": "frontrun_ticks",
            "maxZoneBuffer": "max_zone_buffer", "entryOffsetTicks": "entry_offset_ticks",
            "stopATR": "stop_atr", "approachATR": "approach_atr", "minAge": "min_age",
            "maxZones": "max_zones", "maxTouches": "max_touches",
            "raiseAfterFirstTap": "raise_after_first_tap", "repeatTapATR": "repeat_tap_atr",
            "requireDeparture": "require_departure", "confirmBars": "confirm_bars",
            "confirmRVOL": "confirm_rvol", "confirmCLV": "confirm_clv",
            "confirmBOSLen": "confirm_bos_len", "requireSweep": "require_sweep",
            "sweepLen": "sweep_len",
        }
        names = {f.name for f in dataclasses.fields(cls)}
        out: Dict[str, Any] = {}
        unknown: List[str] = []
        for key, value in data.items():
            if key in pine_alias:
                key = pine_alias[key]
            if key in names:
                out[key] = value
            else:
                unknown.append(str(key))
        if unknown:
            # silently dropping a typo would silently change the strategy
            raise KeyError(
                "unknown indicator parameter(s): " + ", ".join(sorted(unknown))
                + " — see precision_tap/params.py for valid names"
            )
        return cls(**out)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    def replace(self, **overrides: Any) -> "Params":
        return dataclasses.replace(self, **overrides)

    def describe(self) -> str:
        lines = []
        for f in dataclasses.fields(self):
            lines.append(f"  {f.name:<24} = {getattr(self, f.name)!r}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest + alerting configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeConfig:
    """Execution / exit policy for the backtest engine (the indicator is an
    entry-detection tool only — it defines no take-profit, so exits are ours)."""

    entry_trigger: str = "tap1"          # tap1 | tap | confirmed | tap1_or_confirmed
    min_tap: int = 1                     # first tap number to trade (`tap1` => 1)
    max_tap: int = 1
    allow_short: bool = False            # the indicator is long-only by construction
    max_entry_distance_atr: float = 0.0  # skip signals already >X ATR past the entry level (0 = off)
    same_bar_stop: bool = True           # assume worst intrabar path on the entry bar
    stop_mode: str = "touch"             # touch (low<=stop) | close (close<stop, mirrors Pine invalidation)
    target_r: float = 2.0                # primary take-profit in R multiples (0 = disable)
    trail_target: bool = False           # move stop to break-even after target_r/2 is reached
    trail_mode: str = "none"             # none | breakeven | chandelier | structure
    chandelier_atr: float = 3.0          # trail distance for chandelier mode
    structure_lookback: int = 5          # swing-low lookback for structure trailing
    time_stop_bars: int = 20             # flat after N bars (0 = off)
    exit_on_reversal: bool = False       # exit if a bar closes below zone bottom
    commission_bps: float = 2.0          # per-side, basis points of notional
    slippage_bps: float = 2.0            # per-side, applied to fills
    spread_bps: float = 0.0              # extra half-spread penalty per side
    risk_per_trade: float = 1.0          # % of equity risked (for $ sizing in the report)
    max_open_positions: int = 0          # 0 = unlimited
    initial_capital: float = 100_000.0
    allow_rollover: bool = False         # re-enter the same zone after a fresh Tap-1

    def __post_init__(self) -> None:
        ok = {"tap1", "tap", "confirmed", "tap1_or_confirmed"}
        if self.entry_trigger not in ok:
            raise ValueError(f"entry_trigger={self.entry_trigger!r} must be one of {sorted(ok)}")
        if self.stop_mode not in {"touch", "close"}:
            raise ValueError("stop_mode must be 'touch' or 'close'")
        if self.trail_mode not in {"none", "breakeven", "chandelier", "structure"}:
            raise ValueError("trail_mode must be none|breakeven|chandelier|structure")


@dataclass
class AlertConfig:
    """Which events reach the Telegram channel, and how they are throttled."""

    events: List[str] = field(default_factory=lambda: ["tap1", "approach", "confirmed"])
    recent_bars: int = 1                 # only alert on events within the last N bars
    include_invalidations: bool = False
    min_liquidity_dollar_volume: float = 0.0   # 20d median $ volume filter (0 = off)
    min_price: float = 0.0
    max_price: float = 0.0
    max_age_bars: int = 0                      # ignore zones older than N bars (0 = off)
    skip_dead_on_arrival: bool = True          # no buy alert for a level its own bar already broke
    skip_stale_bars: bool = True               # no intraday alert unless the last bar is today's session
    match_indicator_100: bool = True           # EOD scans evaluate closed bars only (== TV "Once Per Bar Close")
    once_per_symbol_per_day: bool = True
    daily_limit: int = 0                       # hard cap on alerts per scan cycle (0 = off)
    quiet_log_only: bool = False               # compute + store, send nothing
    chart: bool = True                         # attach rendered PNG
    chart_bars: int = 120
    top_zones: int = 40                        # rows in the console/markdown watchlist
    link_template: str = "https://in.tradingview.com/chart/?symbol={exchange}:{symbol}"
    default_exchange: str = "NSE"

    def __post_init__(self) -> None:
        ok = {"new_ob", "approach", "tap1", "tap", "confirmed", "invalidated"}
        bad = [e for e in self.events if e not in ok]
        if bad:
            raise ValueError(f"alert.events contains unknown values {bad}; valid: {sorted(ok)}")


@dataclass
class TelegramConfig:
    enabled: bool = True
    bot_token: str = ""
    chat_ids: List[str] = field(default_factory=list)
    parse_mode: str = "HTML"                   # HTML | MarkdownV2 | Markdown | plain
    disable_notification: bool = False
    protect_content: bool = False
    allow_insecure: bool = False
    api_base: str = "https://api.telegram.org"
    timeout_sec: float = 20.0
    max_retries: int = 3
    min_seconds_between_messages: float = 1.2  # stay under Telegram's ~1 msg/s per chat
    proxy: str = ""
    message_thread_id: int = 0                 # forum-topic id, 0 = main chat
    preview: bool = True

    def __post_init__(self) -> None:
        self.chat_ids = [str(c).strip() for c in (self.chat_ids or []) if str(c).strip()]
        self.parse_mode = (self.parse_mode or "HTML").strip().upper()
        if self.parse_mode not in {"HTML", "MARKDOWNV2", "MARKDOWN", "PLAIN"}:
            raise ValueError("telegram.parse_mode must be HTML | MarkdownV2 | Markdown | plain")


@dataclass
class LiveConfig:
    """Scheduling for the always-on `run` command."""

    market_timezone: str = "Asia/Kolkata"      # NSE/BSE session clock
    trading_days: List[str] = field(default_factory=lambda: ["mon", "tue", "wed", "thu", "fri"])
    session_open: str = "09:15"                # NSE continuous session
    session_close: str = "15:30"
    intraday_poll_minutes: int = 10            # 0 disables intraday polling
    scan_times: List[str] = field(default_factory=lambda: ["09:35", "11:30", "13:30", "15:35", "16:10"])
    force_close_buffer_min: int = 5
    run_on_startup: bool = True
    recheck_on_new_bar: bool = True
    heartbeat_daily_time: str = "16:30"        # "scan finished / idle" summary ping; "" disables
    log_file: str = "logs/precision_tap.log"
    pid_file: str = ""
    max_workers: int = 8


@dataclass
class DataConfig:
    provider: str = "yfinance"                 # yfinance | yahoo | csv | synthetic
    fallback_provider: str = "yahoo"           # per-symbol failover when the primary errors ("" = off)
    market: str = "NSE"                        # NSE | BSE — labels + tick-size defaults
    symbol_suffix: str = ".NS"                 # appended when a symbol has no suffix (BSE: .BO)
    tick_sizes: Dict[str, float] = field(default_factory=lambda: {".NS": 0.05, ".BO": 0.05})
    interval: str = "1d"
    lookback_days: int = 800                   # daily bars fetched for scanning
    history_dir: str = "data/csv"              # csv provider root
    cache_dir: str = "data/cache"
    cache_max_age_minutes: int = 30            # TTL for cached bars during live scans
    eod_cache_max_age_minutes: int = 60 * 20   # TTL once the session is closed
    universe_file: str = "universe/nse.txt"
    universe: List[str] = field(default_factory=list)
    live_intraday_bar: bool = True             # rebuild today's forming daily bar from 1m/5m data
    intraday_interval: str = "5m"
    corporate_adjustments: bool = True         # yfinance auto_adjust / Yahoo `?period` raw vs adj
    rate_limit_per_sec: float = 4.0
    retry_max: int = 3
    retry_backoff: float = 1.6
    min_bars: int = 90                         # skip symbols with too little history

    _DAILY = {"1d", "d", "daily", "day", "1day"}

    def __post_init__(self) -> None:
        """Hard constraints for this build: Indian markets, daily timeframe, yfinance live."""
        self.provider = (self.provider or "yfinance").strip().lower()
        if self.provider in {"yahoo_chart", "chart", ""}:
            self.provider = "yahoo"
        self.fallback_provider = (self.fallback_provider or "").strip().lower()
        if self.fallback_provider in {"yahoo_chart", "chart"}:
            self.fallback_provider = "yahoo"
        if self.fallback_provider == self.provider:
            self.fallback_provider = ""        # failing over to yourself is not a fallback
        if self.fallback_provider and self.fallback_provider not in {"yahoo", "yfinance"}:
            raise ValueError(
                f"data.fallback_provider={self.fallback_provider!r}: only 'yahoo'/'yfinance' "
                "can serve the live market (csv/synthetic are frozen replays)")
        if self.provider == "csv":                       # offline replay of exported NSE history
            pass
        self.interval = str(self.interval or "1d").strip().lower()
        if self.interval in self._DAILY:
            self.interval = "1d"
        if self.interval != "1d":
            raise ValueError(
                f"data.interval={self.interval!r}: only the daily timeframe is supported. The "
                "indicator's gates are daily-calibrated (RVOL vs the 20-day volume mean, ATR "
                "ratios, the 8-day structure break, minAge in bars-of-trading) — alerting on a "
                "lower frame would silently mean something else.")
        self.market = (self.market or "NSE").strip().upper()
        if self.market not in {"NSE", "BSE"}:
            raise ValueError(
                f"data.market={self.market!r}: this build scans Indian markets only (NSE or BSE).")
        self.symbol_suffix = (self.symbol_suffix or ".NS").strip().upper()
        if self.symbol_suffix not in {".NS", ".BO"}:
            raise ValueError('data.symbol_suffix must be ".NS" (NSE) or ".BO" (BSE)')
        if float(self.mintick_default() if hasattr(self, "mintick_default") else 0.0):
            pass
        self.universe = [str(s).strip().upper() for s in (self.universe or []) if str(s).strip()]


@dataclass
class ScanConfig:
    """Top-level runtime configuration."""

    params: Params = field(default_factory=Params.default)
    trade: TradeConfig = field(default_factory=TradeConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    state_db: str = "data/state.sqlite3"
    out_dir: str = "results"
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ScanConfig":
        d = dict(d)
        aliases = {"indicator", "params", "backtest", "trade", "alerts", "alert"}
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)} - aliases
        if unknown:
            raise KeyError("unknown top-level config key(s): " + ", ".join(sorted(unknown)))
        def build(cls_, data, section):
            names = {f.name for f in dataclasses.fields(cls_)}
            bad = sorted(set(data or {}) - names)
            if bad:
                raise KeyError(f"config section '{section}' has unknown key(s): "
                               + ", ".join(bad) + f" · valid: {', '.join(sorted(names))}")
            return cls_(**(data or {}))

        return cls(
            params=Params.from_dict(d.get("indicator") or d.get("params")),
            trade=build(TradeConfig, d.get("backtest") or d.get("trade"), "backtest"),
            alert=build(AlertConfig, d.get("alerts") or d.get("alert"), "alerts"),
            telegram=build(TelegramConfig, d.get("telegram"), "telegram"),
            live=build(LiveConfig, d.get("live"), "live"),
            data=build(DataConfig, d.get("data"), "data"),
            state_db=d.get("state_db", cls().state_db),
            out_dir=d.get("out_dir", cls().out_dir),
            log_level=d.get("log_level", "INFO"),
        )
