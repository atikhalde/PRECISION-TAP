"""Configuration: YAML file + environment overrides + CLI dotted overrides."""

from __future__ import annotations

import copy
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .params import AlertConfig, DataConfig, LiveConfig, Params, ScanConfig, TelegramConfig, TradeConfig

log = logging.getLogger("precision_tap.config")

#: Secrets files probed, in order.  ``/etc/precision-tap.env`` is the location the
#: deployment docs use for the systemd ``EnvironmentFile`` — and the only way a
#: cron job (or a bare ``python -m precision_tap scan`` in a shell that never
#: exported the secrets) sees them at all.  Without it the same host scans
#: "successfully" while every alert is logged and dropped, because
#: ``TELEGRAM_BOT_TOKEN`` was never in the process environment.
#: ``PRECISION_TAP_ENV_FILE`` overrides the list entirely.
ENV_FILE_CANDIDATES = (".env", "config/.env", "/etc/precision-tap.env")
_SECRET_RE = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


# ─────────────────────────────────────────────────────────────────────────────
# .env handling (no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def load_dotenv(path: Optional[str] = None, *, override: bool = False) -> List[str]:
    """Load ``KEY=value`` lines into ``os.environ``. Returns the keys set."""
    # NB: not named `override` — that is this function's "replace existing env"
    # keyword, and shadowing it here would silently clobber the process env.
    wanted = str(os.environ.get("PRECISION_TAP_ENV_FILE", "") or "").strip()
    if path:
        candidates = [Path(path)]
    elif wanted:
        candidates = [Path(wanted)]
    else:
        candidates = [Path(c) for c in ENV_FILE_CANDIDATES]
    loaded: List[str] = []
    for p in candidates:
        if not p or not p.exists():
            if path or wanted:
                log.warning("env file %s not found — any secret it should provide "
                            "is missing", p)
            continue
        for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].strip()
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (override or key not in os.environ):
                os.environ[key] = value
                loaded.append(key)
        break
    return loaded


def expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:default}`` inside strings/lists."""
    if isinstance(value, str):
        def sub(m):
            return os.environ.get(m.group(1), m.group(2) or "")
        out = _SECRET_RE.sub(sub, value)
        return out
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


# ─────────────────────────────────────────────────────────────────────────────
# YAML (PyYAML optional — tiny fallback parser keeps the tool usable anywhere)
# ─────────────────────────────────────────────────────────────────────────────

def _mini_yaml(text: str) -> Dict[str, Any]:
    """Very small indentation-based YAML subset: mappings, scalars, inline lists,
    ``- item`` sequences, ``#`` comments. Enough for config.yaml without a dependency."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any]] = [(-1, root)]
    lines = text.splitlines()
    for pos, raw in enumerate(lines):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = raw.split("  #", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if body.startswith("- "):
            val = _coerce(body[2:].strip())
            if isinstance(parent, list):
                parent.append(val)
            else:
                log.warning("mini-yaml: orphan list item %r (install PyYAML for full support)", body)
            continue
        if ":" not in body:
            continue
        key, _, rest = body.partition(":")
        key = key.strip().strip('"').strip("'")
        rest = rest.strip()
        if rest == "":
            # container: look ahead to decide list vs dict
            child: Any = {}
            if isinstance(parent, dict):
                parent[key] = child
            stack.append((indent, child))
            # if the next non-blank line is a "- " item, swap to list
            for nxt in lines[pos + 1:]:
                if not nxt.strip() or nxt.lstrip().startswith("#"):
                    continue
                if nxt.strip().startswith("- "):
                    lst: List[Any] = []
                    parent[key] = lst
                    stack[-1] = (indent, lst)
                break
            continue
        if isinstance(parent, dict):
            parent[key] = _coerce(rest)
    return _expand_lists(root)


def _expand_lists(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_lists(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_lists(v) for v in node]
    return node


def _coerce(text: str) -> Any:
    t = text.strip()
    if t == "":
        return None
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        return [_coerce(x) for x in inner.split(",")]
    if t.lower() in {"true", "yes", "on"}:
        return True
    if t.lower() in {"false", "no", "off"}:
        return False
    if t.lower() in {"null", "none", "~"}:
        return None
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    try:
        if re.fullmatch(r"[+-]?\d+", t):
            return int(t)
        return float(t)
    except ValueError:
        return t


def read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")
        return data
    except ImportError:
        return _mini_yaml(text)


def deep_merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_kv_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """``--set data.provider=csv --set indicator.min_rvol=2.2`` → nested dict."""
    out: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, _, val = item.partition("=")
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part.strip(), {})
        node[parts[-1].strip()] = _coerce(val)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Optional[str] = None, *, overrides: Sequence[str] = (),
                env_path: Optional[str] = None) -> ScanConfig:
    """config.yaml (optional) → env expansion → CLI overrides → ScanConfig."""
    load_dotenv(env_path)
    raw: Dict[str, Any] = {}
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates += [Path("config.yaml"), Path("config/config.yaml"), Path("config.yml")]
    for cand in candidates:
        if cand.exists():
            raw = read_yaml(cand)
            log.debug("loaded config from %s", cand)
            break
    else:
        if path:
            raise FileNotFoundError(f"config file not found: {path}")
    # allow env to point at a universe file
    if os.environ.get("PRECISION_TAP_UNIVERSE"):
        raw.setdefault("data", {})["universe_file"] = os.environ["PRECISION_TAP_UNIVERSE"]
    if os.environ.get("PRECISION_TAP_PROVIDER"):
        raw.setdefault("data", {})["provider"] = os.environ["PRECISION_TAP_PROVIDER"]

    raw = deep_merge(raw, parse_kv_overrides(overrides))
    raw = expand_env(raw)

    # scalars are accepted anywhere a list is expected: `--set telegram.chat_ids=111`
    # or `chat_ids: "111, 222"` in YAML must both work
    for section, fields in (("telegram", ("chat_ids",)),
                            ("alerts", ("events",)),
                            ("live", ("trading_days", "scan_times")),
                            ("data", ("universe",))):
        node = raw.get(section)
        if not isinstance(node, dict):
            continue
        for field in fields:
            if field not in node or node[field] is None:
                continue
            val = node[field]
            if isinstance(val, (list, tuple)):
                node[field] = [v for v in val if str(v).strip() != ""]
            elif isinstance(val, str):
                node[field] = [v.strip() for v in re.split(r"[,\s]+", val) if v.strip()]
            else:
                node[field] = [val]
        if section == "telegram" and "chat_id" in node:
            node.setdefault("chat_ids", node.pop("chat_id"))
        raw[section] = node

    cfg = ScanConfig.from_dict(raw)
    if not cfg.telegram.bot_token:
        cfg.telegram.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not cfg.telegram.chat_ids:
        env_chats = os.environ.get("TELEGRAM_CHAT_ID", "") or os.environ.get("TELEGRAM_CHAT_IDS", "")
        cfg.telegram.chat_ids = [c.strip() for c in re.split(r"[,\s]+", env_chats) if c.strip()]
    if not cfg.telegram.proxy:
        cfg.telegram.proxy = os.environ.get("HTTPS_PROXY", os.environ.get("https_proxy", ""))
    if not cfg.data.universe:
        env_uni = os.environ.get("PRECISION_TAP_SYMBOLS", "")
        cfg.data.universe = [s.strip() for s in re.split(r"[,\s]+", env_uni) if s.strip()]
    return cfg


def config_to_dict(cfg: ScanConfig) -> Dict[str, Any]:
    import dataclasses
    return {
        "indicator": cfg.params.to_dict(),
        "backtest": dataclasses.asdict(cfg.trade),
        "alerts": dataclasses.asdict(cfg.alert),
        "telegram": {k: ("***" if k == "bot_token" and v else v)
                     for k, v in dataclasses.asdict(cfg.telegram).items()},
        "live": dataclasses.asdict(cfg.live),
        "data": dataclasses.asdict(cfg.data),
        "state_db": cfg.state_db,
        "out_dir": cfg.out_dir,
        "log_level": cfg.log_level,
    }


def setup_logging(level: str = "INFO", logfile: Optional[str] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if logfile:
        p = Path(logfile)
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    for noisy in ("urllib3", "yfinance", "peewee", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
