"""precision_tap — Python port of "Institutional OB — Precision Tap & Pre-Order".

Scanner + backtest engine for the daily-timeframe precision order-block / Tap-1
logic, with Telegram delivery.

Quick start::

    from precision_tap.engine import run_engine, Params
    result = run_engine(df, Params.default(), symbol="AAPL")
    for ev in result.tap1:
        print(ev.fmt())
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["Params", "ScanConfig", "run_engine", "EngineResult", "Event", "Zone"]


def __getattr__(name: str):  # lazy re-exports keep `import precision_tap` cheap
    if name in {"Params"}:
        from .params import Params
        return Params
    if name in {"ScanConfig"}:
        from .params import ScanConfig
        return ScanConfig
    if name in {"run_engine", "EngineResult", "Event", "Zone"}:
        from . import engine
        return getattr(engine, name)
    raise AttributeError(name)
