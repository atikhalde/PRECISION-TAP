"""`tools/live_drill.py` — the whole live pipeline, offline, in one assertion.

The scanner's own tests swap out :func:`DataSource.get`, so nothing in them
touches the provider code that runs against Yahoo while the market is open.
This wraps the drill instead: fake chart API → intraday rebuild → scanner →
dispatch → mocked Telegram, for both providers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRILL = ROOT / "tools" / "live_drill.py"


def _load():
    spec = importlib.util.spec_from_file_location("live_drill", DRILL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["live_drill"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("provider", ["yahoo", "yfinance"])
@pytest.mark.parametrize("mode", ["live", "eod"])
def test_live_drill_delivers_every_alert(provider, mode, tmp_path, monkeypatch):
    drill = _load()
    monkeypatch.chdir(tmp_path)
    rc = drill.main(["--provider", provider, "--mode", mode, "--cycles", "2", "--quiet",
                     "--config", str(ROOT / "config.yaml"), "--out", "results_drill"])
    assert rc == 0, "the drill reported a failure (see its stdout)"


def test_live_drill_leaves_no_provider_patches_behind(tmp_path, monkeypatch):
    """The drill installs a fake ``yfinance`` module; it must put it back."""
    drill = _load()
    monkeypatch.chdir(tmp_path)
    before = sys.modules.get("yfinance")
    assert drill.main(["--provider", "yfinance", "--quiet", "--config", str(ROOT / "config.yaml"),
                       "--out", "results_drill"]) == 0
    assert sys.modules.get("yfinance") is before
