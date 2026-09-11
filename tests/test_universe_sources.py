"""Universe resolution: preset source chains, full-market preset, loud failures.

Regression cover for the "scanner is green but scans nothing" failure class:
a preset whose every download source fails used to return an empty list, which
the scanner then happily "scanned" (0 symbols, exit 0, no alerts, no diagnosis).
"""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from precision_tap.data import (PRESET_SOURCES, _http_symbols, read_universe,
                                resolve_preset)
from precision_tap.params import ScanConfig
from precision_tap.scanner import Scanner

ROOT = Path(__file__).resolve().parents[1]


# ── source chain ─────────────────────────────────────────────────────────────

def test_every_preset_has_a_non_nse_fallback():
    """nseindia archives reject datacenter IPs — every chain must survive that."""
    for name, chain in PRESET_SOURCES.items():
        non_nse = [u for u, _mode in chain if "nseindia.com" not in u]
        if name != "allnse":            # the exchange file falls back to nifty500
            assert non_nse, f"preset {name!r} has no datacenter-friendly fallback"


def test_preset_chain_falls_through_to_second_source():
    chain = PRESET_SOURCES["nifty500"]
    with patch("precision_tap.data._http_symbols",
               side_effect=[[], ["RELIANCE", "TCS"]]) as fh:
        out = resolve_preset("nifty500")
    assert out == ["RELIANCE", "TCS"]
    assert fh.call_count == 2


def test_preset_with_all_sources_dead_raises():
    with patch("precision_tap.data._http_symbols", return_value=[]):
        with pytest.raises(RuntimeError, match="resolved to 0 symbols"):
            resolve_preset("nifty500")


def test_read_universe_empty_preset_raises_rather_than_silent_empty():
    with patch("precision_tap.data._http_symbols", return_value=[]):
        with pytest.raises(RuntimeError):
            read_universe("nifty500")


def test_preset_aliases():
    assert resolve_preset.__doc__
    with patch("precision_tap.data._http_symbols", return_value=["X"]):
        for alias in ("nse_all", "all_nse", "all-nse"):
            assert resolve_preset(alias), alias


# ── full-exchange preset ─────────────────────────────────────────────────────

_EQUITY_L = """SYMBOL,NAME OF COMPANY,SERIES,DATE OF LISTING
RELIANCE,Reliance Industries,EQ,01-Jan-1995
YESBANK,Yes Bank Limited,EQ,01-Jan-2005
SOMEBANK,Some Bank Bea,BE,12-Mar-2016
SMALLCO,Small Company Bz,BZ,01-Jan-2020
BONDCO,Some Debentures,N2,01-Jan-2010
ETFCO,Some ETF,N1,01-Jan-2010
"""


def test_http_symbols_equities_only_filters_series():
    resp = pd.NA
    with patch("requests.get") as get:
        get.return_value.status_code = 200
        get.return_value.text = _EQUITY_L
        get.return_value.raise_for_status = lambda: None
        out = _http_symbols("https://x/EQUITY_L.csv", csv=True, equities_only=True)
    assert out == ["RELIANCE", "YESBANK", "SOMEBANK", "SMALLCO"]


def test_allnse_falls_back_to_nifty500_chain_when_exchange_file_is_blocked():
    def fake(url, **kw):
        if "EQUITY_L" in url:
            return []
        return ["RELIANCE", "TCS"]           # the nifty500 chain answered
    with patch("precision_tap.data._http_symbols", side_effect=fake):
        out = resolve_preset("allnse")
    assert out == ["RELIANCE", "TCS"]


# ── scanner behaviour on an empty universe ───────────────────────────────────

def _synthetic_cfg() -> ScanConfig:
    cfg = ScanConfig.from_dict({"data": {"provider": "synthetic"}})
    cfg.data.universe = []
    cfg.data.universe_file = "nifty500"
    cfg.out_dir = "/tmp/pt_test_out"
    return cfg


def test_scan_reports_empty_universe_as_a_note():
    cfg = _synthetic_cfg()
    with patch("precision_tap.scanner.Scanner.universe", return_value=[]):
        sc = Scanner(cfg, dry_run=True, out_dir="/tmp/pt_test_out")
        rep = sc.scan(send=False, progress=False)
    assert rep.universe == 0
    assert any(n.startswith("UNIVERSE EMPTY") for n in rep.notes)


def test_cmd_scan_exits_2_on_empty_universe(capsys, tmp_path):
    """A dead preset raises at resolve time (rc=1, loud message); a source that
    answers with an empty list must still not scan 'successfully' (rc=2)."""
    from precision_tap.cli import main
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "data:\n"
        "  provider: synthetic\n"
        "  universe_file: nifty500\n"
        "state_db: /tmp/pt_test_state.sqlite3\n"
        "out_dir: /tmp/pt_test_out\n"
    )
    with patch("precision_tap.data.resolve_preset",
               side_effect=RuntimeError("every source failed")):
        rc = main(["scan", "-c", str(cfg_file), "--no-send"])
    assert rc == 1
    assert "every source failed" in capsys.readouterr().err

    with patch("precision_tap.scanner.read_universe", return_value=[]):
        rc = main(["scan", "-c", str(cfg_file), "--no-send"])
    assert rc == 2
    assert "resolved to 0 symbols" in capsys.readouterr().err


# ── CLI universe overrides ───────────────────────────────────────────────────
def test_cli_symbols_are_normalised_like_the_universe_file():
    """``-S reliance`` must produce ``RELIANCE.NS``, exactly as the file does.

    An unsuffixed universe still *fetches* (the provider adds the suffix), but
    every downstream identity changes: the Yahoo quote button points at a US
    listing, ``data.market: BSE`` scans ``.NS`` tickers, and the de-duplication
    keys differ from the file's, so the same signal alerts twice.
    """
    from types import SimpleNamespace

    from precision_tap.cli import _cfg

    def args(**kw):
        base = dict(config=None, set=[], env_file=None, symbols=None, days=None,
                    provider=None, limit=0, events=None, recent_bars=None,
                    no_charts=False, min_dollar_volume=None, trigger=None, target_r=None,
                    stop_mode=None, trail=None, time_stop=None, risk=None, capital=None,
                    max_positions=None)
        base.update(kw)
        return SimpleNamespace(**base)

    uni_file = str(ROOT / "universe" / "nse.txt")
    cfg = _cfg(args(symbols=["reliance", "TCS.NS", "^NSEI"],
                    set=[f"data.universe_file={uni_file}"]))
    assert cfg.data.universe == ["RELIANCE.NS", "TCS.NS", "^NSEI"]

    bse = _cfg(args(symbols=["RELIANCE"], set=["data.symbol_suffix=.BO"]))
    assert bse.data.universe == ["RELIANCE.BO"]

    lim = _cfg(args(limit=3, set=[f"data.universe_file={uni_file}"]))
    assert len(lim.data.universe) == 3
    assert all(s.endswith(".NS") for s in lim.data.universe), lim.data.universe
