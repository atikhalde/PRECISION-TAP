"""Regression tests for source mode and live-bar safety in the scanner."""
from __future__ import annotations

from precision_tap.data import Bars, synthetic_frame
from precision_tap.params import AlertConfig, DataConfig, ScanConfig, TelegramConfig
from precision_tap.scanner import Scanner


def _config(tmp_path):
    return ScanConfig(
        data=DataConfig(
            provider="yfinance",
            universe=["RELIANCE.NS"],
            min_bars=90,
            cache_dir=str(tmp_path / "cache"),
        ),
        alert=AlertConfig(chart=False, min_liquidity_dollar_volume=0, min_price=0),
        telegram=TelegramConfig(enabled=False),
        state_db=":memory:",
        out_dir=str(tmp_path / "results"),
    )


def test_single_symbol_live_scan_constructs_a_live_source(monkeypatch, tmp_path):
    """The one-symbol API must fetch the forming bar, not the init EOD source."""
    seen_modes = []
    frame = synthetic_frame(n=520, seed=0, tap_on_last_bar=True)

    def fake_get(self, symbol, **kwargs):
        seen_modes.append(self.live)
        return Bars(symbol=symbol, df=frame, live=self.live, source="fake")

    monkeypatch.setattr("precision_tap.scanner.DataSource.get", fake_get)
    scanner = Scanner(_config(tmp_path), dry_run=True)
    try:
        state = scanner.scan_symbol("RELIANCE.NS", live=True)
    finally:
        scanner.close()

    assert state.ok
    assert state.live is True
    assert seen_modes == [True]


def test_live_scan_does_not_replay_a_closed_bar_as_intraday(monkeypatch, tmp_path):
    """A live-capable feed without today's bar must not create a false live alert."""
    frame = synthetic_frame(n=520, seed=0, tap_on_last_bar=True)

    def fake_get_many(self, symbols, **kwargs):
        return {s: Bars(symbol=s, df=frame, live=False, source="fake") for s in symbols}

    monkeypatch.setattr("precision_tap.scanner.DataSource.get_many", fake_get_many)
    scanner = Scanner(_config(tmp_path), dry_run=True)
    try:
        report = scanner.scan(live=True, progress=False)
    finally:
        scanner.close()

    assert report.alerts == []
    assert any("live bar unavailable" in note for note in report.notes)
