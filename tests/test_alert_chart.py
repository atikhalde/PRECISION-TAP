"""The alert image must actually show the order block and the tap.

The chart is the half of the alert a user reads first — "which zone, which tap" —
and it is drawn from a truncated window (``alerts.chart_bars`` bars of a
multi-year frame), so zone boxes and the tap marker have to be positioned relative
to the *start of that window*.  Compute the offset from the truncated length and
every zone lands hundreds of bars right of the visible range: the picture renders
as bare candles with no OB and no TAP marker, and is still attached to the alert —
so nothing looks wrong from the outside.  A silent failure exactly like the
delivery ones, so it is pinned here.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg", force=True)

import numpy as np                          # noqa: E402
import pytest                               # noqa: E402

from precision_tap.chart import render_chart
from precision_tap.data import synthetic_frame
from precision_tap.engine import EV_TAP, run_engine
from precision_tap.params import Params

BARS = 120


@pytest.fixture
def frame_and_zone():
    """A frame with a live zone and a Tap 1 on the newest bar, plus the engine pass."""
    df = synthetic_frame(n=600, seed=0, tap_on_last_bar=True)
    res = run_engine(df, Params.default(), symbol="RELIANCE.NS", intrabar_last=False)
    tap = [e for e in res.events if e.kind == EV_TAP and e.bar >= len(df) - 2]
    assert tap, "fixture must contain a tap on the newest bar"
    return df, res, tap[-1]


def _render(tmp_path, df, res, event, bars: int = BARS):
    """Render one PNG and hand back the figure that was drawn (savefig is wrapped)."""
    import matplotlib.figure as mf

    seen: dict = {}
    real = mf.Figure.savefig

    def savefig(self, fname, *a, **kw):        # noqa: ANN001
        seen["fig"] = self
        return real(self, fname, *a, **kw)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(mf.Figure, "savefig", savefig)
    try:
        out = render_chart(df, res.live_zones, event, out_path=tmp_path / "c.png",
                           symbol="RELIANCE.NS", bars=bars, timeframe="1d")
    finally:
        monkey.undo()
    assert out and (tmp_path / "c.png").exists()
    return seen["fig"]


def _zone_boxes(ax):
    """Wide rectangles only — candle bodies are ~0.62 units and must be ignored."""
    return [p for p in ax.patches if p.get_width() > 1.0 and p.get_height() > 0]


def test_zone_boxes_land_inside_the_drawn_window(tmp_path, frame_and_zone):
    df, res, tap = frame_and_zone
    n = min(len(df), max(30, BARS))
    fig = _render(tmp_path, df, res, tap)
    ax = fig.axes[0]
    drawn = res.live_zones[-12:]                       # what render_chart overlays
    assert drawn, "fixture needs at least one live zone"

    boxes = _zone_boxes(ax)
    assert len(boxes) >= len(drawn), (len(boxes), len(drawn))
    for p in boxes:
        left, width = p.get_x(), p.get_width()
        assert left < n and left + width > -0.5, \
            f"zone box at x=[{left:.1f},{left + width:.1f}] is outside a {n}-bar window"

    spans = {(round(p.get_y(), 6), round(p.get_height(), 6)) for p in boxes}
    want = {(round(z.bot, 6), round(z.top - z.bot, 6)) for z in drawn}
    assert want & spans, "a drawn box must span the order block itself"


def test_tapped_zone_is_boxed_up_to_the_last_bar(tmp_path, frame_and_zone):
    df, res, tap = frame_and_zone
    n = min(len(df), max(30, BARS))
    z = tap.zone
    fig = _render(tmp_path, df, res, tap)
    ax = fig.axes[0]
    box = [p for p in _zone_boxes(ax)
           if np.isclose(p.get_y(), z.bot) and np.isclose(p.get_height(), z.top - z.bot)]
    assert box, "the tapped zone must be drawn"
    assert any(p.get_x() + p.get_width() >= n - 0.51 for p in box), \
        "the zone box has to extend to the newest bar"


def test_tap_marker_is_drawn_on_the_newest_bar(tmp_path, frame_and_zone):
    df, res, tap = frame_and_zone
    n = min(len(df), max(30, BARS))
    fig = _render(tmp_path, df, res, tap)
    ax = fig.axes[0]
    # the marker is the only point sitting on the tapped level
    pts = [np.asarray(c.get_offsets(), dtype=float) for c in ax.collections if len(c.get_offsets())]
    pts = np.vstack(pts) if pts else np.empty((0, 2))
    hit = pts[np.isclose(pts[:, 1], float(tap.level))] if len(pts) else pts
    assert len(hit) == 1, f"no tap marker at level {tap.level:.2f} (points: {pts.tolist()})"
    assert 0 <= hit[0][0] < n, f"tap marker drawn at x={hit[0][0]} in a {n}-bar window"
    assert ax.texts, "the tap marker needs its label"
    lx, _ly = ax.texts[0].get_position()
    assert 0 <= lx < n, f"tap label drawn at x={lx}"


def test_entry_and_stop_lines_are_drawn(tmp_path, frame_and_zone):
    df, res, tap = frame_and_zone
    fig = _render(tmp_path, df, res, tap)
    assert len(fig.axes[0].lines) >= 2, "each live zone draws its entry and stop line"


def test_frames_shorter_than_the_window_are_untouched(tmp_path, frame_and_zone):
    """Nothing shifts when no truncation happens: the offset must stay 0."""
    df, res, _ = frame_and_zone
    short = df.tail(40)
    res40 = run_engine(short, Params.default(), symbol="RELIANCE.NS", intrabar_last=False)
    ev = [e for e in res40.events if e.kind == EV_TAP]
    if not ev:
        pytest.skip("no tap in this 40-bar slice")
    fig = _render(tmp_path, short, res40, ev[-1])
    ax = fig.axes[0]
    assert _zone_boxes(ax), "zones must still be drawn for a short frame"
    assert ax.texts, "the tap marker needs its label"
    lx, _ = ax.texts[0].get_position()
    assert 0 <= lx < 40, f"marker drawn at x={lx} for a 40-bar frame"
