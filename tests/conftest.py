import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def synthetic_daily():
    from precision_tap.data import synthetic_frame
    return synthetic_frame(n=520, seed=7)


@pytest.fixture
def exchange_clock(monkeypatch):
    """Pin the exchange clock to mid-session of the last trading weekday.

    The live-path tests build synthetic markets whose newest bar is that day,
    while the provider (``data._now_tz``) and the scanner's staleness guard
    (``scanner._session_now``) ask the wall clock whether a bar belongs to
    *today*.  Left free, those two agree only while the IST date matches the
    host date — so every live-merge test flips red after 18:30Z (past midnight
    IST) and on weekends.  Pinning both seams to 11:30 of the last weekday
    (the same rule the fixtures use) makes them deterministic at any hour.
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    tzname = "Asia/Kolkata"
    day = datetime.now(ZoneInfo(tzname))
    while day.weekday() >= 5:                       # Sat/Sun roll back to Friday
        day -= timedelta(days=1)
    now = day.replace(hour=11, minute=30, second=0, microsecond=0)

    def _pinned(tz: str = tzname) -> datetime:
        return now if tz == tzname else now.astimezone(ZoneInfo(tz))

    from precision_tap import data, scanner
    monkeypatch.setattr(data, "_now_tz", _pinned)
    monkeypatch.setattr(scanner, "_session_now", _pinned)
    return now

