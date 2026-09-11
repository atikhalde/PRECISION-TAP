import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def synthetic_daily():
    from precision_tap.data import synthetic_frame
    return synthetic_frame(n=520, seed=7)
