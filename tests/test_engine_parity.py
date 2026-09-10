"""Wraps precision_tap.selftest so pytest runs the hand-computed Pine-parity checks."""
import pytest

from precision_tap.selftest import CHECKS, run_selftest


@pytest.mark.parametrize("fn", CHECKS, ids=[c.__name__ for c in CHECKS])
def test_parity_check(fn):
    fn()


def test_all_green():
    assert run_selftest(verbose=False) == 0
