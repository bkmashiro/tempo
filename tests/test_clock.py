"""Tests for the clock module."""

import pytest
from tempo.clock import ManualClock, MonotonicClock


class TestManualClock:
    def test_initial_time(self):
        c = ManualClock(start=10.0)
        assert c.now() == 10.0

    def test_default_start(self):
        c = ManualClock()
        assert c.now() == 0.0

    def test_advance(self):
        c = ManualClock(start=0.0)
        c.advance(5.0)
        assert c.now() == 5.0
        c.advance(3.0)
        assert c.now() == 8.0

    def test_advance_negative_raises(self):
        c = ManualClock()
        with pytest.raises(ValueError, match="Cannot go backwards"):
            c.advance(-1.0)

    def test_set(self):
        c = ManualClock()
        c.set(42.0)
        assert c.now() == 42.0

    def test_advance_fractional(self):
        c = ManualClock()
        c.advance(0.001)
        assert abs(c.now() - 0.001) < 1e-9


class TestMonotonicClock:
    def test_monotonic(self):
        c = MonotonicClock()
        t1 = c.now()
        t2 = c.now()
        assert t2 >= t1
