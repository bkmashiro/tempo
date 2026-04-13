"""Tests for classical rate limiter implementations."""

import pytest
from tempo.clock import ManualClock
from tempo.limiters import (
    FixedWindowLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    LeakyBucketLimiter,
)


class TestFixedWindowLimiter:
    def test_allows_up_to_limit(self):
        clock = ManualClock()
        lim = FixedWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        for _ in range(5):
            assert lim.allow() is True
        assert lim.allow() is False

    def test_resets_after_window(self):
        clock = ManualClock()
        lim = FixedWindowLimiter(max_requests=3, window_seconds=10.0, clock=clock)
        for _ in range(3):
            lim.allow()
        assert lim.allow() is False
        clock.advance(10.0)
        assert lim.allow() is True

    def test_remaining(self):
        clock = ManualClock()
        lim = FixedWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        assert lim.remaining() == 5
        lim.allow()
        assert lim.remaining() == 4

    def test_reset_at(self):
        clock = ManualClock(start=0.0)
        lim = FixedWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        lim.allow()
        assert lim.reset_at() == 10.0

    def test_separate_keys(self):
        clock = ManualClock()
        lim = FixedWindowLimiter(max_requests=2, window_seconds=10.0, clock=clock)
        assert lim.allow("a") is True
        assert lim.allow("a") is True
        assert lim.allow("a") is False
        assert lim.allow("b") is True  # different key

    def test_cost(self):
        clock = ManualClock()
        lim = FixedWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        assert lim.allow(cost=3) is True
        assert lim.allow(cost=3) is False
        assert lim.allow(cost=2) is True


class TestSlidingWindowLimiter:
    def test_allows_up_to_limit(self):
        clock = ManualClock()
        lim = SlidingWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        for _ in range(5):
            assert lim.allow() is True
        assert lim.allow() is False

    def test_slides(self):
        clock = ManualClock()
        lim = SlidingWindowLimiter(max_requests=3, window_seconds=10.0, clock=clock)
        for _ in range(3):
            assert lim.allow() is True
            clock.advance(1.0)
        # At t=3, window covers [0,3] — all 3 requests are in window
        assert lim.allow() is False
        # Advance past first request's expiry
        clock.advance(8.0)  # now at t=11, window covers (1, 11]
        assert lim.allow() is True

    def test_remaining(self):
        clock = ManualClock()
        lim = SlidingWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        assert lim.remaining() == 5
        lim.allow()
        assert lim.remaining() == 4

    def test_reset_at_empty(self):
        clock = ManualClock()
        lim = SlidingWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        assert lim.reset_at() is None

    def test_reset_at_with_entries(self):
        clock = ManualClock(start=5.0)
        lim = SlidingWindowLimiter(max_requests=5, window_seconds=10.0, clock=clock)
        lim.allow()
        assert lim.reset_at() == 15.0


class TestTokenBucketLimiter:
    def test_allows_up_to_capacity(self):
        clock = ManualClock()
        lim = TokenBucketLimiter(capacity=5, refill_rate=1.0, clock=clock)
        for _ in range(5):
            assert lim.allow() is True
        assert lim.allow() is False

    def test_refills_over_time(self):
        clock = ManualClock()
        lim = TokenBucketLimiter(capacity=5, refill_rate=1.0, clock=clock)
        for _ in range(5):
            lim.allow()
        assert lim.allow() is False
        clock.advance(3.0)
        assert lim.remaining() == 3
        assert lim.allow() is True

    def test_does_not_exceed_capacity(self):
        clock = ManualClock()
        lim = TokenBucketLimiter(capacity=5, refill_rate=1.0, clock=clock)
        clock.advance(100.0)
        assert lim.remaining() == 5

    def test_reset_at_full(self):
        clock = ManualClock()
        lim = TokenBucketLimiter(capacity=5, refill_rate=1.0, clock=clock)
        assert lim.reset_at() is None  # already full

    def test_reset_at_partial(self):
        clock = ManualClock()
        lim = TokenBucketLimiter(capacity=5, refill_rate=1.0, clock=clock)
        for _ in range(5):
            lim.allow()
        # 0 tokens, refill rate 1/s, capacity 5 → 5 seconds
        reset = lim.reset_at()
        assert reset is not None
        assert abs(reset - 5.0) < 0.1


class TestLeakyBucketLimiter:
    def test_allows_up_to_capacity(self):
        clock = ManualClock()
        lim = LeakyBucketLimiter(capacity=5, drain_rate=1.0, clock=clock)
        for _ in range(5):
            assert lim.allow() is True
        assert lim.allow() is False

    def test_drains_over_time(self):
        clock = ManualClock()
        lim = LeakyBucketLimiter(capacity=5, drain_rate=1.0, clock=clock)
        for _ in range(5):
            lim.allow()
        assert lim.allow() is False
        clock.advance(3.0)  # 3 drained
        assert lim.remaining() == 3
        assert lim.allow() is True

    def test_remaining(self):
        clock = ManualClock()
        lim = LeakyBucketLimiter(capacity=5, drain_rate=1.0, clock=clock)
        assert lim.remaining() == 5
        lim.allow()
        assert lim.remaining() == 4

    def test_reset_at_empty(self):
        clock = ManualClock()
        lim = LeakyBucketLimiter(capacity=5, drain_rate=1.0, clock=clock)
        assert lim.reset_at() is None

    def test_reset_at_partial(self):
        clock = ManualClock()
        lim = LeakyBucketLimiter(capacity=5, drain_rate=2.0, clock=clock)
        for _ in range(4):
            lim.allow()
        reset = lim.reset_at()
        assert reset is not None
        assert abs(reset - 2.0) < 0.1  # 4 items / 2 per second
