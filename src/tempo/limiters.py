"""
Classical rate limiter implementations.

Each limiter implements the same interface: allow(key, cost=1) -> bool
and provides introspection via remaining(key) and reset_at(key).
"""

from __future__ import annotations

import abc
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from tempo.clock import Clock, MonotonicClock


class RateLimiter(abc.ABC):
    """Base interface for all rate limiters."""

    @abc.abstractmethod
    def allow(self, key: str = "default", cost: int = 1) -> bool:
        """Return True if the request should be allowed."""
        ...

    @abc.abstractmethod
    def remaining(self, key: str = "default") -> int:
        """Return approximate remaining capacity."""
        ...

    @abc.abstractmethod
    def reset_at(self, key: str = "default") -> Optional[float]:
        """Return the time when capacity will be restored, or None."""
        ...


@dataclass
class _FixedWindow:
    count: int = 0
    window_start: float = 0.0


class FixedWindowLimiter(RateLimiter):
    """
    Fixed window rate limiter.

    Divides time into fixed windows of `window_seconds` and allows
    at most `max_requests` per window.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Optional[Clock] = None,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock or MonotonicClock()
        self._windows: Dict[str, _FixedWindow] = defaultdict(_FixedWindow)

    def _current_window(self, key: str) -> _FixedWindow:
        now = self._clock.now()
        w = self._windows[key]
        window_id = math.floor(now / self.window_seconds)
        current_start = window_id * self.window_seconds
        if w.window_start != current_start:
            w.count = 0
            w.window_start = current_start
        return w

    def allow(self, key: str = "default", cost: int = 1) -> bool:
        w = self._current_window(key)
        if w.count + cost <= self.max_requests:
            w.count += cost
            return True
        return False

    def remaining(self, key: str = "default") -> int:
        w = self._current_window(key)
        return max(0, self.max_requests - w.count)

    def reset_at(self, key: str = "default") -> Optional[float]:
        w = self._current_window(key)
        return w.window_start + self.window_seconds


@dataclass
class _SlidingEntry:
    timestamp: float
    cost: int = 1


class SlidingWindowLimiter(RateLimiter):
    """
    Sliding window log rate limiter.

    Maintains a log of request timestamps and counts requests within
    the trailing window.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Optional[Clock] = None,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock or MonotonicClock()
        self._logs: Dict[str, list[_SlidingEntry]] = defaultdict(list)

    def _prune(self, key: str) -> None:
        now = self._clock.now()
        cutoff = now - self.window_seconds
        entries = self._logs[key]
        # Remove expired entries
        while entries and entries[0].timestamp <= cutoff:
            entries.pop(0)

    def _current_count(self, key: str) -> int:
        self._prune(key)
        return sum(e.cost for e in self._logs[key])

    def allow(self, key: str = "default", cost: int = 1) -> bool:
        if self._current_count(key) + cost <= self.max_requests:
            self._logs[key].append(_SlidingEntry(self._clock.now(), cost))
            return True
        return False

    def remaining(self, key: str = "default") -> int:
        return max(0, self.max_requests - self._current_count(key))

    def reset_at(self, key: str = "default") -> Optional[float]:
        self._prune(key)
        entries = self._logs[key]
        if not entries:
            return None
        return entries[0].timestamp + self.window_seconds


@dataclass
class _Bucket:
    tokens: float = 0.0
    last_refill: float = 0.0


class TokenBucketLimiter(RateLimiter):
    """
    Token bucket rate limiter.

    Tokens are added at a constant rate up to a maximum capacity.
    Each request consumes tokens.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        clock: Optional[Clock] = None,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self._clock = clock or MonotonicClock()
        self._buckets: Dict[str, _Bucket] = {}

    def _get_bucket(self, key: str) -> _Bucket:
        now = self._clock.now()
        if key not in self._buckets:
            self._buckets[key] = _Bucket(tokens=float(self.capacity), last_refill=now)
            return self._buckets[key]
        b = self._buckets[key]
        elapsed = now - b.last_refill
        b.tokens = min(self.capacity, b.tokens + elapsed * self.refill_rate)
        b.last_refill = now
        return b

    def allow(self, key: str = "default", cost: int = 1) -> bool:
        b = self._get_bucket(key)
        if b.tokens >= cost:
            b.tokens -= cost
            return True
        return False

    def remaining(self, key: str = "default") -> int:
        b = self._get_bucket(key)
        return int(b.tokens)

    def reset_at(self, key: str = "default") -> Optional[float]:
        b = self._get_bucket(key)
        if b.tokens >= self.capacity:
            return None
        deficit = self.capacity - b.tokens
        return self._clock.now() + deficit / self.refill_rate


class LeakyBucketLimiter(RateLimiter):
    """
    Leaky bucket rate limiter.

    Requests fill a bucket that drains at a constant rate.
    If the bucket overflows, requests are rejected.
    """

    def __init__(
        self,
        capacity: int,
        drain_rate: float,
        clock: Optional[Clock] = None,
    ):
        self.capacity = capacity
        self.drain_rate = drain_rate  # requests drained per second
        self._clock = clock or MonotonicClock()
        self._levels: Dict[str, _Bucket] = {}

    def _get_level(self, key: str) -> _Bucket:
        now = self._clock.now()
        if key not in self._levels:
            self._levels[key] = _Bucket(tokens=0.0, last_refill=now)
            return self._levels[key]
        b = self._levels[key]
        elapsed = now - b.last_refill
        b.tokens = max(0.0, b.tokens - elapsed * self.drain_rate)
        b.last_refill = now
        return b

    def allow(self, key: str = "default", cost: int = 1) -> bool:
        b = self._get_level(key)
        if b.tokens + cost <= self.capacity:
            b.tokens += cost
            return True
        return False

    def remaining(self, key: str = "default") -> int:
        b = self._get_level(key)
        return max(0, int(self.capacity - b.tokens))

    def reset_at(self, key: str = "default") -> Optional[float]:
        b = self._get_level(key)
        if b.tokens <= 0:
            return None
        return self._clock.now() + b.tokens / self.drain_rate
