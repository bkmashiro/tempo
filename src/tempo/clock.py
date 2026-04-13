"""
Clock abstraction for testability.

Provides a monotonic clock interface so the entire library can be tested
deterministically without sleeping or mocking time.
"""

from __future__ import annotations

import abc
import time


class Clock(abc.ABC):
    """Abstract clock interface."""

    @abc.abstractmethod
    def now(self) -> float:
        """Return current time in seconds (monotonic)."""
        ...


class MonotonicClock(Clock):
    """Production clock backed by time.monotonic()."""

    def now(self) -> float:
        return time.monotonic()


class ManualClock(Clock):
    """Manually-controlled clock for testing."""

    def __init__(self, start: float = 0.0):
        self._time = start

    def now(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        """Advance the clock by the given number of seconds."""
        if seconds < 0:
            raise ValueError("Cannot go backwards in time")
        self._time += seconds

    def set(self, t: float) -> None:
        """Set the clock to an absolute time."""
        self._time = t
