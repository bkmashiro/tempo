"""
Adaptive Rate Limiter — combines classical limiters with rhythm detection.

The AdaptiveRateLimiter wraps a set of policies, each associated with a
behavioral pattern. As the detector classifies the request stream, the
appropriate policy's limiter is applied.

This is the "brain" of Tempo: it decides how strict to be based on
*how* requests arrive, not just how many.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

from tempo.clock import Clock, MonotonicClock
from tempo.detector import (
    DetectorConfig,
    PatternAnalysis,
    RequestPattern,
    RhythmDetector,
)
from tempo.limiters import (
    RateLimiter,
    FixedWindowLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    LeakyBucketLimiter,
)


@dataclass
class Policy:
    """
    A rate limiting policy for a specific behavioral pattern.

    Attributes:
        pattern: The pattern this policy applies to
        limiter_factory: Callable that creates a fresh limiter instance
        description: Human-readable description of the policy
    """

    pattern: RequestPattern
    limiter_factory: Callable[[Clock], RateLimiter]
    description: str = ""

    def create_limiter(self, clock: Clock) -> RateLimiter:
        return self.limiter_factory(clock)


@dataclass
class PolicySet:
    """
    A complete set of policies for all patterns.

    Provides factory methods for common configurations.
    """

    policies: Dict[RequestPattern, Policy] = field(default_factory=dict)
    default_policy: Optional[Policy] = None

    def get(self, pattern: RequestPattern) -> Optional[Policy]:
        return self.policies.get(pattern, self.default_policy)

    @staticmethod
    def lenient(
        base_rate: int = 100,
        window: float = 60.0,
    ) -> "PolicySet":
        """
        A lenient policy set that gives humans extra room and
        only restricts clear bot patterns.
        """
        return PolicySet(
            policies={
                RequestPattern.HUMAN_BURSTY: Policy(
                    pattern=RequestPattern.HUMAN_BURSTY,
                    limiter_factory=lambda c: TokenBucketLimiter(
                        capacity=int(base_rate * 2),
                        refill_rate=base_rate * 2 / window,
                        clock=c,
                    ),
                    description="Generous token bucket for human users",
                ),
                RequestPattern.BOT_PERIODIC: Policy(
                    pattern=RequestPattern.BOT_PERIODIC,
                    limiter_factory=lambda c: LeakyBucketLimiter(
                        capacity=int(base_rate * 0.5),
                        drain_rate=base_rate * 0.5 / window,
                        clock=c,
                    ),
                    description="Strict leaky bucket for bots",
                ),
                RequestPattern.BATCH_RAMP: Policy(
                    pattern=RequestPattern.BATCH_RAMP,
                    limiter_factory=lambda c: SlidingWindowLimiter(
                        max_requests=int(base_rate * 1.5),
                        window_seconds=window,
                        clock=c,
                    ),
                    description="Moderate sliding window for batch ramp-ups",
                ),
                RequestPattern.BATCH_STEADY: Policy(
                    pattern=RequestPattern.BATCH_STEADY,
                    limiter_factory=lambda c: TokenBucketLimiter(
                        capacity=int(base_rate * 1.2),
                        refill_rate=base_rate * 1.2 / window,
                        clock=c,
                    ),
                    description="Moderate token bucket for steady batch",
                ),
            },
            default_policy=Policy(
                pattern=RequestPattern.UNKNOWN,
                limiter_factory=lambda c: FixedWindowLimiter(
                    max_requests=base_rate,
                    window_seconds=window,
                    clock=c,
                ),
                description="Default fixed-window limiter",
            ),
        )

    @staticmethod
    def strict(
        base_rate: int = 60,
        window: float = 60.0,
    ) -> "PolicySet":
        """
        A strict policy set that aggressively throttles bots
        while still accommodating humans.
        """
        return PolicySet(
            policies={
                RequestPattern.HUMAN_BURSTY: Policy(
                    pattern=RequestPattern.HUMAN_BURSTY,
                    limiter_factory=lambda c: TokenBucketLimiter(
                        capacity=base_rate,
                        refill_rate=base_rate / window,
                        clock=c,
                    ),
                    description="Standard token bucket for humans",
                ),
                RequestPattern.BOT_PERIODIC: Policy(
                    pattern=RequestPattern.BOT_PERIODIC,
                    limiter_factory=lambda c: LeakyBucketLimiter(
                        capacity=max(1, int(base_rate * 0.2)),
                        drain_rate=base_rate * 0.2 / window,
                        clock=c,
                    ),
                    description="Very strict leaky bucket for bots",
                ),
                RequestPattern.BATCH_RAMP: Policy(
                    pattern=RequestPattern.BATCH_RAMP,
                    limiter_factory=lambda c: SlidingWindowLimiter(
                        max_requests=int(base_rate * 0.8),
                        window_seconds=window,
                        clock=c,
                    ),
                    description="Tight sliding window for batch ramp-ups",
                ),
                RequestPattern.BATCH_STEADY: Policy(
                    pattern=RequestPattern.BATCH_STEADY,
                    limiter_factory=lambda c: LeakyBucketLimiter(
                        capacity=int(base_rate * 0.6),
                        drain_rate=base_rate * 0.6 / window,
                        clock=c,
                    ),
                    description="Moderate leaky bucket for steady batch",
                ),
            },
            default_policy=Policy(
                pattern=RequestPattern.UNKNOWN,
                limiter_factory=lambda c: FixedWindowLimiter(
                    max_requests=int(base_rate * 0.7),
                    window_seconds=window,
                    clock=c,
                ),
                description="Conservative default limiter",
            ),
        )


@dataclass
class AdaptiveDecision:
    """Result of an adaptive rate limit check."""

    allowed: bool
    pattern: RequestPattern
    confidence: float
    remaining: int
    analysis: PatternAnalysis


class AdaptiveRateLimiter:
    """
    Rate limiter that adapts its behavior based on detected request patterns.

    Each key (client/IP/user) gets its own rhythm detector and, once
    classified, an appropriate rate limiter from the policy set.

    Usage:
        limiter = AdaptiveRateLimiter(PolicySet.lenient())
        decision = limiter.check("client-ip-123")
        if not decision.allowed:
            return 429
    """

    def __init__(
        self,
        policy_set: Optional[PolicySet] = None,
        detector_config: Optional[DetectorConfig] = None,
        clock: Optional[Clock] = None,
        reclassify_interval: int = 50,
    ):
        self._policy_set = policy_set or PolicySet.lenient()
        self._detector_config = detector_config or DetectorConfig()
        self._clock = clock or MonotonicClock()
        self._reclassify_interval = reclassify_interval

        # Per-key state
        self._detectors: Dict[str, RhythmDetector] = {}
        self._limiters: Dict[str, RateLimiter] = {}
        self._patterns: Dict[str, RequestPattern] = {}
        self._request_counts: Dict[str, int] = {}
        self._analyses: Dict[str, PatternAnalysis] = {}

    def _get_detector(self, key: str) -> RhythmDetector:
        if key not in self._detectors:
            self._detectors[key] = RhythmDetector(
                config=self._detector_config, clock=self._clock
            )
        return self._detectors[key]

    def _get_or_create_limiter(self, key: str, pattern: RequestPattern) -> RateLimiter:
        current_pattern = self._patterns.get(key)
        if current_pattern == pattern and key in self._limiters:
            return self._limiters[key]

        # Pattern changed — create new limiter
        policy = self._policy_set.get(pattern)
        if policy is None:
            policy = self._policy_set.default_policy
        if policy is None:
            # Fallback: very basic limiter
            limiter = FixedWindowLimiter(100, 60.0, self._clock)
        else:
            limiter = policy.create_limiter(self._clock)

        self._limiters[key] = limiter
        self._patterns[key] = pattern
        return limiter

    def check(self, key: str = "default", cost: int = 1) -> AdaptiveDecision:
        """
        Check whether a request should be allowed.

        Records the request timestamp, (re)classifies the stream if needed,
        selects the appropriate limiter, and returns the decision.
        """
        detector = self._get_detector(key)
        detector.record()

        count = self._request_counts.get(key, 0) + 1
        self._request_counts[key] = count

        # Analyze / reclassify periodically
        if (
            key not in self._analyses
            or count % self._reclassify_interval == 0
            or count == self._detector_config.min_samples
        ):
            analysis = detector.analyze()
            self._analyses[key] = analysis
        else:
            analysis = self._analyses[key]

        limiter = self._get_or_create_limiter(key, analysis.pattern)
        allowed = limiter.allow(key, cost)

        return AdaptiveDecision(
            allowed=allowed,
            pattern=analysis.pattern,
            confidence=analysis.confidence,
            remaining=limiter.remaining(key),
            analysis=analysis,
        )

    def get_analysis(self, key: str = "default") -> Optional[PatternAnalysis]:
        """Return the latest analysis for a key, or None."""
        return self._analyses.get(key)

    def get_pattern(self, key: str = "default") -> RequestPattern:
        """Return the current classified pattern for a key."""
        return self._patterns.get(key, RequestPattern.UNKNOWN)

    def keys(self) -> List[str]:
        """Return all tracked keys."""
        return list(self._detectors.keys())

    def reset(self, key: str) -> None:
        """Reset all state for a key."""
        self._detectors.pop(key, None)
        self._limiters.pop(key, None)
        self._patterns.pop(key, None)
        self._request_counts.pop(key, None)
        self._analyses.pop(key, None)
