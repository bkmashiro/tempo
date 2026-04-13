"""
Tempo - Adaptive Rate Limiter with Rhythm Detection

A rate limiting library that analyzes the temporal pattern of incoming
requests to classify behavioral archetypes and apply adaptive policies.
"""

from tempo.limiters import (
    FixedWindowLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    LeakyBucketLimiter,
)
from tempo.detector import RhythmDetector, RequestPattern
from tempo.adaptive import AdaptiveRateLimiter, Policy, PolicySet
from tempo.clock import Clock, MonotonicClock, ManualClock

__version__ = "0.1.0"

__all__ = [
    "FixedWindowLimiter",
    "SlidingWindowLimiter",
    "TokenBucketLimiter",
    "LeakyBucketLimiter",
    "RhythmDetector",
    "RequestPattern",
    "AdaptiveRateLimiter",
    "Policy",
    "PolicySet",
    "Clock",
    "MonotonicClock",
    "ManualClock",
]
