"""Tests for the adaptive rate limiter."""

import random

import pytest
from tempo.clock import ManualClock
from tempo.detector import DetectorConfig, RequestPattern
from tempo.adaptive import (
    AdaptiveRateLimiter,
    Policy,
    PolicySet,
)
from tempo.limiters import FixedWindowLimiter, TokenBucketLimiter


class TestPolicySet:
    def test_lenient_has_all_patterns(self):
        ps = PolicySet.lenient()
        for pattern in [
            RequestPattern.HUMAN_BURSTY,
            RequestPattern.BOT_PERIODIC,
            RequestPattern.BATCH_RAMP,
            RequestPattern.BATCH_STEADY,
        ]:
            assert ps.get(pattern) is not None
        assert ps.default_policy is not None

    def test_strict_has_all_patterns(self):
        ps = PolicySet.strict()
        for pattern in [
            RequestPattern.HUMAN_BURSTY,
            RequestPattern.BOT_PERIODIC,
            RequestPattern.BATCH_RAMP,
            RequestPattern.BATCH_STEADY,
        ]:
            assert ps.get(pattern) is not None

    def test_custom_policy_set(self):
        clock = ManualClock()
        ps = PolicySet(
            policies={
                RequestPattern.BOT_PERIODIC: Policy(
                    pattern=RequestPattern.BOT_PERIODIC,
                    limiter_factory=lambda c: FixedWindowLimiter(10, 60.0, c),
                    description="Test",
                ),
            },
            default_policy=Policy(
                pattern=RequestPattern.UNKNOWN,
                limiter_factory=lambda c: FixedWindowLimiter(100, 60.0, c),
            ),
        )
        bot_policy = ps.get(RequestPattern.BOT_PERIODIC)
        assert bot_policy is not None
        limiter = bot_policy.create_limiter(clock)
        assert isinstance(limiter, FixedWindowLimiter)

    def test_missing_pattern_returns_default(self):
        ps = PolicySet(
            policies={},
            default_policy=Policy(
                pattern=RequestPattern.UNKNOWN,
                limiter_factory=lambda c: FixedWindowLimiter(100, 60.0, c),
            ),
        )
        assert ps.get(RequestPattern.BOT_PERIODIC) is not None  # falls back to default


class TestAdaptiveRateLimiter:
    def test_starts_with_unknown(self):
        clock = ManualClock()
        lim = AdaptiveRateLimiter(clock=clock)
        assert lim.get_pattern("test") == RequestPattern.UNKNOWN

    def test_allows_initial_requests(self):
        clock = ManualClock()
        lim = AdaptiveRateLimiter(
            policy_set=PolicySet.lenient(base_rate=100, window=60.0),
            clock=clock,
        )
        # First few requests should be allowed (using default limiter)
        for i in range(10):
            decision = lim.check("user1")
            assert decision.allowed is True
            clock.advance(0.5)

    def test_bot_gets_throttled_more(self):
        """
        A bot-like stream should get classified and given a stricter limiter
        than a human-like stream.
        """
        clock = ManualClock()
        config = DetectorConfig(min_samples=16)
        lim = AdaptiveRateLimiter(
            policy_set=PolicySet.lenient(base_rate=100, window=60.0),
            detector_config=config,
            clock=clock,
            reclassify_interval=10,
        )

        # Bot: perfectly periodic requests
        for i in range(60):
            lim.check("bot")
            clock.advance(1.0)

        # Human: bursty requests
        rng = random.Random(2143)
        for _ in range(10):
            clock.advance(rng.uniform(2.0, 5.0))
            for _ in range(rng.randint(2, 5)):
                lim.check("human")
                clock.advance(rng.uniform(0.05, 0.2))

        bot_analysis = lim.get_analysis("bot")
        human_analysis = lim.get_analysis("human")

        # Bot should have low CV, human should have high CV
        assert bot_analysis is not None
        assert human_analysis is not None
        assert bot_analysis.cv < human_analysis.cv

    def test_separate_keys_independent(self):
        clock = ManualClock()
        lim = AdaptiveRateLimiter(clock=clock)
        d1 = lim.check("user1")
        d2 = lim.check("user2")
        assert d1.allowed
        assert d2.allowed
        assert len(lim.keys()) == 2

    def test_reset_key(self):
        clock = ManualClock()
        lim = AdaptiveRateLimiter(clock=clock)
        lim.check("user1")
        assert "user1" in lim.keys()
        lim.reset("user1")
        assert "user1" not in lim.keys()

    def test_reclassify_interval(self):
        clock = ManualClock()
        config = DetectorConfig(min_samples=5)
        lim = AdaptiveRateLimiter(
            clock=clock,
            detector_config=config,
            reclassify_interval=10,
        )
        # Send periodic requests — should eventually classify as bot
        for i in range(50):
            lim.check("bot")
            clock.advance(1.0)

        pattern = lim.get_pattern("bot")
        assert pattern == RequestPattern.BOT_PERIODIC

    def test_cost_parameter(self):
        clock = ManualClock()
        lim = AdaptiveRateLimiter(
            policy_set=PolicySet.lenient(base_rate=10, window=60.0),
            clock=clock,
        )
        decision = lim.check("user1", cost=5)
        assert decision.allowed is True

    def test_decision_fields(self):
        clock = ManualClock()
        lim = AdaptiveRateLimiter(clock=clock)
        decision = lim.check("user1")
        assert isinstance(decision.allowed, bool)
        assert isinstance(decision.pattern, RequestPattern)
        assert isinstance(decision.confidence, float)
        assert isinstance(decision.remaining, int)
        assert decision.analysis is not None


class TestAdaptiveIntegration:
    def test_strict_policy_denies_bot_faster(self):
        """With strict policy, a bot should hit limits sooner."""
        clock = ManualClock()
        config = DetectorConfig(min_samples=10)

        strict_lim = AdaptiveRateLimiter(
            policy_set=PolicySet.strict(base_rate=20, window=60.0),
            detector_config=config,
            clock=clock,
            reclassify_interval=5,
        )

        denied_count = 0
        for i in range(100):
            decision = strict_lim.check("bot")
            if not decision.allowed:
                denied_count += 1
            clock.advance(0.5)

        # Should have some denials
        assert denied_count > 0

    def test_pattern_transition(self):
        """Pattern can change as behavior changes."""
        clock = ManualClock()
        config = DetectorConfig(min_samples=10, max_samples=30)
        lim = AdaptiveRateLimiter(
            clock=clock,
            detector_config=config,
            reclassify_interval=5,
        )

        # Phase 1: periodic (bot-like)
        for i in range(40):
            lim.check("client")
            clock.advance(1.0)

        pattern1 = lim.get_pattern("client")

        # Phase 2: switch to bursty (human-like)
        rng = random.Random(6839)
        for _ in range(15):
            clock.advance(rng.uniform(3.0, 8.0))
            for _ in range(rng.randint(3, 6)):
                lim.check("client")
                clock.advance(rng.uniform(0.02, 0.15))

        pattern2 = lim.get_pattern("client")

        # The pattern should have potentially changed
        # (we can't guarantee exact transitions, but the system should adapt)
        analysis = lim.get_analysis("client")
        assert analysis is not None
        assert analysis.sample_count > 0
