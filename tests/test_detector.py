"""Tests for the rhythm detector."""

import math
import random

import pytest
from tempo.clock import ManualClock
from tempo.detector import (
    DetectorConfig,
    PatternAnalysis,
    RequestPattern,
    RhythmDetector,
    _mean,
    _std,
)


class TestHelpers:
    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_mean(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_std_single(self):
        assert _std([1.0]) == 0.0

    def test_std(self):
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        m = _mean(vals)
        s = _std(vals, m)
        assert abs(s - 2.0) < 0.01


class TestDetectorBasic:
    def test_unknown_with_few_samples(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(5):
            det.record(float(i))
        result = det.analyze()
        assert result.pattern == RequestPattern.UNKNOWN
        assert result.confidence == 0.0

    def test_sample_count(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        assert det.sample_count == 0
        det.record(0.0)
        det.record(1.0)
        assert det.sample_count == 2

    def test_reset(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(20):
            det.record(float(i))
        det.reset()
        assert det.sample_count == 0
        result = det.analyze()
        assert result.pattern == RequestPattern.UNKNOWN

    def test_max_samples_trimming(self):
        config = DetectorConfig(max_samples=20, min_samples=5)
        clock = ManualClock()
        det = RhythmDetector(config=config, clock=clock)
        for i in range(50):
            det.record(float(i))
        assert det.sample_count == 20


class TestBotPeriodicDetection:
    def test_perfectly_periodic(self):
        """Requests at exact 1-second intervals should be detected as bot."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        # 50 requests at exactly 1.0s intervals
        for i in range(50):
            det.record(float(i))
        result = det.analyze()
        assert result.pattern == RequestPattern.BOT_PERIODIC
        assert result.confidence > 0.3
        assert result.cv < 0.1

    def test_nearly_periodic(self):
        """Requests with small jitter around a fixed interval."""
        clock = ManualClock()
        rng = random.Random(42)
        det = RhythmDetector(clock=clock)
        t = 0.0
        for _ in range(60):
            det.record(t)
            t += 1.0 + rng.gauss(0, 0.05)  # 1s ± 50ms
        result = det.analyze()
        assert result.pattern == RequestPattern.BOT_PERIODIC
        assert result.cv < 0.1


class TestHumanBurstyDetection:
    def test_bursty_pattern(self):
        """Simulate human: bursts of rapid clicks separated by pauses."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(7812)  # seed number!
        t = 0.0
        for _ in range(10):  # 10 bursts
            # Pause of 2-8 seconds
            t += rng.uniform(2.0, 8.0)
            # Burst of 3-6 rapid clicks
            burst_size = rng.randint(3, 6)
            for _ in range(burst_size):
                det.record(t)
                t += rng.uniform(0.05, 0.3)
        result = det.analyze()
        assert result.pattern == RequestPattern.HUMAN_BURSTY
        assert result.cv > 0.4

    def test_irregular_spacing(self):
        """Highly irregular inter-arrival times."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(5262)
        t = 0.0
        for _ in range(50):
            det.record(t)
            # Mix of very short and very long intervals
            if rng.random() < 0.4:
                t += rng.uniform(0.01, 0.1)
            else:
                t += rng.uniform(1.0, 5.0)
        result = det.analyze()
        assert result.pattern == RequestPattern.HUMAN_BURSTY


class TestBatchRampDetection:
    def test_accelerating_requests(self):
        """Intervals that steadily decrease (rate ramps up)."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        t = 0.0
        for i in range(40):
            det.record(t)
            # Interval starts at 2.0s and decreases to 0.1s
            interval = max(0.1, 2.0 - i * 0.05)
            t += interval
        result = det.analyze()
        assert result.pattern == RequestPattern.BATCH_RAMP
        assert result.trend_slope < 0  # negative slope = decreasing intervals

    def test_decelerating_requests(self):
        """Intervals that steadily increase (rate ramps down)."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        t = 0.0
        for i in range(40):
            det.record(t)
            interval = 0.1 + i * 0.05
            t += interval
        result = det.analyze()
        assert result.pattern == RequestPattern.BATCH_RAMP
        assert result.trend_slope > 0


class TestBatchSteadyDetection:
    def test_high_throughput_steady(self):
        """Very fast, steady requests (like a bulk import)."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(424)  # palindrome seed!
        t = 0.0
        for _ in range(60):
            det.record(t)
            t += 0.01 + rng.gauss(0, 0.001)  # 10ms ± 1ms
        result = det.analyze()
        # Should be either BATCH_STEADY or BOT_PERIODIC (both are low-CV)
        assert result.pattern in (RequestPattern.BATCH_STEADY, RequestPattern.BOT_PERIODIC)
        assert result.cv < 0.2


class TestEntropyComputation:
    def test_zero_entropy_identical(self):
        """All identical intervals → zero entropy."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(30):
            det.record(float(i))  # all intervals = 1.0
        result = det.analyze()
        assert result.entropy == 0.0

    def test_high_entropy_varied(self):
        """Widely varied intervals → high entropy."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(8528)
        t = 0.0
        for _ in range(50):
            det.record(t)
            t += rng.uniform(0.01, 10.0)
        result = det.analyze()
        assert result.entropy > 2.0


class TestAutocorrelation:
    def test_periodic_has_high_autocorrelation(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(50):
            det.record(float(i))
        result = det.analyze()
        assert result.autocorrelation_peak > 0.9

    def test_random_has_low_autocorrelation(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(770)
        t = 0.0
        for _ in range(50):
            det.record(t)
            t += rng.expovariate(1.0)
        result = det.analyze()
        # Random exponential intervals should have lower autocorrelation
        assert result.autocorrelation_peak < 0.7


class TestCaching:
    def test_cache_invalidated_on_record(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(20):
            det.record(float(i))
        r1 = det.analyze()
        assert det._cache_valid is True
        det.record(20.0)
        assert det._cache_valid is False
        r2 = det.analyze()
        assert r2.sample_count == 21
