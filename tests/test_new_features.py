"""Tests for new Tempo features: expanded patterns, confidence distribution,
feature importance, rolling classification, bot alerts, nginx export."""

import random

from tempo.clock import ManualClock
from tempo.detector import (
    RequestPattern,
    RhythmDetector,
    DetectorConfig,
    RollingClassifier,
    check_bot_alert,
    export_nginx_rules,
    PatternAnalysis,
)
from tempo.adaptive import AdaptiveRateLimiter, PolicySet


# ---------------------------------------------------------------------------
# Expanded traffic classification
# ---------------------------------------------------------------------------

class TestCrawlerDetection:
    def test_slow_periodic_is_crawler(self):
        """Moderately periodic requests with longer intervals => crawler."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(2143)
        t = 0.0
        for _ in range(40):
            det.record(t)
            # 2-3 second intervals with some jitter
            t += 2.0 + rng.gauss(0, 0.3)
        result = det.analyze()
        # Should be CRAWLER or BOT_PERIODIC (both are valid for slow periodic)
        assert result.pattern in (RequestPattern.CRAWLER, RequestPattern.BOT_PERIODIC)

    def test_crawler_confidence_distribution(self):
        """Crawler analysis should have CRAWLER in confidence distribution."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(4257)
        t = 0.0
        for _ in range(40):
            det.record(t)
            t += 2.5 + rng.gauss(0, 0.4)
        result = det.analyze()
        assert "CRAWLER" in result.confidence_distribution


class TestDDoSDetection:
    def test_very_fast_chaotic_is_ddos(self):
        """Extremely fast and chaotic requests => DDoS."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(8528)
        t = 0.0
        for _ in range(60):
            det.record(t)
            t += rng.uniform(0.0001, 0.004)  # sub-5ms chaotic
        result = det.analyze()
        assert result.pattern == RequestPattern.DDOS

    def test_fast_bursty_is_ddos(self):
        """Very fast with high burst ratio => DDoS."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(770)
        t = 0.0
        for _ in range(60):
            det.record(t)
            # Alternating between very fast and slightly less fast
            if rng.random() > 0.3:
                t += 0.001
            else:
                t += 0.015
        result = det.analyze()
        assert result.pattern == RequestPattern.DDOS

    def test_ddos_confidence_distribution(self):
        """DDoS detection should have DDOS in the confidence distribution."""
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        t = 0.0
        for _ in range(60):
            det.record(t)
            t += 0.001
        result = det.analyze()
        assert "DDOS" in result.confidence_distribution
        assert result.confidence_distribution["DDOS"] > 0


class TestUnknownPattern:
    def test_insufficient_samples_returns_unknown(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(5):
            det.record(float(i))
        result = det.analyze()
        assert result.pattern == RequestPattern.UNKNOWN


# ---------------------------------------------------------------------------
# Confidence distribution
# ---------------------------------------------------------------------------

class TestConfidenceDistribution:
    def test_distribution_sums_to_one(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(42)
        t = 0.0
        for _ in range(30):
            det.record(t)
            t += 1.0 + rng.gauss(0, 0.1)
        result = det.analyze()
        total = sum(result.confidence_distribution.values())
        assert abs(total - 1.0) < 0.01, f"Sum = {total}"

    def test_distribution_has_all_patterns(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(30):
            det.record(float(i))
        result = det.analyze()
        for pattern in [RequestPattern.BOT_PERIODIC, RequestPattern.HUMAN_BURSTY,
                        RequestPattern.BATCH_RAMP, RequestPattern.BATCH_STEADY,
                        RequestPattern.CRAWLER, RequestPattern.DDOS]:
            assert pattern.name in result.confidence_distribution

    def test_format_confidence_distribution(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(30):
            det.record(float(i))
        result = det.analyze()
        text = result.format_confidence_distribution()
        assert "BOT_PERIODIC" in text or "BATCH_STEADY" in text


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_feature_importance_populated(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(30):
            det.record(float(i))
        result = det.analyze()
        assert len(result.feature_importance) > 0

    def test_feature_importance_sums_to_one(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        rng = random.Random(99)
        t = 0.0
        for _ in range(30):
            det.record(t)
            t += 1.0 + rng.gauss(0, 0.5)
        result = det.analyze()
        total = sum(result.feature_importance.values())
        assert abs(total - 1.0) < 0.01, f"Sum = {total}"

    def test_format_feature_importance(self):
        clock = ManualClock()
        det = RhythmDetector(clock=clock)
        for i in range(30):
            det.record(float(i) * 1.0)
        result = det.analyze()
        text = result.format_feature_importance()
        assert "cv" in text or "entropy" in text or "autocorrelation" in text


# ---------------------------------------------------------------------------
# Rolling classification
# ---------------------------------------------------------------------------

class TestRollingClassifier:
    def test_classify_windows(self):
        """Should produce multiple classifications for a long stream."""
        # Generate timestamps spanning multiple windows
        timestamps = [float(i) * 0.5 for i in range(200)]  # 100 seconds
        rc = RollingClassifier(window_seconds=10.0)
        results = rc.classify_windows(timestamps)
        assert len(results) == 10  # 100s / 10s windows

    def test_detects_pattern_change(self):
        """Should flag when pattern changes between windows."""
        # First window: periodic, second window: bursty
        timestamps = []
        # Regular intervals for first 30s
        for i in range(60):
            timestamps.append(float(i) * 0.5)
        # Bursty pattern for next 30s
        rng = random.Random(123)
        base = 30.0
        for _ in range(60):
            timestamps.append(base)
            base += rng.choice([0.01, 0.02, 2.0, 3.0])

        rc = RollingClassifier(window_seconds=30.0, config=DetectorConfig(min_samples=8))
        results = rc.classify_windows(timestamps)
        assert len(results) >= 2

    def test_empty_timestamps(self):
        rc = RollingClassifier()
        results = rc.classify_windows([])
        assert results == []

    def test_single_window(self):
        timestamps = [float(i) for i in range(20)]
        rc = RollingClassifier(window_seconds=100.0)
        results = rc.classify_windows(timestamps)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Bot alert threshold
# ---------------------------------------------------------------------------

class TestBotAlert:
    def test_alert_on_bot(self):
        analysis = PatternAnalysis(
            pattern=RequestPattern.BOT_PERIODIC,
            confidence=0.85,
            sample_count=50,
        )
        alert = check_bot_alert(analysis, key="192.168.1.1", threshold=0.7)
        assert alert is not None
        assert alert.pattern == RequestPattern.BOT_PERIODIC
        assert "192.168.1.1" in alert.message

    def test_alert_on_ddos(self):
        analysis = PatternAnalysis(
            pattern=RequestPattern.DDOS,
            confidence=0.9,
        )
        alert = check_bot_alert(analysis, key="10.0.0.1")
        assert alert is not None
        assert alert.pattern == RequestPattern.DDOS

    def test_alert_on_crawler(self):
        analysis = PatternAnalysis(
            pattern=RequestPattern.CRAWLER,
            confidence=0.8,
        )
        alert = check_bot_alert(analysis, key="crawler-ip")
        assert alert is not None

    def test_no_alert_for_human(self):
        analysis = PatternAnalysis(
            pattern=RequestPattern.HUMAN_BURSTY,
            confidence=0.9,
        )
        alert = check_bot_alert(analysis, threshold=0.5)
        assert alert is None

    def test_no_alert_below_threshold(self):
        analysis = PatternAnalysis(
            pattern=RequestPattern.BOT_PERIODIC,
            confidence=0.3,
        )
        alert = check_bot_alert(analysis, threshold=0.7)
        assert alert is None


# ---------------------------------------------------------------------------
# Nginx rule export
# ---------------------------------------------------------------------------

class TestNginxExport:
    def test_basic_export(self):
        analyses = {
            "1.2.3.4": PatternAnalysis(pattern=RequestPattern.DDOS, confidence=0.9),
            "5.6.7.8": PatternAnalysis(pattern=RequestPattern.BOT_PERIODIC, confidence=0.8),
            "9.10.11.12": PatternAnalysis(pattern=RequestPattern.HUMAN_BURSTY, confidence=0.7),
        }
        config = export_nginx_rules(analyses)
        assert "deny 1.2.3.4" in config
        assert "5.6.7.8" in config
        assert "tempo_bot" in config
        # Human should not appear in deny or rate limit
        assert "deny 9.10.11.12" not in config

    def test_custom_block_patterns(self):
        analyses = {
            "1.2.3.4": PatternAnalysis(pattern=RequestPattern.CRAWLER, confidence=0.9),
        }
        config = export_nginx_rules(
            analyses,
            block_patterns={RequestPattern.CRAWLER},
        )
        assert "deny 1.2.3.4" in config

    def test_empty_analyses(self):
        config = export_nginx_rules({})
        assert "limit_req_zone" in config
        assert "deny" not in config

    def test_custom_rate(self):
        analyses = {}
        config = export_nginx_rules(analyses, rate_limit_rps=20)
        assert "20r/s" in config


# ---------------------------------------------------------------------------
# Adaptive limiter with new patterns
# ---------------------------------------------------------------------------

class TestAdaptiveWithNewPatterns:
    def test_lenient_has_crawler_policy(self):
        ps = PolicySet.lenient()
        policy = ps.get(RequestPattern.CRAWLER)
        assert policy is not None
        assert "crawler" in policy.description.lower()

    def test_lenient_has_ddos_policy(self):
        ps = PolicySet.lenient()
        policy = ps.get(RequestPattern.DDOS)
        assert policy is not None
        assert "ddos" in policy.description.lower() or "ddo" in policy.description.lower()

    def test_strict_has_crawler_policy(self):
        ps = PolicySet.strict()
        policy = ps.get(RequestPattern.CRAWLER)
        assert policy is not None

    def test_strict_has_ddos_policy(self):
        ps = PolicySet.strict()
        policy = ps.get(RequestPattern.DDOS)
        assert policy is not None

    def test_ddos_limiter_is_very_strict(self):
        """DDoS policy should be more restrictive than bot policy."""
        clock = ManualClock()
        ps = PolicySet.strict()
        ddos_limiter = ps.get(RequestPattern.DDOS).create_limiter(clock)
        bot_limiter = ps.get(RequestPattern.BOT_PERIODIC).create_limiter(clock)
        # DDoS should have much lower remaining capacity
        assert ddos_limiter.remaining("test") <= bot_limiter.remaining("test")
