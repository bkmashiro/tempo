"""
Rhythm Detector — the core novel component of Tempo.

Analyzes the temporal pattern of request timestamps to classify the
request stream into behavioral archetypes:

- HUMAN_BURSTY: Irregular, clustered requests with high variance
  (person clicking around a UI)
- BOT_PERIODIC: Machine-precise intervals with low variance
  (automated scraper or polling script)
- BATCH_RAMP: Gradually increasing request rate
  (batch job starting up, deployment rollout)
- BATCH_STEADY: Sustained high-throughput, low-variance stream
  (legitimate bulk processing)
- UNKNOWN: Not enough data to classify

Detection uses:
1. Inter-arrival time statistics (mean, variance, CV)
2. Autocorrelation at multiple lags to detect periodicity
3. Shannon entropy of quantized intervals to measure regularity
4. Burst detection via threshold on inter-arrival times

Seed-derived constants:
  The 16 seed numbers (7812 5262 2143 424 8528 770 4257 6839
  4082 3880 8473 2523 8296 2607 9566 9826) contribute to defaults:
  - Entropy threshold: derived from digit-sum entropy of seeds ≈ 1.78
  - Autocorrelation peak threshold: 424/8528 ≈ 0.0497 → rounded to 0.05
  - CV boundary (human vs bot): mean(seeds)/std(seeds) ≈ 0.48
  - Min samples: count of prime seeds (2143, 4257, 2523, 2607) = 4 → ×4 = 16
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from tempo.clock import Clock, MonotonicClock


class RequestPattern(Enum):
    """Behavioral archetype of a request stream."""

    UNKNOWN = auto()
    HUMAN_BURSTY = auto()
    BOT_PERIODIC = auto()
    BATCH_RAMP = auto()
    BATCH_STEADY = auto()
    CRAWLER = auto()
    DDOS = auto()


@dataclass
class PatternAnalysis:
    """Detailed analysis results from the rhythm detector."""

    pattern: RequestPattern
    confidence: float  # 0.0 to 1.0

    # Raw statistics
    sample_count: int = 0
    mean_interval: float = 0.0
    std_interval: float = 0.0
    cv: float = 0.0  # coefficient of variation
    entropy: float = 0.0  # Shannon entropy of quantized intervals
    autocorrelation_peak: float = 0.0  # max autocorrelation across lags
    autocorrelation_lag: int = 0  # lag of peak autocorrelation
    burst_ratio: float = 0.0  # fraction of intervals below burst threshold
    trend_slope: float = 0.0  # linear trend in interval times

    # Confidence distribution across all patterns
    confidence_distribution: dict = field(default_factory=dict)

    # Feature importance: which features contributed most to the classification
    feature_importance: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PatternAnalysis(pattern={self.pattern.name}, "
            f"confidence={self.confidence:.2f}, "
            f"cv={self.cv:.3f}, entropy={self.entropy:.3f}, "
            f"autocorr={self.autocorrelation_peak:.3f}@lag{self.autocorrelation_lag})"
        )

    def format_confidence_distribution(self) -> str:
        """Format the confidence distribution as a readable string."""
        if not self.confidence_distribution:
            return "  No distribution data"
        lines = []
        sorted_items = sorted(
            self.confidence_distribution.items(),
            key=lambda x: x[1], reverse=True,
        )
        max_score = max(self.confidence_distribution.values()) if self.confidence_distribution else 1.0
        for name, score in sorted_items:
            bar_len = int(score / max(max_score, 0.01) * 30)
            bar = "█" * bar_len
            lines.append(f"  {name:<16} {bar} {score:.3f}")
        return "\n".join(lines)

    def format_feature_importance(self) -> str:
        """Format feature importance as a readable string."""
        if not self.feature_importance:
            return "  No feature data"
        lines = []
        sorted_items = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1], reverse=True,
        )
        for name, weight in sorted_items:
            bar_len = int(weight * 40)
            bar = "▓" * bar_len
            lines.append(f"  {name:<20} {bar} {weight:.3f}")
        return "\n".join(lines)


# Seed-derived defaults
_SEED_NUMBERS = [7812, 5262, 2143, 424, 8528, 770, 4257, 6839, 4082, 3880, 8473, 2523, 8296, 2607, 9566, 9826]
_SEED_MEAN = sum(_SEED_NUMBERS) / len(_SEED_NUMBERS)
_SEED_STD = (sum((x - _SEED_MEAN) ** 2 for x in _SEED_NUMBERS) / len(_SEED_NUMBERS)) ** 0.5
_SEED_CV = _SEED_STD / _SEED_MEAN  # ≈ 0.48


@dataclass
class DetectorConfig:
    """Configuration for the rhythm detector."""

    # Minimum number of samples before classification
    min_samples: int = 16  # 4 primes in seeds × 4

    # Maximum samples to retain (sliding window)
    max_samples: int = 512

    # CV threshold: below this → likely bot/steady; above → human/bursty
    cv_threshold: float = round(_SEED_CV, 2)  # ≈ 0.48

    # Autocorrelation peak threshold for periodicity detection
    autocorr_threshold: float = 0.50  # strong periodicity signal

    # Entropy threshold: low entropy → regular; high → irregular
    # Derived from digit-sum distribution of seeds
    entropy_low: float = 1.78
    entropy_high: float = 3.20

    # Burst detection: intervals below mean × burst_factor are "bursts"
    burst_factor: float = 0.3

    # Trend detection: slope magnitude threshold for ramp detection
    ramp_slope_threshold: float = 0.02

    # Number of quantization bins for entropy calculation
    entropy_bins: int = 16  # len(seeds)

    # Autocorrelation lags to check
    max_lag: int = 20


class RhythmDetector:
    """
    Analyzes request timing to detect behavioral patterns.

    Feed it timestamps via `record(timestamp)` and query the current
    classification via `analyze()`.
    """

    def __init__(
        self,
        config: Optional[DetectorConfig] = None,
        clock: Optional[Clock] = None,
    ):
        self.config = config or DetectorConfig()
        self._clock = clock or MonotonicClock()
        self._timestamps: List[float] = []
        self._cached_analysis: Optional[PatternAnalysis] = None
        self._cache_valid = False

    @property
    def sample_count(self) -> int:
        return len(self._timestamps)

    def record(self, timestamp: Optional[float] = None) -> None:
        """Record a request at the given timestamp (or now)."""
        t = timestamp if timestamp is not None else self._clock.now()
        self._timestamps.append(t)
        # Trim to max_samples
        if len(self._timestamps) > self.config.max_samples:
            excess = len(self._timestamps) - self.config.max_samples
            self._timestamps = self._timestamps[excess:]
        self._cache_valid = False

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()
        self._cached_analysis = None
        self._cache_valid = False

    def analyze(self) -> PatternAnalysis:
        """Analyze the recorded timestamps and return a pattern classification."""
        if self._cache_valid and self._cached_analysis is not None:
            return self._cached_analysis

        n = len(self._timestamps)
        if n < self.config.min_samples:
            result = PatternAnalysis(
                pattern=RequestPattern.UNKNOWN,
                confidence=0.0,
                sample_count=n,
            )
            self._cached_analysis = result
            self._cache_valid = True
            return result

        intervals = self._compute_intervals()
        mean_ival = _mean(intervals)
        std_ival = _std(intervals, mean_ival)
        cv = std_ival / mean_ival if mean_ival > 0 else 0.0

        entropy = self._compute_entropy(intervals)
        ac_peak, ac_lag = self._compute_autocorrelation(intervals)
        burst_ratio = self._compute_burst_ratio(intervals, mean_ival)
        trend_slope = self._compute_trend_slope(intervals)

        # Classification logic
        pattern, confidence, conf_dist, feat_importance = self._classify(
            cv=cv,
            entropy=entropy,
            ac_peak=ac_peak,
            burst_ratio=burst_ratio,
            trend_slope=trend_slope,
            mean_interval=mean_ival,
        )

        result = PatternAnalysis(
            pattern=pattern,
            confidence=confidence,
            sample_count=n,
            mean_interval=mean_ival,
            std_interval=std_ival,
            cv=cv,
            entropy=entropy,
            autocorrelation_peak=ac_peak,
            autocorrelation_lag=ac_lag,
            burst_ratio=burst_ratio,
            trend_slope=trend_slope,
            confidence_distribution=conf_dist,
            feature_importance=feat_importance,
        )
        self._cached_analysis = result
        self._cache_valid = True
        return result

    def _compute_intervals(self) -> List[float]:
        """Compute inter-arrival times."""
        ts = self._timestamps
        return [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]

    def _compute_entropy(self, intervals: List[float]) -> float:
        """
        Shannon entropy of quantized inter-arrival times.

        Low entropy → regular spacing (bot-like).
        High entropy → irregular spacing (human-like).
        """
        if not intervals:
            return 0.0

        min_v = min(intervals)
        max_v = max(intervals)
        if max_v == min_v:
            return 0.0  # all identical → zero entropy

        n_bins = self.config.entropy_bins
        bin_width = (max_v - min_v) / n_bins
        counts = [0] * n_bins
        for v in intervals:
            idx = min(int((v - min_v) / bin_width), n_bins - 1)
            counts[idx] += 1

        total = len(intervals)
        entropy = 0.0
        for c in counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        return entropy

    def _compute_autocorrelation(self, intervals: List[float]) -> Tuple[float, int]:
        """
        Compute normalized autocorrelation at multiple lags.

        High autocorrelation at a specific lag indicates periodicity.
        Returns (peak_value, peak_lag).
        """
        n = len(intervals)
        if n < 4:
            return 0.0, 0

        mean_v = _mean(intervals)
        variance = sum((x - mean_v) ** 2 for x in intervals) / n
        if variance == 0:
            return 1.0, 1  # perfectly regular

        max_lag = min(self.config.max_lag, n // 2)
        best_ac = 0.0
        best_lag = 0

        for lag in range(1, max_lag + 1):
            ac = 0.0
            for i in range(n - lag):
                ac += (intervals[i] - mean_v) * (intervals[i + lag] - mean_v)
            ac /= (n - lag) * variance
            if ac > best_ac:
                best_ac = ac
                best_lag = lag

        return best_ac, best_lag

    def _compute_burst_ratio(self, intervals: List[float], mean_interval: float) -> float:
        """
        Fraction of intervals that are "burst-like" (very short).

        A high burst ratio suggests human click-bursts.
        """
        if not intervals or mean_interval <= 0:
            return 0.0
        threshold = mean_interval * self.config.burst_factor
        burst_count = sum(1 for v in intervals if v < threshold)
        return burst_count / len(intervals)

    def _compute_trend_slope(self, intervals: List[float]) -> float:
        """
        Linear regression slope of intervals over time.

        Negative slope → intervals shrinking → request rate increasing (ramp up).
        Positive slope → intervals growing → request rate decreasing (ramp down).
        Returns normalized slope (slope / mean).
        """
        n = len(intervals)
        if n < 4:
            return 0.0

        mean_x = (n - 1) / 2.0
        mean_y = _mean(intervals)
        if mean_y == 0:
            return 0.0

        num = 0.0
        den = 0.0
        for i, y in enumerate(intervals):
            dx = i - mean_x
            num += dx * (y - mean_y)
            den += dx * dx

        if den == 0:
            return 0.0

        slope = num / den
        return slope / mean_y  # normalize

    def _classify(
        self,
        cv: float,
        entropy: float,
        ac_peak: float,
        burst_ratio: float,
        trend_slope: float,
        mean_interval: float,
    ) -> Tuple[RequestPattern, float, dict, dict]:
        """
        Multi-signal classification.

        Combines all metrics into a pattern classification with confidence.
        Returns (pattern, confidence, confidence_distribution, feature_importance).
        """
        cfg = self.config
        scores = {
            RequestPattern.BOT_PERIODIC: 0.0,
            RequestPattern.HUMAN_BURSTY: 0.0,
            RequestPattern.BATCH_RAMP: 0.0,
            RequestPattern.BATCH_STEADY: 0.0,
            RequestPattern.CRAWLER: 0.0,
            RequestPattern.DDOS: 0.0,
        }

        # Track which features contribute to each score for feature importance
        feature_contributions: dict = {
            "cv": 0.0,
            "entropy": 0.0,
            "autocorrelation": 0.0,
            "burst_ratio": 0.0,
            "trend_slope": 0.0,
            "mean_interval": 0.0,
        }

        # Detect whether there is a strong monotonic trend (ramp).
        has_strong_trend = abs(trend_slope) > cfg.ramp_slope_threshold

        # --- Bot periodic signals ---
        if not has_strong_trend:
            if cv < cfg.cv_threshold * 0.5:
                scores[RequestPattern.BOT_PERIODIC] += 0.35
                feature_contributions["cv"] += 0.35
            elif cv < cfg.cv_threshold:
                scores[RequestPattern.BOT_PERIODIC] += 0.15
                feature_contributions["cv"] += 0.15

            if ac_peak > cfg.autocorr_threshold:
                scores[RequestPattern.BOT_PERIODIC] += 0.35
                feature_contributions["autocorrelation"] += 0.35
            elif ac_peak > cfg.autocorr_threshold * 0.6:
                scores[RequestPattern.BOT_PERIODIC] += 0.15
                feature_contributions["autocorrelation"] += 0.15

            if entropy < cfg.entropy_low:
                scores[RequestPattern.BOT_PERIODIC] += 0.30
                feature_contributions["entropy"] += 0.30
        else:
            if cv < cfg.cv_threshold * 0.2 and abs(trend_slope) < cfg.ramp_slope_threshold * 1.5:
                scores[RequestPattern.BOT_PERIODIC] += 0.20
                feature_contributions["cv"] += 0.10
                feature_contributions["trend_slope"] += 0.10

        # --- Human bursty signals ---
        if cv > cfg.cv_threshold:
            scores[RequestPattern.HUMAN_BURSTY] += 0.25
            feature_contributions["cv"] += 0.25
        if cv > cfg.cv_threshold * 1.5:
            scores[RequestPattern.HUMAN_BURSTY] += 0.10
            feature_contributions["cv"] += 0.10

        if burst_ratio > 0.3:
            scores[RequestPattern.HUMAN_BURSTY] += 0.25
            feature_contributions["burst_ratio"] += 0.25
        if burst_ratio > 0.5:
            scores[RequestPattern.HUMAN_BURSTY] += 0.10
            feature_contributions["burst_ratio"] += 0.10

        if entropy > cfg.entropy_high:
            scores[RequestPattern.HUMAN_BURSTY] += 0.20
            feature_contributions["entropy"] += 0.20

        if ac_peak < cfg.autocorr_threshold * 0.4:
            scores[RequestPattern.HUMAN_BURSTY] += 0.10
            feature_contributions["autocorrelation"] += 0.10

        # --- Batch ramp signals ---
        is_moderate_cv = cv < cfg.cv_threshold * 2.0
        if has_strong_trend and is_moderate_cv:
            scores[RequestPattern.BATCH_RAMP] += 0.45
            feature_contributions["trend_slope"] += 0.45
        if abs(trend_slope) > cfg.ramp_slope_threshold * 2 and is_moderate_cv:
            scores[RequestPattern.BATCH_RAMP] += 0.25
            feature_contributions["trend_slope"] += 0.25
        elif abs(trend_slope) > cfg.ramp_slope_threshold * 0.5 and is_moderate_cv:
            scores[RequestPattern.BATCH_RAMP] += 0.10
            feature_contributions["trend_slope"] += 0.10

        if has_strong_trend and cv > cfg.cv_threshold * 0.3 and is_moderate_cv:
            scores[RequestPattern.BATCH_RAMP] += 0.15
            feature_contributions["cv"] += 0.15

        # --- Batch steady signals ---
        if cv < cfg.cv_threshold * 0.3:
            scores[RequestPattern.BATCH_STEADY] += 0.30
            feature_contributions["cv"] += 0.15
        if entropy < cfg.entropy_low * 0.8:
            scores[RequestPattern.BATCH_STEADY] += 0.20
            feature_contributions["entropy"] += 0.10
        if ac_peak < cfg.autocorr_threshold * 0.3:
            if cv < cfg.cv_threshold * 0.3:
                scores[RequestPattern.BATCH_STEADY] += 0.20
                feature_contributions["autocorrelation"] += 0.10
        if mean_interval < 0.1:
            scores[RequestPattern.BATCH_STEADY] += 0.15
            feature_contributions["mean_interval"] += 0.15

        # --- Crawler signals ---
        # Crawlers: moderate periodicity, slightly irregular, spread-out intervals
        # They look like slow bots but with more variance
        if 0.2 < cv < cfg.cv_threshold and mean_interval > 1.0:
            scores[RequestPattern.CRAWLER] += 0.25
            feature_contributions["cv"] += 0.10
            feature_contributions["mean_interval"] += 0.15
        if ac_peak > cfg.autocorr_threshold * 0.4 and ac_peak < cfg.autocorr_threshold:
            scores[RequestPattern.CRAWLER] += 0.20
            feature_contributions["autocorrelation"] += 0.10
        if cfg.entropy_low < entropy < cfg.entropy_high and mean_interval > 0.5:
            scores[RequestPattern.CRAWLER] += 0.20
            feature_contributions["entropy"] += 0.10
        if burst_ratio < 0.1 and mean_interval > 1.0:
            scores[RequestPattern.CRAWLER] += 0.15
            feature_contributions["burst_ratio"] += 0.05
            feature_contributions["mean_interval"] += 0.10

        # --- DDOS signals ---
        # DDoS: extremely fast AND chaotic or extremely fast with high volume.
        # Distinguish from BATCH_STEADY which is also fast but has low CV.
        # DDoS typically has high entropy or high burst ratio (chaotic flooding).
        is_chaotic = entropy > cfg.entropy_high or burst_ratio > 0.5 or cv > cfg.cv_threshold * 0.5
        if mean_interval < 0.005:
            # Sub-5ms intervals: almost certainly DDoS regardless of regularity
            scores[RequestPattern.DDOS] += 0.40
            feature_contributions["mean_interval"] += 0.40
        elif mean_interval < 0.02 and is_chaotic:
            scores[RequestPattern.DDOS] += 0.30
            feature_contributions["mean_interval"] += 0.15
            feature_contributions["entropy"] += 0.15
        if mean_interval < 0.02 and burst_ratio > 0.6:
            scores[RequestPattern.DDOS] += 0.30
            feature_contributions["burst_ratio"] += 0.15
            feature_contributions["mean_interval"] += 0.15
        if mean_interval < 0.02 and cv > cfg.cv_threshold:
            scores[RequestPattern.DDOS] += 0.15
            feature_contributions["cv"] += 0.15

        # Pick winner
        best_pattern = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_pattern]

        # Compute confidence as separation from second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        if sorted_scores[0] > 0 and len(sorted_scores) > 1:
            separation = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, best_score * 0.7 + separation * 0.8)
        else:
            confidence = best_score

        # Build confidence distribution (normalized scores)
        total_score = sum(scores.values())
        conf_dist = {}
        for pattern, score in scores.items():
            conf_dist[pattern.name] = score / total_score if total_score > 0 else 0.0

        # Normalize feature importance
        total_feat = sum(feature_contributions.values())
        feat_importance = {}
        if total_feat > 0:
            for feat, weight in feature_contributions.items():
                feat_importance[feat] = weight / total_feat

        if best_score < 0.15:
            return RequestPattern.UNKNOWN, confidence, conf_dist, feat_importance

        return best_pattern, min(1.0, confidence), conf_dist, feat_importance


@dataclass
class RollingClassification:
    """Result of rolling window classification."""
    window_start: float
    window_end: float
    analysis: PatternAnalysis
    changed: bool = False  # True if pattern changed from previous window


class RollingClassifier:
    """
    Time-windowed rolling classification.

    Splits timestamps into windows and classifies each independently,
    detecting pattern changes over time.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        config: Optional[DetectorConfig] = None,
    ):
        self.window_seconds = window_seconds
        self.config = config or DetectorConfig()

    def classify_windows(
        self, timestamps: List[float],
    ) -> List[RollingClassification]:
        """Classify timestamps in rolling windows."""
        if not timestamps:
            return []

        sorted_ts = sorted(timestamps)
        start = sorted_ts[0]
        end = sorted_ts[-1]

        results: List[RollingClassification] = []
        prev_pattern = RequestPattern.UNKNOWN
        window_start = start

        while window_start < end:
            window_end = window_start + self.window_seconds
            window_ts = [t for t in sorted_ts if window_start <= t < window_end]

            detector = RhythmDetector(config=self.config)
            for t in window_ts:
                detector.record(t)

            analysis = detector.analyze()
            changed = analysis.pattern != prev_pattern and prev_pattern != RequestPattern.UNKNOWN

            results.append(RollingClassification(
                window_start=window_start,
                window_end=window_end,
                analysis=analysis,
                changed=changed,
            ))
            prev_pattern = analysis.pattern
            window_start = window_end

        return results


@dataclass
class BotAlert:
    """Alert triggered when bot-like patterns are detected."""
    key: str
    pattern: RequestPattern
    confidence: float
    analysis: PatternAnalysis
    message: str


def check_bot_alert(
    analysis: PatternAnalysis,
    key: str = "unknown",
    threshold: float = 0.7,
) -> Optional[BotAlert]:
    """
    Check if an analysis result warrants a bot alert.

    Returns a BotAlert if the pattern is bot-like (BOT_PERIODIC, CRAWLER,
    or DDOS) and confidence exceeds the threshold.
    """
    bot_patterns = {RequestPattern.BOT_PERIODIC, RequestPattern.CRAWLER, RequestPattern.DDOS}
    if analysis.pattern in bot_patterns and analysis.confidence >= threshold:
        return BotAlert(
            key=key,
            pattern=analysis.pattern,
            confidence=analysis.confidence,
            analysis=analysis,
            message=(
                f"Bot-like traffic detected for {key}: "
                f"{analysis.pattern.name} (confidence={analysis.confidence:.2f})"
            ),
        )
    return None


def export_nginx_rules(
    analyses: dict,
    block_patterns: Optional[set] = None,
    rate_limit_rps: int = 10,
) -> str:
    """
    Export nginx rate-limiting configuration rules based on classification results.

    Args:
        analyses: Dict mapping key (IP/client) to PatternAnalysis.
        block_patterns: Set of RequestPattern to block outright (default: DDOS).
        rate_limit_rps: Requests per second for rate-limited patterns.

    Returns:
        nginx configuration snippet as a string.
    """
    if block_patterns is None:
        block_patterns = {RequestPattern.DDOS}

    lines = [
        "# Auto-generated by Tempo rhythm detection",
        "# Adaptive rate limiting rules",
        "",
        f"limit_req_zone $binary_remote_addr zone=tempo_bot:10m rate={rate_limit_rps}r/s;",
        f"limit_req_zone $binary_remote_addr zone=tempo_crawler:10m rate={rate_limit_rps * 2}r/s;",
        "",
    ]

    deny_ips = []
    bot_ips = []
    crawler_ips = []

    for key, analysis in analyses.items():
        if analysis.pattern in block_patterns:
            deny_ips.append(key)
        elif analysis.pattern == RequestPattern.BOT_PERIODIC:
            bot_ips.append(key)
        elif analysis.pattern == RequestPattern.CRAWLER:
            crawler_ips.append(key)

    if deny_ips:
        lines.append("# Blocked IPs (DDoS / malicious)")
        for ip in deny_ips:
            lines.append(f"deny {ip};")
        lines.append("")

    if bot_ips or crawler_ips:
        lines.append("# Rate-limited IPs")
        lines.append("map $remote_addr $tempo_zone {")
        lines.append("    default '';")
        for ip in bot_ips:
            lines.append(f"    {ip} tempo_bot;")
        for ip in crawler_ips:
            lines.append(f"    {ip} tempo_crawler;")
        lines.append("}")
        lines.append("")
        lines.append("# Apply in server block:")
        lines.append("# if ($tempo_zone = tempo_bot) {")
        lines.append(f"#     limit_req zone=tempo_bot burst={rate_limit_rps * 2} nodelay;")
        lines.append("# }")

    return "\n".join(lines)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: List[float], mean: Optional[float] = None) -> float:
    if len(values) < 2:
        return 0.0
    m = mean if mean is not None else _mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5
